"""Hard context-window budgeting for fully assembled LLM requests.

Two ways to size a request. The estimate is local and free, and it can run
several times short of a real tokenizer on dense text. A provider that
implements :class:`~protocore.contracts.llm.IRequestTokenCounter` can say what
the request actually renders to; it is asked only when the estimate is close
enough to a limit for the difference to decide something, and what it says is
kept per request content so a retry of the same request is not counted twice.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from protocore.contracts.llm import LLMContextWindowExceeded, LLMRequest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.logging_utils import get_logger
from protocore.runtime.context.compaction import estimate_history_tokens_uncalibrated
from protocore.runtime.tool_surface import read_tool_surface, tool_surface_tokens

_logger = get_logger(__name__)

#: What a request's size does not depend on. The output cap and the sampling
#: temperature do not change the rendered prompt, and the observability
#: context never reaches the wire; leaving them out lets a refit of the same
#: prompt with a smaller cap reuse the count it already has.
_SIZE_INDEPENDENT_FIELDS: frozenset[str] = frozenset(
    {"max_tokens", "temperature", "observability"}
)

RequestTokenCount = Callable[[LLMRequest], Awaitable[int | None]]


def estimate_request_prompt_tokens_uncalibrated(
    request: LLMRequest,
    rc: LoopConstants,
) -> int:
    """Estimate the complete request before adaptive calibration."""
    surface = read_tool_surface(request.tools)
    raw_tokens = estimate_history_tokens_uncalibrated(list(request.messages), rc)
    raw_tokens += tool_surface_tokens(surface, rc)
    return raw_tokens


def fit_max_tokens(
    *,
    prompt_tokens: int,
    requested_max_tokens: int,
    context_window: int,
    safety_tokens: int = 0,
) -> int:
    """Return the largest output cap after a provider-framing safety margin."""
    usable_window = context_window - safety_tokens
    if prompt_tokens >= usable_window:
        raise LLMContextWindowExceeded(
            "estimated prompt fills the usable model context window "
            f"({prompt_tokens} >= {usable_window}; context={context_window}, "
            f"safety={safety_tokens})"
        )
    return min(requested_max_tokens, usable_window - prompt_tokens)


def estimate_request_prompt_tokens(
    request: LLMRequest,
    rc: LoopConstants,
) -> int:
    """Estimate the current complete prompt with adaptive calibration."""
    return round(estimate_request_prompt_tokens_uncalibrated(request, rc) * rc.token_estimate_calibration)


def fit_request_to_context(
    request: LLMRequest,
    rc: LoopConstants,
) -> LLMRequest:
    """Clip one built request so its prompt and output fit the hard window."""
    prompt_tokens = estimate_request_prompt_tokens(request, rc)
    return _fit_to_prompt_tokens(request, rc, prompt_tokens)


def _fit_to_prompt_tokens(
    request: LLMRequest,
    rc: LoopConstants,
    prompt_tokens: int,
) -> LLMRequest:
    max_tokens = fit_max_tokens(
        prompt_tokens=prompt_tokens,
        requested_max_tokens=request.max_tokens,
        context_window=rc.model_context_window,
        safety_tokens=rc.request_context_safety_tokens,
    )
    if max_tokens == request.max_tokens:
        return request
    return request.model_copy(update={"max_tokens": max_tokens})


def request_token_counter(provider: object) -> RequestTokenCount | None:
    """The provider's exact request counter, or ``None`` when it has none.

    Looked up on the provider's CLASS rather than on the instance. A test double
    built on a mock answers every attribute an instance is asked for, and the
    capability must not appear on a provider because a mock invented it.
    """
    if getattr(type(provider), "count_request_tokens", None) is None:
        return None
    return cast(RequestTokenCount, provider.count_request_tokens)  # type: ignore[attr-defined]


def near_limit(estimate: int, limit: int, rc: LoopConstants) -> bool:
    """Whether ``estimate`` is close enough to ``limit`` to be worth counting."""
    return estimate >= limit * (1.0 - rc.exact_token_count_margin_ratio)


class ExactTokenCountCache:
    """Exact counts one run has already paid for, keyed by request content."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: OrderedDict[str, int] = OrderedDict()

    def get(self, key: str) -> int | None:
        count = self._entries.get(key)
        if count is not None:
            self._entries.move_to_end(key)
        return count

    def put(self, key: str, count: int, *, max_entries: int) -> None:
        self._entries[key] = count
        self._entries.move_to_end(key)
        while len(self._entries) > max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def request_content_key(request: LLMRequest) -> str:
    """A digest of everything in ``request`` that decides its rendered size."""
    payload = request.model_dump_json(exclude=set(_SIZE_INDEPENDENT_FIELDS))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def count_request_tokens_exactly(
    request: LLMRequest,
    provider: object,
    rc: LoopConstants,
    *,
    cache: ExactTokenCountCache | None = None,
) -> int | None:
    """Ask ``provider`` what ``request`` renders to; ``None`` when it cannot say.

    ``None`` covers a provider without the capability, the capability switched
    off, an endpoint that has no counting route, and a count that failed. A
    failure is logged; the others are the ordinary state of most providers and
    are not.
    """
    if not rc.exact_token_count_enabled:
        return None
    counter = request_token_counter(provider)
    if counter is None:
        return None
    key: str | None
    try:
        key = request_content_key(request)
    except (TypeError, ValueError):
        key = None
    if key is not None and cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    try:
        measured = await counter(request)
    except Exception as exc:
        _logger.warning(
            "exact request token count failed for model=%s; using the estimate (err=%s)",
            request.model,
            exc,
        )
        return None
    if measured is None:
        return None
    if isinstance(measured, bool) or not isinstance(measured, int) or measured <= 0:
        _logger.warning(
            "exact request token count for model=%s returned %r; using the estimate",
            request.model,
            measured,
        )
        return None
    if key is not None and cache is not None:
        cache.put(key, measured, max_entries=rc.exact_token_count_cache_max_entries)
    return measured


@dataclass(frozen=True, slots=True)
class FittedRequest:
    """A request fitted to the window, and the sizes the fit was made from."""

    request: LLMRequest
    #: The heuristic's own count of the request, before calibration.
    raw_estimate: int
    #: The calibrated estimate.
    estimate: int
    #: What the provider said the request renders to, when it was asked.
    measured: int | None


async def fit_request_to_context_measured(
    request: LLMRequest,
    rc: LoopConstants,
    provider: object,
    *,
    cache: ExactTokenCountCache | None = None,
) -> FittedRequest:
    """:func:`fit_request_to_context`, sized by the provider near the edge.

    The limit that matters for the fit is the prompt size at which the output
    cap starts being clipped: the usable window less the requested cap. Below
    ``limit * (1 - exact_token_count_margin_ratio)`` the estimate decides alone
    and nothing leaves the process; at or above it, a provider that can count
    the rendered request is asked, and its number replaces the estimate. The
    request itself is not changed by being counted: with no counter, or with a
    count that failed, the result is exactly what :func:`fit_request_to_context`
    returns.
    """
    estimate = estimate_request_prompt_tokens(request, rc)
    # Recovered from the calibrated figure rather than estimated a second time:
    # the two differ only by the factor, and a long history is not walked twice
    # per call for a number that is only read if the provider rejects it.
    raw = round(estimate / rc.token_estimate_calibration)
    measured: int | None = None
    clip_limit = (
        rc.model_context_window - rc.request_context_safety_tokens - request.max_tokens
    )
    if near_limit(estimate, clip_limit, rc):
        measured = await count_request_tokens_exactly(request, provider, rc, cache=cache)
    if measured is not None:
        _logger.warning(
            "DIAG request_budget.exact_count model=%s estimate=%d measured=%d "
            "drift_ratio=%.3f",
            request.model,
            estimate,
            measured,
            measured / estimate if estimate > 0 else 0.0,
        )
    fitted = _fit_to_prompt_tokens(
        request, rc, measured if measured is not None else estimate
    )
    return FittedRequest(
        request=fitted,
        raw_estimate=raw,
        estimate=estimate,
        measured=measured,
    )


__all__ = [
    "ExactTokenCountCache",
    "FittedRequest",
    "RequestTokenCount",
    "count_request_tokens_exactly",
    "estimate_request_prompt_tokens",
    "estimate_request_prompt_tokens_uncalibrated",
    "fit_max_tokens",
    "fit_request_to_context",
    "fit_request_to_context_measured",
    "near_limit",
    "request_content_key",
    "request_token_counter",
]
