"""Hard context-window budgeting for fully assembled LLM requests."""

from __future__ import annotations

from protocore.contracts.llm import LLMContextWindowExceeded, LLMRequest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.context.compaction import estimate_history_tokens_uncalibrated
from protocore.runtime.tool_surface import read_tool_surface, tool_surface_tokens


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
    max_tokens = fit_max_tokens(
        prompt_tokens=prompt_tokens,
        requested_max_tokens=request.max_tokens,
        context_window=rc.model_context_window,
        safety_tokens=rc.request_context_safety_tokens,
    )
    if max_tokens == request.max_tokens:
        return request
    return request.model_copy(update={"max_tokens": max_tokens})


__all__ = [
    "estimate_request_prompt_tokens",
    "estimate_request_prompt_tokens_uncalibrated",
    "fit_max_tokens",
    "fit_request_to_context",
]
