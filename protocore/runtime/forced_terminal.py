"""The terminal call a run owes once its answer is delivered.

A run whose contract ends in a terminal tool, and whose model has written its
answer as ordinary prose and stopped, owes exactly one more thing: the call.
Asking for it in words is a request the model may read differently. Measured on
a thinking model, the request was answered with a round of reasoning and no
output; the runtime then asked the model to continue, and the model, reading
"continue" under a transcript whose summary listed earlier questions, started a
second answer to one of them. The reader saw two answers to one question.

So the call is not asked for, it is forced. From the moment the answer is
delivered, every request the run makes names the terminal tool in the
provider's native ``tool_choice`` and appends nothing to the transcript, so no
request can be read as an invitation to write. The forcing is bounded by
``terminal_tool_forced_max_attempts``; when the bound is spent the run completes
on the answer it already delivered, which is what the terminal call would have
sealed.

This module holds only the state: whether the call is being forced and how many
requests the forcing has spent, both on the engine and both snapshot-persisted.
Where the state is read and what the loop does with it lives in
:mod:`protocore.runtime.query` and the terminal-nudge turn policy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from protocore.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.runtime.query_engine import QueryEngine

_logger = get_logger(__name__)

#: ``state_changed`` reason when the forcing starts on a delivered answer.
REASON_FORCED: str = "terminal_tool_forced"
#: ``state_changed`` reason when a forced request came back without the call.
REASON_RETRY: str = "terminal_tool_forced_retry"
#: ``state_changed`` reason when the run completes on its delivered answer.
REASON_EXHAUSTED: str = "terminal_tool_forced_exhausted"
#: ``state_changed`` reason when the run's terminal tool is not a tool it can call.
REASON_UNAVAILABLE: str = "terminal_tool_unavailable"
#: ``state_changed`` reason when the forcing is lifted so the model can write.
REASON_RELEASED: str = "terminal_tool_forced_released"


def is_armed(engine: object) -> bool:
    """True while the run's terminal call is being forced.

    Takes any object so the turn policies, which see the run only through its
    narrowed state protocol, can ask without a cast.
    """
    return bool(getattr(engine, "_terminal_call_forced", False))


def arm(engine: QueryEngine) -> bool:
    """Start forcing the terminal call. Returns True if it was not already."""
    if is_armed(engine):
        return False
    engine._terminal_call_forced = True
    return True


def release(engine: QueryEngine, *, reason: str) -> None:
    """Stop forcing. The spent attempts are NOT given back."""
    if not is_armed(engine):
        return
    engine._terminal_call_forced = False
    _logger.warning(
        "DIAG forced_terminal.released run=%s reason=%s attempts=%d",
        engine.config.run_id,
        reason,
        attempts_spent(engine),
    )


def attempts_spent(engine: QueryEngine) -> int:
    return int(getattr(engine, "_terminal_call_forced_attempts", 0))


def exhausted(engine: QueryEngine) -> bool:
    """True once the forced requests the run may spend are all spent."""
    return attempts_spent(engine) >= engine.config.rc.terminal_tool_forced_max_attempts


def charge_attempt(engine: QueryEngine) -> int:
    """Spend one forced request and return which one it was."""
    engine._terminal_call_forced_attempts = attempts_spent(engine) + 1
    return engine._terminal_call_forced_attempts


def spend_all(engine: QueryEngine) -> None:
    """Spend the whole bound at once, for a forcing that cannot be carried out."""
    engine._terminal_call_forced_attempts = (
        engine.config.rc.terminal_tool_forced_max_attempts
    )


__all__ = [
    "REASON_EXHAUSTED",
    "REASON_FORCED",
    "REASON_RELEASED",
    "REASON_RETRY",
    "REASON_UNAVAILABLE",
    "arm",
    "attempts_spent",
    "charge_attempt",
    "exhausted",
    "is_armed",
    "release",
    "spend_all",
]
