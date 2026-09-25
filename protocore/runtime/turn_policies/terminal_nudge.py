"""The push towards the run's terminal tool.

Two different runs reach the end of a turn without having called it, and they
need opposite things.

A run that has written its answer owes the call and nothing else. It is not
told so — it is made to: every request from here on names the terminal tool in
the provider's native ``tool_choice``, nothing is appended to the transcript,
and nothing asks the model to "continue". A request that comes back without the
call is retried on a bounded budget, and when the budget is spent the run
completes on the answer it already delivered. No request in this state can
produce a second answer, because none of them can produce prose at all.

A run that has written nothing is different: the terminal call cannot be forced
on it, because the answer it still owes is prose and a forced tool call cannot
write any. That run is told, once, in words, and gets one more message to write
the answer. Once, because a second telling never produced a different answer
and a run that keeps being told burns its budget on being told. When it does
write the answer, the first case takes over.

The latch that spends the single telling is turn-local state the loop can still
see; the forcing's own state is the run's, snapshot-persisted, and everything
else about the decision is here.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.runtime.events import TurnEvent
from protocore.runtime.forced_terminal import (
    REASON_EXHAUSTED,
    REASON_FORCED,
    REASON_RETRY,
    REASON_UNAVAILABLE,
)
from protocore.runtime.turn_policies import (
    HistoryAppender,
    RunPredicate,
    StateChangeEmitter,
)


@dataclass(frozen=True, slots=True)
class ForcedTerminalCall:
    """What the policy may ask and do about a terminal call it forces."""

    #: The run's answer is written after its latest work.
    answer_delivered: RunPredicate
    #: The terminal tool is a tool this run can call at all.
    tool_registered: RunPredicate
    #: Start forcing; True when the forcing was not already on.
    arm: Callable[[Any], bool]
    #: The forcing is on.
    armed: RunPredicate
    #: Lift the forcing without giving back what it spent.
    release: HistoryAppender
    #: The transcript ends in a request that the model write prose — the
    #: prose gate refused the call because the answer is not really there.
    writing_requested: RunPredicate
    #: Spend one forced request.
    charge: Callable[[Any], int]
    #: Every forced request the run may spend is spent.
    exhausted: RunPredicate
    #: Complete the run on the delivered answer, saying why.
    complete: Callable[[Any, str], AsyncIterator[TurnEvent]]


class TerminalNudgePolicy:
    """Force the terminal call after an answer; ask for the answer before one."""

    name = "terminal_nudge"
    coordinates = frozenset({TurnCoordinate.turn_start, TurnCoordinate.finish_nudge})

    __slots__ = ("_append", "_forced", "_required", "_state_change")

    def __init__(
        self,
        *,
        required: RunPredicate,
        append: HistoryAppender,
        state_change: StateChangeEmitter,
        forced: ForcedTerminalCall,
    ) -> None:
        self._required = required
        self._append = append
        self._state_change = state_change
        self._forced = forced

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.coordinate is TurnCoordinate.turn_start:
            async for event in self._before_the_request(turn):
                yield event
            return
        async for event in self._at_the_finish(turn):
            yield event

    async def _before_the_request(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        """Spend a forced request, or end the run when none is left."""
        engine = turn.engine
        forced = self._forced
        if not forced.armed(engine):
            return
        if not self._required(engine):
            # The call landed; nothing is owed.
            forced.release(engine)
            return
        if forced.writing_requested(engine):
            # The prose gate refused the call: what was taken for the answer
            # is not one. The model has to write, which a forced call cannot,
            # so the request goes out free; the answer it writes arms the
            # forcing again on what is left of the same budget.
            forced.release(engine)
            return
        if forced.exhausted(engine):
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = REASON_EXHAUSTED
            async for event in forced.complete(engine, REASON_EXHAUSTED):
                yield event
            return
        forced.charge(engine)

    async def _at_the_finish(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        if not self._required(engine):
            return
        forced = self._forced
        if forced.armed(engine) or forced.answer_delivered(engine):
            if not forced.tool_registered(engine):
                # The run can never make the call it owes, so there is nothing
                # to force and nothing to wait for: the answer stands.
                turn.outcome.directive = TurnDirective.end_turn
                turn.outcome.reason = REASON_UNAVAILABLE
                async for event in forced.complete(engine, REASON_UNAVAILABLE):
                    yield event
                return
            first = forced.arm(engine)
            reason = REASON_FORCED if first else REASON_RETRY
            turn.outcome.directive = TurnDirective.restart_turn
            turn.outcome.extra_turn = True
            turn.outcome.rebuild_context = True
            turn.outcome.reason = reason
            yield self._state_change(engine, reason)
            return
        if turn.flags.terminal_nudge_used:
            return
        turn.flags.terminal_nudge_used = True
        self._append(engine)
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.extra_turn = True
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "terminal_tool_nudge"
        yield self._state_change(engine, "terminal_tool_nudge")


__all__ = ["ForcedTerminalCall", "TerminalNudgePolicy"]
