"""What the model is told when a call is refused, and what the run counts.

Three refusals used to say something other than the truth:

* the wind-down's terminal-only refusal named the wall-clock deadline whatever
  bound had actually ended the run;
* a tool the repeated-error breaker had stopped was refused as being "on the
  tenant blocked list", a list nobody had written;
* a call turned away by the wind-down never reached the dispatcher, so nothing
  the run reported showed that the model had asked for a withdrawn tool.

The refusals are tallied apart from the tool-error count — a withdrawn tool did
not fail — and the model-facing texts are run constants.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pydantic import ValidationError

from protocore.contracts.run import IRunRefusedCallCounter
from protocore.contracts.run_state import RunScopedState
from protocore.contracts.runtime_constants import LoopConstants, override_text
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext, ToolInvocationError
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import EventType
from protocore.runtime.execution_profile import apply_execution_profile
from protocore.runtime.query import (
    _dispatch_tool,
    _terminal_only_blocks,
    _terminal_only_error_message,
)
from protocore.runtime.tool_permission import (
    ToolPermissionGate,
    ToolPermissionOutcome,
)
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

from ._tool_fixtures import MockTool
from .test_query_parallel_safe_tools import (
    _arm_terminal_only_at_the_model_call,
    _queue_multi_tool_stream,
    _ScriptedReadTool,
)

# ---------------------------------------------------------------------------
# The terminal-only refusal names the bound that was reached
# ---------------------------------------------------------------------------


def _wound_down_engine(engine_factory: Any, cause: str) -> Any:
    engine = engine_factory(expected_terminal_tool="final_answer")
    _soft_stop.enter(engine, cause_name=cause)
    return engine


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (_soft_stop.CAUSE_TOOL_CALL_BUDGET, "tool-call budget spent"),
        (_soft_stop.CAUSE_MAX_TURNS, "turn limit reached"),
        (_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET, "output-token budget spent"),
        (_soft_stop.CAUSE_DEADLINE, "wall-clock deadline reached"),
        (_soft_stop.CAUSE_PROVIDER_ERROR, "the model provider stopped answering"),
        (_soft_stop.CAUSE_MODEL_NO_PROGRESS, "the run stopped making progress"),
    ],
)
def test_the_refusal_names_the_bound_that_ended_the_run(
    engine_factory: Any, cause: str, expected: str
) -> None:
    engine = _wound_down_engine(engine_factory, cause)

    message = _terminal_only_error_message(engine, "PageReader")

    assert message == (
        "tool 'PageReader' execution blocked: [terminal-only mode: "
        f"{expected} — call final_answer now with your best-evidence answer]"
    )


@pytest.mark.parametrize(
    "cause",
    sorted(_soft_stop.CAUSES - {_soft_stop.CAUSE_DEADLINE}),
)
def test_only_a_deadline_is_called_a_deadline(
    engine_factory: Any, cause: str
) -> None:
    engine = _wound_down_engine(engine_factory, cause)

    assert "deadline" not in _terminal_only_error_message(engine, "Read")


def test_every_cause_has_its_own_refusal_wording() -> None:
    assert set(_soft_stop.CAUSE_REFUSAL_REASONS) == set(_soft_stop.CAUSES)
    assert len(set(_soft_stop.CAUSE_REFUSAL_REASONS.values())) == len(
        _soft_stop.CAUSES
    )


def test_a_wind_down_without_a_recorded_cause_still_gets_a_true_sentence(
    engine_factory: Any,
) -> None:
    engine = engine_factory(expected_terminal_tool="final_answer")
    engine._soft_stop_stage = _soft_stop.STAGE_WITHDRAWN
    engine._soft_stop_cause = None

    message = _terminal_only_error_message(engine, "Read")

    assert _soft_stop.REFUSAL_REASON_UNKNOWN in message
    assert "deadline" not in message


# ---------------------------------------------------------------------------
# The wording is a run constant, overridable per cause
# ---------------------------------------------------------------------------


def test_the_default_reasons_are_read_only() -> None:
    with pytest.raises(TypeError):
        _soft_stop.CAUSE_REFUSAL_REASONS["deadline"] = "x"  # type: ignore[index]


def _engine_with(engine_factory: Any, **overrides: Any) -> Any:
    rc = LoopConstants(model_context_window=4_096, **overrides)
    engine = engine_factory(rc=rc, expected_terminal_tool="final_answer")
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_MAX_TURNS)
    return engine


def test_a_run_can_reword_one_cause_without_touching_the_others(
    engine_factory: Any,
) -> None:
    engine = _engine_with(
        engine_factory, soft_stop_refusal_reason_max_turns="out of rounds"
    )

    assert "[terminal-only mode: out of rounds — call final_answer" in (
        _terminal_only_error_message(engine, "Read")
    )
    other = engine_factory(rc=engine.config.rc, expected_terminal_tool="final_answer")
    _soft_stop.enter(other, cause_name=_soft_stop.CAUSE_DEADLINE)
    assert "wall-clock deadline reached" in _terminal_only_error_message(other, "Read")


def test_a_causeless_wind_down_takes_the_general_reason_and_then_its_default(
    engine_factory: Any,
) -> None:
    # A wind-down that recorded no cause takes the general reason, and a blank
    # general reason takes its own default.
    general = _engine_with(
        engine_factory, soft_stop_refusal_reason="winding down by order"
    )
    general._soft_stop_cause = None
    assert _soft_stop.refusal_reason(general) == "winding down by order"

    default = _engine_with(engine_factory, soft_stop_refusal_reason="")
    default._soft_stop_cause = None
    assert _soft_stop.refusal_reason(default) == _soft_stop.REFUSAL_REASON_UNKNOWN


def test_a_run_can_reword_the_whole_refusal(engine_factory: Any) -> None:
    engine = _engine_with(
        engine_factory,
        soft_stop_refusal_text="no {tool}: {reason}; use {terminal_tool}",
    )

    assert _terminal_only_error_message(engine, "Read") == (
        "no Read: turn limit reached; use final_answer"
    )


def test_a_blank_refusal_text_is_the_default_not_silence() -> None:
    rc = LoopConstants(model_context_window=4_096, soft_stop_refusal_text="")

    assert override_text(rc, "soft_stop_refusal_text") == (
        LoopConstants.model_fields["soft_stop_refusal_text"].default
    )
    assert override_text(None, "tool_disabled_refusal_text").startswith("tool '")


def test_a_whitespace_only_override_is_the_default_not_silence() -> None:
    rc = LoopConstants(model_context_window=4_096, soft_stop_refusal_text="  \n\t ")

    assert override_text(rc, "soft_stop_refusal_text") == (
        LoopConstants.model_fields["soft_stop_refusal_text"].default
    )
    assert override_text(rc, "soft_stop_refusal_text").strip()


def test_a_cleared_cause_reason_is_that_causes_own_default(
    engine_factory: Any,
) -> None:
    # Clearing the field to reset it must not lose the fact the refusal states.
    engine = _engine_with(
        engine_factory,
        soft_stop_refusal_reason_deadline="",
        soft_stop_refusal_reason="winding down by order",
    )
    engine._soft_stop_cause = _soft_stop.CAUSE_DEADLINE

    assert _soft_stop.refusal_reason(engine) == (
        _soft_stop.CAUSE_REFUSAL_REASONS[_soft_stop.CAUSE_DEADLINE]
    )
    assert "wall-clock deadline reached" in _terminal_only_error_message(
        engine, "Read"
    )

    blank = _engine_with(engine_factory, soft_stop_refusal_reason_max_turns="   ")
    assert _soft_stop.refusal_reason(blank) == (
        _soft_stop.CAUSE_REFUSAL_REASONS[_soft_stop.CAUSE_MAX_TURNS]
    )


# ---------------------------------------------------------------------------
# A tool the breaker stopped is disabled for the run, not tenant-blocked
# ---------------------------------------------------------------------------


def _ctx(rc: LoopConstants | None) -> ToolContext:
    state = RunScopedState()
    state.rc = rc
    return ToolContext(
        run_id="run-gate", tenant_id="t", session_id="s", run_state=state
    )


async def _gate_reason(policy: ToolVisibilityPolicy, ctx: ToolContext) -> str:
    decision = await ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES).check(
        tool=MockTool(tool_name="PageReader"),
        arguments={},
        ctx=ctx,
        visibility_policy=policy,
    )
    assert decision.outcome is ToolPermissionOutcome.deny
    return decision.reason


@pytest.mark.asyncio
async def test_a_run_disabled_tool_says_so_and_quotes_the_streak_that_tripped_it() -> None:
    # The run's own cap is not the number quoted: the breaker's is.
    rc = LoopConstants(model_context_window=4_096, max_consecutive_tool_errors=9)
    policy = ToolVisibilityPolicy(blocked={"PageReader"}, run_disabled={"PageReader": 5})

    reason = await _gate_reason(policy, _ctx(rc))

    assert reason == (
        "tool 'PageReader' is disabled for the rest of this run after 5 "
        "consecutive failures"
    )
    assert "tenant" not in reason
    assert "identical" not in reason


@pytest.mark.asyncio
async def test_a_run_disabled_tool_with_no_recorded_streak_still_says_so() -> None:
    policy = ToolVisibilityPolicy(blocked={"PageReader"}, run_disabled={"PageReader": None})

    reason = await _gate_reason(policy, _ctx(None))

    assert reason == (
        "tool 'PageReader' is disabled for the rest of this run after repeated failures"
    )


@pytest.mark.asyncio
async def test_the_disabled_wording_is_a_run_constant() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        tool_disabled_refusal_text="{tool} is off ({count} failures in a row)",
    )
    policy = ToolVisibilityPolicy(blocked={"PageReader"}, run_disabled={"PageReader": 4})

    assert await _gate_reason(policy, _ctx(rc)) == "PageReader is off (4 failures in a row)"


@pytest.mark.asyncio
async def test_a_tool_the_tenant_blocked_keeps_the_tenant_wording() -> None:
    reason = await _gate_reason(ToolVisibilityPolicy(blocked={"PageReader"}), _ctx(None))

    assert reason == "tool 'PageReader' is on the tenant blocked list"


def test_a_run_disabled_tool_that_is_not_blocked_is_refused() -> None:
    with pytest.raises(ValidationError, match="not blocked"):
        ToolVisibilityPolicy(run_disabled={"Sense": 3})


def test_run_disabled_survives_a_serialisation_round_trip() -> None:
    policy = ToolVisibilityPolicy(
        blocked={"A", "B", "Tenant"}, run_disabled={"A": 3, "B": None}
    )

    restored = ToolVisibilityPolicy.model_validate_json(policy.model_dump_json())

    assert restored == policy
    assert restored.run_disabled == {"A": 3, "B": None}


def test_the_effective_policy_separates_the_breakers_tools_from_the_tenants(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    # A host that pre-filled the field cannot relabel its own block as the run's.
    engine.config = dataclasses.replace(
        engine.config,
        tool_visibility_policy=ToolVisibilityPolicy(
            blocked={"Tenant"}, run_disabled={"Tenant": 7}
        ),
    )
    engine._circuit_broken_tools.update({"Broken", "Tenant"})
    engine._circuit_breaker_trip_caps["Broken"] = 4

    policy = engine.effective_tool_policy

    assert policy.blocked == {"Tenant", "Broken"}
    assert policy.run_disabled == {"Broken": 4}


def test_the_execution_profile_keeps_what_the_breaker_disabled() -> None:
    rc = LoopConstants(model_context_window=4_096, execution_profile_plan_enabled=True)
    policy = ToolVisibilityPolicy(blocked={"Broken"}, run_disabled={"Broken": 3})

    narrowed = apply_execution_profile(
        policy, profile="plan", rc=rc, roles=CONVENTIONAL_TOOL_ROLES
    )

    assert narrowed.run_disabled == {"Broken": 3}


async def _trip_the_breaker(engine: Any) -> int:
    engine._helpers = {}
    engine.tools.register(
        MockTool(tool_name="Sense", raise_exception=ToolInvocationError("busy"))
    )
    cap: int = engine.config.rc.max_consecutive_tool_errors
    for index in range(cap):
        async for _ in _dispatch_tool(
            engine, ToolCall(id=f"c{index}", name="Sense", arguments={"v": "x"})
        ):
            pass
    assert "Sense" in engine._circuit_broken_tools
    return cap


async def _refusal_to(engine: Any, call_id: str = "after") -> str:
    events = [
        evt
        async for evt in _dispatch_tool(
            engine, ToolCall(id=call_id, name="Sense", arguments={"v": "x"})
        )
    ]
    message: str = next(
        evt.payload["error"]["message"]
        for evt in events
        if evt.type is EventType.TOOL_RESULT
    )
    return message


@pytest.mark.asyncio
async def test_the_model_reads_the_real_cause_after_the_breaker_trips(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    cap = await _trip_the_breaker(engine)

    refusal = await _refusal_to(engine)

    assert (
        f"tool 'Sense' is disabled for the rest of this run after {cap} "
        "consecutive failures"
    ) in refusal
    assert "tenant blocked list" not in refusal


@pytest.mark.asyncio
async def test_the_quoted_streak_is_the_one_that_tripped_the_breaker(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    cap = await _trip_the_breaker(engine)
    # The tenant's override changes later; the tool was disabled at `cap`.
    engine.config = dataclasses.replace(
        engine.config,
        rc=engine.config.rc.model_copy(
            update={"max_consecutive_tool_errors": cap + 6}
        ),
    )
    snapshot = engine.snapshot()

    resumed = engine_factory(rc=engine.config.rc)
    resumed._helpers = {}
    resumed.tools.register(
        MockTool(tool_name="Sense", raise_exception=ToolInvocationError("busy"))
    )
    await resumed.resume_from_snapshot(snapshot)

    for candidate in (engine, resumed):
        assert f"after {cap} consecutive failures" in await _refusal_to(candidate)


# ---------------------------------------------------------------------------
# A call the wind-down turns away is tallied apart from the tool errors
# ---------------------------------------------------------------------------


class _Errors:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def increment_tool_errors_count(self, run_id: str, by: int = 1) -> None:
        self.calls.append((run_id, by))


class _Refused:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def increment_refused_tool_calls_count(
        self, run_id: str, by: int = 1
    ) -> None:
        self.calls.append((run_id, by))


def test_the_refused_call_counter_is_a_contract() -> None:
    assert isinstance(_Refused(), IRunRefusedCallCounter)
    assert not isinstance(_Errors(), IRunRefusedCallCounter)


def _wound_down_run(
    engine_factory: Any,
    in_memory_runtime: Any,
    *,
    terminal: str | None = "pcm_answer",
    latch: bool = True,
    cause: str = _soft_stop.CAUSE_TOOL_CALL_BUDGET,
) -> tuple[Any, _Refused, _Errors]:
    rc = LoopConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        terminal_tool_nudge_enabled=True,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool=terminal)
    engine.run_state.rc = rc
    refused, errors = _Refused(), _Errors()
    engine.run_state.refused_call_counter = refused
    engine.run_state.tool_error_counter = errors
    if latch:
        _arm_terminal_only_at_the_model_call(engine, in_memory_runtime["llm"])
    _soft_stop.enter(engine, cause_name=cause)
    in_memory_runtime["tools"].register(
        _ScriptedReadTool(tool_name="Read", description="read", response_content="ok")
    )
    return engine, refused, errors


async def _run(engine: Any) -> list[Any]:
    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    return [evt async for evt in engine.run(user)]


def _refusals(events: list[Any]) -> list[Any]:
    return [
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT
        and evt.payload.get("error", {}).get("kind") == "terminal_only"
    ]


def _terminal_stop(events: list[Any]) -> Any:
    return next(
        evt
        for evt in reversed(events)
        if evt.type is EventType.MESSAGE_STOP
        and evt.payload.get("stop_reason") != "tool_use"
    )


@pytest.mark.asyncio
async def test_a_refused_serial_call_is_tallied_apart_from_tool_errors(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    engine, refused, errors = _wound_down_run(engine_factory, in_memory_runtime)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-solo", tool_name="Read", tool_input={"path": "x"}
    )

    events = await _run(engine)

    found = _refusals(events)
    assert len(found) == 1
    assert "tool-call budget spent" in found[0].payload["error"]["message"]
    assert refused.calls == [("run-test", 1)]
    # The tool did not fail: what a host downgrades a run on stays untouched.
    assert errors.calls == []
    assert engine.refused_tool_calls_count == 1
    assert _terminal_stop(events).payload["refused_tool_calls_count"] == 1
    # And the breaker and the streak stay quiet too.
    assert engine._circuit_broken_tools == set()
    assert engine._circuit_breaker_streak is None


@pytest.mark.asyncio
async def test_every_refused_call_of_a_parallel_batch_is_tallied_once(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    engine, refused, errors = _wound_down_run(engine_factory, in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-r1", "Read", {"path": "a"}),
            ("call-r2", "Read", {"path": "b"}),
        ],
    )

    events = await _run(engine)

    assert len(_refusals(events)) == 2
    assert refused.calls == [("run-test", 2)]
    assert errors.calls == []
    assert engine.refused_tool_calls_count == 2
    assert engine._circuit_broken_tools == set()


@pytest.mark.asyncio
async def test_the_terminal_call_itself_is_never_refused(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    engine, refused, _ = _wound_down_run(engine_factory, in_memory_runtime)

    assert not _terminal_only_blocks(
        engine, ToolCall(id="t", name="pcm_answer", arguments={})
    )
    assert refused.calls == []


@pytest.mark.asyncio
async def test_a_counter_that_raises_does_not_break_the_refusal(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    engine, refused, _ = _wound_down_run(engine_factory, in_memory_runtime)

    async def _boom(run_id: str, by: int = 1) -> None:
        raise RuntimeError("telemetry down")

    refused.increment_refused_tool_calls_count = _boom  # type: ignore[method-assign]
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-solo", tool_name="Read", tool_input={"path": "x"}
    )

    events = await _run(engine)

    results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "call-solo"
    ]
    assert len(results) == 1 and results[0].is_error
    assert len(_refusals(events)) == 1


async def _ask_for_withdrawn_tools(engine: Any, times: int) -> list[str]:
    messages = []
    for index in range(times):
        events = [
            evt
            async for evt in _dispatch_tool(
                engine, ToolCall(id=f"stale-{index}", name="Sense", arguments={})
            )
        ]
        result = next(e for e in events if e.type is EventType.TOOL_RESULT)
        assert result.payload["error"]["kind"] == "terminal_only"
        messages.append(result.payload["error"]["message"])
    return messages


@pytest.mark.asyncio
async def test_a_wind_down_with_no_terminal_tool_refuses_and_says_to_answer(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    # The leader default of a host that treats no tool as terminal: the surface
    # is withdrawn to nothing, and a call off a stale schema used to reach the
    # permission gate and be told the tool was not in a tenant set nobody wrote.
    engine, refused, errors = _wound_down_run(
        engine_factory, in_memory_runtime, terminal=None, latch=False
    )
    engine.tools.register(MockTool(tool_name="Sense"))
    cap = engine.config.rc.max_consecutive_tool_errors

    messages = await _ask_for_withdrawn_tools(engine, cap + 1)

    for message in messages:
        assert "tool-call budget spent" in message
        assert "answer now" in message
        assert "tenant" not in message
    assert engine.refused_tool_calls_count == cap + 1
    assert sum(by for _, by in refused.calls) == cap + 1
    assert errors.calls == []
    # Refused, not failed: the breaker never fed on them, so nothing was
    # disabled and no later text claims a tool failed.
    assert engine._circuit_broken_tools == set()
    assert engine._circuit_breaker_streak is None


@pytest.mark.asyncio
async def test_a_run_continued_after_its_wind_down_began_still_gets_the_refusal(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    rc = LoopConstants(model_context_window=4_096, max_turns_per_run=2)
    first = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    first.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    _soft_stop.enter(first, cause_name=_soft_stop.CAUSE_MAX_TURNS)
    snapshot = first.snapshot()

    # Another pod picks the run up: ``run(None)`` lowers the per-turn latch.
    second = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    refused, errors = _Refused(), _Errors()
    second.run_state.refused_call_counter = refused
    second.run_state.tool_error_counter = errors
    in_memory_runtime["tools"].register(
        _ScriptedReadTool(tool_name="Read", description="read", response_content="ok")
    )
    await second.resume_from_snapshot(snapshot)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-stale", tool_name="Read", tool_input={"path": "x"}
    )

    events = [evt async for evt in second.run(None)]

    found = _refusals(events)
    assert len(found) == 1
    assert "turn limit reached" in found[0].payload["error"]["message"]
    assert "tenant visible set" not in found[0].payload["error"]["message"]
    assert refused.calls == [("run-test", 1)]
    assert errors.calls == []
    assert second._circuit_broken_tools == set()


@pytest.mark.asyncio
async def test_the_stale_calls_of_a_continued_run_never_feed_the_breaker(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    rc = LoopConstants(model_context_window=4_096)
    first = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    _soft_stop.enter(first, cause_name=_soft_stop.CAUSE_DEADLINE)
    second = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    await second.resume_from_snapshot(first.snapshot())
    second._reset_per_turn_state()  # what ``run()`` does at entry
    assert second._terminal_only_active is False
    second.tools.register(MockTool(tool_name="Sense"))

    messages = await _ask_for_withdrawn_tools(
        second, rc.max_consecutive_tool_errors + 1
    )

    assert all("wall-clock deadline reached" in m for m in messages)
    assert second._circuit_broken_tools == set()
    assert second._circuit_breaker_streak is None
    assert second.refused_tool_calls_count == rc.max_consecutive_tool_errors + 1


@pytest.mark.asyncio
async def test_a_subagents_refusals_go_to_the_root_run(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    engine = engine_factory(
        run_id="child-run",
        expected_terminal_tool="SubmitAnswer",
        parent_run_id="root-run",
        subagent_id="child-1",
        root_run_id="root-run",
    )
    refused, errors = _Refused(), _Errors()
    engine.run_state.root_run_id = "root-run"
    engine.run_state.refused_call_counter = refused
    engine.run_state.tool_error_counter = errors
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_TOOL_CALL_BUDGET)
    engine.tools.register(MockTool(tool_name="Sense"))

    await _ask_for_withdrawn_tools(engine, 2)

    assert refused.calls == [("root-run", 1), ("root-run", 1)]
    assert errors.calls == []


@pytest.mark.asyncio
async def test_a_replayed_call_is_counted_once_and_after_its_snapshot(
    engine_factory: Any
) -> None:
    engine = engine_factory(expected_terminal_tool="pcm_answer")
    engine.tools.register(MockTool(tool_name="Sense"))
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_TOOL_CALL_BUDGET)
    order: list[str] = []
    refused = _Refused()
    engine.run_state.refused_call_counter = refused
    persist = engine._persist_snapshot

    async def _spy() -> None:
        order.append(f"persist:{len(refused.calls)}")
        await persist()

    engine._persist_snapshot = _spy

    call = ToolCall(id="same", name="Sense", arguments={})
    for _ in range(2):
        async for _evt in _dispatch_tool(engine, call):
            pass

    # The first refusal reached the counter only once its snapshot was out, and
    # the same call id seen again — a re-drive — added nothing.
    assert order == ["persist:0", "persist:1"]
    assert refused.calls == [("run-test", 1)]
    assert engine.refused_tool_calls_count == 1

    # The tally travels with the snapshot, so another pod does not recount it.
    resumed = engine_factory(expected_terminal_tool="pcm_answer")
    await resumed.resume_from_snapshot(engine.snapshot())
    _soft_stop.record_refusal(resumed, "same")
    assert resumed.refused_tool_calls_count == 1
    assert resumed._refused_calls_unreported == 0


# ---------------------------------------------------------------------------
# What the wind-down refuses is what it withdrew
# ---------------------------------------------------------------------------


def _open_an_artifact(engine: Any) -> None:
    engine._longfile_active_path = "/workspace/big.py"
    engine._longfile_active_file_bytes = 12_000
    engine._longfile_truncated_paths.add("/workspace/big.py")


@pytest.mark.asyncio
@pytest.mark.parametrize("latch", [True, False])
async def test_the_artifact_sealer_is_dispatched_not_refused_in_a_wind_down(
    engine_factory: Any, in_memory_runtime: Any, latch: bool
) -> None:
    # ``latch=False`` is a run continued after its wind-down began: the per-turn
    # latch is down, and the sealer must be admitted all the same.
    rc = LoopConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        longfile_convergence_enabled=True,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    refused, errors = _Refused(), _Errors()
    engine.run_state.refused_call_counter = refused
    engine.run_state.tool_error_counter = errors
    if latch:
        _arm_terminal_only_at_the_model_call(engine, in_memory_runtime["llm"])
    _open_an_artifact(engine)
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_TOOL_CALL_BUDGET)
    sealer = MockTool(tool_name="FinalizeFile")
    engine.tools.register(sealer)
    engine.tools.register(MockTool(tool_name="Sense"))
    assert "FinalizeFile" in _soft_stop.terminal_surface(engine)

    sealing = ToolCall(id="seal-1", name="FinalizeFile", arguments={})
    assert _terminal_only_blocks(engine, sealing) is False
    events = [evt async for evt in _dispatch_tool(engine, sealing)]
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload.get("error", {}).get("kind") != "terminal_only"
    assert refused.calls == []
    assert engine.refused_tool_calls_count == 0

    # Everything else off the surface is still refused, and says why.
    other = ToolCall(id="other-1", name="Sense", arguments={})
    assert _terminal_only_blocks(engine, other) is True
    assert errors.calls == []


@pytest.mark.asyncio
async def test_the_sealer_is_refused_once_no_artifact_is_open(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    engine, _, _ = _wound_down_run(engine_factory, in_memory_runtime)
    engine.tools.register(MockTool(tool_name="FinalizeFile"))

    assert _soft_stop.terminal_surface(engine) == frozenset({"pcm_answer"})
    assert _terminal_only_blocks(
        engine, ToolCall(id="seal-2", name="FinalizeFile", arguments={})
    )


@pytest.mark.asyncio
async def test_a_subagent_keeps_its_answer_tool_only_when_the_host_names_it(
    engine_factory: Any, in_memory_runtime: Any
) -> None:
    def _child(terminal: str | None) -> Any:
        engine = engine_factory(
            run_id="child-run",
            expected_terminal_tool=terminal,
            parent_run_id="root-run",
            subagent_id="child-1",
            root_run_id="root-run",
        )
        _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_TOOL_CALL_BUDGET)
        return engine

    named = _child("SubmitAnswer")
    assert _soft_stop.terminal_surface(named) == frozenset({"SubmitAnswer"})
    assert not _terminal_only_blocks(
        named, ToolCall(id="c-1", name="SubmitAnswer", arguments={})
    )
    assert "call SubmitAnswer now" in _terminal_only_error_message(named, "Read")

    unnamed = _child(None)
    assert _soft_stop.terminal_surface(unnamed) == frozenset()
    assert _terminal_only_blocks(
        unnamed, ToolCall(id="c-2", name="SubmitAnswer", arguments={})
    )
    assert "answer now" in _terminal_only_error_message(unnamed, "SubmitAnswer")
