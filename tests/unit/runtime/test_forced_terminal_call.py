"""The terminal call is forced once the run's answer is delivered.

A run whose contract ends in a terminal tool, and whose model wrote the answer
as prose and stopped, owes only the call. These tests hold the loop to the
property that every request after that point names the terminal tool in
``extra['forced_tool_choice']``, appends nothing to the transcript, never asks
the model to continue, and — when the forcing is spent — completes on the
answer already delivered rather than letting the model write another one.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import ToolContext
from protocore.contracts.turn_policy import TurnCoordinate, TurnDirective, TurnFlags
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    TERMINAL_TOOL_METADATA_KEY,
    TERMINAL_TOOL_STATUS_COMPLETED,
    TERMINAL_TOOL_STATUS_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResult,
)
from protocore.runtime import forced_terminal
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _CORE_TURN_POLICIES, _turn_at
from protocore.runtime.query_engine import QueryEngine
from protocore.tests_support.adapters import InMemoryLLMProvider

from ._tool_fixtures import MockTool

ANSWER = "PostgreSQL keeps row versions in the table; InnoDB keeps them in undo."


class _FinalizeTool(MockTool):
    """A background terminal tool: it records the call and ends the run."""

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content="finalized",
            is_error=False,
            metadata={
                TERMINAL_TOOL_METADATA_KEY: True,
                TERMINAL_TOOL_STATUS_METADATA_KEY: TERMINAL_TOOL_STATUS_COMPLETED,
            },
        )


def _text_stream(text: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": text, "kind": "text"}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


def _reasoning_only_stream(stop_reason: StopReason) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "thinking"}),
        LLMStreamEvent(
            name="content_block_delta",
            payload={"text": "The user asked earlier about installing it...", "kind": "thinking"},
        ),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": stop_reason.value}),
    ]


def _rc(**overrides: Any) -> LoopConstants:
    values: dict[str, Any] = {
        "model_context_window": 8_192,
        "terminal_tool_nudge_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)


def _engine(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
    *,
    rc: LoopConstants,
    register_finalize: bool = True,
) -> tuple[QueryEngine, _FinalizeTool]:
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    finalize = _FinalizeTool(tool_name="Finalize", description="End the run")
    if register_finalize:
        in_memory_runtime["tools"].register(finalize)
    in_memory_runtime["tools"].register(MockTool(tool_name="WebSearch", description="Search"))
    # The run thinks; the forced call must not.
    engine._live_thinking_enabled = True
    return engine, finalize


async def _run(engine: QueryEngine, text: str = "Compare them.") -> list[TurnEvent]:
    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])
    return [event async for event in engine.run(user)]


def _reasons(events: Sequence[TurnEvent]) -> list[Any]:
    return [
        event.payload.get("reason")
        for event in events
        if event.type is EventType.STATE_CHANGED
    ]


def _request_texts(request: LLMRequest) -> list[str]:
    return [
        block.text
        for message in request.messages
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    ]


def _answers(engine: QueryEngine) -> list[str]:
    return [
        block.text
        for message in engine.history
        if message.role is MessageRole.assistant
        for block in message.content_blocks
        if isinstance(block, TextBlock) and block.text.strip()
    ]


def _assert_forced_finalize(request: LLMRequest, rc: LoopConstants) -> None:
    assert request.extra.get("forced_tool_choice") == "Finalize"
    # Thinking is off on the forced call: the only output it can have is the
    # call's arguments.
    assert request.extra.get("enable_thinking") is False
    texts = _request_texts(request)
    assert rc.continue_prompt_text not in texts
    assert rc.reasoning_length_cut_nudge_text not in texts
    assert not any("has not been called yet" in text for text in texts)
    # Nothing was appended after the delivered answer.
    assert request.messages[-1].role is MessageRole.assistant
    assert _request_texts(request)[-1] == ANSWER


@pytest.mark.asyncio
async def test_a_text_answer_is_followed_by_a_forced_terminal_call(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc()
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    assert len(llm.calls) == 2
    first, second = llm.calls
    assert "forced_tool_choice" not in first.extra
    assert first.extra.get("enable_thinking") is True
    _assert_forced_finalize(second, rc)
    reasons = _reasons(events)
    assert forced_terminal.REASON_FORCED in reasons
    assert "terminal_tool_nudge" not in reasons
    assert finalize.calls == [{"declared_deliverables": []}]
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", [StopReason.end_turn, StopReason.max_tokens])
async def test_reasoning_only_after_the_answer_retries_the_forced_call(
    engine_factory, in_memory_runtime, stop_reason: StopReason
) -> None:
    """The shape that produced two answers: the call was asked for, the model
    only reasoned, and the runtime said "continue". Now the round is simply
    forced again."""
    rc = _rc()
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_reasoning_only_stream(stop_reason))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    assert len(llm.calls) == 3
    for request in llm.calls[1:]:
        _assert_forced_finalize(request, rc)
    reasons = _reasons(events)
    assert forced_terminal.REASON_RETRY in reasons
    assert "continue_prompt_injected" not in reasons
    assert "reasoning_length_cut_retry" not in reasons
    assert "max_output_token_recovery" not in reasons
    assert len(finalize.calls) == 1
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_spent_forcing_completes_on_the_delivered_answer(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_max_attempts=2)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_reasoning_only_stream(StopReason.end_turn))
    llm._scripted_streams.append(_reasoning_only_stream(StopReason.max_tokens))
    # A provider that would answer again if asked once more.
    llm._scripted_streams.append(_text_stream("A second answer to an earlier question."))

    events = await _run(engine)

    assert len(llm.calls) == 3
    for request in llm.calls[1:]:
        _assert_forced_finalize(request, rc)
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    assert finalize.calls == []
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops[-1].payload["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_text_on_a_forced_turn_is_never_a_second_answer(
    engine_factory, in_memory_runtime
) -> None:
    """A provider that ignores the forced choice and writes prose instead: the
    prose reaches neither the reader nor the transcript, and the forced call
    is retried."""
    rc = _rc(terminal_tool_forced_max_attempts=1)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    second = "Answering the earlier question again: run apt install."
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream(second))

    events = await _run(engine)

    assert len(llm.calls) == 2
    streamed = "".join(
        str(e.payload["delta"].get("text", ""))
        for e in events
        if e.type is EventType.CONTENT_BLOCK_DELTA
        and isinstance(e.payload.get("delta"), dict)
        and e.payload["delta"].get("type") == "text_delta"
    )
    assert second not in streamed
    assert ANSWER in streamed
    assert _answers(engine) == [ANSWER]
    assert finalize.calls == []
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_failed_terminal_call_is_forced_again_then_bounded(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_max_attempts=2)
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    failing = MockTool(
        tool_name="Finalize",
        description="End the run",
        response_content="declared_deliverables is required",
        response_is_error=True,
    )
    in_memory_runtime["tools"].register(failing)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    for index in range(3):
        llm.queue_tool_call_response(
            tool_call_id=f"fin-{index}", tool_name="Finalize", tool_input={}
        )

    events = await _run(engine)

    assert len(llm.calls) == 3
    assert all(
        request.extra.get("forced_tool_choice") == "Finalize"
        for request in llm.calls[1:]
    )
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    # A failed terminal call is not work: the answer before it still counts,
    # so the prose gate never asks for another one.
    assert not any(
        message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
        for message in engine.history
    )
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_zero_attempts_completes_without_another_request(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_max_attempts=0)
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream("never requested"))

    events = await _run(engine)

    assert len(llm.calls) == 1
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_an_unregistered_terminal_tool_completes_on_the_answer(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc()
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc, register_finalize=False)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream("never requested"))

    events = await _run(engine)

    assert len(llm.calls) == 1
    assert forced_terminal.REASON_UNAVAILABLE in _reasons(events)
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_no_answer_yet_is_asked_for_in_words_then_forced(
    engine_factory, in_memory_runtime
) -> None:
    """Below the answer floor there is nothing to seal, and a forced call cannot
    write prose — so the run is told once, and the answer it then writes is
    followed by the forced call."""
    rc = _rc(finalize_prose_gate_min_chars=20, terminal_tool_nudge_write_first_enabled=False)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream("Working on it."))
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    assert len(llm.calls) == 3
    reasons = _reasons(events)
    assert reasons.index("terminal_tool_nudge") < reasons.index(forced_terminal.REASON_FORCED)
    assert "forced_tool_choice" not in llm.calls[1].extra
    assert any("has not been called yet" in text for text in _request_texts(llm.calls[1]))
    forced = llm.calls[2]
    assert forced.extra.get("forced_tool_choice") == "Finalize"
    assert forced.extra.get("enable_thinking") is False
    # The one telling stays where it was; nothing was added after the answer.
    assert forced.messages[-1].role is MessageRole.assistant
    assert _request_texts(forced)[-1] == ANSWER
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_thinking_can_be_kept_on_the_forced_call(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_thinking_enabled=True)
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    await _run(engine)

    assert llm.calls[1].extra.get("forced_tool_choice") == "Finalize"
    assert llm.calls[1].extra.get("enable_thinking") is True


@pytest.mark.asyncio
async def test_the_wind_down_answer_is_sealed_by_a_forced_call(
    engine_factory, in_memory_runtime
) -> None:
    """A run cut short by its turn cap is told in words to write its answer
    (the answer is prose, which no forced call can produce) and the answer it
    writes is then sealed by force rather than by a second request in words."""
    rc = _rc(max_turns_per_run=1)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="WebSearch", tool_input={"query": "oltp"}
    )
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    reasons = _reasons(events)
    assert "soft_stop_notified" in reasons
    assert forced_terminal.REASON_FORCED in reasons
    assert "terminal_tool_nudge" not in reasons
    _assert_forced_finalize(llm.calls[-1], rc)
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_prose_gate_refusal_lifts_the_forcing(
    engine_factory, in_memory_runtime
) -> None:
    """When the gate refuses the call because the answer is not really there,
    the next request must let the model write."""
    rc = _rc()
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc)
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=ANSWER)])
    )
    forced_terminal.arm(engine)
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="Write the answer first.")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR},
        )
    )

    turn = _turn_at(engine, TurnFlags(), TurnCoordinate.turn_start, turn_budget=10)
    async for _ in _CORE_TURN_POLICIES.apply(turn):
        pass

    assert turn.outcome.directive is TurnDirective.proceed

    assert forced_terminal.is_armed(engine) is False
    assert forced_terminal.attempts_spent(engine) == 0


def test_forcing_pins_the_terminal_tool_to_the_surface(
    engine_factory, in_memory_runtime
) -> None:
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=_rc())
    assert "Finalize" not in engine.effective_tool_policy.forced_pinned
    forced_terminal.arm(engine)
    assert "Finalize" in engine.effective_tool_policy.forced_pinned


@pytest.mark.asyncio
async def test_forcing_survives_a_snapshot_round_trip(
    engine_factory, in_memory_runtime
) -> None:
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=_rc())
    forced_terminal.arm(engine)
    forced_terminal.charge_attempt(engine)
    snapshot = engine.snapshot()

    restored = engine_factory(rc=_rc(), expected_terminal_tool="Finalize")
    await restored.resume_from_snapshot(snapshot)

    assert forced_terminal.is_armed(restored) is True
    assert forced_terminal.attempts_spent(restored) == 1


@pytest.mark.asyncio
async def test_a_blocked_terminal_tool_spends_the_forcing_at_once(
    engine_factory, in_memory_runtime
) -> None:
    """A surface that blocks the terminal tool outright cannot carry the forced
    call. The bound is spent on the spot, the stray turn's text stays hidden,
    and the run completes on its answer."""
    rc = _rc()
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    engine._circuit_broken_tools.add("Finalize")
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream("Another answer."))
    llm._scripted_streams.append(_text_stream("never requested"))

    events = await _run(engine)

    assert len(llm.calls) == 2
    assert "forced_tool_choice" not in llm.calls[1].extra
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    assert _answers(engine) == [ANSWER]
    assert finalize.calls == []
    assert engine.state is LoopState.COMPLETED
