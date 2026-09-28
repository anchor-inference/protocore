"""A permanent refusal caused by the tool surface is exactly what the wind-down can rescue.

The wind-down's request is not "the same history and one more message": it
narrows the tool surface to the finalizing tool. A provider that refuses a
request for a reason carried by the surface (a tool definition its schema
check rejects, a surface over its tool-count limit) answers the wind-down's
request. Skipping the wind-down on every non-retryable classification turns a
run that could have reported its evidence into a failure.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from protocore.contracts.llm import LLMProviderError, LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState
from tests.unit.runtime.test_soft_stop import (
    TERMINAL_TOOL,
    _build_engine,
    _FailsOnLLM,
    _FinalizeTool,
    _NamedTool,
    _ScriptedLLM,
    _user,
)


class _Verdict:
    reason = "format_error"
    retryable = False


class _RefusesTheWideSurface:
    """Answers the first request; then refuses, for good, any request carrying a tool besides the terminal one."""

    def __init__(self) -> None:
        self._scripted = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "Section a is fine; details above."}])
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        names = {getattr(tool, "name", None) for tool in request.tools}
        if len(self.calls) >= 2 and names - {TERMINAL_TOOL}:
            refusal = LLMProviderError("HTTP 400: invalid schema for function 'Read'")
            object.__setattr__(refusal, "classified", _Verdict())
            raise refusal
        async for event in self._scripted.stream_with_tools(request):
            yield event

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        raise AssertionError("not used")

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


async def test_a_refusal_of_the_tool_surface_is_rescued_by_the_wind_down() -> None:
    llm = _RefusesTheWideSurface()
    engine = _build_engine(rc=LoopConstants(model_context_window=4_096), llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    async for _ in engine.run(_user()):
        pass

    assert engine.state is LoopState.COMPLETED


async def test_a_surface_refusal_the_wind_down_meets_again_still_fails_on_the_providers_words() -> None:
    """The wind-down is tried; when its narrowed request is refused too, the refusal is the outcome."""
    refusal = LLMProviderError("HTTP 400: request rejected by the upstream validator")
    object.__setattr__(refusal, "classified", _Verdict())
    llm = _FailsOnLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "unused"}], refusal, fail_on={2, 3, 4})
    engine = _build_engine(rc=LoopConstants(model_context_window=4_096), llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    reasons = [e.payload.get("reason") for e in events if e.type is EventType.STATE_CHANGED]
    assert "soft_stop_notified" in reasons
    assert "transient_llm_error_retry" not in reasons
    assert engine.state is LoopState.FAILED
    errors = [e.payload for e in events if e.type is EventType.ERROR]
    assert errors and "upstream validator" in str(errors[-1].get("message"))
