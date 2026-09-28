"""The wind-down notice must not come back for a turn that has its tools again.

A finished run's history no longer carries the notice, but its snapshot keeps
the wind-down state. ``resume_from_snapshot`` puts the notice back whenever
that state is armed, even when the snapshot is of a TERMINAL run. A host that
resumes a settled living agent from its snapshot and re-arms it for the next
turn then hands that turn a history whose last instruction is "your tools are
gone": ``rearm`` clears the wind-down state but not the message ``restore``
appended.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import SYNTHETIC_RECOVERY_METADATA_KEY
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.loop_state import LoopState
from tests.unit.runtime.test_soft_stop import _build_engine, _FinalizeTool, _NamedTool, _ScriptedLLM, _user


def _notices(engine) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for m in engine.history
        if m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY) == _soft_stop.SYNTHETIC_RECOVERY_SOFT_STOP
    )


@pytest.mark.xfail(
    strict=True,
    reason="restore() re-adds the notice for a terminal snapshot; rearm() keeps it in history",
)
async def test_a_settled_wound_down_run_resumed_and_rearmed_has_no_notice() -> None:
    rc = LoopConstants(model_context_window=4_096, leader_tool_call_soft_cap=1)
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "done"}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])
    async for _ in engine.run(_user()):
        pass
    assert engine.is_terminal
    assert _soft_stop.is_armed(engine)
    assert _notices(engine) == 0
    snapshot = engine.snapshot()

    # A pod restart: the host rebuilds the engine from the settled snapshot and
    # re-arms it for the operator's next turn.
    fresh = _build_engine(rc=rc, llm=_ScriptedLLM([]), tools=[_NamedTool("Read"), _FinalizeTool()])
    await fresh.resume_from_snapshot(snapshot)
    assert fresh.state in {LoopState.COMPLETED, LoopState.FAILED} or fresh.is_terminal
    fresh.rearm()

    assert not _soft_stop.is_armed(fresh)
    assert _notices(fresh) == 0
