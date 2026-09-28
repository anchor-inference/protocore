"""The kept tail is bounded in tokens, and a pass that cannot reach the trigger calls no summariser.

The routine keep window protects the last ``compaction_keep_recent_turns``
messages whatever they weigh. A run whose recent messages carry large tool
results then has a fixed part above the trigger: every pass ends at the floor,
the prompt never goes back under the trigger, and the summariser is called on
every iteration for nothing it can fix.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.runtime.context.budgets import derive_budgets
from protocore.tests_support.adapters import InMemoryLLMProvider

from .conftest import ScenarioFactory, ScriptedTool, default_rc

pytestmark = pytest.mark.asyncio

_PAGE = "страница учебника про тайгу " * 75


def _scenario(scenario: ScenarioFactory, **rc_overrides: object):
    summariser = InMemoryLLMProvider()
    for _ in range(40):
        summariser.queue_response(text="## Summary\nearlier pages were read")
    rc = default_rc(
        model_context_window=8_192,
        compaction_trigger_ratio=0.3,
        compaction_keep_recent_turns=8,
        **rc_overrides,
    )
    run = scenario(rc=rc, tools=[ScriptedTool(tool_name="PageRead", content=_PAGE)], compaction_provider=summariser)
    for i in range(6):
        run.llm.queue_tool_call_response(tool_call_id=f"c{i}", tool_name="PageRead", tool_input={"v": str(i)})
    run.llm.queue_response(text="done")
    return run, summariser


def _completed(run) -> list[dict[str, Any]]:
    return [evt.payload for evt in run.events if evt.type.value == "compaction_completed"]


async def test_large_recent_messages_do_not_hold_the_fixed_part_above_the_trigger(
    scenario: ScenarioFactory,
) -> None:
    run, _ = _scenario(scenario)
    await run.run("read six pages")

    trigger = derive_budgets(run.engine.config.rc).compaction_trigger_tokens
    passes = _completed(run)
    assert passes, "the pages must put the prompt over the trigger"
    assert all(int(p["fixed_tokens"]) < trigger for p in passes)
    assert all(p["outcome"] != "at_floor" for p in passes)


async def test_a_pass_that_cannot_reach_the_trigger_calls_no_summariser(
    scenario: ScenarioFactory,
) -> None:
    # An unbounded tail puts the fixed part over the trigger on purpose.
    run, summariser = _scenario(scenario, compaction_keep_recent_max_ratio=1.0)
    await run.run("read six pages")

    trigger = derive_budgets(run.engine.config.rc).compaction_trigger_tokens
    passes = _completed(run)
    assert passes and all(int(p["fixed_tokens"]) >= trigger for p in passes)
    assert len(summariser.calls) == 0
    assert run.engine.state.value == "completed"
