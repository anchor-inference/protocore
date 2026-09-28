"""Edge cases of tool groups: loading whole, rules and their cost, the rules mark.

Each case states what a host or a model observes, and is marked as an expected
failure where the runtime does not do it yet.
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import pytest

from protocore.contracts.types import MessageRole, TextBlock
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.tool_surface import forget_tool_surfaces
from protocore.tools import ToolSearchTool

from .conftest import Scenario, ScenarioFactory, ScriptedTool, default_rc


@pytest.fixture(autouse=True)
def _forget_surfaces() -> Iterator[None]:
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


def _system_text(run: Scenario, index: int) -> str:
    return "\n".join(
        block.text
        for message in run.requests[index].messages
        if message.role is MessageRole.system
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    )


def _browser(count: int) -> list[ScriptedTool]:
    return [
        ScriptedTool(tool_name=f"Browser{index:02d}", description=f"browser action {index}")
        for index in range(count)
    ]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the count limit reserves room for pinned_tool_max_count loaded TOOLS while a "
        "group loaded whole is one ENTRY of many tools; _fit_loaded then trims the "
        "loaded tail tool by tool and leaves half of a group the model was told is loaded"
    ),
)
async def test_a_group_loaded_whole_stays_whole_under_the_provider_tool_limit(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(
        tools=[
            ScriptedTool(tool_name="Note", description="record a note"),
            ScriptedTool(tool_name="Zeta", description="the last tool"),
            *_browser(6),
        ],
        rc=default_rc(max_advertised_tools=6, pinned_tool_max_count=2),
    )
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group("browser", "Drive a web browser", prefix="Browser", load="lazy")
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"group": "browser"}
    )
    run.llm.queue_response(text="loaded")
    await run.run("open a page")

    (search_result,) = run.tool_results()
    # The model is told all six are loaded and callable ...
    assert search_result.content.startswith(
        "Loaded, and callable from your next step: " + ", ".join(f"Browser{i:02d}" for i in range(6))
    )
    # ... and the next request must then carry all six, or none of them.
    advertised = run.advertised_tool_names(1)
    loaded = [name for name in advertised if name.startswith("Browser")]
    assert loaded in ([], [f"Browser{i:02d}" for i in range(6)])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a group's rules are written into the system prompt but never counted against "
        "tool_definitions_ratio: a group whose definitions fit stays on the surface and "
        "its rules land in every request, however long they are"
    ),
)
async def test_rules_of_a_group_on_the_surface_count_against_the_tool_budget(
    scenario: ScenarioFactory,
) -> None:
    rules = "Always double-check the page before acting on it. " * 400  # ~20 000 chars
    run = scenario(
        tools=[
            ScriptedTool(tool_name="Note", description="record a note"),
            ScriptedTool(tool_name="BrowserOpen", description="open a page"),
        ],
        rc=default_rc(model_context_window=4_096),
    )
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group(
        "browser", "Drive a web browser", prefix="Browser", load="auto", instructions=rules
    )
    run.llm.queue_response(text="done")
    await run.run("hello")

    # The 1 024-token budget for tools (a quarter of the window) is exceeded
    # five times over by the rules alone, so the group should be held back and
    # its rules given only when it is loaded. Today the rules go into the
    # system prompt and the first request is refused as over the window
    # (LLMContextWindowExceeded) before it is sent.
    assert run.requests, "no request was sent: the rules filled the context window"
    assert "Always double-check the page" not in _system_text(run, 0)
    assert "BrowserOpen" not in run.advertised_tool_names(0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a blind call of one tool of a group with rules loads every tool of the group, "
        "with no check against tool_definitions_ratio: a large group loaded whole "
        "puts the request's definitions far over the budget the deferral was enforcing"
    ),
)
async def test_a_blind_call_does_not_load_past_the_tool_budget(
    scenario: ScenarioFactory,
) -> None:
    long = "performs one specific browser action and reports what happened on the page. " * 3
    run = scenario(
        tools=[
            ScriptedTool(tool_name="Note", description="record a note"),
            *[
                ScriptedTool(tool_name=f"Browser{index:02d}", description=f"{long} ({index})")
                for index in range(30)
            ],
        ],
        rc=default_rc(model_context_window=8_192),
    )
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group(
        "browser",
        "Drive a web browser",
        prefix="Browser",
        load="lazy",
        instructions="Ask before submitting a form.",
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Browser00", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="read the rules")
    await run.run("open a page")

    rc = run.engine.config.rc
    budget = derive_budgets(rc).tool_definitions_budget_tokens
    sent = sum(
        estimate_tokens(definition.model_dump_json(), rc)
        for definition in run.requests[-1].tools
    )
    assert sent <= budget
