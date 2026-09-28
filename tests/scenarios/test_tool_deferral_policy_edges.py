"""Held-back and loaded tools at the edges of the visibility policy.

A loaded tool is admitted by being folded into the run's ``pinned`` set, and
``pinned`` is admitted past a ``visible`` whitelist. What loads a tool must
therefore be checked against the policy before the load, or the load itself
widens what the run may call.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import pytest

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import ToolResult
from protocore.runtime.tool_surface import forget_tool_surfaces
from protocore.tools import ToolSearchTool

from .conftest import Scenario, ScenarioFactory, ScriptedTool


@pytest.fixture(autouse=True)
def _forget_surfaces() -> Iterator[None]:
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


def _tools() -> list[ScriptedTool]:
    return [
        ScriptedTool(tool_name="Note", description="record a note"),
        ScriptedTool(tool_name="Mcp_Github_list_issues", description="list the issues"),
        ScriptedTool(tool_name="Mcp_Github_create_issue", description="open an issue"),
        ScriptedTool(tool_name="Zeta", description="the last tool by name"),
    ]


def _with_search(run: Scenario) -> Scenario:
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group("github", "GitHub issues", dynamic=True, prefix="Mcp_Github_")
    return run


@pytest.mark.xfail(
    strict=True,
    reason=(
        "QueryEngineConfig.discovered_tools is loaded without a policy check; "
        "the loaded name is then folded into pinned, which a visible whitelist admits"
    ),
)
async def test_a_seeded_tool_outside_the_visible_whitelist_is_neither_advertised_nor_run(
    scenario: ScenarioFactory,
) -> None:
    """A host hands the last run's loaded tools back as the seed; the operator
    has since narrowed the tenant's whitelist so the tool is no longer visible.
    The seed must not make it callable again."""
    run = _with_search(
        scenario(
            tools=_tools(),
            tool_visibility_policy=ToolVisibilityPolicy(visible={"Note", "Zeta", "ToolSearch"}),
            discovered_tools=("Mcp_Github_create_issue",),
        )
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_create_issue", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="done")
    await run.run("file it")

    assert "Mcp_Github_create_issue" not in run.advertised_tool_names(0)
    result = run.tool_results()[0]
    assert result.is_error, result.content


@dataclass
class _Narrow(ScriptedTool):
    """Stands in for the operator narrowing the tenant's whitelist mid-run."""

    tool_name: str = "Narrow"
    description: str = "narrow the whitelist"
    engine: Any = None
    visible: frozenset[str] = frozenset()

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.engine.config = replace(
            self.engine.config,
            tool_visibility_policy=ToolVisibilityPolicy(visible=set(self.visible)),
        )
        return await super().invoke(context, arguments)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a loaded tool stays in the effective pinned set after the host's whitelist "
        "drops it, and pinned is admitted past visible; the re-made deferral decision "
        "does not unload it"
    ),
)
async def test_a_loaded_tool_the_whitelist_later_drops_is_no_longer_callable(
    scenario: ScenarioFactory,
) -> None:
    narrow = _Narrow(visible=frozenset({"Note", "Zeta", "ToolSearch", "Narrow"}))
    run = _with_search(scenario(tools=[*_tools(), narrow]))
    narrow.engine = run.engine
    run.llm.queue_tool_call_response(
        tool_call_id="s-1",
        tool_name="ToolSearch",
        tool_input={"query": "select:Mcp_Github_create_issue"},
    )
    run.llm.queue_tool_call_response(tool_call_id="n-1", tool_name="Narrow", tool_input={})
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_create_issue", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="done")
    await run.run("load, narrow, call")

    assert "Mcp_Github_create_issue" in run.advertised_tool_names(1)
    assert "Mcp_Github_create_issue" not in run.advertised_tool_names(2)
    by_id = {block.tool_call_id: block for block in run.tool_results()}
    assert by_id["c-1"].is_error, by_id["c-1"].content
