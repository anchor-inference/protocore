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
from protocore.contracts.types import MessageRole, TextBlock, ToolResult
from protocore.runtime.events import EventType
from protocore.runtime.tool_surface import forget_tool_surfaces
from protocore.tools import ToolSearchTool

from .conftest import Scenario, ScenarioFactory, ScriptedTool, default_rc


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
    # Nor is the seed announced as having loaded it.
    assert not run.events_of(EventType.TOOL_GROUP_LOADED)
    assert not run.events_of(EventType.TOOL_DISCOVERED)


async def test_a_seeded_tool_inside_the_whitelist_is_loaded_as_before(
    scenario: ScenarioFactory,
) -> None:
    """The check admits what the policy admits: a seed the whitelist covers
    is advertised, announced and callable exactly as it was."""
    run = _with_search(
        scenario(
            tools=_tools(),
            tool_visibility_policy=ToolVisibilityPolicy(
                visible={"Note", "Zeta", "ToolSearch", "Mcp_Github_create_issue"}
            ),
            discovered_tools=("Mcp_Github_create_issue",),
        )
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_create_issue", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="done")
    await run.run("file it")

    assert run.advertised_tool_names(0)[-1] == "Mcp_Github_create_issue"
    assert not run.tool_results()[0].is_error
    (loaded,) = run.events_of(EventType.TOOL_GROUP_LOADED)
    assert loaded.payload == {
        "group": "github",
        "via": "seed",
        "tools": ["Mcp_Github_create_issue"],
    }


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


async def test_a_loaded_tool_the_whitelist_drops_and_readmits_comes_back_where_it_was(
    scenario: ScenarioFactory,
) -> None:
    """The loaded list keeps the name while the policy excludes it, so when the
    host admits the tool again it is advertised again, at its old place in the
    loaded tail, without another search."""
    narrow = _Narrow(visible=frozenset({"Note", "Zeta", "ToolSearch", "Narrow", "Widen"}))
    widen = _Narrow(tool_name="Widen", description="widen the whitelist", visible=frozenset())
    run = _with_search(scenario(tools=[*_tools(), narrow, widen]))
    narrow.engine = widen.engine = run.engine
    run.llm.queue_tool_call_response(
        tool_call_id="s-1",
        tool_name="ToolSearch",
        tool_input={"query": "select:Mcp_Github_create_issue"},
    )
    run.llm.queue_tool_call_response(tool_call_id="n-1", tool_name="Narrow", tool_input={})
    run.llm.queue_tool_call_response(tool_call_id="w-1", tool_name="Widen", tool_input={})
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_create_issue", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="done")
    await run.run("load, narrow, widen, call")

    assert run.advertised_tool_names(1)[-1] == "Mcp_Github_create_issue"
    assert "Mcp_Github_create_issue" not in run.advertised_tool_names(2)
    assert run.advertised_tool_names(3)[-1] == "Mcp_Github_create_issue"
    by_id = {block.tool_call_id: block for block in run.tool_results()}
    assert not by_id["c-1"].is_error, by_id["c-1"].content
    assert run.engine.context_manager.called_discovered_tool_names() == (
        "Mcp_Github_create_issue",
    )


async def test_a_clipped_surface_keeps_its_search_tool(scenario: ScenarioFactory) -> None:
    """With the per-message clip on, the tools it leaves off can only be found
    by a search; the search tool is always-load for exactly that reason."""
    run = scenario(tools=_tools(), rc=default_rc(tool_retrieval_top_k=1))
    run.tools.register(ToolSearchTool(run.tools))
    run.llm.queue_response(text="done")
    await run.run("note this down")

    advertised = run.advertised_tool_names(0)
    assert len(advertised) < len(_tools()) + 1  # the clip did leave tools off
    assert "ToolSearch" in advertised


async def test_a_clipped_tool_is_found_and_loaded_by_the_search(
    scenario: ScenarioFactory,
) -> None:
    """The search tool kept on a clipped surface does its job: a tool the clip
    left off is found, loaded, and appended after the clipped base."""
    run = scenario(tools=_tools(), rc=default_rc(tool_retrieval_top_k=1))
    run.tools.register(ToolSearchTool(run.tools))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"query": "select:Zeta"}
    )
    run.llm.queue_response(text="done")
    await run.run("note this down")

    assert "Zeta" not in run.advertised_tool_names(0)
    assert run.advertised_tool_names(1)[-1] == "Zeta"
    assert "ToolSearch" in run.advertised_tool_names(1)


async def test_a_clip_that_leaves_nothing_off_hides_the_search_tool(
    scenario: ScenarioFactory,
) -> None:
    """A clip wide enough for every admitted tool leaves the search nothing to
    find, and the surface is what it would be with the clip off."""
    run = scenario(tools=_tools(), rc=default_rc(tool_retrieval_top_k=len(_tools())))
    run.tools.register(ToolSearchTool(run.tools))
    run.llm.queue_response(text="done")
    await run.run("note this down")

    advertised = run.advertised_tool_names(0)
    assert "ToolSearch" not in advertised
    assert sorted(advertised) == sorted(tool.name for tool in _tools())


async def test_a_clip_that_leaves_off_only_tools_outside_the_declared_set_hides_the_search(
    scenario: ScenarioFactory,
) -> None:
    """A child declared to use Note alone has all of it on a clipped surface;
    the tools the clip left off are not the child's to load, so a search
    could find nothing and is not advertised."""
    run = scenario(
        tools=_tools(),
        rc=default_rc(tool_retrieval_top_k=1),
        subagent_tool_allowlist=("Note", "ToolSearch"),
    )
    run.tools.register(ToolSearchTool(run.tools))
    run.llm.queue_response(text="done")
    await run.run("note this down")

    assert run.advertised_tool_names(0) == ["Note"]


async def test_with_the_clip_off_the_surface_is_built_once_per_request(
    scenario: ScenarioFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is left off for a clip that is off, so no unclipped surface is
    built beside the one advertised to see what it left off."""
    run = scenario(tools=_tools())
    run.tools.register(ToolSearchTool(run.tools))
    calls: list[int | None] = []
    compute = run.tools.compute_effective_surface

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("top_k"))
        return compute(*args, **kwargs)

    monkeypatch.setattr(run.tools, "compute_effective_surface", counted)
    run.llm.queue_response(text="done")
    await run.run("note this down")

    assert "ToolSearch" not in run.advertised_tool_names(0)
    assert calls and all(top_k is None for top_k in calls)
    clipped = scenario(tools=_tools(), rc=default_rc(tool_retrieval_top_k=1))
    clipped.tools.register(ToolSearchTool(clipped.tools))
    clipped_calls: list[int | None] = []
    clipped_compute = clipped.tools.compute_effective_surface

    def clipped_counted(*args: Any, **kwargs: Any) -> Any:
        clipped_calls.append(kwargs.get("top_k"))
        return clipped_compute(*args, **kwargs)

    monkeypatch.setattr(clipped.tools, "compute_effective_surface", clipped_counted)
    clipped.llm.queue_response(text="done")
    await clipped.run("note this down")
    # Built with the clip, and once more without it to see what it left off.
    assert 1 in clipped_calls and None in clipped_calls
    assert len(clipped_calls) > len(calls)


async def test_the_catalogue_names_no_group_the_declared_tool_set_cannot_reach(
    scenario: ScenarioFactory,
) -> None:
    """A child declared to use Note alone is told the github tools are there to
    load; every search for them comes back empty and every call is refused."""
    run = _with_search(
        scenario(tools=_tools(), subagent_tool_allowlist=("Note", "ToolSearch"))
    )
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert "Mcp_Github_" not in _system_text(run)
    # Nothing is held back, so there is nothing for a search to find either.
    assert "ToolSearch" not in run.advertised_tool_names(0)


async def test_the_catalogue_lists_only_the_tools_of_a_group_the_declared_set_reaches(
    scenario: ScenarioFactory,
) -> None:
    """A child declared to use one github tool has the group held back with
    that tool alone in its line: the other is not the child's to load."""
    run = _with_search(
        scenario(
            tools=_tools(),
            subagent_tool_allowlist=("Note", "ToolSearch", "Mcp_Github_list_issues"),
        )
    )
    run.llm.queue_response(text="done")
    await run.run("hello")

    text = _system_text(run)
    assert "Mcp_Github_list_issues" in text
    assert "Mcp_Github_create_issue" not in text
    assert "ToolSearch" in run.advertised_tool_names(0)


def _system_text(run: Scenario) -> str:
    return "\n".join(
        block.text
        for message in run.requests[0].messages
        if message.role is MessageRole.system
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    )


@dataclass
class _LeavePlan(ScriptedTool):
    """Stands in for the host ending plan mode mid-run: the profile, not the
    visibility policy, is what changes."""

    tool_name: str = "LeavePlan"
    description: str = "leave plan mode"
    engine: Any = None

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.engine.config = replace(self.engine.config, execution_profile="default")
        return await super().invoke(context, arguments)


async def test_a_dynamic_group_a_profile_change_admits_is_held_back(
    scenario: ScenarioFactory,
) -> None:
    """Planned under the plan profile, the github tools were not admitted and
    nothing was held back; the profile ends and the decision is not made again,
    so the whole server lands on the surface instead of in the catalogue."""
    leave = _LeavePlan()
    run = _with_search(
        scenario(
            tools=[*_tools(), leave],
            execution_profile="plan",
            rc=default_rc(
                execution_profile_plan_enabled=True,
                execution_profile_plan_tools="Note,Zeta,LeavePlan,ToolSearch",
            ),
        )
    )
    leave.engine = run.engine
    run.llm.queue_tool_call_response(tool_call_id="l-1", tool_name="LeavePlan", tool_input={})
    run.llm.queue_response(text="done")
    await run.run("plan, then act")

    assert not any(name.startswith("Mcp_Github_") for name in run.advertised_tool_names(0))
    assert not any(name.startswith("Mcp_Github_") for name in run.advertised_tool_names(1))
