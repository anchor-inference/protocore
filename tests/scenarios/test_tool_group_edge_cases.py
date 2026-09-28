"""Edge cases of tool groups: loading whole, rules and their cost, the rules mark.

Each case states what a host or a model observes, and is marked as an expected
failure where the runtime does not do it yet.
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from protocore.contracts.tools import ToolContext
from protocore.contracts.types import MessageRole, TextBlock, ToolResult
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.tool_deferral import tool_rules_mark
from protocore.runtime.tool_surface import forget_tool_surfaces
from protocore.tests_support.adapters import InMemoryToolRegistry
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


def _six_browser_tools_under_a_limit(scenario: ScenarioFactory, limit: int) -> Scenario:
    run = scenario(
        tools=[
            ScriptedTool(tool_name="Note", description="record a note"),
            ScriptedTool(tool_name="Zeta", description="the last tool"),
            *_browser(6),
        ],
        rc=default_rc(max_advertised_tools=limit, pinned_tool_max_count=2),
    )
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group("browser", "Drive a web browser", prefix="Browser", load="lazy")
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"group": "browser"}
    )
    run.llm.queue_response(text="loaded")
    return run


async def test_a_group_loaded_whole_stays_whole_under_the_provider_tool_limit(
    scenario: ScenarioFactory,
) -> None:
    """What the model is told about a group is true of the next tool list:
    a group that fits the provider's limit is carried whole, and one that
    would not is not loaded at all rather than loaded and then cut in half."""
    fits = _six_browser_tools_under_a_limit(scenario, limit=9)
    await fits.run("open a page")
    (result,) = fits.tool_results()
    assert result.content.startswith(
        "Loaded, and callable from your next step: " + ", ".join(f"Browser{i:02d}" for i in range(6))
    )
    assert [n for n in fits.advertised_tool_names(1) if n.startswith("Browser")] == [
        f"Browser{i:02d}" for i in range(6)
    ]

    # Note, Zeta and ToolSearch leave room for three: six cannot come whole.
    over = _six_browser_tools_under_a_limit(scenario, limit=6)
    await over.run("open a page")
    (result,) = over.tool_results()
    assert result.content.startswith("Nothing was loaded.\n\nThe browser group (6 tools) is not loaded")
    assert "over the provider's limit on the number of tools" in result.content
    assert not any(n.startswith("Browser") for n in over.advertised_tool_names(1))
    assert over.engine.context_manager.loaded_tool_group_names() == ()


@dataclass
class _Register(ScriptedTool):
    """Stands in for the host registering a tool mid-run, which grows the base
    surface under the loaded tail."""

    tool_name: str = "Register"
    description: str = "register another tool"
    registry: Any = None

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.registry.register(ScriptedTool(tool_name="Extra", description="an extra tool"))
        return await super().invoke(context, arguments)


async def test_a_loaded_tail_over_the_provider_limit_is_trimmed_by_whole_entries(
    scenario: ScenarioFactory,
) -> None:
    """When the base grows under a loaded tail that fit before, the tail is
    trimmed by entries: a group loaded whole leaves the request whole, and a
    single tool loaded after it — more recent, and small enough — stays."""
    register = _Register()
    run = scenario(
        tools=[
            ScriptedTool(tool_name="Note", description="record a note"),
            ScriptedTool(tool_name="Solo", description="a tool on its own"),
            *_browser(2),
            register,
        ],
        rc=default_rc(max_advertised_tools=6, pinned_tool_max_count=3),
    )
    register.registry = run.tools
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group("browser", "Drive a web browser", prefix="Browser", load="lazy")
    run.tools.declare_group("solo", "A tool of its own", prefix="Solo", load="lazy")
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"group": "browser"}
    )
    run.llm.queue_tool_call_response(
        tool_call_id="s-2", tool_name="ToolSearch", tool_input={"select": ["Solo"]}
    )
    run.llm.queue_tool_call_response(tool_call_id="r-1", tool_name="Register", tool_input={})
    run.llm.queue_response(text="done")
    await run.run("load, then grow")

    # Base Note, Register, ToolSearch plus the group and Solo: six, at the limit.
    before = run.advertised_tool_names(2)
    assert len(before) == 6 and before[-3:] == ["Browser00", "Browser01", "Solo"]
    # Extra makes the base four; two places are left. The group of two is the
    # older entry and does not fit beside the more recent Solo, so it leaves
    # whole; Solo stays. Nothing is unloaded: both come back when there is room.
    after = run.advertised_tool_names(3)
    assert "Extra" in after and after[-1] == "Solo" and len(after) == 5
    assert not any(name.startswith("Browser") for name in after)
    assert run.engine.context_manager.loaded_tool_group_names() == ("browser",)


async def test_rules_of_a_group_on_the_surface_count_against_the_tool_budget(
    scenario: ScenarioFactory,
) -> None:
    rules = "Always double-check the page before acting on it. " * 150  # 7 500 chars
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
    # by the rules alone (about 1 900 tokens), so the group is held back and
    # its rules are given only when it is loaded. Counted as definitions only,
    # the group stayed on the surface, the rules went into the system prompt,
    # and the first request was refused as over the window before it was sent.
    assert run.requests, "no request was sent: the rules filled the context window"
    assert "Always double-check the page" not in _system_text(run, 0)
    assert "BrowserOpen" not in run.advertised_tool_names(0)
    assert "- browser: Drive a web browser. Tools: BrowserOpen" in _system_text(run, 0)


def test_rules_longer_than_the_cap_are_refused_at_the_declaration() -> None:
    """Rules of a group that cannot be held back go into every request, so
    their length is bounded where the host declares them, not discovered at
    a run's first request."""
    registry = InMemoryToolRegistry()
    registry.declare_group("browser", "Drive a web browser", instructions="x" * 8_000)
    with pytest.raises(ValueError, match="instructions run to 8001 characters"):
        registry.declare_group("browser", "Drive a web browser", instructions="x" * 8_001)
    assert len(registry.tool_groups()[0].instructions) == 8_000


_MARK_IN_A_FRESH_PROCESS = (
    "from protocore.runtime.tool_deferral import tool_rules_mark;"
    "print(tool_rules_mark('session-1'))"
)


def test_the_rules_mark_of_a_session_is_the_same_in_every_process() -> None:
    marks = {
        subprocess.run(
            [sys.executable, "-c", _MARK_IN_A_FRESH_PROCESS],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(marks) == 1


def test_the_rules_mark_follows_its_scope_and_the_hosts_key() -> None:
    """Eight hex digits; another scope or another key is another mark, and the
    same scope under the same key is the same mark wherever it is computed."""
    mark = tool_rules_mark("tenant-a")
    assert len(mark) == 8 and int(mark, 16) >= 0
    assert tool_rules_mark("tenant-a") == mark
    assert tool_rules_mark("tenant-b") != mark
    assert tool_rules_mark("tenant-a", key="deployment-secret") != mark
    assert tool_rules_mark("tenant-a", key="deployment-secret") == tool_rules_mark(
        "tenant-a", key="deployment-secret"
    )


async def test_two_sessions_with_the_same_setup_send_the_same_system_prompt(
    scenario: ScenarioFactory,
) -> None:
    prompts = []
    for session in ("session-a", "session-b"):
        run = scenario(
            session_id=session,
            tools=[
                ScriptedTool(tool_name="Note", description="record a note"),
                ScriptedTool(tool_name="BrowserOpen", description="open a page"),
            ],
        )
        run.tools.register(ToolSearchTool(run.tools))
        run.tools.declare_group(
            "browser",
            "Drive a web browser",
            prefix="Browser",
            load="lazy",
            instructions="Ask before submitting a form.",
        )
        run.llm.queue_response(text="done")
        await run.run("hello")
        prompts.append(_system_text(run, 0))
    assert prompts[0] == prompts[1]


def _large_browser_run(scenario: ScenarioFactory) -> Scenario:
    """Thirty browser tools whose definitions together are well over the
    2 048-token budget an 8k window gives, held back as a lazy group with rules."""
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
    return run


def _sent_definition_tokens(run: Scenario) -> tuple[int, int]:
    rc = run.engine.config.rc
    budget = derive_budgets(rc).tool_definitions_budget_tokens
    sent = sum(
        estimate_tokens(definition.model_dump_json(), rc)
        for definition in run.requests[-1].tools
    )
    return sent, budget


async def test_a_blind_call_does_not_load_past_the_tool_budget(
    scenario: ScenarioFactory,
) -> None:
    run = _large_browser_run(scenario)
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Browser00", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="read the rules")
    await run.run("open a page")

    sent, budget = _sent_definition_tokens(run)
    assert sent <= budget
    # Only the called tool is loaded, on its own and not as the group, and the
    # answer says how many more there are and how to load the ones needed.
    (held,) = run.tool_results()
    assert "Rules for the browser tools" in held.content
    assert (
        "The rest of the browser group (29 tools) is not loaded: the whole group would "
        "not fit your tool list, so load the ones you need with ToolSearch ('select:' "
        "and their exact names)." in held.content
    )
    assert "callable too" not in held.content
    assert [n for n in run.advertised_tool_names(1) if n.startswith("Browser")] == ["Browser00"]
    assert run.engine.context_manager.loaded_tool_group_names() == ()


async def test_a_group_load_over_the_budget_loads_nothing_and_says_so(
    scenario: ScenarioFactory,
) -> None:
    """``ToolSearch(group=...)`` is held to the same budget: a group that would
    not fit is listed for the model to pick from, and nothing is loaded."""
    run = _large_browser_run(scenario)
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"group": "browser"}
    )
    run.llm.queue_response(text="picked")
    await run.run("open a page")

    (result,) = run.tool_results()
    assert result.content.startswith("Nothing was loaded.\n\nThe browser group (30 tools) is not loaded")
    assert "over the budget for tool definitions" in result.content
    assert "... and 22 more; describe what you need to find them." in result.content
    assert "Rules for the browser tools" not in result.content
    assert not any(n.startswith("Browser") for n in run.advertised_tool_names(1))
    sent, budget = _sent_definition_tokens(run)
    assert sent <= budget


def _ruled_browser_run(scenario: ScenarioFactory) -> Scenario:
    run = scenario(
        tools=[
            ScriptedTool(tool_name="Note", description="record a note"),
            ScriptedTool(tool_name="BrowserOpen", description="open a page"),
            ScriptedTool(tool_name="BrowserClick", description="click on a page"),
        ],
    )
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group(
        "browser",
        "Drive a web browser",
        prefix="Browser",
        load="lazy",
        instructions="Ask the user before submitting a form.",
    )
    return run


async def test_rules_are_given_once_when_one_message_loads_a_group_twice(
    scenario: ScenarioFactory,
) -> None:
    run = _ruled_browser_run(scenario)
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("s-1", "ToolSearch", {"select": "BrowserOpen"}),
            ("s-2", "ToolSearch", {"select": "BrowserClick"}),
        ]
    )
    run.llm.queue_response(text="loaded")
    await run.run("load the browser")

    results = run.tool_results()
    assert sum("Ask the user before submitting a form." in r.content for r in results) == 1
    # The first load gives them; the second, folded in after it, owes none.
    assert "Rules for the browser tools" in results[0].content
    assert "Rules for the browser tools" not in results[1].content
    assert run.advertised_tool_names(1)[-2:] == ["BrowserOpen", "BrowserClick"]


async def test_a_blind_call_beside_a_search_of_its_group_is_pointed_at_the_search(
    scenario: ScenarioFactory,
) -> None:
    """A search that loads the group and a blind call of one of its tools in
    the same message: the call is held, as decided before the message ran, and
    its answer points at the search's rules rather than repeating them."""
    run = _ruled_browser_run(scenario)
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("s-1", "ToolSearch", {"group": "browser"}),
            ("c-1", "BrowserClick", {"v": "x"}),
        ]
    )
    run.llm.queue_response(text="loaded")
    await run.run("load the browser and click")

    searched, held = run.tool_results()
    assert "Rules for the browser tools" in searched.content
    assert held.content.startswith(
        "Not run yet: BrowserClick was not in your tool list. The rules for the "
        "browser tools are in another result of this step."
    )
    assert sum("Ask the user before submitting a form." in r.content for r in (searched, held)) == 1
