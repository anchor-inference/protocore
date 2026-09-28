"""What retrieval shares across registries is built once per process.

A host may build a registry per request. Reading and building the bundled
lexicon, and tokenising and stemming every tool, cost far more than a query, so
neither may be repeated for a registry over tools the process has already
analysed. The assertions count builds, not wall time, so they hold on any
machine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime import tool_retrieval
from protocore.runtime.tool_registry import ToolRegistry
from protocore.runtime.tool_retrieval import AnalyzedCatalogue, Lexicon, ToolIndex

from ._tool_fixtures import MockTool


def _tools(prefix: str) -> list[MockTool]:
    return [
        MockTool(tool_name=f"{prefix}Read", description="Read a text file from the workspace."),
        MockTool(tool_name=f"{prefix}Edit", description="Replace an exact string in an existing file."),
        MockTool(tool_name=f"{prefix}Search", description="Search the internet and return links."),
    ]


def test_the_bundled_lexicon_is_built_once_for_every_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[int] = []
    original = Lexicon.from_translations.__func__  # type: ignore[attr-defined]

    def counting(cls: type[Lexicon], translations: Mapping[str, Sequence[str]]) -> Lexicon:
        builds.append(1)
        result: Lexicon = original(cls, translations)
        return result

    monkeypatch.setattr(Lexicon, "from_translations", classmethod(counting))
    tool_retrieval._bundled_lexicon.cache_clear()
    try:
        for _ in range(3):
            registry = ToolRegistry(_tools("Lexicon"))
            assert [tool.name for tool in registry.search("отредактируй файл", top_k=1)] == ["LexiconEdit"]
        assert Lexicon.bundled() is Lexicon.bundled()
        assert len(builds) == 1
    finally:
        # Leave a lexicon built by the real classmethod for later tests.
        monkeypatch.undo()
        tool_retrieval._bundled_lexicon.cache_clear()


def test_the_bundled_lexicon_cannot_be_changed() -> None:
    lexicon = Lexicon.bundled()
    with pytest.raises(TypeError):
        lexicon._expansions["файл"] = ("changed",)  # type: ignore[index]


def test_a_registry_over_known_tools_analyses_no_tool_again() -> None:
    first = ToolRegistry(_tools("Rebuild"))
    first.search("edit a file", top_k=1)
    before = tool_retrieval._analyze_document.cache_info()

    second = ToolRegistry(_tools("Rebuild"))
    assert [tool.name for tool in second.search("edit a file", top_k=1)] == ["RebuildEdit"]
    after = tool_retrieval._analyze_document.cache_info()

    assert after.misses == before.misses
    assert after.hits - before.hits == 3


def test_a_changed_tool_is_the_only_one_analysed_again() -> None:
    documents = [
        ToolDocument("ChangeRead", "Read a text file."),
        ToolDocument("ChangeEdit", "Edit a text file."),
        ToolDocument("ChangeList", "List a directory."),
    ]
    AnalyzedCatalogue(documents)
    before = tool_retrieval._analyze_document.cache_info()

    changed = [*documents[:2], ToolDocument("ChangeList", "List a directory, recursively.")]
    AnalyzedCatalogue(changed)
    after = tool_retrieval._analyze_document.cache_info()

    assert after.misses - before.misses == 1


def test_the_fallback_matcher_is_built_on_first_use_and_kept() -> None:
    catalogue = AnalyzedCatalogue([ToolDocument("Remember", "Save a fact to long-term memory.")])
    index = ToolIndex(catalogue, RetrievalSettings.from_constants(LoopConstants()), None)
    assert catalogue._fallback is None
    assert index.fallback("памяти memo", 5) == ["Remember"]
    matcher = catalogue._fallback
    assert matcher is not None
    assert index.fallback("memor", 5) == ["Remember"]
    assert catalogue._fallback is matcher


def test_a_rebuilt_index_ranks_like_the_first() -> None:
    documents = [
        ToolDocument("mcp__github__get_issue", "Get details of a specific issue."),
        ToolDocument("mcp__github__add_issue_comment", "Add a comment to an existing issue."),
        ToolDocument("BrowserScreenshot", "Take a screenshot of the current browser page."),
    ]
    settings = RetrievalSettings.from_constants(LoopConstants())
    queries = ("get_issue", "сделай скриншот", "comment on an issue")
    first = ToolIndex(AnalyzedCatalogue(documents), settings, Lexicon.bundled())
    second = ToolIndex(AnalyzedCatalogue(list(reversed(documents))), settings, Lexicon.bundled())
    for query in queries:
        assert first.rank(query, 3) == second.rank(query, 3)
        assert first.scores(query) == second.scores(query)
