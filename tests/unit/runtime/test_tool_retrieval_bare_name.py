"""A tool asked for by its name without the server prefix is found first.

MCP tools are registered as ``mcp__<server>__<tool>``, and a model searching
for one writes the part it knows, ``get_issue``. "get" is a stopword, so
without more the query is "issue", which every issue tool shares. The analyser
therefore indexes the joined form of every tail of an identifier's parts: the
document carries ``getissue`` as well as ``mcpgithubgetissue``, and the tool
that was named exactly ranks first.
"""

from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime.text_analysis import raw_tokens
from protocore.runtime.tool_retrieval import AnalyzedCatalogue, Lexicon, ToolIndex

_CATALOGUE = (
    ToolDocument("mcp__github__get_issue", "Get details of a specific issue in a GitHub repository."),
    ToolDocument("mcp__github__create_issue", "Create a new issue in a GitHub repository."),
    ToolDocument("mcp__github__list_issues", "List issues in a GitHub repository with filtering options."),
    ToolDocument("mcp__github__update_issue", "Update an existing issue in a GitHub repository."),
    ToolDocument("mcp__github__add_issue_comment", "Add a comment to an existing issue."),
    ToolDocument("mcp__github__get_pull_request", "Get details of a specific pull request."),
    ToolDocument("mcp__github__merge_pull_request", "Merge a pull request."),
    ToolDocument("mcp__github__list_pull_requests", "List and filter repository pull requests."),
)


@pytest.mark.parametrize("name", ["get_issue", "get_pull_request"])
def test_the_bare_name_of_an_mcp_tool_ranks_it_first(name: str) -> None:
    index = ToolIndex(
        AnalyzedCatalogue(_CATALOGUE),
        RetrievalSettings.from_constants(LoopConstants()),
        Lexicon.bundled(),
    )
    assert index.rank(name, 1) == [f"mcp__github__{name}"]


def test_every_tail_of_an_identifier_is_indexed_joined() -> None:
    assert raw_tokens("mcp__github__get_issue") == [
        "mcp", "github", "get", "issue", "mcpgithubgetissue", "githubgetissue", "getissue",
    ]
    assert raw_tokens("get_issue") == ["get", "issue", "getissue"]
