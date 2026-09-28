"""A tool asked for by its name without the server prefix is found first.

MCP tools are registered as ``mcp__<server>__<tool>``, and a model searching
for one writes the part it knows, ``get_issue``. The analyser keeps the
identifier's parts and the whole joined identifier, but not the joined tail:
the document carries ``mcpgithubgetissue`` and never ``getissue``, while the
query carries ``getissue`` and not the prefix. "get" is a stopword, so what is
left of the query is "issue", which every issue tool shares, and a
neighbour outranks the tool that was named exactly.
"""

from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "only the whole identifier is indexed joined, and 'get' is a stopword, "
        "so a bare MCP tool name scores like any tool sharing its other words"
    ),
)
@pytest.mark.parametrize("name", ["get_issue", "get_pull_request"])
def test_the_bare_name_of_an_mcp_tool_ranks_it_first(name: str) -> None:
    index = ToolIndex(
        AnalyzedCatalogue(_CATALOGUE),
        RetrievalSettings.from_constants(LoopConstants()),
        Lexicon.bundled(),
    )
    assert index.rank(name, 1) == [f"mcp__github__{name}"]
