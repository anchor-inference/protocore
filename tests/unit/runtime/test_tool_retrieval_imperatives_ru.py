"""Russian imperatives must reach the English tools their infinitives reach.

A Russian operator asks a tool to act in the imperative ("отредактируй",
"найди", "поищи"), while the bundled lexicon lists infinitives. Snowball
stems the two apart ("редактировать" -> "редактирова", "редактируй" ->
"редактир"; "найти" -> "найт", "найди" -> "найд"), so the expansion that works
for the infinitive never fires for the imperative, although the lexicon
promises to match "whatever inflection the query uses".
"""

from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime.stemmers import stem
from protocore.runtime.tool_retrieval import AnalyzedCatalogue, Lexicon, ToolIndex

_REASON = (
    "the bundled lexicon is keyed by infinitives; Snowball stems imperatives "
    "(-ируй, -ди, -щи) apart from them, so the expansion never fires"
)

_CATALOGUE = (
    ToolDocument("Read", "Read a text file from the workspace.", parameters="path"),
    ToolDocument("Edit", "Replace an exact string in an existing file with new text.", parameters="path old_string new_string"),
    ToolDocument("UploadFile", "Upload a local file and return a link to share it.", parameters="path"),
    ToolDocument("ArchiveFiles", "Pack files into a zip archive.", parameters="paths destination"),
    ToolDocument("CopyFile", "Copy a file to a new path.", parameters="source destination"),
    ToolDocument("RenameFile", "Rename a file.", parameters="source new_name"),
    ToolDocument("WebSearch", "Search the internet and return result titles and links.", parameters="query"),
)


def _index() -> ToolIndex:
    return ToolIndex(
        AnalyzedCatalogue(_CATALOGUE),
        RetrievalSettings.from_constants(LoopConstants()),
        Lexicon.bundled(),
    )


@pytest.mark.xfail(strict=True, reason=_REASON)
@pytest.mark.parametrize(
    ("imperative", "infinitive", "english"),
    [
        ("найди", "найти", "find"),
        ("отредактируй", "изменить", "edit"),
        ("редактируй", "изменить", "edit"),
        ("поищи", "поиск", "search"),
        ("скопируй", "копировать", "copi"),
        ("переименуй", "переименовать", "renam"),
    ],
)
def test_an_imperative_expands_like_its_infinitive(imperative: str, infinitive: str, english: str) -> None:
    lexicon = Lexicon.bundled()
    assert english in lexicon.expand(stem(infinitive))
    assert english in lexicon.expand(stem(imperative))


@pytest.mark.xfail(strict=True, reason=_REASON)
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("отредактируй файл", "Edit"),
        ("скопируй файл", "CopyFile"),
    ],
)
def test_an_imperative_request_finds_its_tool(query: str, expected: str) -> None:
    assert _index().rank(query, 1) == [expected]
