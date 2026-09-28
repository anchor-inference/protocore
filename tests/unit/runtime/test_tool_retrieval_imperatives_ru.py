"""Russian imperatives must reach the English tools their infinitives reach.

A Russian operator asks a tool to act in the imperative ("отредактируй",
"найди", "поищи"), while the bundled lexicon lists infinitives. Snowball
stems the two apart ("редактировать" -> "редактирова", "редактируй" ->
"редактир"; "найти" -> "найт", "найди" -> "найд"), so the lexicon registers
each infinitive under its imperatives' stems too, and a prefixed perfective
("отредактируй") is looked up once more without its prefix.
"""
# ruff: noqa: RUF003 — Russian word forms are the point of this file

from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime.stemmers import stem
from protocore.runtime.tool_retrieval import AnalyzedCatalogue, Lexicon, ToolIndex

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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("отредактируй файл", "Edit"),
        ("скопируй файл", "CopyFile"),
    ],
)
def test_an_imperative_request_finds_its_tool(query: str, expected: str) -> None:
    assert _index().rank(query, 1) == [expected]


@pytest.mark.parametrize(
    ("imperative", "infinitive", "english"),
    [
        ("удали", "удалить", "delet"),
        ("сохрани", "сохранить", "save"),
        ("открой", "открыть", "open"),
        ("покажи", "показать", "show"),
        ("напиши", "написать", "write"),
        ("отправь", "отправить", "send"),
        ("проверь", "проверить", "check"),
        ("запусти", "запустить", "run"),
        ("обнови", "обновить", "updat"),
        ("посмотри", "посмотреть", "look"),
        ("сравни", "сравнить", "compar"),
        ("переведи", "перевести", "translat"),
        ("ищи", "искать", "search"),
        ("разверни", "развернуть", "deploi"),
        ("проанализируй", "анализировать", "analyz"),
    ],
)
def test_common_imperatives_expand_like_their_infinitive(imperative: str, infinitive: str, english: str) -> None:
    lexicon = Lexicon.bundled()
    assert english in lexicon.expand(stem(infinitive))
    assert english in lexicon.expand(stem(imperative))


def test_imperatives_are_registered_from_a_custom_lexicon() -> None:
    lexicon = Lexicon.from_translations({"archive": ["архивировать"], "shrink": ["сжать"]})
    assert lexicon.expand(stem("архивируй")) == ("archiv",)
    assert lexicon.expand(stem("заархивируй")) == ("archiv",)
    assert lexicon.expand(stem("сжать")) == ("shrink",)


def test_a_listed_stem_is_not_looked_up_without_its_prefix() -> None:
    lexicon = Lexicon.from_translations({"send": ["отправить"], "prepare": ["править"]})
    assert lexicon.expand(stem("отправь")) == ("send",)


def test_a_prefix_leaving_too_short_a_remainder_is_not_stripped() -> None:
    lexicon = Lexicon.from_translations({"code": ["код"], "patch": ["правка"]})
    assert lexicon.expand(stem("поправка")) == ("patch",)
    # "с" + "код": three letters left, too short to be a word of its own.
    assert lexicon.expand("скод") == ()


def test_a_word_that_is_no_infinitive_gets_no_imperatives() -> None:
    lexicon = Lexicon.from_translations({"file": ["файл"]})
    assert len(lexicon) == 1
