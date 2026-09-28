"""A long Russian message must not lose the words that name the tool.

Past 25 distinct terms the query keeps the terms that can add the most to a
score. A Russian word reaches an English catalogue only through the lexicon, so
its own idf is 0; ranked by that alone, every Russian term ties and the 25
alphabetically first stems survive, whichever words named the tool. A term is
therefore worth the idf of its rarest expansion, at the expansion weight.
"""
# ruff: noqa: RUF001 — Russian queries are the point of this file

from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime.tool_retrieval import AnalyzedCatalogue, Lexicon, ToolIndex

_CATALOGUE = (
    ToolDocument("BrowserOpen", "Open a page in the controlled browser and wait until it loads.", parameters="url"),
    ToolDocument("BrowserScreenshot", "Take a screenshot of the current browser page.", parameters="full_page"),
    ToolDocument("SqlQuery", "Run a read-only SQL query against the database and return the rows.", parameters="sql"),
    ToolDocument("TodoWrite", "Replace the task checklist of the current run with an updated list."),
    ToolDocument("Recall", "Look up facts saved in long-term memory that match a query."),
    ToolDocument("DockerLogs", "Show the logs of a running container.", parameters="container tail"),
)

_CONTEXT = (
    "У нас вчера вечером после обновления базы данных упал сервис оплаты, клиенты жалуются, "
    "что заказы висят в статусе ожидания, а менеджеры не видят новых заявок в админке, бухгалтерия "
    "тоже ругается, поддержка завалена обращениями, партнёры звонят директору. "
)


def test_the_request_at_the_end_of_a_long_russian_message_is_kept() -> None:
    index = ToolIndex(
        AnalyzedCatalogue(_CATALOGUE),
        RetrievalSettings.from_constants(LoopConstants()),
        Lexicon.bundled(),
    )
    request = "Сделай скриншот страницы."
    assert index.rank(request, 1) == ["BrowserScreenshot"]
    assert index.rank(_CONTEXT + request, 1) == ["BrowserScreenshot"]


def _index(lexicon: Lexicon | None) -> ToolIndex:
    return ToolIndex(
        AnalyzedCatalogue(_CATALOGUE),
        RetrievalSettings.from_constants(LoopConstants()),
        lexicon,
    )


def test_terms_that_can_add_nothing_are_dropped_before_the_cap() -> None:
    weights = _index(Lexicon.bundled()).weighted_query(_CONTEXT + "Сделай скриншот страницы.")
    assert "скриншот" in weights
    # Words with no catalogue term and no expansion are not kept at all.
    assert "бухгалтер" not in weights
    assert "screenshot" in weights


def test_without_a_lexicon_the_cap_keeps_the_rarest_catalogue_terms() -> None:
    filler = " ".join(f"word{index}" for index in range(40))
    weights = _index(None).weighted_query(f"{filler} screenshot container")
    assert set(weights) == {"screenshot", "contain"}
