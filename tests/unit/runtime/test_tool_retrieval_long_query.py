"""A long Russian message must not lose the words that name the tool.

Past 25 distinct terms the query keeps the terms with the highest catalogue
idf. A Russian word reaches an English catalogue only through the lexicon, so
its own idf is 0, every Russian term ties, and the tie is broken by the term
itself: the 25 alphabetically first stems survive. Which Russian words decide
the ranking then depends on the alphabet, not on the words.
"""
# ruff: noqa: RUF001 — Russian queries are the point of this file

from __future__ import annotations

import pytest

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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the query-term cap ranks terms by catalogue idf only; lexicon-bridged "
        "Russian terms all score 0 and are cut alphabetically"
    ),
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
