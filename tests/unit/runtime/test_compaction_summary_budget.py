"""The word budget a summary is asked for is sized for the script it is written in."""
from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.context.compaction import _summary_word_budget


def test_the_budget_never_exceeds_what_the_output_cap_holds_at_the_going_rate() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_summary_max_output_tokens=4_096,
        compaction_summary_output_tokens_per_word=4,
    )
    # A large unit: scaled by its size the budget would be far above the ceiling.
    assert _summary_word_budget(200_000, rc) == 4_096 // 4


def test_a_small_unit_keeps_the_floor() -> None:
    rc = LoopConstants(model_context_window=4_096)
    assert _summary_word_budget(100, rc) == rc.compaction_summary_min_words


def test_the_per_word_cost_is_configurable_not_the_english_figure() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_summary_max_output_tokens=4_096,
    )
    cyrillic = _summary_word_budget(200_000, rc)
    english = _summary_word_budget(
        200_000, rc.model_copy(update={"compaction_summary_output_tokens_per_word": 2})
    )
    # Two tokens a word was the English figure; the default is four, so the
    # budget asked for on the same unit is half of what it used to be.
    assert rc.compaction_summary_output_tokens_per_word == 4
    assert english == cyrillic * 2


def test_the_stock_budget_fits_the_stock_output_cap_in_a_non_latin_script() -> None:
    rc = LoopConstants(model_context_window=65_536)
    budget = _summary_word_budget(1_000_000, rc)
    assert budget * rc.compaction_summary_output_tokens_per_word <= (
        rc.compaction_summary_max_output_tokens
    )
