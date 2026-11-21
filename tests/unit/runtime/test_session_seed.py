"""The session seed: raw earlier turns, older tool results masked, summary last.

Pins the layout, the masking (and that it never orphans a tool call or result),
the budget fit, prefix stability across successive runs of one session, the
fold decision, and head protection counted in turns.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    CompactionSourceRef,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import estimate_message_tokens
from protocore.runtime.context.session_memory import (
    END_OF_SUMMARY_MARKER,
    ArtifactLedger,
    SessionMemory,
    SessionSummary,
    SummaryDelta,
    fold_run,
)
from protocore.runtime.context.session_seed import (
    MaskingRule,
    SessionSeed,
    build_session_seed,
    default_masking_rule,
    mask_boundary,
    mask_turn,
    opens_turn,
    plan_session_fold,
    repair_turn,
    split_turns,
)
from protocore.runtime.wire_format import render_compacted_placeholder

RC = LoopConstants()
#: The seed budget a host derives for a 49k-token window.
BUDGET = 13_722


# --- builders ---------------------------------------------------------------


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(text: str) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


def _calls(*calls: tuple[str, str, dict[str, Any]]) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(tool_call_id=cid, name=name, arguments_json=json.dumps(args))
            for cid, name, args in calls
        ],
    )


def _results(*results: tuple[str, str]) -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[ToolResultBlock(tool_call_id=cid, content=text) for cid, text in results],
    )


def _bulky(tag: str, chars: int) -> str:
    lines = [f"Cite: record {tag}-cite", f"see https://example.org/{tag}/item"]
    filler = f"{tag} body line with prose about the record and nothing else of note"
    while sum(len(x) + 1 for x in lines) < chars:
        lines.append(f"{filler} {len(lines)}")
    return "\n".join(lines)


def _turn(index: int, *, tools: int = 3, result_chars: int = 9000, answer_chars: int = 3000) -> list[Message]:
    calls = [(f"c{index}-{k}", "Search", {"query": f"topic {index} part {k}"}) for k in range(tools)]
    answer = (f"Answer {index}: the second book is B{index}. " * (answer_chars // 40 + 1))[:answer_chars]
    turn = [_user(f"question {index}: find the books about topic {index}")]
    if tools:
        turn.append(_calls(*calls))
        turn.append(_results(*[(cid, _bulky(f"t{index}k{k}", result_chars)) for k, (cid, _, _) in enumerate(calls)]))
    turn.append(_assistant(answer))
    return turn


def _result_text(message: Message) -> str:
    return "\n".join(b.content for b in message.content_blocks if isinstance(b, ToolResultBlock))


def _wire(message: Message) -> tuple[Any, ...]:
    """What reaches a provider: role, blocks, reasoning — never the metadata."""
    return (
        message.role,
        tuple(json.dumps(b.model_dump(mode="json", exclude={"metadata"}), sort_keys=True) for b in message.content_blocks),
        message.reasoning_content,
    )


def _common_prefix(a: Sequence[Message], b: Sequence[Message]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if _wire(x) != _wire(y):
            break
        n += 1
    return n


def _pairs_valid(messages: Sequence[Message]) -> bool:
    """Every tool result follows its call, and every call has its result."""
    open_calls: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                open_calls.add(block.tool_call_id)
            elif isinstance(block, ToolResultBlock):
                if block.tool_call_id not in open_calls:
                    return False
                answered.add(block.tool_call_id)
    return open_calls == answered


class _Coverage:
    """The host's record: the summary covers every message up to a position."""

    def __init__(self) -> None:
        self.through = -1
        self.positions: dict[int, int] = {}

    def index(self, history: Sequence[Message]) -> None:
        self.positions = {id(m): i for i, m in enumerate(history)}

    def __call__(self, turn: Sequence[Message]) -> bool:
        return all(self.positions.get(id(m), 10**9) <= self.through for m in turn)

    def advance(self, plan_messages: Sequence[Message]) -> None:
        self.through = max(self.through, max(self.positions[id(m)] for m in plan_messages))


# --- turns --------------------------------------------------------------------


def test_split_turns_keeps_tool_exchanges_inside_their_turn() -> None:
    history = [*_turn(0), *_turn(1, tools=0)]
    turns = split_turns(history)
    assert [len(t) for t in turns] == [4, 2]
    assert all(opens_turn(t[0]) for t in turns)


def test_a_compaction_record_does_not_open_a_turn() -> None:
    record = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="<summary>")],
        metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
    )
    history = [_user("q"), record, _assistant("a")]
    assert [len(t) for t in split_turns(history)] == [3]


def test_repair_drops_unpaired_calls_and_results_but_keeps_text() -> None:
    turn = [
        _user("q"),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                TextBlock(text="looking"),
                ToolUseBlock(tool_call_id="a", name="Search", arguments_json="{}"),
                ToolUseBlock(tool_call_id="b", name="Search", arguments_json="{}"),
            ],
        ),
        _results(("a", "found")),
        _results(("ghost", "no call")),
    ]
    repaired, removed = repair_turn(turn)
    assert removed == 2
    assert _pairs_valid(repaired)
    assert any(isinstance(b, TextBlock) and b.text == "looking" for m in repaired for b in m.content_blocks)
    assert len(repaired) == 3  # the ghost-only tool message is gone


# --- masking -----------------------------------------------------------------


def test_masking_keeps_pairs_args_citations_and_all_prose() -> None:
    turn = _turn(0)
    masked = mask_turn(turn, RC, default_masking_rule(RC))
    assert _pairs_valid(masked)
    assert masked[0] is turn[0] and masked[1] is turn[1] and masked[-1] is turn[-1]
    block = masked[2].content_blocks[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_call_id == "c0-0"
    assert "Search(" in block.content and "topic 0 part 0" in block.content
    assert "Cite: record t0k0-cite" in block.content
    assert "https://example.org/t0k0/item" in block.content
    assert estimate_message_tokens(masked[2], RC) * 10 < estimate_message_tokens(turn[2], RC)


def test_masking_is_deterministic_bytes() -> None:
    rule = default_masking_rule(RC)
    a = mask_turn(_turn(4), RC, rule)
    b = mask_turn(_turn(4), RC, rule)
    assert [_wire(m) for m in a] == [_wire(m) for m in b]


def test_small_results_and_kept_whole_results_are_not_masked() -> None:
    turn = [_user("q"), _calls(("s", "Small", {}), ("p", "Pinned", {})), _results(("s", "ok"), ("p", _bulky("p", 6000))), _assistant("a")]
    rule = MaskingRule(
        kept_lines=default_masking_rule(RC).kept_lines,
        keep_whole=lambda call, result: call is not None and call.name == "Pinned",
    )
    masked = mask_turn(turn, RC, rule)
    assert masked[2] is turn[2]


def test_a_host_rule_decides_which_lines_survive() -> None:
    turn = [_user("q"), _calls(("x", "Lookup", {"id": 7})), _results(("x", "RECORD-77 alpha\n" + "noise\n" * 2000)), _assistant("a")]
    rule = MaskingRule(kept_lines=lambda call, result: [line for line in result.content.splitlines() if line.startswith("RECORD-")])
    block = mask_turn(turn, RC, rule)[2].content_blocks[0]
    assert isinstance(block, ToolResultBlock)
    assert "Kept lines:\nRECORD-77 alpha" in block.content


def test_an_already_compacted_result_is_left_alone() -> None:
    ref = CompactionSourceRef(
        blob_ref="blob-1", sha256="", original_tokens=9000, label="tool_result", tool_name="Search", preview=""
    )
    placeholder = render_compacted_placeholder(ref) + "\n" + "kept line of the original\n" * 200
    turn = [_user("q"), _calls(("x", "Search", {})), _results(("x", placeholder)), _assistant("a")]
    assert mask_turn(turn, RC, default_masking_rule(RC))[2] is turn[2]


def test_mask_boundary_moves_in_batches() -> None:
    rc = LoopConstants(session_seed_full_turns=1, session_seed_mask_batch_turns=3)
    boundaries = [mask_boundary(n, rc) for n in range(1, 9)]
    assert boundaries == [0, 0, 0, 3, 3, 3, 6, 6]
    assert [mask_boundary(n, RC) for n in range(1, 5)] == [0, 1, 2, 3]


# --- the seed -------------------------------------------------------------------


def test_empty_session_is_a_cold_start() -> None:
    seed = build_session_seed([], SessionMemory(), BUDGET, RC)
    assert seed.messages == [] and seed.tokens == 0


def test_layout_head_summary_window_registry_and_tags() -> None:
    history = [*_turn(0), *_turn(1), *_turn(2), *_turn(3)]
    coverage = _Coverage()
    coverage.index(history)
    coverage.through = len(_turn(0)) + len(_turn(1)) - 1  # turns 0..1 summarised
    memory = SessionMemory(
        summary=SessionSummary(entries={"KEY FACTS & IDENTIFIERS": ("B1 is the second book",)}),
        ledger=ArtifactLedger(files=["notes.md"], content={"notes.md": "x"}),
        turn_index=4,
    )
    seed = build_session_seed(history, memory, 40_000, RC, covered=coverage)
    texts = [m.text for m in seed.messages]
    assert texts[0] == "question 0: find the books about topic 0"  # the head turn
    summary_at = next(i for i, t in enumerate(texts) if END_OF_SUMMARY_MARKER in t)
    assert summary_at == len(_turn(0))
    assert texts[summary_at + 1] == "question 2: find the books about topic 2"
    assert seed.messages[-1].text.startswith("ARTIFACT REGISTRY")
    assert (seed.head_turns, seed.summarized_turns, seed.window_turns) == (1, 1, 2)
    assert (seed.full_turns, seed.masked_turns) == (1, 2)
    for message in seed.messages:
        assert message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
    refs = [m for m in seed.messages if m.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True]
    assert len(refs) == 2  # the summary and the registry only
    assert _pairs_valid(seed.messages)


def test_the_newest_turn_is_full_and_older_ones_masked() -> None:
    history = [*_turn(0), *_turn(1), *_turn(2)]
    seed = build_session_seed(history, SessionMemory(turn_index=3), 40_000, RC)
    tool_messages = [m for m in seed.messages if m.role is MessageRole.tool]
    assert "not carried here in full" in _result_text(tool_messages[0])
    assert "not carried here in full" in _result_text(tool_messages[1])
    assert "not carried here in full" not in _result_text(tool_messages[2])
    assert seed.over_budget_tokens == 0


def test_under_pressure_the_newest_turn_is_cut_to_fit_rather_than_masked() -> None:
    history = [*_turn(0), *_turn(1, result_chars=20_000)]
    seed = build_session_seed(history, SessionMemory(turn_index=2), 6_000, RC)
    assert seed.trimmed_turns == 1 and seed.full_turns == 0
    assert seed.tokens <= 6_000
    newest = [m for m in seed.messages if m.role is MessageRole.tool][-1]
    assert "characters:" in _result_text(newest)
    assert _pairs_valid(seed.messages)


def test_under_severe_pressure_everything_is_masked_and_reported_over_budget() -> None:
    history = [*_turn(0), *_turn(1), *_turn(2)]
    seed = build_session_seed(history, SessionMemory(turn_index=3), 500, RC)
    assert seed.full_turns == 0 and seed.trimmed_turns == 0
    assert seed.masked_turns == 3
    assert seed.over_budget_tokens == seed.tokens - 500 > 0
    # Nothing un-summarised is dropped by default: every question is still there.
    assert sum(1 for m in seed.messages if m.text.startswith("question")) == 3


def test_drop_uncovered_leaves_out_oldest_window_turns_never_head_or_newest() -> None:
    history = [t for i in range(6) for t in _turn(i)]
    seed = build_session_seed(history, SessionMemory(turn_index=6), 4_000, RC, drop_uncovered=True)
    questions = [m.text for m in seed.messages if m.text.startswith("question")]
    assert questions[0].startswith("question 0") and questions[-1].startswith("question 5")
    assert seed.dropped_turns > 0
    assert seed.tokens <= 4_000 or seed.window_turns == 1


def test_seed_token_count_is_the_shared_estimate_of_its_messages() -> None:
    history = [*_turn(0), *_turn(1)]
    seed = build_session_seed(history, SessionMemory(turn_index=2), BUDGET, RC)
    assert seed.tokens == sum(estimate_message_tokens(m, RC) for m in seed.messages)


def test_a_leading_fragment_is_not_carried() -> None:
    history = [_assistant("hello, how can I help?"), *_turn(0, tools=0)]
    seed = build_session_seed(history, SessionMemory(turn_index=1), BUDGET, RC)
    assert seed.fragment_messages == 1
    assert seed.messages[0].role is MessageRole.user


def test_an_incomplete_turn_is_carried_without_its_orphans() -> None:
    turn = _turn(0)
    broken = [turn[0], turn[1], turn[3]]  # the results never made it to the store
    seed = build_session_seed(broken, SessionMemory(turn_index=1), BUDGET, RC)
    assert _pairs_valid(seed.messages)
    assert seed.repaired_blocks == 3
    assert [m.role for m in seed.messages] == [MessageRole.user, MessageRole.assistant]


# --- head protection in turns ----------------------------------------------------


def test_head_protection_counts_turns_not_messages() -> None:
    # A first turn with a tool call is four messages: a message count of 3
    # never protected it. One protected turn now keeps it ahead of the summary.
    history = [t for i in range(5) for t in _turn(i)]
    coverage = _Coverage()
    coverage.index(history)
    coverage.through = len(_turn(0)) * 3 - 1
    memory = SessionMemory(summary=SessionSummary(entries={"REQUESTS": ("find books",)}), turn_index=5)
    seed = build_session_seed(history, memory, BUDGET, RC, covered=coverage)
    assert seed.head_turns == 1
    assert seed.messages[0].text.startswith("question 0")
    assert seed.messages[len(_turn(0))].text.startswith("[Running summary")


def test_head_too_large_is_not_protected() -> None:
    rc = LoopConstants(session_memory_head_max_tokens=10)
    seed = build_session_seed([*_turn(0), *_turn(1)], SessionMemory(turn_index=2), BUDGET, rc)
    assert seed.head_turns == 0


def test_head_protection_off() -> None:
    rc = LoopConstants(session_memory_head_protect_turns=0)
    seed = build_session_seed([*_turn(0), *_turn(1)], SessionMemory(turn_index=2), BUDGET, rc)
    assert seed.head_turns == 0 and seed.window_turns == 2


# --- the fold decision -------------------------------------------------------------


def test_no_fold_while_the_masked_history_fits() -> None:
    history = [t for i in range(3) for t in _turn(i)]
    plan = plan_session_fold(history, SessionMemory(turn_index=3), BUDGET, RC)
    assert not plan.needed and plan.messages == []
    assert plan.projected_tokens <= plan.trigger_tokens


def test_one_raw_turn_larger_than_the_old_tail_allowance_does_not_force_a_fold() -> None:
    # A turn of ~10k tokens raw used to trip a gate sized on a 4k "tail" after
    # every run. The plan sizes what the seed carries: that turn, masked.
    history = _turn(0, result_chars=12_000)
    raw = sum(estimate_message_tokens(m, RC) for m in history)
    assert raw > 0.3 * BUDGET
    assert not plan_session_fold(history, SessionMemory(turn_index=1), BUDGET, RC).needed


def test_fold_plans_oldest_window_turns_down_to_the_target_and_spares_head_and_newest() -> None:
    history = [t for i in range(14) for t in _turn(i)]
    plan = plan_session_fold(history, SessionMemory(turn_index=14), BUDGET, RC)
    assert plan.needed
    assert plan.projected_tokens > plan.trigger_tokens
    assert plan.remaining_tokens <= plan.target_tokens
    first = plan.turns[0][0].text
    assert first.startswith("question 1:")  # turn 0 is the protected head
    assert all(not t[0].text.startswith("question 13") for t in plan.turns)
    # contiguous, oldest first
    numbers = [int(t[0].text.split()[1].rstrip(":")) for t in plan.turns]
    assert numbers == list(range(1, 1 + len(numbers)))


def test_fold_source_budget_defers_turns_but_takes_at_least_one() -> None:
    history = [t for i in range(14) for t in _turn(i)]
    plan = plan_session_fold(history, SessionMemory(turn_index=14), BUDGET, RC, max_source_tokens=1)
    assert len(plan.turns) == 1 and plan.deferred_turns > 0


def test_the_plan_and_the_seed_agree_on_what_is_carried() -> None:
    history = [t for i in range(14) for t in _turn(i)]
    coverage = _Coverage()
    coverage.index(history)
    memory = SessionMemory(turn_index=14)
    plan = plan_session_fold(history, memory, BUDGET, RC, covered=coverage)
    coverage.advance(plan.messages)
    memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("x" * 4000,)}), RC).memory
    after = plan_session_fold(history, memory, BUDGET, RC, covered=coverage)
    assert not after.needed
    seed = build_session_seed(history, memory, BUDGET, RC, covered=coverage)
    assert seed.summarized_turns == len(plan.turns)
    assert seed.tokens <= BUDGET


# --- prefix stability across runs -----------------------------------------------------


def _simulate(rc: LoopConstants, runs: int, *, budget: int = BUDGET) -> tuple[list[SessionSeed], list[bool], list[int]]:
    """Run a session ``runs`` times: seed, append the run's turn, plan, fold.

    Returns the seeds, whether a fold landed before each seed, and the message
    count of each turn.
    """
    history: list[Message] = []
    memory = SessionMemory()
    coverage = _Coverage()
    seeds: list[SessionSeed] = []
    folded_before: list[bool] = []
    landed = False
    sizes: list[int] = []
    for run in range(runs):
        coverage.index(history)
        seeds.append(build_session_seed(history, memory, budget, rc, covered=coverage))
        folded_before.append(landed)
        turn = _turn(run, tools=2 + run % 3, result_chars=6000 + 1500 * (run % 4))
        sizes.append(len(turn))
        history = [*history, *turn]
        coverage.index(history)
        plan = plan_session_fold(history, memory, budget, rc, covered=coverage)
        delta = SummaryDelta(entries={"KEY FACTS & IDENTIFIERS": (f"fact from fold at run {run}",)}) if plan.needed else None
        memory = fold_run(memory, turn, delta, rc).memory
        if plan.needed:
            coverage.advance(plan.messages)
        landed = plan.needed
    return seeds, folded_before, sizes


def test_consecutive_seeds_share_everything_but_the_full_tail() -> None:
    seeds, folded_before, sizes = _simulate(RC, 12)
    folds = sum(folded_before)
    assert 1 <= folds <= 3  # hysteresis: several turns between folds, not one per turn
    for run in range(2, len(seeds)):
        previous, current = seeds[run - 1], seeds[run]
        common = _common_prefix(previous.messages, current.messages)
        if folded_before[run]:
            assert common >= sizes[0]  # the head survives a fold
            continue
        # Only the previous seed's newest turn changes form (full → masked),
        # and only from its first tool result on: its question and its calls
        # are the same bytes either way.
        assert common == len(previous.messages) - sizes[run - 2] + 2, run
        assert current.tokens <= BUDGET


def test_with_batched_masking_most_runs_reuse_the_whole_previous_seed() -> None:
    rc = LoopConstants(session_seed_mask_batch_turns=3)
    seeds, folded_before, _sizes = _simulate(rc, 7, budget=200_000)
    assert not any(folded_before)
    whole = 0
    for run in range(2, len(seeds)):
        previous, current = seeds[run - 1], seeds[run]
        if _common_prefix(previous.messages, current.messages) == len(previous.messages):
            whole += 1
    # Of five transitions the boundary moves on two (at 4 and 7 turns).
    assert whole >= 3


def test_the_summary_never_sits_in_front_of_carried_turns_it_does_not_cover() -> None:
    seeds, _folded, sizes = _simulate(RC, 12)
    for seed in seeds:
        texts = [m.text for m in seed.messages]
        marks = [i for i, t in enumerate(texts) if END_OF_SUMMARY_MARKER in t]
        if marks:
            assert marks == [sizes[0]]  # right after the head, nowhere else
