"""The session seed: raw earlier turns, older tool exchanges masked, summary last.

Pins the layout, the masking (and that it never orphans a tool call or result),
the budget fit, prefix stability across successive runs of one session (on the
messages and on the bytes an OpenAI-compatible endpoint receives), the fold
decision, coverage held by the summary, head protection counted in turns, and
the cost of a call with a session cache.
"""
from __future__ import annotations

import itertools
import json
import random
from collections.abc import Sequence
from typing import Any

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    IMAGE_REFS_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    SESSION_SEED_MASKED_FORM_METADATA_KEY,
    CompactionSourceRef,
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolResultImage,
    ToolUseBlock,
)
from protocore.providers.openai_compat.spec import ProviderCapabilities, ReasoningReplay
from protocore.providers.openai_compat.wire import messages_to_wire
from protocore.runtime.context import session_seed as seed_module
from protocore.runtime.context.compaction import estimate_message_tokens
from protocore.runtime.context.session_memory import (
    END_OF_SUMMARY_MARKER,
    ArtifactLedger,
    Coverage,
    CoverageStep,
    SessionMemory,
    SessionSummary,
    SummaryDelta,
    build_summary_delta_message,
    fold_run,
    fold_source_tokens,
)
from protocore.runtime.context.session_seed import (
    CUT_ARGUMENT_SUFFIX,
    MaskingRule,
    SeedCache,
    SessionSeed,
    build_session_seed,
    default_masking_rule,
    has_cut_marker,
    mask_boundary,
    mask_turn,
    opens_turn,
    plan_session_fold,
    recommend_seed_budget,
    repair_turn,
    seed_floor_tokens,
    split_turns,
)
from protocore.runtime.wire_format import render_compacted_placeholder
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

RC = LoopConstants()
BUDGET = 16_000
POS = "test.position"


# --- builders ---------------------------------------------------------------


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(text: str, reasoning: str | None = None) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)], reasoning_content=reasoning)


def _calls(*calls: tuple[str, str, dict[str, Any]], reasoning: str | None = None) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(tool_call_id=cid, name=name, arguments_json=json.dumps(args)) for cid, name, args in calls
        ],
        reasoning_content=reasoning,
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


def _turn(
    index: int,
    *,
    tools: int = 3,
    rounds: int = 1,
    result_chars: int = 9000,
    answer_chars: int = 3000,
    reasoning: int = 0,
) -> list[Message]:
    answer = (f"Answer {index}: the second item is I{index}. " * (answer_chars // 40 + 1))[:answer_chars]
    turn = [_user(f"question {index}: find the items about topic {index}")]
    think = ("weighing the options " * reasoning) or None
    for r in range(rounds if tools else 0):
        calls = [(f"c{index}-{r}-{k}", "Search", {"query": f"topic {index} part {r}.{k}"}) for k in range(tools)]
        turn.append(_calls(*calls, reasoning=think))
        turn.append(_results(*[(cid, _bulky(f"t{index}r{r}k{k}", result_chars)) for k, (cid, _, _) in enumerate(calls)]))
    turn.append(_assistant(answer, reasoning=think))
    return turn


def _stamp(messages: Sequence[Message], first: int) -> list[Message]:
    """Give each message its session position, as a host's store does."""
    return [m.model_copy(update={"metadata": {**m.metadata, POS: first + i}}) for i, m in enumerate(messages)]


def _position(message: Message) -> int | None:
    value = message.metadata.get(POS)
    return value if isinstance(value, int) else None


def _session(*turns: list[Message]) -> list[Message]:
    history: list[Message] = []
    for turn in turns:
        history.extend(_stamp(turn, len(history)))
    return history


def _covering(first: int, last: int, **entries: tuple[str, ...]) -> SessionMemory:
    summary = SessionSummary(entries=dict(entries) or {"REQUESTS": ("earlier request",)}, coverage=Coverage(first, last))
    return SessionMemory(summary=summary, turn_index=1)


def _result_text(message: Message) -> str:
    return "\n".join(b.content for b in message.content_blocks if isinstance(b, ToolResultBlock))


def _wire(message: Message) -> tuple[Any, ...]:
    """What reaches a provider: role, blocks, reasoning — never the metadata."""
    return (
        message.role,
        tuple(json.dumps(b.model_dump(mode="json", exclude={"metadata"}), sort_keys=True) for b in message.content_blocks),
        message.reasoning_content,
        message.reasoning_items,
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
    open_calls: list[str] = []
    answered = 0
    calls = 0
    for message in messages:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                open_calls.append(block.tool_call_id)
                calls += 1
            elif isinstance(block, ToolResultBlock):
                if block.tool_call_id not in open_calls:
                    return False
                open_calls.remove(block.tool_call_id)
                answered += 1
    return calls == answered


# --- turns --------------------------------------------------------------------


def test_split_turns_keeps_tool_exchanges_inside_their_turn() -> None:
    history = [*_turn(0, rounds=2), *_turn(1, tools=0)]
    turns = split_turns(history)
    assert [len(t) for t in turns] == [6, 2]
    assert all(opens_turn(t[0]) for t in turns)


def test_a_compaction_record_does_not_open_a_turn() -> None:
    record = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="<summary>")],
        metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
    )
    assert [len(t) for t in split_turns([_user("q"), record, _assistant("a")])] == [3]


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
    assert len(repaired) == 3


def test_repair_drops_a_result_stored_before_its_call() -> None:
    turn = [_user("q"), _results(("a", "early")), _calls(("a", "Search", {})), _assistant("done")]
    repaired, removed = repair_turn(turn)
    assert removed == 2
    assert _pairs_valid(repaired)
    assert [m.role for m in repaired] == [MessageRole.user, MessageRole.assistant]


def test_repeated_call_ids_are_paired_in_order() -> None:
    turn = [
        _user("q"),
        _calls(("call_0", "Alpha", {"q": 1})),
        _results(("call_0", "Cite: ALPHA-1\n" + "alpha body\n" * 200)),
        _calls(("call_0", "Beta", {"q": 2})),
        _results(("call_0", "Cite: BETA-2\n" + "beta body\n" * 300)),
        _assistant("a"),
    ]
    repaired, removed = repair_turn(turn)
    assert removed == 0
    masked = mask_turn(repaired, RC, default_masking_rule(RC))
    first, second = _result_text(masked[2]), _result_text(masked[4])
    assert "Output of Alpha" in first and "ALPHA-1" in first and "BETA" not in first
    assert "Output of Beta" in second and "BETA-2" in second and "ALPHA" not in second


# --- masking -----------------------------------------------------------------


def test_masking_keeps_pairs_citations_and_all_prose() -> None:
    turn = _turn(0)
    masked = mask_turn(turn, RC, default_masking_rule(RC))
    assert _pairs_valid(masked)
    assert masked[0] is turn[0] and masked[-1] is turn[-1]
    text = _result_text(masked[2])
    assert "Output of Search" in text
    assert "Cite: record t0r0k0-cite" in text
    assert "https://example.org/t0r0k0/item" in text
    assert estimate_message_tokens(masked[2], RC) * 10 < estimate_message_tokens(turn[2], RC)


def test_the_placeholder_states_what_is_missing_and_asks_nothing() -> None:
    text = _result_text(mask_turn(_turn(0), RC, default_masking_rule(RC))[2])
    header = text.splitlines()[0]
    assert header.startswith("[Output of Search from an earlier turn,") and header.endswith("not in this context.]")
    lowered = text.lower()
    assert "call" not in lowered and "again" not in lowered and "re-run" not in lowered


def test_an_error_result_says_so() -> None:
    turn = [_user("q"), _calls(("x", "Run", {})), _results(("x", "boom\n" * 500)), _assistant("a")]
    turn[2] = turn[2].model_copy(
        update={"content_blocks": (ToolResultBlock(tool_call_id="x", content="boom\n" * 500, is_error=True),)}
    )
    masked = mask_turn(turn, RC, default_masking_rule(RC))[2].content_blocks[0]
    assert isinstance(masked, ToolResultBlock) and masked.is_error
    assert "it reported an error" in masked.content


def test_masking_is_deterministic_bytes() -> None:
    rule = default_masking_rule(RC)
    a = mask_turn(_turn(4, rounds=2, reasoning=20), RC, rule)
    b = mask_turn(_turn(4, rounds=2, reasoning=20), RC, rule)
    assert [_wire(m) for m in a] == [_wire(m) for m in b]


def test_masking_never_makes_a_result_larger() -> None:
    dense = "\n".join(f"Cite: entry {i} of a citation-dense listing" for i in range(30))
    turn = [_user("q"), _calls(("x", "Search", {})), _results(("x", dense)), _assistant("a")]
    masked = mask_turn(turn, RC, default_masking_rule(RC))
    assert masked[2] is turn[2]


def test_kept_lines_past_the_cap_are_counted_not_hidden() -> None:
    many = "\n".join(f"Cite: entry {i}" for i in range(120)) + "\n" + "filler line\n" * 2000
    turn = [_user("q"), _calls(("x", "Search", {})), _results(("x", many)), _assistant("a")]
    text = _result_text(mask_turn(turn, RC, default_masking_rule(RC))[2])
    assert "Cite: entry 39" in text and "Cite: entry 40" not in text
    assert "(+80 more such lines not kept)" in text


def test_short_and_kept_whole_results_are_not_masked() -> None:
    turn = [_user("q"), _calls(("s", "Small", {}), ("p", "Pinned", {})), _results(("s", "ok"), ("p", _bulky("p", 6000))), _assistant("a")]
    rule = MaskingRule(
        kept_lines=default_masking_rule(RC).kept_lines,
        keep_whole=lambda call, result: call is not None and call.name == "Pinned",
    )
    assert mask_turn(turn, RC, rule)[2] is turn[2]


def test_a_host_rule_decides_which_lines_survive() -> None:
    turn = [_user("q"), _calls(("x", "Lookup", {"id": 7})), _results(("x", "RECORD-77 alpha\n" + "noise\n" * 2000)), _assistant("a")]
    rule = MaskingRule(kept_lines=lambda call, result: [line for line in result.content.splitlines() if line.startswith("RECORD-")])
    assert "Lines kept from it:\nRECORD-77 alpha" in _result_text(mask_turn(turn, RC, rule)[2])


def test_an_already_compacted_result_is_left_alone() -> None:
    ref = CompactionSourceRef(blob_ref="blob-1", sha256="", original_tokens=9000, label="tool_result", tool_name="Search", preview="")
    placeholder = render_compacted_placeholder(ref) + "\n" + "kept line of the original\n" * 200
    turn = [_user("q"), _calls(("x", "Search", {})), _results(("x", placeholder)), _assistant("a")]
    assert mask_turn(turn, RC, default_masking_rule(RC))[2] is turn[2]


def _with_images(turn: list[Message], index: int, count: int, text: str) -> list[Message]:
    images = tuple(ToolResultImage(blob_ref=f"img-{k}", mime_type="image/png") for k in range(count))
    block = turn[index].content_blocks[0]
    assert isinstance(block, ToolResultBlock)
    new = ToolResultBlock(tool_call_id=block.tool_call_id, content=text, images=images)
    return [*turn[:index], turn[index].model_copy(update={"content_blocks": (new,)}), *turn[index + 1 :]]


def test_a_masked_turn_drops_every_tool_image_and_says_so() -> None:
    short = _with_images(_turn(0, tools=1), 2, 2, "Screenshot taken.")
    long = _with_images(_turn(1, tools=1), 2, 1, "page text\n" * 900)
    for turn in (short, long):
        block = mask_turn(turn, RC, default_masking_rule(RC))[2].content_blocks[0]
        assert isinstance(block, ToolResultBlock)
        assert block.images == ()
        assert "image/png) from this output not carried.]" in block.content
    block = mask_turn(short, RC, default_masking_rule(RC))[2].content_blocks[0]
    assert isinstance(block, ToolResultBlock) and block.content.startswith("Screenshot taken.\n[2 images")


def test_the_full_and_the_cut_newest_turn_keep_their_images() -> None:
    old = _turn(0, tools=1)
    newest = _with_images(_turn(1, tools=1, result_chars=40_000), 2, 1, _bulky("big", 40_000))
    history = _session(old, newest)
    roomy = build_session_seed(history, SessionMemory(turn_index=2), 200_000, RC, position=_position)
    assert roomy.full_turns == 1
    tight = build_session_seed(history, SessionMemory(turn_index=2), 7_000, RC, position=_position)
    assert tight.trimmed_turns == 1
    for seed in (roomy, tight):
        newest_result = [m for m in seed.messages if m.role is MessageRole.tool][-1].content_blocks[0]
        assert isinstance(newest_result, ToolResultBlock) and len(newest_result.images) == 1


def test_long_call_arguments_are_cut_but_stay_json_and_keep_short_values() -> None:
    body = "x = 1\n" * 4000
    write = _calls(("w", "Write", {"path": "/srv/app/main.py", "content": body}))
    turn = [_user("q"), write, _results(("w", "written")), _assistant("done")]
    masked = mask_turn(turn, RC, default_masking_rule(RC))
    call = masked[1].content_blocks[0]
    assert isinstance(call, ToolUseBlock)
    args = json.loads(call.arguments_json)
    assert args["path"] == "/srv/app/main.py"
    assert args["content"].endswith(f"[{len(body)} characters, not carried]")
    assert len(call.arguments_json) < 600
    assert _pairs_valid(masked)
    # The registry is built from the run as it happened, not from a seed.
    folded = fold_run(SessionMemory(), turn, None, RC, roles=CONVENTIONAL_TOOL_ROLES).memory
    assert folded.ledger.content["/srv/app/main.py"] == body[:4000]


def test_reasoning_is_dropped_in_masked_turns_and_kept_in_full_ones() -> None:
    turn = _turn(0, rounds=2, reasoning=50)
    turn.insert(1, Message(role=MessageRole.assistant, content_blocks=[ThinkingBlock(text="only thinking")]))
    masked = mask_turn(turn, RC, default_masking_rule(RC))
    assert all(m.reasoning_content is None for m in masked)
    assert not any(isinstance(b, ThinkingBlock) for m in masked for b in m.content_blocks)
    assert len(masked) == len(turn) - 1  # the thinking-only message is gone
    assert _pairs_valid(masked)
    history = _session(turn)
    seed = build_session_seed(history, SessionMemory(turn_index=1), 200_000, RC, position=_position)
    assert seed.full_turns == 1 and any(m.reasoning_content for m in seed.messages)


@pytest.mark.asyncio
async def test_masked_turns_go_through_a_reasoning_requiring_endpoint_with_pairs_intact() -> None:
    history = _session(_turn(0, rounds=2, reasoning=30), _turn(1, reasoning=30))
    seed = build_session_seed(history, SessionMemory(turn_index=2), 200_000, RC, position=_position)
    capabilities = ProviderCapabilities(reasoning_replay=ReasoningReplay.required)
    wire = await messages_to_wire(
        [*seed.messages, _user("next")], capabilities=capabilities, thinking=True, endpoint_name="e", model="m"
    )
    entries = wire.entries
    assistants = [e for e in entries if e["role"] == "assistant"]
    assert all("reasoning_content" in e for e in assistants)
    assert any(e["reasoning_content"] == "" for e in assistants)  # a masked turn
    asked = [c["id"] for e in assistants for c in e.get("tool_calls", [])]
    answered = [e["tool_call_id"] for e in entries if e["role"] == "tool"]
    assert sorted(asked) == sorted(answered)


def test_mask_boundary_moves_in_batches() -> None:
    rc = LoopConstants(session_seed_full_turns=1, session_seed_mask_batch_turns=3)
    assert [mask_boundary(n, rc) for n in range(1, 9)] == [0, 0, 0, 3, 3, 3, 6, 6]
    assert [mask_boundary(n, RC) for n in range(1, 5)] == [0, 1, 2, 3]


# --- the seed -------------------------------------------------------------------


def test_empty_session_is_a_cold_start() -> None:
    seed = build_session_seed([], SessionMemory(), BUDGET, RC, position=_position)
    assert seed.messages == [] and seed.tokens == 0


def test_layout_head_summary_window_registry_and_tags() -> None:
    history = _session(_turn(0), _turn(1), _turn(2), _turn(3))
    through = len(_turn(0)) + len(_turn(1)) - 1  # turns 0..1 summarised
    memory = SessionMemory(
        summary=SessionSummary(
            entries={"KEY FACTS & IDENTIFIERS": ("I1 is the second item",)}, coverage=Coverage(len(_turn(0)), through)
        ),
        ledger=ArtifactLedger(files=["notes.md"], content={"notes.md": "x"}),
        turn_index=4,
    )
    seed = build_session_seed(history, memory, 60_000, RC, position=_position)
    texts = [m.text for m in seed.messages]
    assert texts[0] == "question 0: find the items about topic 0"
    summary_at = next(i for i, t in enumerate(texts) if END_OF_SUMMARY_MARKER in t)
    assert summary_at == len(_turn(0))
    assert texts[summary_at + 1] == "question 2: find the items about topic 2"
    assert seed.messages[-1].text.startswith("ARTIFACT REGISTRY")
    assert (seed.head_turns, seed.summarized_turns, seed.window_turns) == (1, 1, 2)
    assert (seed.full_turns, seed.masked_turns) == (1, 2)
    for message in seed.messages:
        assert message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
    refs = [m for m in seed.messages if m.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True]
    assert len(refs) == 2
    assert _pairs_valid(seed.messages)


def test_a_reset_summary_brings_every_turn_back() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    covering = build_session_seed(history, _covering(len(_turn(0)), len(history) - 1 - len(_turn(2))), 60_000, RC, position=_position)
    assert covering.summarized_turns == 1
    reset = SessionMemory(summary=SessionSummary(), turn_index=3)
    seed = build_session_seed(history, reset, 60_000, RC, position=_position)
    assert seed.summarized_turns == 0 and seed.window_turns == 2 and seed.head_turns == 1


def test_a_turn_with_no_known_position_is_never_recorded_but_one_with_some_is() -> None:
    history = _session(_turn(0), _turn(1), _turn(2), _turn(3))
    start = len(_turn(0))
    history[start + 1] = history[start + 1].model_copy(update={"metadata": {}})  # one message of turn 1
    for k in range(len(_turn(2))):  # every message of turn 2
        at = start * 2 + k
        history[at] = history[at].model_copy(update={"metadata": {}})
    seed = build_session_seed(history, _covering(start, 10**6), 60_000, RC, position=_position)
    questions = [m.text.split(":")[0] for m in seed.messages if m.text.startswith("question")]
    assert questions == ["question 0", "question 2"]
    assert seed.summarized_turns == 2  # turns 1 and 3


def test_an_empty_summary_that_covers_turns_still_says_they_were_there() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    memory = SessionMemory(summary=SessionSummary(coverage=Coverage(len(_turn(0)), len(_turn(0)) * 2 - 1)), turn_index=3)
    seed = build_session_seed(history, memory, 60_000, RC, position=_position)
    assert seed.summarized_turns == 1
    assert any("the summary holds no entries for those turns" in m.text for m in seed.messages)


def test_the_newest_turn_is_full_and_older_ones_masked() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    seed = build_session_seed(history, SessionMemory(turn_index=3), 60_000, RC, position=_position)
    tool_messages = [m for m in seed.messages if m.role is MessageRole.tool]
    assert "not in this context" in _result_text(tool_messages[0])
    assert "not in this context" in _result_text(tool_messages[1])
    assert "not in this context" not in _result_text(tool_messages[2])
    assert seed.over_budget_tokens == 0


def test_under_pressure_the_newest_turn_is_cut_to_fit_rather_than_masked() -> None:
    history = _session(_turn(0), _turn(1, result_chars=20_000))
    seed = build_session_seed(history, SessionMemory(turn_index=2), 6_000, RC, position=_position)
    assert seed.trimmed_turns == 1 and seed.full_turns == 0
    assert seed.tokens <= 6_000
    assert "characters:" in _result_text([m for m in seed.messages if m.role is MessageRole.tool][-1])
    assert _pairs_valid(seed.messages)


def test_under_severe_pressure_everything_is_masked_before_anything_is_left_out() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    rule = default_masking_rule(RC)
    masked_total = sum(estimate_message_tokens(m, RC) for t in split_turns(history) for m in mask_turn(t, RC, rule))
    seed = build_session_seed(history, SessionMemory(turn_index=3), masked_total, RC, position=_position)
    assert seed.full_turns == 0 and seed.trimmed_turns + seed.masked_turns == 3
    assert seed.dropped_turns == 0 and seed.tokens <= masked_total
    smaller = build_session_seed(history, SessionMemory(turn_index=3), masked_total - 1, RC, position=_position)
    assert smaller.dropped_turns == 1 and smaller.tokens < masked_total


def test_the_seed_always_fits_leaving_out_oldest_window_turns_with_a_notice() -> None:
    history = _session(*(_turn(i) for i in range(8)))
    seed = build_session_seed(history, SessionMemory(turn_index=8), 6_000, RC, position=_position)
    questions = [m.text.split(":")[0] for m in seed.messages if m.text.startswith("question")]
    assert questions[0] == "question 0" and questions[-1] == "question 7"
    assert seed.dropped_turns == 8 - len(questions) >= 3
    dropped = [f"question {i}" for i in range(1, 1 + seed.dropped_turns)]
    assert not set(dropped) & set(questions)
    notices = [m.text for m in seed.messages if "not in this context" in m.text and m.text.startswith("[")]
    assert notices == [f"[{seed.dropped_turns} earlier turns of this conversation are not in this context and not in the summary.]"]
    assert seed.tokens <= 6_000 and seed.over_budget_tokens == 0
    assert seed.tokens == sum(estimate_message_tokens(m, RC) for m in seed.messages)


def test_under_extreme_pressure_head_newest_registry_and_summary_give_way_in_order() -> None:
    history = _session(*(_turn(i) for i in range(3)))
    memory = SessionMemory(
        summary=SessionSummary(entries={"REQUESTS": ("r " * 200,)}),
        ledger=ArtifactLedger(files=["a.md"], content={"a.md": "line\n" * 400}),
        turn_index=3,
    )
    seed = build_session_seed(history, memory, 120, RC, position=_position)
    assert seed.dropped_turns == 3 and seed.dropped_registry and seed.dropped_summary
    assert [m.text for m in seed.messages] == [
        "[3 earlier turns of this conversation are not in this context and not in the summary. "
        "The summary of earlier turns is not in this context. "
        "The registry of files written in this conversation is not in this context.]"
    ]
    assert seed.tokens <= 120


def test_seed_token_count_is_the_shared_estimate_of_its_messages() -> None:
    seed = build_session_seed(_session(_turn(0), _turn(1)), SessionMemory(turn_index=2), BUDGET, RC, position=_position)
    assert seed.tokens == sum(estimate_message_tokens(m, RC) for m in seed.messages)


def test_a_leading_fragment_is_not_carried() -> None:
    history = _session([_assistant("hello, how can I help?")], _turn(0, tools=0))
    seed = build_session_seed(history, SessionMemory(turn_index=1), BUDGET, RC, position=_position)
    assert seed.fragment_messages == 1
    assert seed.messages[0].role is MessageRole.user


def test_an_incomplete_turn_is_carried_without_its_orphans() -> None:
    turn = _turn(0)
    seed = build_session_seed(_session([turn[0], turn[1], turn[3]]), SessionMemory(turn_index=1), BUDGET, RC, position=_position)
    assert _pairs_valid(seed.messages)
    assert seed.repaired_blocks == 3
    assert [m.role for m in seed.messages] == [MessageRole.user, MessageRole.assistant]


# --- head protection in turns ----------------------------------------------------


def test_head_protection_counts_turns_not_messages() -> None:
    history = _session(*(_turn(i) for i in range(5)))
    memory = _covering(len(_turn(0)), len(_turn(0)) * 3 - 1, REQUESTS=("find items",))
    seed = build_session_seed(history, memory, BUDGET, RC, position=_position)
    assert seed.head_turns == 1
    assert seed.messages[0].text.startswith("question 0")
    assert seed.messages[len(_turn(0))].text.startswith("[Running summary")


def test_head_too_large_is_not_protected() -> None:
    rc = LoopConstants(session_memory_head_max_tokens=10)
    seed = build_session_seed(_session(_turn(0), _turn(1)), SessionMemory(turn_index=2), BUDGET, rc, position=_position)
    assert seed.head_turns == 0


def test_head_protection_off() -> None:
    rc = LoopConstants(session_memory_head_protect_turns=0)
    seed = build_session_seed(_session(_turn(0), _turn(1)), SessionMemory(turn_index=2), BUDGET, rc, position=_position)
    assert seed.head_turns == 0 and seed.window_turns == 2


# --- the fold decision -------------------------------------------------------------


def test_no_fold_while_the_seed_fits() -> None:
    plan = plan_session_fold(_session(*(_turn(i, result_chars=2000) for i in range(3))), SessionMemory(turn_index=3), BUDGET, RC, position=_position)
    assert not plan.needed and plan.messages == [] and plan.step is None
    assert plan.projected_tokens <= plan.trigger_tokens


def test_a_single_large_turn_forces_no_fold() -> None:
    plan = plan_session_fold(_session(_turn(0, result_chars=30_000)), SessionMemory(turn_index=1), BUDGET, RC, position=_position)
    assert not plan.needed


def test_fold_plans_oldest_window_turns_and_spares_head_and_newest() -> None:
    history = _session(*(_turn(i) for i in range(14)))
    plan = plan_session_fold(history, SessionMemory(turn_index=14), BUDGET, RC, position=_position)
    assert plan.needed and plan.projected_tokens > plan.trigger_tokens
    numbers = [int(t[0].text.split()[1].rstrip(":")) for t in plan.turns]
    assert numbers == list(range(1, 1 + len(numbers)))
    assert 13 not in numbers
    assert plan.step is not None and plan.step.after == Coverage(_position(plan.turns[0][0]) or 0, _position(plan.turns[-1][-1]) or 0)


def test_the_fold_is_shown_what_the_seed_carried_including_kept_lines() -> None:
    deep = "body line\n" * 400 + "Cite: Order No. 1234-3\nhttps://example.org/deep/42\n" + "body line\n" * 400
    turn = [_user("question 1: deep"), _calls(("d", "Search", {"q": "deep"})), _results(("d", deep)), _assistant("see [3]")]
    history = _session(_turn(0), turn, *(_turn(i) for i in range(2, 14)))
    plan = plan_session_fold(history, SessionMemory(turn_index=14), BUDGET, RC, position=_position)
    assert plan.turns[0][0].text == "question 1: deep"
    text = build_summary_delta_message(SessionSummary(), plan.source_messages)
    assert text is not None
    assert "Cite: Order No. 1234-3" in text and "https://example.org/deep/42" in text
    assert "body line\n" * 50 not in text


def test_fold_source_budget_defers_turns_but_takes_at_least_one() -> None:
    history = _session(*(_turn(i) for i in range(14)))
    plan = plan_session_fold(history, SessionMemory(turn_index=14), BUDGET, RC, position=_position, max_source_tokens=1)
    assert len(plan.turns) == 1 and plan.deferred_turns > 0


def test_a_turn_without_positions_does_not_stop_later_turns_from_folding() -> None:
    history = _session(*(_turn(i) for i in range(14)))
    size = len(_turn(0))
    for k in range(size):  # every message of turn 2
        history[size * 2 + k] = history[size * 2 + k].model_copy(update={"metadata": {}})
    memory = SessionMemory(turn_index=14)
    plan = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    numbers = [int(t[0].text.split()[1].rstrip(":")) for t in plan.turns]
    assert numbers[:3] == [1, 3, 4]
    memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("r",)}), RC, step=plan.step).memory
    again = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    assert not again.needed or all(t[0].text != history[size * 2].text for t in again.turns)
    seed = build_session_seed(history, memory, BUDGET, RC, position=_position)
    assert any(m.text == history[size * 2].text for m in seed.messages) or seed.dropped_turns >= 1


def test_the_plan_keeps_room_for_the_newest_turn_uncut() -> None:
    # Near the trigger the masked history alone fits, but not with the newest
    # turn in full: the planner folds older turns rather than cut the newest.
    history = _session(*(_turn(i) for i in range(9)))
    memory = SessionMemory(turn_index=9)
    plan = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    assert plan.needed and plan.fresh_tokens > 5000
    memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("x" * 400,) * 1}), RC, step=plan.step).memory
    seed = build_session_seed(history, memory, BUDGET, RC, position=_position)
    assert seed.full_turns == 1 and seed.trimmed_turns == 0
    assert seed.summarized_turns == len(plan.turns)
    assert seed.tokens <= BUDGET
    assert not plan_session_fold(history, memory, BUDGET, RC, position=_position).needed


def test_the_summary_reserve_includes_its_wrapper() -> None:
    from protocore.runtime.context.session_seed import _summary_message, _summary_reserve

    rc = LoopConstants(session_memory_running_summary_token_cap=200, session_memory_fold_max_output_tokens=200)
    full = SessionSummary(entries={"KEY FACTS & IDENTIFIERS": tuple(f"fact {i} with words" for i in range(60))})
    from protocore.runtime.context.session_memory import cap_summary

    capped, _, _ = cap_summary(full, rc)
    memory = SessionMemory(summary=capped, turn_index=1)
    carried = sum(estimate_message_tokens(m, rc) for m in _summary_message(memory, 1, 0))
    assert _summary_reserve(memory, rc) >= carried


# --- prefix stability across runs -----------------------------------------------------


def _simulate(rc: LoopConstants, runs: int, *, budget: int = BUDGET, seed: int = 7, reasoning: bool = False) -> tuple[list[SessionSeed], list[bool], list[list[Message]]]:
    """Run a session ``runs`` times: seed, append the run's turn, plan, fold."""
    rng = random.Random(seed)
    history: list[Message] = []
    memory = SessionMemory()
    cache = SeedCache()
    seeds: list[SessionSeed] = []
    folded_before: list[bool] = []
    turns: list[list[Message]] = []
    landed = False
    for run in range(runs):
        seeds.append(build_session_seed(history, memory, budget, rc, position=_position, cache=cache))
        folded_before.append(landed)
        turn = _stamp(
            _turn(run, tools=rng.randint(1, 3), rounds=rng.randint(1, 3), result_chars=rng.randint(2000, 9000), reasoning=40 if reasoning else 0),
            len(history),
        )
        turns.append(turn)
        history = [*history, *turn]
        plan = plan_session_fold(history, memory, budget, rc, position=_position, cache=cache)
        delta = SummaryDelta(entries={"KEY FACTS & IDENTIFIERS": (f"fact from fold at run {run}",)}) if plan.needed else None
        memory = fold_run(memory, turn, delta, rc, step=plan.step).memory
        landed = plan.needed
    return seeds, folded_before, turns


def _unchanged_lead(turn: Sequence[Message], rc: LoopConstants) -> int:
    """How many leading messages of a turn masking leaves byte-identical."""
    full, _ = repair_turn(turn)
    masked = mask_turn(full, rc, default_masking_rule(rc))
    return _common_prefix(full, masked)


def test_consecutive_seeds_share_everything_but_the_full_tail_with_multi_round_turns() -> None:
    for reasoning in (False, True):
        seeds, folded_before, turns = _simulate(RC, 16, reasoning=reasoning)
        assert 1 <= sum(folded_before) < 16
        for run in range(2, len(seeds)):
            previous, current = seeds[run - 1], seeds[run]
            common = _common_prefix(previous.messages, current.messages)
            if folded_before[run]:
                assert common >= len(turns[0])  # the head survives a fold
                continue
            if previous.full_turns == 0 and previous.trimmed_turns == 0:
                assert common == len(previous.messages), run
                continue
            newest = turns[run - 2]
            expected = len(previous.messages) - len(newest) + _unchanged_lead(newest, RC)
            assert common == expected, (run, reasoning)


@pytest.mark.asyncio
async def test_consecutive_seeds_share_their_bytes_on_the_wire() -> None:
    seeds, folded_before, turns = _simulate(RC, 10, reasoning=True)
    capabilities = ProviderCapabilities(reasoning_replay=ReasoningReplay.when_present)

    async def wire_text(messages: Sequence[Message]) -> str:
        wire = await messages_to_wire(list(messages), capabilities=capabilities, thinking=None, endpoint_name="e", model="m")
        return "".join(json.dumps(entry, sort_keys=True, ensure_ascii=False) for entry in wire.entries)

    checked = 0
    for run in range(2, len(seeds)):
        if folded_before[run]:
            continue
        previous = seeds[run - 1].messages
        stable = previous[: len(previous) - len(turns[run - 2]) + _unchanged_lead(turns[run - 2], RC)]
        assert (await wire_text(seeds[run].messages)).startswith(await wire_text(stable))
        checked += 1
    assert checked >= 5


def test_with_batched_masking_most_runs_reuse_the_whole_previous_seed() -> None:
    rc = LoopConstants(session_seed_mask_batch_turns=3)
    seeds, folded_before, _turns = _simulate(rc, 7, budget=400_000)
    assert not any(folded_before)
    whole = sum(
        1
        for run in range(2, len(seeds))
        if _common_prefix(seeds[run - 1].messages, seeds[run].messages) == len(seeds[run - 1].messages)
    )
    assert whole >= 3


def test_the_summary_never_sits_in_front_of_carried_turns_it_does_not_cover() -> None:
    seeds, _folded, turns = _simulate(RC, 14)
    for seed in seeds:
        marks = [i for i, m in enumerate(seed.messages) if END_OF_SUMMARY_MARKER in m.text]
        if marks:
            assert marks == [len(turns[0])]


# --- cost ------------------------------------------------------------------------------


def test_with_a_session_cache_a_run_estimates_only_its_new_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = seed_module.estimate_message_tokens

    def counting(message: Message, rc: LoopConstants) -> int:
        calls["n"] += 1
        return real(message, rc)

    monkeypatch.setattr(seed_module, "estimate_message_tokens", counting)
    big = LoopConstants(session_memory_running_summary_token_cap=0)
    history = _session(*(_turn(i, result_chars=1500) for i in range(30)))
    memory = SessionMemory(turn_index=30)
    cache = SeedCache()
    build_session_seed(history, memory, 400_000, big, position=_position, cache=cache)
    first = calls["n"]
    assert first > 0
    calls["n"] = 0
    plan_session_fold(history, memory, 400_000, big, position=_position, cache=cache)
    assert calls["n"] <= 3  # the summary reserve and the registry, nothing per turn
    new_turn = _stamp(_turn(30, result_chars=1500), len(history))
    history = [*history, *new_turn]
    calls["n"] = 0
    build_session_seed(history, memory, 400_000, big, position=_position, cache=cache)
    plan_session_fold(history, memory, 400_000, big, position=_position, cache=cache)
    # Each message of the new turn once in full and once per masked message
    # that differs, plus a handful of fixed blocks — not the 30 old turns.
    assert calls["n"] <= 2 * len(new_turn) + 6
    assert len(cache) >= 31


def test_the_default_rule_has_a_stable_cache_key() -> None:
    assert default_masking_rule(RC).key == default_masking_rule(RC).key != ""
    other = LoopConstants(session_seed_mask_identifier_lines=2)
    assert default_masking_rule(other).key != default_masking_rule(RC).key


def test_nothing_covered_is_the_default() -> None:
    assert SessionSummary().coverage is None


# --- the cache key -------------------------------------------------------------------


def test_a_preparation_setting_misses_the_cache_and_an_unrelated_one_does_not() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    cache = SeedCache()
    build_session_seed(history, SessionMemory(turn_index=3), 60_000, RC, position=_position, cache=cache)
    assert len(cache) == 3
    unrelated = LoopConstants(max_turns_per_run=7)
    build_session_seed(history, SessionMemory(turn_index=3), 60_000, unrelated, position=_position, cache=cache)
    assert len(cache) == 3
    for changed in (LoopConstants(session_seed_mask_preview_chars=50), LoopConstants(token_count_chars_per_token_latin=2.0)):
        before = len(cache)
        seed = build_session_seed(history, SessionMemory(turn_index=3), 60_000, changed, position=_position, cache=cache)
        assert len(cache) == before + 3
        if changed.session_seed_mask_preview_chars == 50:
            first_tool = next(m for m in seed.messages if m.role is MessageRole.tool)
            assert "It began with 50 characters:" in _result_text(first_tool)


def test_a_change_anywhere_in_a_turn_misses_the_cache() -> None:
    turn = _turn(0)
    history = _session(turn, _turn(1))
    cache = SeedCache()
    build_session_seed(history, SessionMemory(turn_index=2), 60_000, RC, position=_position, cache=cache)
    edited = list(history)
    last = len(turn) - 1
    edited[last] = edited[last].model_copy(update={"content_blocks": (TextBlock(text="a corrected answer"),)})
    seed = build_session_seed(edited, SessionMemory(turn_index=2), 60_000, RC, position=_position, cache=cache)
    assert any(m.text == "a corrected answer" for m in seed.messages)
    assert len(cache) == 3


def test_a_rule_without_a_key_is_never_served_another_rules_masking() -> None:
    history = _session(_turn(0), _turn(1))
    cache = SeedCache()
    for tag in ("FIRST", "SECOND", "THIRD"):
        rule = MaskingRule(kept_lines=lambda call, result, tag=tag: [f"{tag} line"])
        seed = build_session_seed(history, SessionMemory(turn_index=2), 60_000, RC, position=_position, rule=rule, cache=cache)
        masked = _result_text(next(m for m in seed.messages if m.role is MessageRole.tool))
        assert f"{tag} line" in masked
    assert len(cache) == 0


def test_identical_turns_keep_their_own_metadata() -> None:
    same = [_user("continue"), _assistant("Done.")]
    history = _session(same, same, same, _turn(3))
    seed = build_session_seed(history, SessionMemory(turn_index=4), 60_000, RC, position=_position, cache=SeedCache())
    positions = [_position(m) for m in seed.messages if m.text in ("continue", "Done.")]
    assert positions == [0, 1, 2, 3, 4, 5]


def test_the_cache_is_safe_across_threads() -> None:
    import threading

    histories = [_session(*(_turn(i + k) for i in range(4))) for k in range(6)]
    cache = SeedCache(max_entries=3)
    errors: list[BaseException] = []

    def work(history: list[Message]) -> None:
        try:
            for _ in range(5):
                build_session_seed(history, SessionMemory(turn_index=4), 60_000, RC, position=_position, cache=cache)
                plan_session_fold(history, SessionMemory(turn_index=4), 60_000, RC, position=_position, cache=cache)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work, args=(h,)) for h in histories]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(cache) <= 3


def test_every_cache_access_holds_its_lock() -> None:
    class Recording:
        def __init__(self) -> None:
            self.held = 0
            self.entered = 0

        def __enter__(self) -> None:
            self.held += 1
            self.entered += 1

        def __exit__(self, *exc: object) -> None:
            self.held -= 1

    cache = SeedCache(max_entries=2)
    lock = Recording()
    cache._lock = lock  # type: ignore[assignment]
    entries = cache._entries

    class Checked(type(entries)):  # type: ignore[misc]
        def __getitem__(self, key: object) -> object:
            assert lock.held, "read without the lock"
            return super().__getitem__(key)

        def get(self, key: object, default: object = None) -> object:
            assert lock.held, "read without the lock"
            return super().get(key, default)

        def move_to_end(self, key: object, last: bool = True) -> None:
            assert lock.held, "reorder without the lock"
            super().move_to_end(key, last)

        def popitem(self, last: bool = True) -> object:
            assert lock.held, "evict without the lock"
            return super().popitem(last)

    cache._entries = Checked()  # type: ignore[assignment]
    history = _session(_turn(0), _turn(1), _turn(2))
    build_session_seed(history, SessionMemory(turn_index=3), 60_000, RC, position=_position, cache=cache)
    build_session_seed(history, SessionMemory(turn_index=3), 60_000, RC, position=_position, cache=cache)
    assert lock.entered >= 6 and lock.held == 0


def test_kept_lines_and_keep_whole_are_exported_where_documented() -> None:
    from protocore.runtime import context

    assert context.KeptLines is seed_module.KeptLines and context.KeepWhole is seed_module.KeepWhole


# --- argument masking edge cases ----------------------------------------------------


def _masked_args(raw: str, rc: LoopConstants = RC) -> str:
    turn = [_user("q"), Message(role=MessageRole.assistant, content_blocks=[ToolUseBlock(tool_call_id="w", name="Write", arguments_json=raw)]), _results(("w", "ok")), _assistant("a")]
    call = mask_turn(turn, rc, default_masking_rule(rc))[1].content_blocks[0]
    assert isinstance(call, ToolUseBlock)
    return call.arguments_json


def test_arguments_with_numbers_json_cannot_carry_are_left_alone() -> None:
    for raw in ('{"x": 1e400, "body": "' + "a" * 2000 + '"}', '{"x": NaN, "body": "' + "a" * 2000 + '"}'):
        assert _masked_args(raw) == raw


def test_long_keys_and_long_lists_of_short_strings_are_cut() -> None:
    raw = json.dumps({"k" * 1000: 1, "items": [f"item number {i:04d} with some words" for i in range(500)]})
    out = _masked_args(raw)
    parsed = json.loads(out)
    assert len(out) < 1000
    assert any(key.endswith("[1000 characters, not carried]") for key in parsed)
    assert parsed["items"][-1].endswith("more items, not carried]")
    assert parsed["items"][0] == "item number 0000 with some words"


def test_a_cut_argument_never_splits_a_character() -> None:
    raw = json.dumps({"body": "\U0001f600" * 1000}, ensure_ascii=True)
    out = _masked_args(raw, LoopConstants(session_seed_mask_arg_preview_chars=7))
    out.encode("utf-8")
    assert json.loads(out)["body"].startswith("\U0001f600" * 7 + "…")


def test_an_escaped_lone_surrogate_in_arguments_is_left_escaped() -> None:
    raw = '{"content": "\\ud83d' + "x" * 2000 + '"}'
    out = _masked_args(raw)
    out.encode("utf-8")
    assert out == raw
    head_only = '{"content": "' + "y" * 2000 + '\\ud83d"}'
    _masked_args(head_only).encode("utf-8")


def test_argument_masking_never_makes_the_call_longer() -> None:
    fields = {f"f{i}": i for i in range(300)}
    fields["body"] = "z" * 401
    raw = json.dumps(fields, separators=(",", ":"))
    # Cutting one 401-character value saves less than re-spelling 300 fields
    # would cost if the output were not compact; with compact output it still
    # must not grow, and an output that would is not used.
    out = _masked_args(raw, LoopConstants(session_seed_mask_arg_preview_chars=380))
    assert len(out) <= len(raw)
    assert out == raw


# --- coverage as a range: the head --------------------------------------------------


def test_a_head_whose_protection_lapses_is_carried_and_then_folded_never_lost() -> None:
    history = _session(*(_turn(i) for i in range(14)))
    memory = SessionMemory(turn_index=14)
    plan = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    assert plan.needed and plan.turns[0][0].text.startswith("question 1")
    memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("r",)}), RC, step=plan.step).memory
    assert memory.summary.coverage is not None and memory.summary.coverage.first == len(_turn(0))
    for lapsed in (LoopConstants(session_memory_head_protect_turns=0), LoopConstants(session_memory_head_max_tokens=10)):
        seed = build_session_seed(history, memory, BUDGET, lapsed, position=_position)
        assert seed.head_turns == 0
        assert seed.messages[0].text.startswith("question 0")  # carried, before the summary
        assert END_OF_SUMMARY_MARKER in seed.messages[len(mask_turn(_turn(0), lapsed, default_masking_rule(lapsed)))].text
        again = plan_session_fold(history, memory, BUDGET, lapsed, position=_position)
        if again.needed:
            assert again.turns[0][0].text.startswith("question 0")
            folded = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("q0",)}), lapsed, step=again.step).memory
            assert folded.summary.coverage is not None and folded.summary.coverage.first == 0


def test_lapsed_head_turns_are_folded_nearest_first_within_the_source_bound_and_never_leave_a_gap() -> None:
    two_heads = LoopConstants(session_memory_head_protect_turns=2)
    history = _session(*(_turn(i) for i in range(14)))
    memory = SessionMemory(turn_index=14)
    plan = plan_session_fold(history, memory, BUDGET, two_heads, position=_position)
    assert plan.turns[0][0].text.startswith("question 2")
    memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("r",)}), two_heads, step=plan.step).memory
    for i in range(14, 22):
        history = [*history, *_stamp(_turn(i), len(history))]
    lapsed = LoopConstants(session_memory_head_protect_turns=0)
    seen: list[str] = []
    for _ in range(3):
        again = plan_session_fold(history, memory, BUDGET, lapsed, position=_position, max_source_tokens=1)
        assert again.needed and len(again.turns) == 1  # the source bound holds, lapsed turns too
        seen.append(again.turns[0][0].text.split(":")[0])
        memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("q",)}), lapsed, step=again.step).memory
        seed = build_session_seed(history, memory, 400_000, lapsed, position=_position)
        carried = sum(1 for m in seed.messages if m.text.startswith("question"))
        assert carried + seed.summarized_turns == 22  # nothing in neither place, at every step
    assert seen[:2] == ["question 1", "question 0"]


# --- convergence -----------------------------------------------------------------------


def _fold_rate(result_chars: int, runs: int = 20) -> tuple[int, int, list[int]]:
    history: list[Message] = []
    memory = SessionMemory()
    cache = SeedCache()
    folds: list[int] = []
    cut = 0
    for run in range(runs):
        seed = build_session_seed(history, memory, BUDGET, RC, position=_position, cache=cache)
        assert seed.tokens <= BUDGET
        cut += seed.trimmed_turns
        turn = _stamp(_turn(run, tools=3, result_chars=result_chars, answer_chars=1500), len(history))
        history = [*history, *turn]
        plan = plan_session_fold(history, memory, BUDGET, RC, position=_position, cache=cache)
        if plan.needed:
            folds.append(run)
        delta = SummaryDelta(entries={"KEY FACTS & IDENTIFIERS": (f"fact {run}",)}) if plan.needed else None
        memory = fold_run(memory, turn, delta, RC, step=plan.step).memory
    return len(folds), cut, folds


@pytest.mark.parametrize("result_chars", [6000, 12_000, 13_000, 14_000, 16_000, 20_000])
def test_folds_never_run_on_every_run_whatever_the_newest_turn_size(result_chars: int) -> None:
    count, _cut, folds = _fold_rate(result_chars)
    assert count <= 10, folds
    assert all(b - a >= 2 for a, b in itertools.pairwise(folds)), folds


def test_the_newest_turn_stays_uncut_where_that_costs_at_most_a_fold_every_second_run() -> None:
    for result_chars in (6000, 12_000, 13_000):
        _count, cut, _folds = _fold_rate(result_chars)
        assert cut == 0, result_chars


# --- catching up after a reset ------------------------------------------------------------


def test_after_a_reset_every_seed_fits_and_the_backlog_is_folded_in_bounded_batches() -> None:
    history = _session(*(_turn(i, result_chars=4000) for i in range(60)))
    memory = SessionMemory(turn_index=60)
    cache = SeedCache()
    source = 8000
    first = plan_session_fold(history, memory, BUDGET, RC, position=_position, cache=cache, max_source_tokens=source)
    backlog = first.backlog_turns
    bound = backlog  # at least one turn per fold
    runs = 0
    for runs in range(1, bound + 2):
        seed = build_session_seed(history, memory, BUDGET, RC, position=_position, cache=cache)
        assert seed.tokens <= BUDGET
        if seed.dropped_turns:
            assert any("not in this context and not in the summary" in m.text for m in seed.messages)
        plan = plan_session_fold(history, memory, BUDGET, RC, position=_position, cache=cache, max_source_tokens=source)
        if not plan.needed:
            break
        assert fold_source_tokens(plan.source_messages, RC) <= source or len(plan.turns) == 1
        memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": (f"batch {runs}",)}), RC, step=plan.step).memory
    assert runs <= bound
    assert build_session_seed(history, memory, BUDGET, RC, position=_position, cache=cache).dropped_turns == 0


# --- round-three behaviour ------------------------------------------------------------


def test_the_cut_marker_is_exported_and_recognised() -> None:
    body = "x = 1\n" * 4000
    turn = [_user("q"), _calls(("w", "Write", {"path": "/srv/a.py", "content": body})), _results(("w", "ok")), _assistant("a")]
    call = mask_turn(turn, RC, default_masking_rule(RC))[1].content_blocks[0]
    assert isinstance(call, ToolUseBlock)
    content = json.loads(call.arguments_json)["content"]
    assert content.endswith(CUT_ARGUMENT_SUFFIX) and has_cut_marker(content)
    assert not has_cut_marker(body) and not has_cut_marker("[many characters, not carried]")
    assert has_cut_marker(json.loads(_masked_args(json.dumps({"items": [f"short {i}" for i in range(400)]})))["items"][-1])


def test_two_turns_that_differ_only_in_an_attached_image_are_not_confused_by_the_cache() -> None:
    images = [ImageRefBlock(blob_ref=f"img-{k}").model_dump(mode="json") for k in range(8)]
    plain = [_user("what is in the picture?"), _assistant("A cat.")]
    with_images = [plain[0].model_copy(update={"metadata": {IMAGE_REFS_METADATA_KEY: images}}), plain[1]]
    cache = SeedCache()
    first = build_session_seed(_session(plain), SessionMemory(turn_index=1), 400_000, RC, position=_position, cache=cache)
    second = build_session_seed(_session(with_images), SessionMemory(turn_index=1), 400_000, RC, position=_position, cache=cache)
    assert second.tokens == sum(estimate_message_tokens(m, RC) for m in second.messages)
    assert second.tokens > first.tokens + 8 * RC.token_count_image_tokens - 10


def test_every_seed_encodes_whatever_the_input_holds() -> None:
    rng = random.Random(11)
    pieces = ["plain", "\ud83d", "\udc00", "\U0001f600", "кириллица", "\\ud83d", "\n", "Cite: x"]
    failures = 0
    for case in range(160):
        def text(n: int) -> str:
            return "".join(rng.choice(pieces) for _ in range(n))
        turns = []
        for t in range(rng.randint(1, 4)):
            args_raw = json.dumps({"q": text(3), "body": text(rng.randint(0, 120)) * rng.randint(1, 12)})
            turns.append([
                _user(text(5)),
                Message(role=MessageRole.assistant, content_blocks=[ToolUseBlock(tool_call_id=f"c{case}-{t}", name="Tool", arguments_json=args_raw)], reasoning_content=text(4) or None),
                _results((f"c{case}-{t}", text(rng.randint(1, 300)) * rng.randint(1, 4))),
                _assistant(text(10)),
            ])
        history = _session(*turns)
        memory = SessionMemory(summary=SessionSummary(entries={"REQUESTS": (text(4) or "r",)}), turn_index=len(turns))
        seed = build_session_seed(history, memory, rng.choice([300, 3000, 60_000]), RC, position=_position)
        plan = plan_session_fold(history, memory, rng.choice([300, 3000]), RC, position=_position)
        try:
            for message in seed.messages:
                message.model_dump_json().encode("utf-8")
            text_in = build_summary_delta_message(memory.summary, plan.source_messages or [m for t in turns for m in t])
            if text_in is not None:
                text_in.encode("utf-8")
        except (UnicodeEncodeError, ValueError):
            failures += 1
    assert failures == 0


def test_a_large_registry_never_costs_the_previous_answer() -> None:
    files = [f"src/module_{i}.py" for i in range(40)]
    ledger = ArtifactLedger(files=files, content={f: "def f():\n    return 1\n" * 200 for f in files})
    first = [_user("list the three reports"), _assistant("1. Alpha report\n2. Beta report\n3. Gamma report")]
    history = _session(*(_turn(i) for i in range(3)), first)
    memory = SessionMemory(summary=SessionSummary(entries={"REQUESTS": ("r " * 300,)}), ledger=ledger, turn_index=4)
    rc = LoopConstants(session_memory_registry_max_tokens=12_000)
    for budget in (16_000, 4_000, 900, 200):
        seed = build_session_seed(history, memory, budget, rc, position=_position)
        assert seed.tokens <= budget
        assert any("2. Beta report" in m.text for m in seed.messages), budget
    tight = build_session_seed(history, memory, 900, rc, position=_position)
    assert tight.registry_reduced and not tight.dropped_registry and tight.dropped_turns >= 3
    tighter = build_session_seed(history, memory, 200, rc, position=_position)
    assert tighter.dropped_registry and tighter.dropped_summary


def test_the_newest_turn_reduced_to_its_words_keeps_question_and_answer() -> None:
    history = _session(_turn(0, result_chars=30_000, answer_chars=200))
    seed = build_session_seed(history, SessionMemory(turn_index=1), 400, RC, position=_position)
    assert seed.words_only_turns == 1 and seed.tokens <= 400
    texts = [m.text for m in seed.messages]
    assert texts[0].startswith("question 0") and texts[1].startswith("Answer 0")
    assert "tool calls and results of the latest earlier turn are not in this context" in texts[2]
    assert all(m.role is not MessageRole.tool for m in seed.messages)


def test_repeated_cut_folds_shrink_the_batch_to_one_turn_and_then_pin_it() -> None:
    history = _session(*(_turn(i) for i in range(16)))
    memory = SessionMemory(turn_index=16)
    sizes: list[int] = []
    pinned = False
    for _ in range(12):
        plan = plan_session_fold(history, memory, BUDGET, RC, position=_position)
        assert plan.step is not None
        sizes.append(len(plan.turns))
        if plan.pin:
            assert not plan.needed
            memory = fold_run(memory, [], None, RC, step=plan.step).memory
            pinned = True
            break
        cut = SummaryDelta(entries={"KEY FACTS & IDENTIFIERS": ("partial fact",)}, truncated=True)
        memory = fold_run(memory, [], cut, RC, step=plan.step, fold_attempted=True).memory
    assert pinned
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] == 1
    assert sizes[0] > 1 and len(sizes) <= 2 + sizes[0].bit_length() + RC.session_seed_fold_cuts_before_pin
    seed = build_session_seed(history, memory, 400_000, RC, position=_position)
    assert seed.pinned_turns == 1
    assert any("could not be summarised" in m.text for m in seed.messages)
    after = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    assert after.step is None or after.turns[0] != plan.turns[0]
    assert memory.summary.entries["KEY FACTS & IDENTIFIERS"] == ("partial fact",)  # not re-added per retry


def _check_no_gap(history: list[Message], memory: SessionMemory, before: SessionMemory, planned: set[int]) -> None:
    turns = split_turns(history)
    if turns and not opens_turn(turns[0][0]):
        turns = turns[1:]
    known = seed_module._known_positions(turns, _position)
    for index, positions in enumerate(known):
        if not positions:
            continue
        was = all(before.summary.covers(p) for p in positions)
        now = all(memory.summary.covers(p) for p in positions)
        partly = any(memory.summary.covers(p) for p in positions) and not now
        assert not partly, index
        if now and not was:
            assert index in planned, index


def test_coverage_never_spans_a_turn_it_did_not_plan_under_random_histories_and_stale_steps() -> None:
    rng = random.Random(3)
    for case in range(80):
        turns = [_turn(i, tools=rng.randint(0, 2), result_chars=rng.randint(500, 6000)) for i in range(rng.randint(2, 14))]
        history = _session(*turns)
        for _ in range(rng.randint(0, 4)):  # missing or broken positions
            at = rng.randrange(len(history))
            meta = {} if rng.random() < 0.5 else {POS: rng.randint(0, len(history))}
            history[at] = history[at].model_copy(update={"metadata": meta})
        rc = LoopConstants(session_memory_head_protect_turns=rng.randint(0, 2))
        memory = SessionMemory(turn_index=len(turns))
        old_steps: list[tuple[CoverageStep, set[int]]] = []
        for _ in range(6):
            budget = rng.choice([2_000, 6_000, 16_000])
            plan = plan_session_fold(history, memory, budget, rc, position=_position, max_source_tokens=rng.choice([0, 500, 4000]))
            if plan.step is None:
                continue
            firsts = {id(turn[0]) for turn in plan.turns}
            mine = {index for index, turn in enumerate(_turns_of(history)) if id(turn[0]) in firsts}
            old_steps.append((plan.step, mine))
            step, planned = old_steps[-1] if rng.random() < 0.7 else rng.choice(old_steps)
            delta = SummaryDelta(entries={"REQUESTS": (f"case {case}",)}, truncated=rng.random() < 0.2)
            before = memory
            result = fold_run(memory, [], None if step.pin else delta, rc, step=step, fold_attempted=True)
            memory = result.memory
            _check_no_gap(history, memory, before, planned)
            if rng.random() < 0.3:
                rc = LoopConstants(session_memory_head_protect_turns=rng.randint(0, 2))


def _turns_of(history: list[Message]) -> list[list[Message]]:
    turns = split_turns(history)
    return turns[1:] if turns and not opens_turn(turns[0][0]) else turns


def test_positions_that_break_the_contract_count_as_unknown() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    size = len(_turn(0))
    swapped = list(history)
    swapped[size], swapped[size + 1] = (
        swapped[size].model_copy(update={"metadata": {POS: size + 1}}),
        swapped[size + 1].model_copy(update={"metadata": {POS: size}}),
    )
    turns = _turns_of(swapped)
    known = seed_module._known_positions(turns, _position)
    assert known[0] and known[1] == [] and known[2]
    seed = build_session_seed(swapped, _covering(size, 10**6), 60_000, RC, position=_position)
    assert sum(1 for m in seed.messages if m.text.startswith("question 1")) == 1


def test_in_run_compaction_remasks_the_seeded_newest_turn_first_to_the_next_seeds_form() -> None:
    import asyncio

    from protocore.runtime.context.compaction import remask_seeded_turns, run_tier1_truncation
    from protocore.tests_support.adapters import InMemoryBlobStore

    history = _session(_turn(0), _turn(1, rounds=2, reasoning=20))
    seed = build_session_seed(history, SessionMemory(turn_index=2), 200_000, RC, position=_position)
    assert seed.full_turns == 1 and seed.reclaimable_tokens > 5_000
    tagged = [m for m in seed.messages if SESSION_SEED_MASKED_FORM_METADATA_KEY in m.metadata]
    assert tagged
    later = [*history, *_stamp(_turn(2, tools=0), len(history))]
    next_seed = build_session_seed(later, SessionMemory(turn_index=3), 200_000, RC, position=_position)
    run = [*seed.messages, _user("the new task")]
    before = sum(estimate_message_tokens(m, RC) for m in run)
    freed = remask_seeded_turns(list(run), RC).tokens_freed
    assert freed > 5_000
    result = asyncio.run(
        run_tier1_truncation(run, InMemoryBlobStore(), "t", RC, truncation_threshold_tokens=10**9, keep_recent_turns=1)
    )
    assert result.tokens_freed >= freed
    assert sum(estimate_message_tokens(m, RC) for m in run) <= before - freed
    wire_run = [_wire(m) for m in run[:-1]]
    wire_next = [_wire(m) for m in next_seed.messages if not m.text.startswith("question 2")][: len(wire_run)]
    assert wire_run == wire_next
    assert not any(SESSION_SEED_MASKED_FORM_METADATA_KEY in m.metadata for m in run)
    assert _pairs_valid(run)


def test_recommend_seed_budget_splits_the_room_up_to_the_trigger() -> None:
    rc = LoopConstants(model_context_window=49_152)
    budget = recommend_seed_budget(rc, 8_000)
    assert budget.trigger_tokens == 39_117
    assert budget.seed_tokens + budget.run_reserve_tokens == 39_117 - 8_000
    assert budget.run_reserve_tokens >= rc.session_seed_run_reserve_min_tokens
    assert budget.seed_tokens == int((39_117 - 8_000) * rc.session_seed_budget_ratio)
    tight = recommend_seed_budget(rc, 36_000)
    assert tight.seed_tokens == 1_559 and tight.run_reserve_tokens == 1_558  # the reserve scales down
    assert recommend_seed_budget(rc, -5).overhead_tokens == 0


def test_recommend_seed_budget_never_starves_a_follow_up_of_the_previous_answer() -> None:
    history = _session(_turn(0, answer_chars=2_000))
    floor = seed_floor_tokens(history, RC)
    assert floor > 400
    for window, overhead in ((8_192, 2_000), (8_192, 8_000), (16_384, 8_000), (49_152, 39_117)):
        rc = LoopConstants(model_context_window=window)
        budget = recommend_seed_budget(rc, overhead, history=history)
        assert budget.floor_tokens == floor
        if floor <= window - overhead:
            assert budget.seed_tokens >= floor, (window, overhead)
            seed = build_session_seed(history, SessionMemory(turn_index=1), budget.seed_tokens, rc, position=_position)
            assert any(m.text.startswith("Answer 0") for m in seed.messages)
    assert recommend_seed_budget(LoopConstants(model_context_window=8_192), 8_192, history=history).seed_tokens == 0
    assert seed_floor_tokens([], RC) == 0


def test_without_fold_spacing_a_large_newest_turn_folds_every_run() -> None:
    def rate(rc: LoopConstants) -> int:
        history: list[Message] = []
        memory = SessionMemory()
        folds = 0
        for run in range(16):
            turn = _stamp(_turn(run, tools=3, result_chars=14_400, answer_chars=1500), len(history))
            history = [*history, *turn]
            plan = plan_session_fold(history, memory, BUDGET, rc, position=_position)
            folds += plan.needed
            delta = SummaryDelta(entries={"KEY FACTS & IDENTIFIERS": (f"fact {run}",)}) if plan.needed else None
            memory = fold_run(memory, turn, delta, rc, step=plan.step if delta else None).memory
        return folds

    assert rate(LoopConstants(session_seed_fold_min_turns_between=0)) >= 12
    assert rate(RC) <= 8


def test_the_convergence_bound_holds() -> None:
    history = _session(*(_turn(i, result_chars=rng_size) for i, rng_size in enumerate([2000, 9000, 4000, 7000] * 10)))
    memory = SessionMemory(turn_index=40)
    source = 2_500
    first = plan_session_fold(history, memory, BUDGET, RC, position=_position, max_source_tokens=source)
    backlog = first.backlog_turns
    largest = max(fold_source_tokens(seed_module._Preparer(RC, default_masking_rule(RC), None).prepare(t).masked, RC) for t in _turns_of(history))
    folds = 0
    folded_source = 0
    while True:
        plan = plan_session_fold(history, memory, BUDGET, RC, position=_position, max_source_tokens=source)
        if not plan.needed:
            break
        folds += 1
        folded_source += fold_source_tokens(plan.source_messages, RC)
        memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": (f"b{folds}",)}), RC, step=plan.step).memory
    assert folds <= backlog
    if source > largest:
        assert folds <= -(-folded_source // (source - largest)) + 1


# --- in-run gate and partial re-mask -------------------------------------------------


def _gate_run() -> tuple[list[Message], int, LoopConstants]:
    rc = LoopConstants(model_context_window=49_152)
    budget = recommend_seed_budget(rc, 8_000)
    history = _session(_turn(4, result_chars=3_000), _turn(5, tools=6, result_chars=9_000))
    seed = build_session_seed(history, SessionMemory(turn_index=2), budget.seed_tokens, rc, position=_position)
    assert seed.full_turns == 1 and seed.reclaimable_tokens > 10_000
    run = [*seed.messages, _user("the new task")]
    run += [_calls(("n0", "Search", {"query": "q0"})), _results(("n0", _bulky("n0", 70_000)))]
    return run, len(run) - 2, rc


def test_the_routine_gate_opens_for_a_seeded_full_turn_and_the_pass_gets_under_the_trigger() -> None:
    import asyncio

    from protocore.runtime.context.budgets import derive_budgets
    from protocore.runtime.context.compaction import CompactionState
    from protocore.runtime.context.manager import ContextManager
    from protocore.tests_support.adapters import InMemoryBlobStore

    run, protect, rc = _gate_run()
    trigger = derive_budgets(rc).compaction_trigger_tokens
    prompt = sum(estimate_message_tokens(m, rc) for m in run) + 8_000
    assert prompt > trigger
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    assert manager.has_proactive_work(run, CompactionState(), force=False, protect_tail_from_index=protect, llm_tiers=False)
    attempt = asyncio.run(
        manager.run_compaction(
            history=run, compaction_state=CompactionState(), tenant_id="t", model_name="m",
            overhead_tokens=8_000, llm_tiers=False, protect_tail_from_index=protect,
        )
    )
    assert attempt.prompt_after < trigger
    assert attempt.target_tokens < attempt.prompt_before  # the re-mask counts as removable
    assert any(isinstance(b, ToolResultBlock) and b.content.startswith("Cite: record n0") for m in run for b in m.content_blocks)


def test_remask_gives_back_only_what_is_needed_oldest_first() -> None:
    from protocore.runtime.context.compaction import is_seeded_full, remask_seeded_turns, remask_surplus_tokens

    history = _session(_turn(0), _turn(1, rounds=3, tools=3))
    seed = build_session_seed(history, SessionMemory(turn_index=2), 400_000, RC, position=_position)
    run = list(seed.messages)
    surplus = remask_surplus_tokens(run, RC)
    tagged = [i for i, m in enumerate(run) if is_seeded_full(m)]
    result = remask_seeded_turns(run, RC, free_target_tokens=500)
    assert 500 <= result.tokens_freed < surplus
    assert 0 < result.tagged_left < len(tagged)
    still = [i for i, m in enumerate(run) if is_seeded_full(m)]
    assert still and min(still) > tagged[0]  # the oldest went first, the rest kept their detail


def test_a_removed_reasoning_only_message_does_not_shift_the_protected_batch() -> None:
    import asyncio

    from protocore.runtime.context.compaction import run_tier1_truncation
    from protocore.tests_support.adapters import InMemoryBlobStore

    turn = [_user("question 1"), Message(role=MessageRole.assistant, content_blocks=[ThinkingBlock(text="only thinking")])]
    turn += [_calls(("s1", "Search", {"query": "x"})), _results(("s1", _bulky("s1", 9_000))), _assistant("answer 1")]
    history = _session(_turn(0), turn)
    seed = build_session_seed(history, SessionMemory(turn_index=2), 200_000, RC, position=_position)
    run = [*seed.messages, _user("task"), _calls(("n1", "Search", {"query": "q"}), reasoning="current loop reasoning " * 50)]
    run.append(_results(("n1", _bulky("n1", 6_000))))
    protect = len(run) - 2
    asyncio.run(
        run_tier1_truncation(
            run, InMemoryBlobStore(), "t", RC, truncation_threshold_tokens=500, keep_recent_turns=0,
            protect_tail_from_index=protect, mask_by_age=True,
        )
    )
    call = next(m for m in run if any(isinstance(b, ToolUseBlock) and b.tool_call_id == "n1" for b in m.content_blocks))
    assert call.reasoning_content is not None


# --- head question, registry bound, pins, source halving, registry reserve ------------------


def test_the_first_question_is_protected_even_when_the_first_turn_is_too_large() -> None:
    history = _session(_turn(0, answer_chars=14_000), *(_turn(i) for i in range(1, 14)))
    memory = SessionMemory(turn_index=14)
    seed = build_session_seed(history, memory, BUDGET, RC, position=_position)
    assert seed.head_turns == 0
    plan = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    assert plan.turns[0][0].text.startswith("question 0")  # the turn itself is folded
    memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("r",)}), RC, step=plan.step).memory
    after = build_session_seed(history, memory, BUDGET, RC, position=_position)
    assert after.messages[0].text == "question 0: find the items about topic 0"
    assert sum(1 for m in after.messages if m.text.startswith("question 0")) == 1


def test_the_registrys_reduced_form_respects_its_bound_and_spares_the_newest_turn() -> None:
    files = [f"/srv/projects/very/deep/directory/structure/module_{i:02d}/implementation_file_{i}.py" for i in range(60)]
    ledger = ArtifactLedger(files=files, content={f: "x = 1\n" * 300 for f in files})
    rc = LoopConstants(session_memory_registry_max_tokens=2_000)
    history = _session(_turn(0, result_chars=2_000), _turn(1, result_chars=2_000))
    memory = SessionMemory(ledger=ledger, turn_index=2)
    roomy = build_session_seed(history, memory, 400_000, rc, position=_position)
    assert roomy.full_turns == 1
    for budget in (9_000, 8_000, 6_000):
        seed = build_session_seed(history, memory, budget, rc, position=_position)
        registry = next((m for m in seed.messages if m.text.startswith("ARTIFACT REGISTRY")), None)
        if registry is not None:
            assert estimate_message_tokens(registry, rc) <= rc.session_memory_registry_max_tokens + 10
        if roomy.tokens - estimate_message_tokens(next(m for m in roomy.messages if m.text.startswith("ARTIFACT")), rc) + 60 <= budget:
            assert seed.full_turns == 1, budget


def test_a_pin_goes_to_the_turn_that_was_cut_not_the_one_that_comes_first() -> None:
    history = _session(*(_turn(i) for i in range(16)))
    size = len(_turn(0))
    span_3 = Coverage(size * 3, size * 4 - 1)
    memory = SessionMemory(turn_index=16, cut_turn=span_3, cut_count=RC.session_seed_fold_cuts_before_pin)
    plan = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    assert plan.pin and plan.step is not None and plan.step.pinned == (span_3,)  # the cut turn, not turn 1
    other = SessionMemory(turn_index=16, cut_turn=Coverage(10**6, 10**6 + 1), cut_count=5)
    assert not plan_session_fold(history, other, BUDGET, RC, position=_position).pin
    span_1 = Coverage(size, size * 2 - 1)
    first = SessionMemory(turn_index=16, cut_turn=span_1, cut_count=RC.session_seed_fold_cuts_before_pin)
    pinned = plan_session_fold(history, first, BUDGET, RC, position=_position)
    assert pinned.pin and pinned.step is not None and pinned.step.pinned == (span_1,)


def test_a_pin_is_released_when_its_basis_changes_or_after_enough_runs() -> None:
    history = _session(*(_turn(i) for i in range(16)))
    size = len(_turn(0))
    span_1 = Coverage(size, size * 2 - 1)
    memory = SessionMemory(turn_index=16, cut_turn=span_1, cut_count=RC.session_seed_fold_cuts_before_pin)
    plan = plan_session_fold(history, memory, BUDGET, RC, position=_position, summariser="model-a")
    memory = fold_run(memory, [], None, RC, step=plan.step, summariser="model-a").memory
    assert memory.summary.pinned == (span_1,)
    same = plan_session_fold(history, memory, BUDGET, RC, position=_position, summariser="model-a")
    assert all(t[0].text != "question 1: find the items about topic 1" for t in same.turns)
    for released in (
        plan_session_fold(history, memory, BUDGET, RC, position=_position, summariser="model-b"),
        plan_session_fold(history, memory, BUDGET, LoopConstants(session_memory_running_summary_token_cap=1_800), position=_position, summariser="model-a"),
        plan_session_fold(
            history,
            replace_memory(memory, turn_index=memory.pinned_at_run + RC.session_seed_pin_release_runs),
            BUDGET, RC, position=_position, summariser="model-a",
        ),
    ):
        assert released.needed and released.turns[0][0].text.startswith("question 1")
        assert released.step is not None and released.step.unpin == (span_1,)
    landed = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("q1",)}), RC, step=released.step, summariser="model-a")
    assert landed.memory.summary.pinned == () and landed.memory.summary.covers(span_1.first)


def replace_memory(memory: SessionMemory, **changes: Any) -> SessionMemory:
    data = memory.to_dict()
    data.update(changes)
    return SessionMemory.from_dict(data)


def test_the_source_bound_halves_after_a_fold_that_did_not_land() -> None:
    history = _session(*(_turn(i) for i in range(16)))
    calm = plan_session_fold(history, SessionMemory(turn_index=16), BUDGET, RC, position=_position, max_source_tokens=6_000)
    stale = plan_session_fold(
        history, SessionMemory(turn_index=16, stale_fold_count=1), BUDGET, RC, position=_position, max_source_tokens=6_000
    )
    assert fold_source_tokens(calm.source_messages, RC) > 3_000
    assert fold_source_tokens(stale.source_messages, RC) <= 3_000 or len(stale.turns) == 1
    assert len(stale.turns) < len(calm.turns)


def test_the_planner_reserves_room_for_the_registry() -> None:
    files = [f"src/module_{i}.py" for i in range(12)]
    ledger = ArtifactLedger(files=files, content={f: "def f():\n    return 1\n" * 60 for f in files})
    history = _session(*(_turn(i) for i in range(16)))
    bare = plan_session_fold(history, SessionMemory(turn_index=16), BUDGET, RC, position=_position)
    with_files = plan_session_fold(history, SessionMemory(ledger=ledger, turn_index=16), BUDGET, RC, position=_position)
    assert with_files.projected_tokens - bare.projected_tokens > 1_000
    roomy = 24_000
    bare = plan_session_fold(history, SessionMemory(turn_index=16), roomy, RC, position=_position)
    with_files = plan_session_fold(history, SessionMemory(ledger=ledger, turn_index=16), roomy, RC, position=_position)
    assert with_files.needed and len(with_files.turns) > len(bare.turns)
