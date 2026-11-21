"""The session seed: raw earlier turns, older tool exchanges masked, summary last.

Pins the layout, the masking (and that it never orphans a tool call or result),
the budget fit, prefix stability across successive runs of one session (on the
messages and on the bytes an OpenAI-compatible endpoint receives), the fold
decision, coverage held by the summary, head protection counted in turns, and
the cost of a call with a session cache.
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Any

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    CompactionSourceRef,
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
    NOTHING_COVERED,
    ArtifactLedger,
    SessionMemory,
    SessionSummary,
    SummaryDelta,
    build_summary_delta_message,
    fold_run,
)
from protocore.runtime.context.session_seed import (
    MaskingRule,
    SeedCache,
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


def _covering(through: int, **entries: tuple[str, ...]) -> SessionMemory:
    summary = SessionSummary(entries=dict(entries) or {"REQUESTS": ("earlier request",)}, covered_through=through)
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
        summary=SessionSummary(entries={"KEY FACTS & IDENTIFIERS": ("I1 is the second item",)}, covered_through=through),
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
    covering = build_session_seed(history, _covering(len(history) - 1 - len(_turn(2))), 60_000, RC, position=_position)
    assert covering.summarized_turns == 1
    reset = SessionMemory(summary=SessionSummary(), turn_index=3)
    seed = build_session_seed(history, reset, 60_000, RC, position=_position)
    assert seed.summarized_turns == 0 and seed.window_turns == 2 and seed.head_turns == 1


def test_a_turn_with_an_unknown_position_is_never_covered() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    history[len(_turn(0)) + 1] = history[len(_turn(0)) + 1].model_copy(update={"metadata": {}})
    seed = build_session_seed(history, _covering(10**6), 60_000, RC, position=_position)
    assert seed.summarized_turns == 0 and seed.window_turns == 2


def test_an_empty_summary_that_covers_turns_still_says_they_existed() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    memory = SessionMemory(summary=SessionSummary(covered_through=len(_turn(0)) * 2 - 1), turn_index=3)
    seed = build_session_seed(history, memory, 60_000, RC, position=_position)
    assert seed.summarized_turns == 1
    assert any("added nothing that needed recording" in m.text for m in seed.messages)


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


def test_under_severe_pressure_everything_is_masked_and_reported_over_budget() -> None:
    history = _session(_turn(0), _turn(1), _turn(2))
    seed = build_session_seed(history, SessionMemory(turn_index=3), 500, RC, position=_position)
    assert seed.full_turns == 0 and seed.trimmed_turns == 0 and seed.masked_turns == 3
    assert seed.over_budget_tokens == seed.tokens - 500 > 0
    assert sum(1 for m in seed.messages if m.text.startswith("question")) == 3


def test_drop_uncovered_leaves_out_oldest_window_turns_and_says_how_many() -> None:
    history = _session(*(_turn(i) for i in range(8)))
    seed = build_session_seed(history, SessionMemory(turn_index=8), 6_000, RC, position=_position, drop_uncovered=True)
    questions = [m.text.split(":")[0] for m in seed.messages if m.text.startswith("question")]
    assert questions[0] == "question 0" and questions[-1] == "question 7"
    assert seed.dropped_turns == 8 - len(questions)
    assert seed.dropped_turns >= 3
    dropped = [f"question {i}" for i in range(1, 1 + seed.dropped_turns)]
    assert not set(dropped) & set(questions)  # the oldest window turns went, in order
    notices = [m.text for m in seed.messages if "not available here" in m.text]
    assert notices == [f"[{seed.dropped_turns} earlier turns of this conversation are not available here and not in the summary.]"]
    assert seed.tokens <= 6_000
    assert seed.tokens == sum(estimate_message_tokens(m, RC) for m in seed.messages)


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
    memory = _covering(len(_turn(0)) * 3 - 1, REQUESTS=("find items",))
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
    assert not plan.needed and plan.messages == [] and plan.covers_through is None
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
    assert plan.covers_through == _position(plan.turns[-1][-1])


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


def test_a_turn_without_positions_is_not_planned() -> None:
    history = _session(*(_turn(i) for i in range(14)))
    at = len(_turn(0)) * 2
    history[at] = history[at].model_copy(update={"metadata": {}})  # first message of turn 2
    plan = plan_session_fold(history, SessionMemory(turn_index=14), BUDGET, RC, position=_position)
    assert len(plan.turns) == 1 and plan.deferred_turns > 0


def test_the_plan_keeps_room_for_the_newest_turn_uncut() -> None:
    # Near the trigger the masked history alone fits, but not with the newest
    # turn in full: the planner folds older turns rather than cut the newest.
    history = _session(*(_turn(i) for i in range(9)))
    memory = SessionMemory(turn_index=9)
    plan = plan_session_fold(history, memory, BUDGET, RC, position=_position)
    assert plan.needed and plan.fresh_tokens > 5000
    memory = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("x" * 400,) * 1}), RC, covers_through=plan.covers_through).memory
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
    carried = sum(estimate_message_tokens(m, rc) for m in _summary_message(memory, 1))
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
        memory = fold_run(memory, turn, delta, rc, covers_through=plan.covers_through).memory
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


def test_a_host_identity_keys_the_cache() -> None:
    history = _session(_turn(0), _turn(1))
    cache = SeedCache(identity=lambda m: str(m.metadata.get(POS)))
    build_session_seed(history, SessionMemory(turn_index=2), BUDGET, RC, position=_position, cache=cache)
    assert len(cache) == 2
    small = SeedCache(max_entries=1)
    build_session_seed(history, SessionMemory(turn_index=2), BUDGET, RC, position=_position, cache=small)
    assert len(small) == 1


def test_the_default_rule_has_a_stable_cache_key() -> None:
    assert default_masking_rule(RC).key == default_masking_rule(RC).key != ""
    other = LoopConstants(session_seed_mask_identifier_lines=2)
    assert default_masking_rule(other).key != default_masking_rule(RC).key


def test_nothing_covered_is_the_default() -> None:
    assert SessionSummary().covered_through == NOTHING_COVERED
