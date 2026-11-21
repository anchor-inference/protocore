"""Unit tests for the pure session memory: the delta summary and the registry.

Covers the artifact registry (files and content read from tool-call arguments
by role), the delta fold — its prompt, the parsing of a reply, the append
merge — the cap that sheds whole entries oldest first instead of cutting the
end, and the storage round trip.
"""
from __future__ import annotations

import json

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.session_memory import (
    PENDING_NOTHING_OPEN,
    SUMMARY_DELTA_SYSTEM,
    SUMMARY_FORMAT_VERSION,
    SUMMARY_SECTIONS,
    ArtifactLedger,
    Coverage,
    SessionMemory,
    SessionSummary,
    SummaryDelta,
    SummaryDeltaError,
    build_summary_delta_message,
    build_summary_delta_request,
    cap_summary,
    extract_artifacts,
    fold_run,
    is_stored_summary_text,
    merge_summary_delta,
    parse_summary_delta,
    render_ledger,
)
from protocore.runtime.token_counting import estimate_tokens
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

RC = LoopConstants()


# --- helpers ---------------------------------------------------------------


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(text: str) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


def _write_call(path: str, content: str, *, name: str = "Write", call_id: str = "tc1") -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id=call_id,
                name=name,
                arguments_json=json.dumps({"path": path, "content": content}),
            )
        ],
    )


def _tool_result(text: str, *, call_id: str = "tc1") -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[ToolResultBlock(tool_call_id=call_id, content=text)],
    )


# --- structured artifact registry (files + content by key) -----------------


def test_extract_file_path_from_write_tool_call() -> None:
    led = extract_artifacts([_write_call("src/app.py", "def main():\n    pass\n")], roles=CONVENTIONAL_TOOL_ROLES)
    assert led.files == ["src/app.py"]
    assert led.content["src/app.py"] == "def main():\n    pass\n"


def test_extract_is_language_agnostic_cyrillic_path() -> None:
    # A cyrillic path is read verbatim from the structured ``path`` arg — no text
    # scanning, no language assumption.
    content = "Документация процедуры отката. Версия 3.12."
    led = extract_artifacts([_write_call("docs/откат.md", content)], roles=CONVENTIONAL_TOOL_ROLES)
    assert led.files == ["docs/откат.md"]
    assert led.content["docs/откат.md"] == content


def test_extract_edit_tool_file_path_key() -> None:
    edit = Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id="e1",
                name="Edit",
                arguments_json=json.dumps(
                    {"file_path": "main.py", "new_string": "def fixed():\n    pass"}
                ),
            )
        ],
    )
    led = extract_artifacts([edit], roles=CONVENTIONAL_TOOL_ROLES)
    assert led.files == ["main.py"]
    assert led.content["main.py"] == "def fixed():\n    pass"


def test_extract_ignores_non_file_tools() -> None:
    # A non-file tool (e.g. Bash) with a path-looking arg is NOT recorded: we key
    # on the structured tool NAME, not text shape.
    bash = Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id="b1", name="Bash",
                arguments_json=json.dumps({"command": "cat src/app.py"}),
            )
        ],
    )
    led = extract_artifacts([bash], roles=CONVENTIONAL_TOOL_ROLES)
    assert led.files == []


def test_extract_dedup_keeps_latest_content() -> None:
    base = ArtifactLedger(files=["a.py"], content={"a.py": "old"})
    led = extract_artifacts([_write_call("a.py", "new")], base=base, roles=CONVENTIONAL_TOOL_ROLES)
    assert led.files == ["a.py"]  # not duplicated, order preserved
    assert led.content["a.py"] == "new"  # latest write wins


def test_extract_no_files_from_plain_prose() -> None:
    # No tool call → nothing in the registry (text is never scanned for facts).
    led = extract_artifacts([_user("Please write notes about https://x.com v3.12 `BUILD::`")], roles=CONVENTIONAL_TOOL_ROLES)
    assert led.files == []
    assert led.content == {}


def test_extract_content_snapshot_capped() -> None:
    huge = "x" * 100_000
    led = extract_artifacts([_write_call("big.txt", huge)], roles=CONVENTIONAL_TOOL_ROLES)
    assert len(led.content["big.txt"]) <= 4000


# --- the delta prompt -------------------------------------------------------


def test_delta_prompt_names_every_section_heading() -> None:
    for section in SUMMARY_SECTIONS:
        assert f"  {section.heading}:\n" in SUMMARY_DELTA_SYSTEM


def test_delta_prompt_asks_only_for_what_is_new_and_is_multilingual() -> None:
    assert "ONLY what the NEW TURNS add" in SUMMARY_DELTA_SYSTEM
    assert "Never repeat" in SUMMARY_DELTA_SYSTEM
    assert "VERBATIM" in SUMMARY_DELTA_SYSTEM
    assert "Русский" in SUMMARY_DELTA_SYSTEM


def test_delta_message_carries_the_existing_summary_and_the_turns() -> None:
    summary = SessionSummary(entries={"REQUESTS": ("find items on topic 1",)})
    text = build_summary_delta_message(summary, [_user("new question"), _tool_result("result body")])
    assert text is not None
    assert "EXISTING SUMMARY (already recorded — do not repeat):\nREQUESTS:\n- find items on topic 1" in text
    assert "NEW TURNS" in text and "new question" in text and "[tool result: result body]" in text


def test_delta_message_with_no_summary_says_empty() -> None:
    text = build_summary_delta_message(SessionSummary(), [_user("task")])
    assert text is not None and "(empty)" in text


def test_delta_request_is_bounded_by_what_the_summary_can_store() -> None:
    rc = LoopConstants(session_memory_fold_max_output_tokens=700)
    request = build_summary_delta_request(SessionSummary(), [_user("task")], rc, model="m", thinking=False, temperature=0.1)
    assert request is not None
    assert request.max_tokens == 700 and request.temperature == 0.1 and not request.tools
    assert request.extra == {"enable_thinking": False}
    assert request.messages[0].role is MessageRole.system
    assert request.messages[0].text == SUMMARY_DELTA_SYSTEM
    default = build_summary_delta_request(SessionSummary(), [_user("task")], rc, model="m", thinking=None)
    assert default is not None and "enable_thinking" not in default.extra
    assert build_summary_delta_request(SessionSummary(), [], rc, model="m", thinking=False) is None


def test_delta_message_for_nothing_is_none() -> None:
    assert build_summary_delta_message(SessionSummary(), []) is None


def test_delta_message_renders_what_it_is_handed_without_cutting() -> None:
    # The fold is handed the masked turns (already bounded, kept lines
    # included); cutting them again here is what lost deep citation lines.
    tail = "Cite: deep line 9"
    text = build_summary_delta_message(SessionSummary(), [_tool_result("y" * 3000 + "\n" + tail)])
    assert text is not None and tail in text


# --- parsing a reply -------------------------------------------------------------


def test_parse_reads_entries_under_headings_in_any_order() -> None:
    reply = (
        "KEY FACTS & IDENTIFIERS:\n- The second item is I1 (code 12-3456-7)\n"
        "REQUESTS:\n- find items on topic 1\n\n"
        "PENDING / NEXT:\n- compare editions\n"
    )
    delta = parse_summary_delta(reply)
    assert delta.entries["KEY FACTS & IDENTIFIERS"] == ("The second item is I1 (code 12-3456-7)",)
    assert delta.entries["REQUESTS"] == ("find items on topic 1",)
    assert delta.entries["PENDING / NEXT"] == ("compare editions",)


def test_parse_tolerates_presentation() -> None:
    reply = (
        "## 1. **Constraints and Decisions**\n* answer in Russian\n"
        "**key facts & identifiers:** port 8080\n"
        "3) Pending/Next:\n1. check the index\n"
        "> ARTIFACTS\n\u2013 notes.md: draft\n"
    )
    delta = parse_summary_delta(reply)
    assert delta.entries["CONSTRAINTS & DECISIONS"] == ("answer in Russian",)
    assert delta.entries["KEY FACTS & IDENTIFIERS"] == ("port 8080",)
    assert delta.entries["PENDING / NEXT"] == ("check the index",)
    assert delta.entries["ARTIFACTS"] == ("notes.md: draft",)


def test_parse_drops_punctuation_only_entries_and_counts_stray_lines() -> None:
    delta = parse_summary_delta("preamble line\nREQUESTS:\n- —\n- real one\n")
    assert delta.entries == {"REQUESTS": ("real one",)}
    assert delta.unattributed_lines == 1


def test_an_empty_or_all_reasoning_reply_is_not_a_fold() -> None:
    for reply in ("", "   \n", "<think>the bound ran out while thinking", "<think>done</think>\n  "):
        with pytest.raises(SummaryDeltaError):
            parse_summary_delta(reply)
        with pytest.raises(SummaryDeltaError):
            parse_summary_delta(reply, truncated=True)


def test_nothing_new_is_a_valid_empty_delta() -> None:
    delta = parse_summary_delta("NOTHING NEW")
    assert delta.is_empty() and delta.nothing_new
    assert parse_summary_delta("<think>checked</think>\nNothing new.").nothing_new
    assert parse_summary_delta("REQUESTS:\n\nARTIFACTS:\n").is_empty()


def test_a_truncated_reply_without_a_complete_entry_is_not_a_fold() -> None:
    with pytest.raises(SummaryDeltaError):
        parse_summary_delta("REQUESTS:\n- find th", truncated=True)
    with pytest.raises(SummaryDeltaError):
        parse_summary_delta("NOTHING NEW", truncated=True)


def test_reasoning_before_the_reply_is_not_an_entry() -> None:
    delta = parse_summary_delta("<think>REQUESTS:\n- not a real one</think>\nREQUESTS:\n- real one")
    assert delta.entries == {"REQUESTS": ("real one",)}


def test_placeholder_entries_are_dropped() -> None:
    delta = parse_summary_delta("PENDING / NEXT:\n- none\n- None.\n- N/A\n- нет\nREQUESTS:\n- keep this")
    assert delta.entries == {"REQUESTS": ("keep this",)}


def test_a_bulleted_entry_naming_a_section_stays_in_its_section() -> None:
    reply = "KEY FACTS & IDENTIFIERS:\n- Requests: limited to 3 per minute\n- ID-1234\n- db-7.internal:5432"
    delta = parse_summary_delta(reply)
    assert delta.entries == {
        "KEY FACTS & IDENTIFIERS": ("Requests: limited to 3 per minute", "ID-1234", "db-7.internal:5432")
    }


def test_an_unbulleted_entry_naming_a_section_stays_in_its_section() -> None:
    reply = "KEY FACTS & IDENTIFIERS:\nRequests: limited to 3 per minute\nID-1234\nREQUESTS: find the report"
    delta = parse_summary_delta(reply)
    assert delta.entries["KEY FACTS & IDENTIFIERS"] == ("Requests: limited to 3 per minute", "ID-1234")
    assert delta.entries["REQUESTS"] == ("find the report",)


def test_parse_refuses_a_reply_with_no_heading() -> None:
    with pytest.raises(SummaryDeltaError):
        parse_summary_delta("Here is a summary of the conversation in prose.")


def test_parse_truncated_reply_drops_only_its_last_entry() -> None:
    delta = parse_summary_delta("REQUESTS:\n- one\n- two\nKEY FACTS & IDENTIFIERS:\n- cut in the mid", truncated=True)
    assert delta.entries == {"REQUESTS": ("one", "two")}


def test_parse_clips_a_huge_entry() -> None:
    delta = parse_summary_delta("KEY FACTS & IDENTIFIERS:\n- " + "z" * 5000)
    assert len(delta.entries["KEY FACTS & IDENTIFIERS"][0]) <= 1000


def test_summary_text_round_trip_is_exact() -> None:
    summary = SessionSummary(
        entries={"REQUESTS": ("a",), "KEY FACTS & IDENTIFIERS": ("b", "PENDING / NEXT: was discussed")},
        shed={"ARTIFACTS": 4, "REQUESTS": 2},
        coverage=Coverage(5, 17),
    )
    text = summary.to_text()
    assert text.startswith(f"[session summary format {SUMMARY_FORMAT_VERSION}; covers 5-17]\n")
    assert SessionSummary.from_text(text) == summary
    assert is_stored_summary_text(text)


def test_an_empty_summary_that_covers_turns_is_still_self_describing() -> None:
    summary = SessionSummary(coverage=Coverage(3, 9))
    text = summary.to_text()
    assert text == f"[session summary format {SUMMARY_FORMAT_VERSION}; covers 3-9]"
    assert SessionSummary.from_text(text) == summary
    fresh = SessionSummary().to_text()
    assert SessionSummary.from_text(fresh) == SessionSummary() and is_stored_summary_text(fresh)


def test_blank_old_or_foreign_text_loads_as_a_fresh_summary() -> None:
    for text in (
        "",
        "   ",
        "ORIGINAL ASK:\nfind the items\nCONSTRAINTS & DECISIONS:\nanswer briefly\n",
        "free prose",
        "[session summary format 999; covers 0-40]",
        f"[session summary format {SUMMARY_FORMAT_VERSION}; covers 9-3]",
        f"[session summary format {SUMMARY_FORMAT_VERSION}; covers 0-4]\nnot a line of this format",
    ):
        assert SessionSummary.from_text(text) == SessionSummary(), text
        assert not is_stored_summary_text(text)


def test_a_memory_dict_without_the_format_marker_loads_covering_nothing() -> None:
    legacy = {"running_summary": "ORIGINAL ASK: x", "ledger": {"files": ["a.md"]}, "turn_index": 4, "cumulative_raw_tokens": 9}
    memory = SessionMemory.from_dict(legacy)
    assert memory.summary == SessionSummary() and memory.summary.coverage is None
    assert memory.ledger.files == ["a.md"] and memory.turn_index == 4
    unmarked = {"summary": {"entries": {"REQUESTS": ["r"]}, "coverage": [0, 30]}}
    assert SessionMemory.from_dict(unmarked).summary == SessionSummary()
    marked = SessionMemory(summary=SessionSummary(entries={"REQUESTS": ("r",)}, coverage=Coverage(0, 30)))
    assert SessionMemory.from_dict(json.loads(json.dumps(marked.to_dict()))).summary == marked.summary


def test_coverage_is_a_valid_range() -> None:
    with pytest.raises(ValueError):
        Coverage(5, 4)
    with pytest.raises(ValueError):
        Coverage(-1, 4)
    assert Coverage(2, 5).union(Coverage(6, 9)) == Coverage(2, 9)
    assert Coverage(0, 9).includes(Coverage(2, 5)) and not Coverage(2, 5).includes(Coverage(0, 9))


def test_parse_reads_its_own_rendering_back() -> None:
    summary = SessionSummary(
        entries={"REQUESTS": ("a",), "CONSTRAINTS & DECISIONS": ("b", "c"), "ARTIFACTS": ("d",)}
    )
    assert parse_summary_delta(summary.render()).entries == summary.entries


# --- merging and the cap ----------------------------------------------------------


def test_merge_appends_and_skips_duplicates() -> None:
    summary = SessionSummary(entries={"KEY FACTS & IDENTIFIERS": ("I1 is the second item",)})
    delta = SummaryDelta(entries={"KEY FACTS & IDENTIFIERS": ("i1  is the SECOND item", "I2 is the third")})
    merged, report = merge_summary_delta(summary, delta, RC)
    assert merged.section("KEY FACTS & IDENTIFIERS") == ("I1 is the second item", "I2 is the third")
    assert (report.added, report.duplicates) == (1, 1)


def test_merge_replaces_pending_only_when_the_delta_writes_it() -> None:
    summary = SessionSummary(entries={"PENDING / NEXT": ("old step",), "REQUESTS": ("r",)})
    kept, _ = merge_summary_delta(summary, SummaryDelta(entries={"REQUESTS": ("r2",)}), RC)
    assert kept.section("PENDING / NEXT") == ("old step",)
    replaced, report = merge_summary_delta(summary, SummaryDelta(entries={"PENDING / NEXT": ("new step",)}), RC)
    assert replaced.section("PENDING / NEXT") == ("new step",)
    assert report.replaced_sections == ("PENDING / NEXT",)


def _big_summary() -> SessionSummary:
    return SessionSummary(
        entries={
            "REQUESTS": tuple(f"request {i} " + "r" * 120 for i in range(6)),
            "CONSTRAINTS & DECISIONS": tuple(f"constraint {i} " + "c" * 120 for i in range(6)),
            "KEY FACTS & IDENTIFIERS": tuple(f"fact {i} " + "f" * 120 for i in range(6)),
            "PENDING / NEXT": ("next step",),
            "ARTIFACTS": tuple(f"file{i}.md: notes " + "a" * 120 for i in range(6)),
        }
    )


def test_cap_sheds_whole_entries_oldest_first_and_keeps_every_section() -> None:
    rc = LoopConstants(session_memory_running_summary_token_cap=600, session_memory_fold_max_output_tokens=600)
    capped, shed, clipped = cap_summary(_big_summary(), rc)
    assert estimate_tokens(capped.render(), rc) <= 600
    assert shed > 0 and clipped == 0
    rendered = capped.render()
    # Nothing is cut from the end: the trailing sections are still there.
    assert "PENDING / NEXT:\n- next step" in rendered
    # ARTIFACTS give way first, then later REQUESTS; the original ask stays.
    assert capped.section("REQUESTS")[0].startswith("request 0 ")
    assert capped.shed.get("ARTIFACTS") == 6
    assert "earlier entries of this section are no longer carried" in rendered
    # Within a section the oldest go first.
    facts = capped.section("KEY FACTS & IDENTIFIERS")
    if facts and len(facts) < 6:
        assert facts[-1].startswith("fact 5 ")


def test_cap_order_is_artifacts_requests_facts_constraints_pending() -> None:
    rc = LoopConstants(session_memory_running_summary_token_cap=200, session_memory_fold_max_output_tokens=200)
    capped, _, _ = cap_summary(_big_summary(), rc)
    # Facts went whole before constraints were touched; the original request
    # and the pending list outlive both.
    assert capped.shed["ARTIFACTS"] == 6 and capped.shed["KEY FACTS & IDENTIFIERS"] == 6
    assert 0 < len(capped.section("CONSTRAINTS & DECISIONS")) < 6
    assert capped.section("REQUESTS") == (_big_summary().section("REQUESTS")[0],)
    assert capped.section("PENDING / NEXT") == ("next step",)
    rc = LoopConstants(session_memory_running_summary_token_cap=60, session_memory_fold_max_output_tokens=60)
    last, _, _ = cap_summary(_big_summary(), rc)
    assert set(last.entries) == {"REQUESTS"}  # the original ask is the last to go


def test_cap_clips_a_single_oversized_entry_instead_of_dropping_it() -> None:
    rc = LoopConstants(session_memory_running_summary_token_cap=50, session_memory_fold_max_output_tokens=50)
    summary = SessionSummary(entries={"REQUESTS": ("x " * 2000,)})
    capped, shed, clipped = cap_summary(summary, rc)
    assert shed == 0 and clipped == 1
    assert capped.section("REQUESTS") and len(capped.section("REQUESTS")[0]) < 4000


def test_cap_off_keeps_everything() -> None:
    rc = LoopConstants(session_memory_running_summary_token_cap=0)
    assert cap_summary(_big_summary(), rc)[0] == _big_summary()


def test_fold_output_bound_cannot_exceed_what_can_be_stored() -> None:
    with pytest.raises(ValueError, match="session_memory_fold_max_output_tokens"):
        LoopConstants(session_memory_running_summary_token_cap=500, session_memory_fold_max_output_tokens=800)
    LoopConstants(session_memory_running_summary_token_cap=0, session_memory_fold_max_output_tokens=8000)


def test_fold_target_must_be_below_trigger() -> None:
    with pytest.raises(ValueError, match="session_seed_fold_target_ratio"):
        LoopConstants(session_seed_fold_trigger_ratio=0.5, session_seed_fold_target_ratio=0.5)


# --- fold_run -----------------------------------------------------------------------


def test_fold_run_without_a_delta_updates_only_the_registry_and_turn() -> None:
    memory = SessionMemory(summary=SessionSummary(entries={"REQUESTS": ("r",)}), turn_index=2, stale_fold_count=1)
    result = fold_run(memory, [_write_call("a.py", "x")], None, RC, roles=CONVENTIONAL_TOOL_ROLES)
    assert not result.summary_updated
    assert result.memory.summary == memory.summary
    assert result.memory.ledger.files == ["a.py"]
    assert result.memory.turn_index == 3
    assert result.memory.stale_fold_count == 1  # the host owns this counter


def test_fold_run_appends_a_delta_and_advances_coverage() -> None:
    memory = SessionMemory(summary=SessionSummary(entries={"REQUESTS": ("r",)}, coverage=Coverage(2, 3)), turn_index=2)
    result = fold_run(memory, [], SummaryDelta(entries={"REQUESTS": ("r2",)}), RC, coverage=Coverage(4, 9))
    assert result.summary_updated
    assert result.memory.summary.section("REQUESTS") == ("r", "r2")
    assert result.memory.summary.coverage == Coverage(2, 9)
    assert result.report.added == 1
    assert result.notes["entries_added"] == 1


def test_fold_run_needs_the_coverage_a_delta_records() -> None:
    with pytest.raises(ValueError, match="coverage"):
        fold_run(SessionMemory(), [], SummaryDelta(entries={"REQUESTS": ("r",)}), RC)


def test_applying_the_same_delta_twice_changes_nothing_the_second_time() -> None:
    rc = LoopConstants(session_memory_running_summary_token_cap=200, session_memory_fold_max_output_tokens=200)
    delta = SummaryDelta(entries={"ARTIFACTS": tuple(f"file{i}.md: " + "a" * 100 for i in range(6))})
    once = fold_run(SessionMemory(), [], delta, rc, coverage=Coverage(0, 10)).memory
    twice = fold_run(once, [], delta, rc, coverage=Coverage(0, 10))
    assert twice.report.already_applied
    assert twice.memory.summary == once.summary
    assert twice.memory.turn_index == once.turn_index + 1


_STORED_PENDING = ("send the report", "fix the build")


def _pending_after(reply: str, *, truncated: bool = False) -> tuple[tuple[str, ...], object]:
    stored = SessionSummary(entries={"PENDING / NEXT": _STORED_PENDING})
    merged, report = merge_summary_delta(stored, parse_summary_delta(reply, truncated=truncated), RC)
    return merged.section("PENDING / NEXT"), report


def test_pending_can_be_emptied() -> None:
    for reply in (
        f"PENDING / NEXT:\n- {PENDING_NOTHING_OPEN}",
        "PENDING / NEXT:\n- none",
        "PENDING / NEXT:\n- nothing",
        "PENDING / NEXT: none",
        "PENDING / NEXT:\n- Nothing open.",
    ):
        pending, report = _pending_after(reply)
        assert pending == (), reply
        assert report.cleared_sections == ("PENDING / NEXT",)  # type: ignore[attr-defined]


def test_a_bare_pending_heading_leaves_the_list_as_stored() -> None:
    assert _pending_after("PENDING / NEXT:\nREQUESTS:\n- something new")[0] == _STORED_PENDING


def test_closed_items_leave_the_stored_list() -> None:
    pending, report = _pending_after("PENDING / NEXT:\n- DONE: send the report")
    assert pending == ("fix the build",)
    assert report.closed_items == 1  # type: ignore[attr-defined]


def test_a_new_complete_list_replaces_the_stored_one() -> None:
    assert _pending_after("PENDING / NEXT:\n- fix the build\n- call the supplier")[0] == ("fix the build", "call the supplier")


def test_the_prompt_says_how_to_empty_and_close_pending() -> None:
    assert f"- {PENDING_NOTHING_OPEN}" in SUMMARY_DELTA_SYSTEM
    assert "DONE:" in SUMMARY_DELTA_SYSTEM


def test_a_truncated_reply_adds_new_pending_items_and_removes_none() -> None:
    reply = "PENDING / NEXT:\n- send the report\n- send the invoice to the supplier\n- book the ro"
    pending, report = _pending_after(reply, truncated=True)
    assert pending == ("send the report", "fix the build", "send the invoice to the supplier")
    assert report.kept_on_truncation == ("PENDING / NEXT",)  # type: ignore[attr-defined]
    for reply in ("PENDING / NEXT:\n- none\nREQUESTS:\n- a\n- b", "PENDING / NEXT:\n- DONE: fix the build\nREQUESTS:\n- a\n- b"):
        assert _pending_after(reply, truncated=True)[0] == _STORED_PENDING


def test_a_truncated_fold_keeps_its_entries_but_does_not_advance_coverage() -> None:
    memory = SessionMemory(summary=SessionSummary(entries={"REQUESTS": ("r",)}, coverage=Coverage(0, 4)), stale_fold_count=2)
    delta = parse_summary_delta("REQUESTS:\n- open request from the user\n- cut mid-sen", truncated=True)
    result = fold_run(memory, [], delta, RC, coverage=Coverage(5, 20), fold_attempted=True)
    assert result.memory.summary.section("REQUESTS") == ("r", "open request from the user")
    assert result.memory.summary.coverage == Coverage(0, 4)
    assert result.report.coverage_held
    assert result.memory.stale_fold_count == 3


def test_fold_run_keeps_the_stale_fold_count() -> None:
    memory = SessionMemory(stale_fold_count=2)
    failed = fold_run(memory, [], None, RC, fold_attempted=True).memory
    assert failed.stale_fold_count == 3
    landed = fold_run(failed, [], SummaryDelta(entries={"REQUESTS": ("r",)}), RC, coverage=Coverage(0, 3), fold_attempted=True).memory
    assert landed.stale_fold_count == 0
    assert fold_run(failed, [], None, RC).memory.stale_fold_count == 3  # no fold attempted


def test_fold_run_with_nothing_new_advances_coverage_and_says_so() -> None:
    memory = SessionMemory(summary=SessionSummary(entries={"REQUESTS": ("r",)}), turn_index=2)
    result = fold_run(memory, [], parse_summary_delta("NOTHING NEW"), RC, coverage=Coverage(0, 5))
    assert result.memory.summary.entries == memory.summary.entries
    assert result.memory.summary.coverage == Coverage(0, 5)
    assert result.report.nothing_new and result.notes["nothing_new"]


def test_fold_run_does_not_mutate_its_input() -> None:
    memory = SessionMemory(turn_index=1)
    fold_run(memory, [_write_call("a.py", "x")], SummaryDelta(entries={"REQUESTS": ("r",)}), RC, roles=CONVENTIONAL_TOOL_ROLES, coverage=Coverage(0, 1))
    assert memory.summary.is_empty() and memory.ledger.is_empty() and memory.turn_index == 1


# --- registry rendering and storage ---------------------------------------------------


def test_render_ledger_verbatim() -> None:
    led = ArtifactLedger(files=["a.py", "b.md"], content={"a.py": "code", "b.md": "# Title"})
    rendered = render_ledger(led)
    assert "a.py" in rendered and "b.md" in rendered
    assert "code" in rendered
    assert "# Title" in rendered


def test_render_empty_ledger_is_blank() -> None:
    assert render_ledger(ArtifactLedger()) == ""


def test_session_memory_round_trip() -> None:
    mem = SessionMemory(
        summary=SessionSummary(entries={"REQUESTS": ("s",)}, shed={"ARTIFACTS": 2}),
        ledger=ArtifactLedger(files=["x.py"], content={"x.py": "def f(): ..."}),
        turn_index=3,
        stale_fold_count=3,
    )
    restored = SessionMemory.from_dict(json.loads(json.dumps(mem.to_dict())))
    assert restored.summary == mem.summary
    assert restored.ledger.files == ["x.py"]
    assert restored.ledger.content == {"x.py": "def f(): ..."}
    assert restored.turn_index == 3
    assert restored.stale_fold_count == 3


def test_session_memory_from_empty_dict() -> None:
    assert SessionMemory.from_dict(None).is_empty()
    assert SessionMemory.from_dict({}).is_empty()


def test_summary_from_dict_ignores_unknown_sections() -> None:
    restored = SessionSummary.from_dict(
        {"format": SUMMARY_FORMAT_VERSION, "entries": {"NOT A SECTION": ["x"], "REQUESTS": ["r"]}, "shed": {"X": 1}}
    )
    assert restored.entries == {"REQUESTS": ("r",)} and restored.shed == {}
    assert SessionSummary.from_dict(None).is_empty()


def test_artifact_ledger_round_trip_via_json() -> None:
    led = ArtifactLedger(files=["b"], content={"b": "c"})
    raw = json.dumps(led.to_dict())
    restored = ArtifactLedger.from_dict(json.loads(raw))
    assert restored.to_dict() == led.to_dict()
