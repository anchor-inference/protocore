
"""What a session remembers across runs: a running summary and an artifact registry.

PURE core module. It never imports the host, a store or a model client (the
import-boundary guard ``tests/test_core_import_boundary.py`` enforces this). The
host loads and stores :class:`SessionMemory`, makes the summariser call, and
hands the reply back; this module only computes.

A session's next run is seeded from its RAW earlier turns
(:mod:`protocore.runtime.context.session_seed`), with the tool results of the
older ones masked. The memory here stands in only for what that raw window no
longer carries:

1. **The running summary** (:class:`SessionSummary`) — entries under five fixed
   sections, recording the turns that have left the raw window, and
   :attr:`SessionSummary.covered_through`, the session position of the last
   message those turns hold. Coverage lives on the summary on purpose: a
   summary that is reset (``SessionSummary()``) covers nothing, so no turn can
   be left out of a seed on the strength of a summary that no longer records
   it. The summary grows by *delta folds*: the summariser is shown the
   existing summary and the turns leaving the window (in the masked form the
   seed carried them in), and asked only for what those turns add
   (:data:`SUMMARY_DELTA_SYSTEM`, :func:`build_summary_delta_request`). The
   reply is parsed into entries (:func:`parse_summary_delta`) and appended by
   :func:`fold_run`. The summary is never rewritten whole, so a fold costs the
   decode of a few hundred tokens, and nothing already recorded can be lost to
   a rewrite that forgot it.
2. **The artifact registry** (:class:`ArtifactLedger`) — ``path`` -> last
   verbatim ``content`` of every file a workspace-mutating tool wrote, read by
   role from the tool-call arguments of the run as it happened (never from a
   masked seed). Updated after every run, with or without a fold.

The cap. Past ``session_memory_running_summary_token_cap`` the summary sheds
whole entries, oldest first, from the sections in :data:`SUMMARY_SECTIONS`
order of ``shed_rank`` — ARTIFACTS (the registry keeps the files) before
later REQUESTS before KEY FACTS & IDENTIFIERS before CONSTRAINTS & DECISIONS,
PENDING / NEXT last — and every section that lost entries says how many. The
first REQUESTS entry, the user's original ask, is the last thing to go.

Concurrency contract. One fold per session at a time. The host loads the
memory once, plans the fold against that memory and the stored history
(:func:`~protocore.runtime.context.session_seed.plan_session_fold`), builds the
request from the same memory, and calls :func:`fold_run` on that same memory
before saving it with a compare-and-set on its own revision. A delta carries
the coverage it advances to, so applying it to a memory that already covers
that far changes nothing: a retry after a lost acknowledgement cannot apply
the same delta twice. A save that loses the compare-and-set reloads and plans
again.

Parsing a reply is reading the format the prompt asked for — headings this
module named and one entry per line — not mining text for meaning. There is no
regular expression here; semantic extraction is the summariser's job.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from protocore.contracts.llm import LLMRequest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    WORKSPACE_MUTATION_ROLES,
    ToolArgumentSlot,
    ToolRoleMap,
)
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.tool_arguments import string_argument

# ---------------------------------------------------------------------------
# Render-shape constants. They shape strings, not behaviour an operator tunes
# per tenant, so they are module constants; every budget is a LoopConstants
# field.
# ---------------------------------------------------------------------------

NOTHING_COVERED: Final[int] = -1
""":attr:`SessionSummary.covered_through` of a summary that covers no turn.
Session positions are the host's and must be non-negative and increase in
session order."""

#: Cap on how many files the registry pins, and on each content snapshot.
_MAX_FILES: Final[int] = 60
_CONTENT_SNAPSHOT_MAX_CHARS: Final[int] = 4000

#: Longest single summary entry kept. An entry is one fact; a reply that put a
#: page on one line is clipped rather than allowed to crowd out every other.
_ENTRY_MAX_CHARS: Final[int] = 1000

#: Shortest an entry is clipped to when shedding whole entries could not bring
#: the summary under its cap (one entry larger than the cap by itself).
_ENTRY_MIN_CLIP_CHARS: Final[int] = 80

_ELLIPSIS: Final[str] = " …"

#: The words of the line a section ends with when the cap shed some of its
#: entries: ``  (N earlier entries of this section are no longer carried)``.
_SHED_NOTE: Final[str] = "earlier entries of this section are no longer carried"

#: What the summariser answers when the turns add nothing (and what makes that
#: answer different from an empty or cut-off reply).
NOTHING_NEW: Final[str] = "NOTHING NEW"

#: Characters a heading line may be decorated with by a model (markdown marks,
#: quote marks) and that are stripped before it is compared.
_HEADING_DECORATION: Final[str] = "#*_`> \t"

#: Bullet marks an entry line starts with. A bulleted line is always an entry,
#: even when its text begins with a section's name.
_ENTRY_BULLETS: Final[tuple[str, ...]] = ("- ", "* ", "\u2022 ", "\u2013 ", "\u2014 ", "+ ")

#: Entries that say "there is nothing here" in the words a model uses for an
#: empty section. Compared case-folded, with punctuation and spaces removed.
_NULL_ENTRIES: Final[frozenset[str]] = frozenset(
    {"none", "na", "nil", "nothing", "empty", "nothingnew", "нет", "ничего", "нетданных"}
)

_THINK_OPEN: Final[str] = "<think>"
_THINK_CLOSE: Final[str] = "</think>"


class SectionMerge(StrEnum):
    """How a section takes a fold's new entries."""

    accumulate = "accumulate"
    """New entries are added after the stored ones (duplicates skipped)."""
    supersede = "supersede"
    """A fold that writes the section replaces its entries; one that leaves it
    out keeps them. For state that is current, not cumulative. A reply cut at
    its output bound never supersedes: its list may be the cut one."""


@dataclass(frozen=True, slots=True)
class SummarySection:
    """One fixed section of the running summary."""

    heading: str
    merge: SectionMerge
    shed_rank: int
    """Lower sheds first when the summary is over its cap."""
    keep_first: bool = False
    """The section's first entry is shed only after every other entry of every
    section is gone (the user's original request)."""


SUMMARY_SECTIONS: Final[tuple[SummarySection, ...]] = (
    SummarySection("REQUESTS", SectionMerge.accumulate, shed_rank=1, keep_first=True),
    SummarySection("CONSTRAINTS & DECISIONS", SectionMerge.accumulate, shed_rank=3),
    SummarySection("KEY FACTS & IDENTIFIERS", SectionMerge.accumulate, shed_rank=2),
    SummarySection("PENDING / NEXT", SectionMerge.supersede, shed_rank=4),
    SummarySection("ARTIFACTS", SectionMerge.accumulate, shed_rank=0),
)
"""The sections, in render order. The order a reply writes them in does not
matter; the order they are shown in does, and is fixed."""

_SECTION_BY_HEADING: Final[Mapping[str, SummarySection]] = {s.heading: s for s in SUMMARY_SECTIONS}

SUMMARY_DELTA_SYSTEM: Final[str] = (
    "You keep the running summary of a conversation between a user and an "
    "agent that spans many runs. The agent's context carries the recent turns "
    "verbatim; older turns leave it and survive only in this summary.\n"
    "You receive the EXISTING SUMMARY (what is already recorded; it may be "
    "empty) and NEW TURNS (the turns leaving the agent's context now). Write "
    "ONLY what the NEW TURNS add: entries that are not already in the EXISTING "
    "SUMMARY. Never repeat, rephrase, reorder or correct an existing entry.\n"
    "Write each entry on its own line starting with \"- \", under one of these "
    "headings, each heading at most once, in any order:\n"
    "  REQUESTS:\n"
    "  CONSTRAINTS & DECISIONS:\n"
    "  KEY FACTS & IDENTIFIERS:\n"
    "  PENDING / NEXT:\n"
    "  ARTIFACTS:\n"
    "Leave out a heading you have nothing new for. Do not write placeholder "
    "entries such as \"none\". If the NEW TURNS add nothing at all, reply with "
    f"exactly: {NOTHING_NEW}\n"
    "RULES:\n"
    "- REQUESTS: what the user asked for in the NEW TURNS, one entry per "
    "request, in the user's own words where they are short.\n"
    "- CONSTRAINTS & DECISIONS: every rule, limit, preference or decision the "
    "user or the agent stated.\n"
    "- KEY FACTS & IDENTIFIERS: facts the agent found or stated that a later "
    "answer may need — names, titles, numbers, dates, ids, codes, URLs, paths, "
    "versions, citation lines, and which item was which in any list the agent "
    "gave — copied VERBATIM, exact characters.\n"
    "- PENDING / NEXT: the complete list of what is still open after the NEW "
    "TURNS. It replaces the previous list, so write it whole, or leave the "
    "heading out if nothing changed.\n"
    "- ARTIFACTS: files created or edited, one line each: the path and what it "
    "is. Not their contents; those are kept elsewhere.\n"
    "- Parts of tool output marked as not carried are gone; record only what "
    "is shown.\n"
    "- One fact per entry, self-contained: it must make sense without the "
    "transcript.\n"
    "- The conversation may be in Russian, English or mixed (Русский, English "
    "или смешанный). Copy names, codes, paths and numbers exactly in their "
    "original script; do not translate them.\n"
    "- Output only headings and entries. Do not address the user."
)
"""The system prompt of a delta fold. The headings it names are
:data:`SUMMARY_SECTIONS`; a test pins the two together."""

END_OF_SUMMARY_MARKER: Final[str] = (
    "--- END OF SUMMARY — the turns that follow continue the conversation "
    "verbatim; it is reference for what came before them ---"
)
"""Closes the summary block in a seed, so a model reads the summary as
reference data and not as a turn to continue."""


# ---------------------------------------------------------------------------
# The running summary
# ---------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(1, limit - len(_ELLIPSIS))].rstrip() + _ELLIPSIS


def _entry_key(entry: str) -> str:
    """What makes two entries the same entry: case and spacing aside."""
    return " ".join(entry.split()).casefold()


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """The running summary: entries per section, shed counts, and what it covers.

    Immutable; every operation returns a new value. ``entries`` holds only the
    sections that have entries, in no particular order (rendering imposes the
    fixed one). ``covered_through`` is the session position of the last
    message of the turns this summary records — the host's positions, opaque
    here. A summary built fresh covers nothing, which is what makes a reset
    safe: the seed carries every turn again.
    """

    entries: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    shed: Mapping[str, int] = field(default_factory=dict)
    """Entries a section no longer carries because of the cap."""
    covered_through: int = NOTHING_COVERED

    def is_empty(self) -> bool:
        """No entries and no shed note — it renders to nothing. Coverage aside."""
        return not any(self.entries.values()) and not any(self.shed.values())

    def covers(self, position: int | None) -> bool:
        """Whether the message at session ``position`` is recorded here."""
        return position is not None and position <= self.covered_through

    def section(self, heading: str) -> tuple[str, ...]:
        return tuple(self.entries.get(heading, ()))

    def render(self) -> str:
        """The summary as the seed shows it and the summariser is shown it.

        Deterministic: the same summary renders to the same bytes, which is
        what keeps the seed's prefix stable between folds. Every entry line
        starts with ``- ``, so an entry that begins with a section's name is
        never read back as a heading (:meth:`from_text`).
        """
        lines: list[str] = []
        for section in SUMMARY_SECTIONS:
            items = self.entries.get(section.heading, ())
            shed = self.shed.get(section.heading, 0)
            if not items and not shed:
                continue
            lines.append(f"{section.heading}:")
            lines.extend(f"- {item}" for item in items)
            if shed:
                lines.append(f"  ({shed} {_SHED_NOTE})")
        return "\n".join(lines)

    @classmethod
    def from_text(cls, text: str, *, covered_through: int) -> SessionSummary:
        """Read back exactly what :meth:`render` wrote, for a host that stores text.

        Strict: every non-blank line must be a heading line (``HEADING:``), an
        entry line (``- …``) or a shed note, as :meth:`render` writes them.
        Anything else — text stored by an earlier format of this summary, or
        not by this module at all — raises :class:`SummaryDeltaError`; the
        host's answer to that is ``SessionSummary()``, which covers nothing,
        so every turn is carried again rather than any being dropped.
        ``covered_through`` is the coverage the host stored beside the text.
        """
        entries: dict[str, list[str]] = {}
        shed: dict[str, int] = {}
        current: str | None = None
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.endswith(":") and line[:-1] in _SECTION_BY_HEADING:
                current = line[:-1]
                entries.setdefault(current, [])
                continue
            if current is None:
                raise SummaryDeltaError("stored summary text has a line outside every section")
            if line.startswith("- "):
                entries[current].append(line[2:])
                continue
            count = _shed_count(line)
            if count is None:
                raise SummaryDeltaError("stored summary text is not in the current format")
            shed[current] = shed.get(current, 0) + count
        return cls(
            entries={k: tuple(v) for k, v in entries.items() if v},
            shed={k: v for k, v in shed.items() if v},
            covered_through=covered_through,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {k: list(v) for k, v in self.entries.items() if v},
            "shed": {k: int(v) for k, v in self.shed.items() if v},
            "covered_through": self.covered_through,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SessionSummary:
        """The summary :meth:`to_dict` wrote. ``None`` or ``{}`` is a fresh summary."""
        if not data:
            return cls()
        raw_entries = data.get("entries") or {}
        raw_shed = data.get("shed") or {}
        entries = {
            str(k): tuple(str(x) for x in v)
            for k, v in raw_entries.items()
            if str(k) in _SECTION_BY_HEADING and isinstance(v, list | tuple) and v
        }
        shed = {
            str(k): int(v)
            for k, v in raw_shed.items()
            if str(k) in _SECTION_BY_HEADING and isinstance(v, int) and v > 0
        }
        covered = data.get("covered_through")
        return cls(
            entries=entries,
            shed=shed,
            covered_through=covered if isinstance(covered, int) else NOTHING_COVERED,
        )


class SummaryDeltaError(ValueError):
    """Text that is not a usable fold reply (or not a stored summary)."""


@dataclass(frozen=True, slots=True)
class SummaryDelta:
    """A fold's reply, parsed: new entries per section."""

    entries: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    unattributed_lines: int = 0
    """Non-empty lines outside any known heading (dropped; worth logging)."""
    truncated: bool = False
    """The reply was cut at its output bound; its last entry was dropped and it
    supersedes no section."""
    nothing_new: bool = False
    """The summariser said the turns add nothing (:data:`NOTHING_NEW`)."""

    def is_empty(self) -> bool:
        return not any(self.entries.values())


def _heading_of(line: str) -> tuple[str, str] | None:
    """``(heading, rest of the line)`` when ``line`` opens a section, else ``None``.

    A bulleted line is never a heading. Presentation a model varies freely is
    folded away first: markdown marks, an ordinal, case, ``and`` for ``&``,
    spacing around ``/`` and a trailing colon. Text after the colon on the same
    line is the section's first entry.
    """
    stripped = line.strip()
    if any(stripped.startswith(bullet) for bullet in _ENTRY_BULLETS):
        return None
    text = stripped.lstrip(_HEADING_DECORATION)
    digits = len(text) - len(text.lstrip("0123456789"))
    if digits and text[digits : digits + 1] in {".", ")"}:
        text = text[digits + 1 :]
    head, sep, rest = text.partition(":")
    label = head
    for mark in "*_`":
        label = label.replace(mark, "")
    label = " ".join(label.upper().replace(" AND ", " & ").replace("/", " / ").split())
    if label in _SECTION_BY_HEADING:
        return label, rest.strip().strip("*_` ") if sep else ""
    return None


def _entry_of(line: str) -> str:
    text = line.strip()
    for bullet in _ENTRY_BULLETS:
        if text.startswith(bullet):
            text = text[len(bullet) :].strip()
            break
    else:
        digits = len(text) - len(text.lstrip("0123456789"))
        if digits and text[digits : digits + 2] in {". ", ") "}:
            text = text[digits + 2 :].strip()
    if not any(ch.isalnum() for ch in text):
        return ""
    if "".join(ch for ch in text.casefold() if ch.isalnum()) in _NULL_ENTRIES:
        return ""
    return _clip(text, _ENTRY_MAX_CHARS)


def _shed_count(line: str) -> int | None:
    """N when ``line`` is the shed note :meth:`SessionSummary.render` writes."""
    text = line.strip()
    if not (text.startswith("(") and text.endswith(f" {_SHED_NOTE})")):
        return None
    number = text[1 : -len(f" {_SHED_NOTE})")]
    return int(number) if number.isdigit() else None


def _without_reasoning(text: str) -> str:
    """The reply after a leading reasoning block; empty when the reply is all reasoning."""
    close = text.rfind(_THINK_CLOSE)
    if close >= 0:
        return text[close + len(_THINK_CLOSE) :]
    if text.lstrip().startswith(_THINK_OPEN):
        return ""
    return text


def parse_summary_delta(text: str, *, truncated: bool = False) -> SummaryDelta:
    """Parse a delta fold's reply into new entries per section — PURE.

    Each non-empty line under a known heading is one entry; a bulleted line is
    always an entry. Placeholder entries ("none", "n/a", "—") are dropped. A
    leading reasoning block (``<think>…</think>``) is not part of the reply.
    Headings may come in any order and more than once. A reply of exactly
    :data:`NOTHING_NEW` is a valid empty delta.

    ``truncated`` says the reply was cut at its output bound: its last entry
    is probably cut mid-sentence and is dropped; the delta is marked
    :attr:`SummaryDelta.truncated` and supersedes no section.

    Raises :class:`SummaryDeltaError` when nothing usable is left, so the host
    advances no coverage and the planned turns stay in the seed:

    * an empty reply, or one that is all reasoning — the output bound spent
      thinking leaves exactly this;
    * text with none of the headings;
    * a truncated reply with no complete entry left.
    """
    body = _without_reasoning(text)
    if not body.strip():
        raise SummaryDeltaError("the reply is empty")
    if " ".join(body.split()).upper().strip(".") == NOTHING_NEW and not truncated:
        return SummaryDelta(nothing_new=True)
    entries: dict[str, list[str]] = {}
    order: list[tuple[str, int]] = []
    current: str | None = None
    unattributed = 0
    seen_heading = False
    for line in body.splitlines():
        if not line.strip():
            continue
        heading = _heading_of(line)
        if heading is not None:
            seen_heading = True
            current, rest = heading
            entries.setdefault(current, [])
            if not rest:
                continue
            line = rest
        elif current is None:
            unattributed += 1
            continue
        entry = _entry_of(line)
        if entry:
            entries[current].append(entry)
            order.append((current, len(entries[current]) - 1))
    if not seen_heading:
        raise SummaryDeltaError("the reply names none of the summary sections")
    if truncated:
        if order:
            last_heading, last_index = order[-1]
            del entries[last_heading][last_index]
        if not any(entries.values()):
            raise SummaryDeltaError("the reply was cut off before its first complete entry")
    return SummaryDelta(
        entries={k: tuple(v) for k, v in entries.items() if v},
        unattributed_lines=unattributed,
        truncated=truncated,
    )


@dataclass(frozen=True, slots=True)
class MergeReport:
    """What one merge did to the summary."""

    added: int = 0
    duplicates: int = 0
    replaced_sections: tuple[str, ...] = ()
    kept_on_truncation: tuple[str, ...] = ()
    """Supersede sections a truncated reply wrote and that were left as stored."""
    shed: int = 0
    clipped: int = 0
    already_applied: bool = False
    """The memory already covered what the delta covers; nothing was merged."""


def _over_cap(summary: SessionSummary, cap: int, rc: LoopConstants) -> bool:
    return cap > 0 and estimate_tokens(summary.render(), rc) > cap


def _shed_one(entries: dict[str, list[str]], shed: dict[str, int]) -> bool:
    """Remove the oldest entry of the first section in shed order; False if none can go."""
    for section in sorted(SUMMARY_SECTIONS, key=lambda s: s.shed_rank):
        items = entries.get(section.heading, [])
        floor = 1 if section.keep_first else 0
        if len(items) > floor:
            del items[floor]
            shed[section.heading] = shed.get(section.heading, 0) + 1
            return True
    for section in sorted(SUMMARY_SECTIONS, key=lambda s: s.shed_rank):
        items = entries.get(section.heading, [])
        if items:
            del items[0]
            shed[section.heading] = shed.get(section.heading, 0) + 1
            return True
    return False


def cap_summary(summary: SessionSummary, rc: LoopConstants) -> tuple[SessionSummary, int, int]:
    """Bring ``summary`` under ``session_memory_running_summary_token_cap`` — PURE.

    Whole entries go first, oldest first, in shed order. Only when the summary
    is down to a single entry and still over the cap is that entry clipped,
    by halves, to no less than a short floor — a cap set below one short entry
    leaves the summary as close to it as that allows. Coverage is kept.
    Returns the summary, the number of entries shed and of entries clipped.
    """
    cap = rc.session_memory_running_summary_token_cap
    if not _over_cap(summary, cap, rc):
        return summary, 0, 0
    entries = {k: list(v) for k, v in summary.entries.items()}
    shed = dict(summary.shed)
    count = 0
    clipped = 0

    def current() -> SessionSummary:
        return SessionSummary(
            entries={k: tuple(v) for k, v in entries.items() if v},
            shed=dict(shed),
            covered_through=summary.covered_through,
        )

    while _over_cap(current(), cap, rc) and sum(len(v) for v in entries.values()) > 1:
        if not _shed_one(entries, shed):
            break  # pragma: no cover - guarded by the entry count above
        count += 1
    while _over_cap(current(), cap, rc):
        remaining = [(k, v) for k, v in entries.items() if v]
        if not remaining:
            break
        heading, items = remaining[0]
        limit = max(_ENTRY_MIN_CLIP_CHARS, len(items[0]) // 2)
        if len(items[0]) <= _ENTRY_MIN_CLIP_CHARS:
            break
        items[0] = _clip(items[0], limit)
        entries[heading] = items
        clipped = 1
    return current(), count, clipped


def merge_summary_delta(
    summary: SessionSummary,
    delta: SummaryDelta,
    rc: LoopConstants,
) -> tuple[SessionSummary, MergeReport]:
    """Append ``delta`` to ``summary`` and hold the result to the cap — PURE.

    Accumulating sections skip an entry already present (case and spacing
    aside); a superseding section takes the delta's entries whole when it has
    any — unless the delta is :attr:`~SummaryDelta.truncated`, when it is left
    as stored and named in :attr:`MergeReport.kept_on_truncation`. Then
    :func:`cap_summary`. Coverage is the caller's (:func:`fold_run`).
    """
    entries = {k: list(v) for k, v in summary.entries.items()}
    added = 0
    duplicates = 0
    replaced: list[str] = []
    kept: list[str] = []
    for section in SUMMARY_SECTIONS:
        new = list(delta.entries.get(section.heading, ()))
        if not new:
            continue
        if section.merge is SectionMerge.supersede:
            if delta.truncated:
                kept.append(section.heading)
                continue
            entries[section.heading] = []
            replaced.append(section.heading)
        items = entries.setdefault(section.heading, [])
        keys = {_entry_key(item) for item in items}
        for entry in new:
            key = _entry_key(entry)
            if key in keys:
                duplicates += 1
                continue
            keys.add(key)
            items.append(entry)
            added += 1
    shed = dict(summary.shed)
    for heading in replaced:
        shed.pop(heading, None)
    merged = SessionSummary(
        entries={k: tuple(v) for k, v in entries.items() if v},
        shed=shed,
        covered_through=summary.covered_through,
    )
    capped, shed_count, clipped = cap_summary(merged, rc)
    return capped, MergeReport(
        added=added,
        duplicates=duplicates,
        replaced_sections=tuple(replaced),
        kept_on_truncation=tuple(kept),
        shed=shed_count,
        clipped=clipped,
    )


# ---------------------------------------------------------------------------
# The summariser's input
# ---------------------------------------------------------------------------


def _serialize_message(message: Message) -> str:
    """Render a message as ``[ROLE] text [calls Tool(args)] [tool result: …]``.

    Summary INPUT, so the model treats it as material, not as a chat to
    continue. Nothing is cut here: the messages handed in are the masked form
    the seed carried (:attr:`~protocore.runtime.context.session_seed.FoldPlan.source`),
    already bounded, with the lines masking kept.
    """
    role = message.role.value.upper()
    pieces: list[str] = []
    for block in message.content_blocks:
        if isinstance(block, TextBlock):
            if block.text:
                pieces.append(block.text)
        elif isinstance(block, ToolUseBlock):
            pieces.append(f"[calls {block.name}({block.arguments_json})]")
        elif isinstance(block, ToolResultBlock):
            pieces.append(f"[tool result: {block.content}]")
    return f"[{role}] " + " ".join(p for p in pieces if p).strip()


def _serialize_turns(messages: Sequence[Message]) -> str:
    return "\n".join(_serialize_message(m) for m in messages)


def build_summary_delta_message(
    summary: SessionSummary,
    turn_messages: Sequence[Message],
) -> str | None:
    """The user message of a delta fold, to send under :data:`SUMMARY_DELTA_SYSTEM` — PURE.

    ``turn_messages`` are the turns leaving the raw window, oldest first, in
    the form the seed carried them —
    :attr:`~protocore.runtime.context.session_seed.FoldPlan.source_messages`.
    ``None`` when they render to nothing: there is no call to make.
    """
    serialized = _serialize_turns(turn_messages).strip()
    if not serialized:
        return None
    existing = summary.render() or "(empty)"
    return (
        f"EXISTING SUMMARY (already recorded — do not repeat):\n{existing}\n\n"
        f"NEW TURNS (leaving the agent's context now):\n{serialized}"
    )


def build_summary_delta_request(
    summary: SessionSummary,
    turn_messages: Sequence[Message],
    rc: LoopConstants,
    *,
    model: str,
    thinking: bool | None,
    temperature: float | None = None,
) -> LLMRequest | None:
    """The whole delta-fold request, ready for a provider's ``complete_text`` — PURE.

    System prompt :data:`SUMMARY_DELTA_SYSTEM`, the user message of
    :func:`build_summary_delta_message`, no tools, and an output bound of
    ``session_memory_fold_max_output_tokens``. ``thinking`` is the request's
    reasoning switch (``extra["enable_thinking"]``) and the host must decide
    it: ``False`` keeps a reasoning model from spending the bound thinking and
    replying with nothing; ``None`` leaves it to the provider's default.
    ``None`` when there is nothing to fold.
    """
    user = build_summary_delta_message(summary, turn_messages)
    if user is None:
        return None
    return LLMRequest(
        model=model,
        messages=[
            Message(role=MessageRole.system, content_blocks=[TextBlock(text=SUMMARY_DELTA_SYSTEM)]),
            Message(role=MessageRole.user, content_blocks=[TextBlock(text=user)]),
        ],
        tools=[],
        max_tokens=rc.session_memory_fold_max_output_tokens,
        temperature=temperature,
        extra={} if thinking is None else {"enable_thinking": thinking},
    )


def fold_source_tokens(turn_messages: Sequence[Message], rc: LoopConstants) -> int:
    """Estimated tokens of the turns as the summariser is shown them."""
    return estimate_tokens(_serialize_turns(turn_messages), rc)


# ---------------------------------------------------------------------------
# Artifact registry — structured tool-call field reads only
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ArtifactLedger:
    """Files written in the session, read from tool-call arguments by role.

    ``files`` is an ordered, de-duplicated ``path`` list (first-seen order);
    ``content`` maps ``path`` -> the last verbatim content snapshot (capped).
    """

    files: list[str] = field(default_factory=list)
    content: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.files

    def to_dict(self) -> dict[str, Any]:
        return {"files": list(self.files), "content": dict(self.content)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ArtifactLedger:
        if not data:
            return cls()
        content = data.get("content") or {}
        return cls(
            files=[str(x) for x in (data.get("files") or [])],
            content={str(k): str(v) for k, v in content.items()} if isinstance(content, dict) else {},
        )


def _parse_args(block: ToolUseBlock) -> dict[str, Any]:
    try:
        parsed = json.loads(block.arguments_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_file_write_tool(name: str, roles: ToolRoleMap) -> bool:
    """A tool whose call records a written or edited file — by its registered role."""
    return bool(roles.roles_of(name) & WORKSPACE_MUTATION_ROLES)


def extract_artifacts(
    messages: Sequence[Message],
    *,
    base: ArtifactLedger | None = None,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> ArtifactLedger:
    """The file registry of ``messages``, folded into ``base``.

    Reads the ``path`` and ``content`` argument slots of workspace-mutating
    tool calls. Order-preserving and de-duplicated; the latest write of a path
    wins its snapshot. Hand it the run's messages as they happened, never a
    seed: a masked call's long arguments are no longer there.
    """
    ledger = ArtifactLedger(
        files=list(base.files) if base else [],
        content=dict(base.content) if base else {},
    )
    seen = set(ledger.files)
    for message in messages:
        for block in message.content_blocks:
            if not isinstance(block, ToolUseBlock) or not _is_file_write_tool(block.name, roles):
                continue
            args = _parse_args(block)
            path_value = string_argument(args, ToolArgumentSlot.path, roles=roles) or ""
            if not path_value:
                continue
            if path_value not in seen and len(ledger.files) < _MAX_FILES:
                seen.add(path_value)
                ledger.files.append(path_value)
            if path_value in seen:
                content_value = string_argument(args, ToolArgumentSlot.content, roles=roles)
                if content_value:
                    ledger.content[path_value] = content_value[:_CONTENT_SNAPSHOT_MAX_CHARS]
    return ledger


def render_ledger(ledger: ArtifactLedger) -> str:
    """The file registry as one compact block; empty when there are no files."""
    if ledger.is_empty():
        return ""
    lines = ["ARTIFACT REGISTRY — files created/edited this session (verbatim, authoritative):"]
    for path in ledger.files:
        snapshot = ledger.content.get(path)
        if snapshot:
            lines.append(f"- {path}\n  current content:\n{snapshot}")
        else:
            lines.append(f"- {path}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The durable memory and the post-run update
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionMemory:
    """The durable, per-session memory carried across runs.

    ``turn_index`` counts the runs recorded (0 = a fresh memory). Which turns
    the summary covers is part of the summary
    (:attr:`SessionSummary.covered_through`), so the two are stored, rolled
    back and reset together.

    ``stale_fold_count`` counts consecutive folds that tried to add to the
    summary and could not (a refused reply). The host, which knows which kind
    of fold ran, owns the increment; :func:`fold_run` carries it through.

    Loading a memory stored by an earlier format: :meth:`from_dict` reads only
    the keys :meth:`to_dict` writes, so a dict without ``"summary"`` loads with
    a fresh summary that covers nothing (its registry and counters still
    load), and :meth:`SessionSummary.from_text` refuses text it did not
    write. Either way every stored turn is carried in the next seed again.
    """

    summary: SessionSummary = field(default_factory=SessionSummary)
    ledger: ArtifactLedger = field(default_factory=ArtifactLedger)
    turn_index: int = 0
    stale_fold_count: int = 0

    def is_empty(self) -> bool:
        return (
            self.turn_index == 0
            and self.summary.is_empty()
            and self.summary.covered_through == NOTHING_COVERED
            and self.ledger.is_empty()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "ledger": self.ledger.to_dict(),
            "turn_index": self.turn_index,
            "stale_fold_count": self.stale_fold_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SessionMemory:
        if not data:
            return cls()
        return cls(
            summary=SessionSummary.from_dict(data.get("summary")),
            ledger=ArtifactLedger.from_dict(data.get("ledger")),
            turn_index=int(data.get("turn_index") or 0),
            stale_fold_count=int(data.get("stale_fold_count") or 0),
        )


@dataclass(slots=True)
class FoldResult:
    """Outcome of :func:`fold_run`."""

    memory: SessionMemory
    summary_updated: bool = False
    """The delta added, replaced or shed something, or advanced coverage."""
    report: MergeReport = field(default_factory=MergeReport)
    notes: dict[str, Any] = field(default_factory=dict)


def fold_run(
    memory: SessionMemory,
    run_messages: Sequence[Message],
    delta: SummaryDelta | None,
    rc: LoopConstants,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
    *,
    covers_through: int | None = None,
) -> FoldResult:
    """Record one finished run in ``memory`` — PURE; the input is not mutated.

    Called ONCE per run, with or without a fold:

    * The artifact registry takes the run's file writes (always; no model).
    * ``delta`` — the parsed reply of the fold the planner asked for, or
      ``None`` when no fold ran or it failed (an unparsable reply raised
      :class:`SummaryDeltaError` before this point and must be passed as
      ``None``) — is appended to the summary and the result held to its cap,
      and the summary's coverage advances to ``covers_through``
      (:attr:`~protocore.runtime.context.session_seed.FoldPlan.covers_through`),
      which a delta requires. A delta whose ``covers_through`` the summary
      already reaches is not applied again (:attr:`MergeReport.already_applied`).
    * ``turn_index`` advances.
    """
    ledger = extract_artifacts(run_messages, base=memory.ledger, roles=roles)
    summary = memory.summary
    report = MergeReport()
    if delta is not None:
        if covers_through is None:
            raise ValueError("a delta needs the coverage it advances to (covers_through)")
        if covers_through <= summary.covered_through:
            report = MergeReport(already_applied=True)
        else:
            merged, report = merge_summary_delta(summary, delta, rc)
            summary = SessionSummary(
                entries=merged.entries, shed=merged.shed, covered_through=covers_through
            )
    updated = summary != memory.summary
    new_memory = SessionMemory(
        summary=summary,
        ledger=ledger,
        turn_index=memory.turn_index + 1,
        stale_fold_count=memory.stale_fold_count,
    )
    return FoldResult(
        memory=new_memory,
        summary_updated=updated,
        report=report,
        notes={
            "folded_messages": len(run_messages),
            "ledger_files": len(ledger.files),
            "summary_tokens": estimate_tokens(summary.render(), rc),
            "covered_through": summary.covered_through,
            "entries_added": report.added,
            "entries_shed": report.shed,
        },
    )


__all__ = [
    "END_OF_SUMMARY_MARKER",
    "NOTHING_COVERED",
    "NOTHING_NEW",
    "SUMMARY_DELTA_SYSTEM",
    "SUMMARY_SECTIONS",
    "ArtifactLedger",
    "FoldResult",
    "MergeReport",
    "SectionMerge",
    "SessionMemory",
    "SessionSummary",
    "SummaryDelta",
    "SummaryDeltaError",
    "SummarySection",
    "build_summary_delta_message",
    "build_summary_delta_request",
    "cap_summary",
    "extract_artifacts",
    "fold_run",
    "fold_source_tokens",
    "merge_summary_delta",
    "parse_summary_delta",
    "render_ledger",
]
