"""Seeding a run from its session's earlier turns: raw, masked, summarised last.

PURE core module: no host, no store, no model client. The host loads the
session's earlier messages and its :class:`~protocore.runtime.context.session_memory.SessionMemory`,
and says where each stored message sits in the session (``position``); this
module decides what the next run is shown and which turns, if any, need a
summary fold now.

Both entry points are CPU-bound: they split, mask and estimate messages and do
no I/O. An async host runs them off its event loop (``asyncio.to_thread``)
and keeps a :class:`SeedCache` per session, so that the work per run is
proportional to the turns added since the last call, not to the session's
length. The cache is thread-safe; the seed of one run and the fold plan of
the previous one may use it at the same time. A cold call — the first after a
process start or a summary reset — prepares every carried turn: about 1 s for
200 heavy turns and 6 s for 1000 on the pure-Python estimator, CPU that holds
the interpreter lock even in a worker thread; warm calls take milliseconds.
:func:`recommend_seed_budget` sizes the seed from the model's window and the
run's measured overhead.

Everything the seed emits encodes as UTF-8: a message holding a lone
surrogate is rewritten with that code point as its backslash-u escape, and
argument masking never turns an escaped surrogate into a raw one.

Positions
---------

``position(message)`` is the host's session position of a stored message: a
non-negative integer, strictly increasing in session order, the same number
for the same message on every call, or ``None`` when the host does not know
it. A turn whose known positions do not increase, or do not lie after every
position of the turns before it, is treated as having none. Coverage
(:class:`~protocore.runtime.context.session_memory.Coverage`) is one range of
these positions. A turn is recorded by the summary when it has at least one
known position and every known position lies in the range and in no pinned
range; a message without a position rides along with its turn. A turn none
of whose messages has a position can never be recorded: it is carried while
the budget allows and otherwise counted among the turns not in context, and
it does not stop later turns from being folded. Pinned turns — ones the
summariser was cut off on again and again, one at a time — lie inside the
range, are never recorded and are carried masked for good.

What a seed carries
-------------------

The session's earlier messages are split into turns (:func:`split_turns`): a
user message and everything that answered it. A turn is carried whole or not
at all. A tool call without a result after it, or a result without a call
before it, is taken out (:func:`repair_turn`), so the seed never puts an
orphan in front of a provider; ids are paired in order, so a server that
reuses an id across steps is paired correctly.

Every user and assistant word of a carried turn is carried verbatim. What
changes with age is the tool exchanges. The newest
``session_seed_full_turns`` turns carry them in full. In older (masked) turns:

* each tool result whose text is long enough is replaced by a short
  placeholder: the tool's name, the size of what is not carried, the lines a
  :class:`MaskingRule` keeps (by default lines starting with a configured
  citation prefix and lines carrying an identifier), and the head of the
  result. A placeholder is used only when it is shorter than the result. It
  states what is missing; it never tells the model to do anything;
* every tool image is dropped, and the result says how many and of which
  type;
* long tool-call arguments are cut to a head (strings, keys and lists past
  ``session_seed_mask_arg_max_chars``); the arguments stay valid JSON and
  shorter values (paths, ids) stay whole. The artifact registry is built
  from the run as it happened and keeps the files;
* the reasoning of assistant messages (``reasoning_content``, thinking
  blocks, opaque reasoning records) is dropped. The core's OpenAI-compatible
  adapter does send earlier turns' reasoning back when an endpoint is
  configured to (``ReasoningReplay.when_present``, or ``required`` with
  thinking on, where a message without it is sent with an empty field), so
  dropping it changes what such an endpoint receives. It is safe for the
  request because a masked turn is a completed turn — its tool calls are all
  answered and a later user message follows it — and the reasoning that
  providers require is that of the tool loop in progress, which is never in
  a masked turn: the newest full turns keep theirs, and the run's own turns
  are not seeded at all.

The masked form of a turn depends only on that turn and the configuration, so
the same turn is the same bytes in every seed.

Only turns that no longer fit even masked are represented by the running
summary, and the summary is grown by a delta fold of exactly those turns,
shown in their masked form (:func:`plan_session_fold`).

Layout::

    [head]     the first session_memory_head_protect_turns turns, never folded,
               and any turn older than the summary's range it does not record
    [summary]  the running summary, when it has entries or covers turns
                                            (dual-tagged reference block)
    [notice]   only when something did not fit: how many turns (and whether
               the summary or the registry) are not in this context
    [window]   every later turn the summary does not record, oldest first
               (pinned turns among them, masked):
               masked … masked | full (newest session_seed_full_turns)
    [registry] the artifact registry, when the session wrote files
                                            (dual-tagged reference block)
    ← the host then appends its preludes and the new task

Which turns are in the window is decided by the summary's coverage, not by
the budget: a turn leaves the window only when the summary records it. A
summary that is reset covers nothing, so a reset brings every turn back
rather than losing any; and coverage is a range that starts after the head,
so a head turn whose protection lapses (a settings or rule change, a head
that grew past ``session_memory_head_max_tokens``) moves to the window and is
folded like any other, never into neither place.

The seed always fits its budget. Under pressure, in this order: older full
turns are masked; the registry shrinks within its bound (to its paths, then
to a count of its files); the newest
turn's result text is cut to the longest head that fits (its images stay);
it is masked; window turns are left out oldest first; head turns are left
out; the registry is left out; the summary is left out; the newest turn is
reduced to its words (question and answer); last, the newest turn is left
out. So the previous answer outranks the registry, the summary detail and
every older turn. One notice says what is not in context. Turns left out
this way are not recorded anywhere yet; the fold plan, which counts every
turn the summary does not record, folds them in bounded batches over the
following runs (see :func:`plan_session_fold`).

In-run compaction and the seeded previous turn: every message of an
unmasked newest turn carries
:data:`~protocore.contracts.types.SESSION_SEED_MASKED_FORM_METADATA_KEY`, its
masked form. The first thing in-run compaction does is put those forms in
place (:func:`~protocore.runtime.context.compaction.remask_seeded_turns`),
before it touches the run's own work, so the seed can carry the previous
turn uncut without starving a run that does new tool work, and what the run
keeps of it is exactly what the next seed carries.

Adjacent user messages: the summary, the notice and the registry are
user-role blocks, so a seed can put two user messages next to each other
(summary before the first window question, registry before the host's task),
as the core's own in-run compaction also does. The core does not merge them,
and its OpenAI-compatible adapter sends them as separate user messages. An
endpoint whose chat template rejects consecutive user messages is not
supported by that adapter as it stands; a host serving one must merge
adjacent user messages itself before the request is sent.

Prefix stability
----------------

A provider's prefix cache serves a request up to its first differing byte, so
the seed is built to grow by appending. Between two consecutive runs of one
session, when no fold has landed in between and nothing had to be left out,
the next seed starts with the previous one byte for byte through its last
masked turn: the head, the summary and every masked window turn are
unchanged and in the same positions, and the next seed adds turns after
them. What is NOT stable, and when:

* **The full tail.** A turn that crosses the masking boundary is shown masked
  from then on, so the prefix breaks at its first message that masking
  changes (an assistant message with reasoning or long arguments, else its
  first long tool result). The boundary moves in steps of
  ``session_seed_mask_batch_turns`` turns, and it is at the end of the seed.
  Budget pressure (a long new message) may mask or cut the full turns of one
  run: again the end of the seed only.
* **Images.** Masked turns carry no tool image, so an endpoint's or the
  loop's limit on how many recent images a request carries can elide only
  images in the full tail. Images a user attached to an older turn are user
  content and stay; on an endpoint with an image limit, the prefix breaks at
  the first such image the limit elides once newer images arrive.
* **The registry.** It changes when a file is written, and sits after the
  window for that reason.
* **A fold landing.** The summary changes and the turns it now records leave
  the window: the prefix is stable only through the head. Folds are planned
  with hysteresis (see below).
* **A held fold.** A fold reply cut off at its output bound adds its
  complete entries to the summary but advances no coverage: the summary
  changes (the prefix breaks after the head) and the next run folds again,
  in a smaller batch.
* **Leaving turns out.** When the budget cannot hold the masked window, the
  front of the window moves on every run that does it, until the folds catch
  up.
* **A changed history.** A turn edited, regenerated or removed changes the
  seed from that turn on.
* **In-run compaction.** Re-masking the seeded newest turn changes the run's
  request from that turn on; it happens only on a run whose own work needed
  the room, and it is what the next seed carries anyway.

The masking rule must be a pure function of its arguments; one that is not
makes masked turns differ between runs and gives the cache nothing.

When a fold is needed
---------------------

:func:`plan_session_fold`, after a run, measures the seed the next run would
get with this run included: the head (masked), the summary at its cap, the
registry, every turn the summary does not record (masked), and the newest
full turns. Answer quality comes first, so the newest turn is counted at its
full size — older turns are folded early so that the previous turn's tool
results reach the next run uncut — whenever, with every older turn folded,
there would still be room below the trigger for
``session_seed_fold_min_turns_between`` more turns of the newest turn's masked
size: each run turns the newest turn into an older, masked one, so that room
is what keeps two folds at least that many runs apart. When the newest turn is
too large for that, counting it in full would fold on every run, so it is
counted masked instead and the seed cuts it when the window has grown. Pinned
turns and turns without known positions count as fixed: folding cannot
move them. Once the measured size passes
``session_seed_fold_trigger_ratio`` of the budget, the oldest unrecorded
turns are planned until it is at or below ``session_seed_fold_target_ratio``
(or, with the newest counted in full, at or below what the head, summary,
registry and newest turn need). Otherwise the plan is empty and the host
makes no summariser call.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

from pydantic_core import PydanticSerializationError

from protocore.constants import MAX_SESSION_SEED_CACHE_ENTRIES
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    SESSION_SEED_MASKED_FORM_METADATA_KEY,
    Message,
    MessageOrigin,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.context.compaction import estimate_message_tokens
from protocore.runtime.context.ledger import extract_identifiers
from protocore.runtime.context.session_memory import (
    END_OF_SUMMARY_MARKER,
    Coverage,
    CoverageStep,
    SessionMemory,
    encodable,
    fold_source_tokens,
    pin_basis,
    render_ledger,
    render_ledger_count,
)
from protocore.runtime.wire_format import is_compacted_placeholder

# Render-shape caps of the placeholder: they bound a string, not a behaviour.
_KEPT_LINE_MAX_CHARS: Final[int] = 400
_KEPT_LINES_MAX: Final[int] = 40
_ELLIPSIS: Final[str] = "…"

#: The constants preparing a turn reads (masking and the token estimate). The
#: cache is keyed on these and nothing else, so a change to an unrelated
#: setting keeps every prepared turn.
_PREPARATION_FIELDS: Final[tuple[str, ...]] = (
    "session_seed_mask_min_chars",
    "session_seed_mask_preview_chars",
    "session_seed_mask_arg_max_chars",
    "session_seed_mask_arg_preview_chars",
    "token_count_chars_per_token_latin",
    "token_count_chars_per_token_cyrillic",
    "token_count_chars_per_token_cyrillic_json_escape",
    "token_count_chars_per_token_cjk",
    "token_count_chars_per_token_json_struct",
    "token_count_image_tokens",
    "token_estimate_calibration",
)

MessagePosition = Callable[[Message], int | None]
"""Where a stored message sits in the session — see "Positions" above."""

KeptLines = Callable[[ToolUseBlock | None, ToolResultBlock], Sequence[str]]
"""Lines of a tool result its placeholder must keep verbatim."""

KeepWhole = Callable[[ToolUseBlock | None, ToolResultBlock], bool]
"""Whether a tool result is never masked."""


def _keep_nothing_whole(call: ToolUseBlock | None, result: ToolResultBlock) -> bool:
    return False


@dataclass(frozen=True, slots=True)
class MaskingRule:
    """What of an older turn's tool result survives masking. Injected by the host.

    ``kept_lines`` returns the lines its placeholder keeps (citation lines,
    record identifiers — whatever later answers rely on); ``keep_whole`` names
    results that are never masked (their text and images are carried as they
    are). Both are called with the call the result answers (``None`` when the
    turn holds no such call) and must be pure: the seed's prefix stability
    rests on a masked turn being the same bytes every time.

    ``key`` names what the rule does, for :class:`SeedCache`: two rules with
    the same key must mask identically, and a rule whose behaviour depends on
    configuration puts that configuration into its key. A rule with an empty
    key is never cached — every call prepares its turns afresh — so it cannot
    be served another rule's masking.
    """

    kept_lines: KeptLines
    keep_whole: KeepWhole = _keep_nothing_whole
    key: str = ""


def _prefixes(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def default_masking_rule(rc: LoopConstants) -> MaskingRule:
    """Keep the lines a deployment marks as citations, and lines carrying identifiers.

    A line is kept when it starts with one of
    ``tool_result_stale_trim_protected_prefixes`` (all of them), or when it
    carries an identifier of a shape tool output rarely holds as noise — a
    URL, a UUID, an absolute path, a host:port, a version — up to
    ``session_seed_mask_identifier_lines`` such lines, in the order they
    appear. It names no tool: what a deployment's tools print is the host's to
    add (wrap this rule's ``kept_lines``, and give the wrapper its own key).
    """
    prefixes = _prefixes(rc.tool_result_stale_trim_protected_prefixes)
    limit = rc.session_seed_mask_identifier_lines

    def kept_lines(call: ToolUseBlock | None, result: ToolResultBlock) -> list[str]:
        kept: list[str] = []
        identified = 0
        for raw in result.content.splitlines():
            line = raw.strip()
            if not line:
                continue
            if any(line.startswith(prefix) for prefix in prefixes):
                kept.append(line)
            elif identified < limit and extract_identifiers(line, from_tool_output=True):
                kept.append(line)
                identified += 1
        return kept

    return MaskingRule(kept_lines=kept_lines, key=json.dumps(["default", list(prefixes), limit]))


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def opens_turn(message: Message) -> bool:
    """A user message of the conversation that is not carrying tool results back.

    Compaction records are user-role and written mid-turn, often right after a
    tool result; the loop's scaffolding (nudges, recovery notes) is user-role
    too. Read as the start of a turn either would split a run in two.
    """
    return (
        message.role is MessageRole.user
        and message.origin is MessageOrigin.conversation
        and not any(isinstance(b, ToolResultBlock) for b in message.content_blocks)
    )


def split_turns(messages: Sequence[Message]) -> list[list[Message]]:
    """``messages`` as turns: each a user message and everything that answered it.

    Messages before the first user message form a leading fragment of their
    own, which the seed does not carry.
    """
    turns: list[list[Message]] = []
    for message in messages:
        if opens_turn(message) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def _pairing(turn: Sequence[Message]) -> dict[tuple[int, int], tuple[int, int]]:
    """Result position -> call position, pairing each result with the earliest
    unanswered call of its id that comes BEFORE it. Positions are
    ``(message index, block index)``."""
    pending: dict[str, list[tuple[int, int]]] = {}
    answered: dict[tuple[int, int], tuple[int, int]] = {}
    for mi, message in enumerate(turn):
        for bi, block in enumerate(message.content_blocks):
            if isinstance(block, ToolUseBlock):
                pending.setdefault(block.tool_call_id, []).append((mi, bi))
            elif isinstance(block, ToolResultBlock):
                calls = pending.get(block.tool_call_id)
                if calls:
                    answered[(mi, bi)] = calls.pop(0)
    return answered


def repair_turn(turn: Sequence[Message]) -> tuple[list[Message], int]:
    """The turn without tool calls that lack a later result or results that lack an earlier call.

    Returns the repaired messages and how many blocks were taken out. A
    message left with no blocks is dropped. Text is never removed.
    """
    answered = _pairing(turn)
    kept_calls = set(answered.values())
    out: list[Message] = []
    removed = 0
    for mi, message in enumerate(turn):
        blocks = []
        for bi, block in enumerate(message.content_blocks):
            if isinstance(block, ToolUseBlock) and (mi, bi) not in kept_calls:
                continue
            if isinstance(block, ToolResultBlock) and (mi, bi) not in answered:
                continue
            blocks.append(block)
        dropped = len(message.content_blocks) - len(blocks)
        if not dropped:
            out.append(message)
            continue
        removed += dropped
        if blocks:
            out.append(message.model_copy(update={"content_blocks": tuple(blocks)}))
    return out, removed


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + _ELLIPSIS


def _images_note(result: ToolResultBlock) -> str:
    kinds = ", ".join(sorted({image.mime_type for image in result.images}))
    count = len(result.images)
    return f"[{count} image{'s' if count != 1 else ''} ({kinds}) from this output not carried.]"


def _placeholder(
    call: ToolUseBlock | None,
    result: ToolResultBlock,
    kept: Sequence[str],
    preview_chars: int,
) -> str:
    content = result.content
    name = call.name if call is not None else "a tool"
    error = "; it reported an error" if result.is_error else ""
    lines = [f"[Output of {name} from an earlier turn, {len(content)} characters, not in this context{error}.]"]
    if kept:
        lines.append("Lines kept from it:")
        lines.extend(_clip(line, _KEPT_LINE_MAX_CHARS) for line in kept[:_KEPT_LINES_MAX])
        if len(kept) > _KEPT_LINES_MAX:
            lines.append(f"(+{len(kept) - _KEPT_LINES_MAX} more such lines not kept)")
    if preview_chars > 0 and content.strip():
        head = content[:preview_chars]
        lines.append(f"It began with {len(head)} characters:")
        lines.append(head + (_ELLIPSIS if len(head) < len(content) else ""))
    return "\n".join(lines)


class _NotFinite(ValueError):
    """A number JSON cannot carry (``NaN``, ``Infinity``, or a float past the range)."""


def _finite_float(token: str) -> float:
    value = float(token)
    if value in (float("inf"), float("-inf")):
        raise _NotFinite(token)
    return value


def _reject_constant(token: str) -> Any:
    raise _NotFinite(token)


CUT_ARGUMENT_SUFFIX: Final[str] = " characters, not carried]"
"""How a cut tool-call argument ends in a masked turn: ``<head>… [N characters, not carried]``."""

CUT_LIST_SUFFIX: Final[str] = " more items, not carried]"
"""How a cut list argument ends in a masked turn: its last item is ``[N more items, not carried]``."""


def _marked_count(text: str, suffix: str) -> bool:
    end = text.find(suffix)
    while end >= 0:
        start = text.rfind("[", 0, end)
        number = text[start + 1 : end] if start >= 0 else ""
        if number and number.isascii() and number.isdigit():
            return True
        end = text.find(suffix, end + 1)
    return False


def has_cut_marker(text: str) -> bool:
    """Whether ``text`` holds the marker a masked turn leaves in a cut argument.

    For a host's tool layer: a model that copies the shape of its own masked
    call into a new one would write the marker into a file. A tool can refuse
    an argument for which this is true.
    """
    return _marked_count(text, CUT_ARGUMENT_SUFFIX) or _marked_count(text, CUT_LIST_SUFFIX)


def _head(text: str, chars: int) -> str:
    """The first ``chars`` characters."""
    return text[:chars]


def _mask_args(call: ToolUseBlock, rc: LoopConstants) -> ToolUseBlock:
    """``call`` with its long arguments cut to a head — always valid JSON, never longer.

    In the parsed arguments: a string longer than
    ``session_seed_mask_arg_max_chars`` becomes its first
    ``session_seed_mask_arg_preview_chars`` characters and the count of the
    rest; a key that long is cut the same way (unless that would make two
    keys equal); a list whose JSON is that long keeps its leading items up to
    the preview and ends with one string counting the items not carried.
    Shorter values (paths, ids) stay whole. Arguments that are not JSON, that
    hold a number JSON cannot carry, or that masking would not make shorter
    are returned unchanged, and so are arguments whose cut form would hold
    a lone surrogate (one the original carried escaped). Output is compact
    JSON, so number spelling may change (``1E2`` becomes ``100.0``) but
    nothing else does. Cut values end in :data:`CUT_ARGUMENT_SUFFIX` and
    :data:`CUT_LIST_SUFFIX` (:func:`has_cut_marker`).
    """
    limit = rc.session_seed_mask_arg_max_chars
    raw = call.arguments_json
    if len(raw) <= limit:
        return call
    try:
        parsed = json.loads(raw, parse_float=_finite_float, parse_constant=_reject_constant)
    except (_NotFinite, json.JSONDecodeError, TypeError, ValueError):
        return call
    preview = rc.session_seed_mask_arg_preview_chars

    def cut(text: str) -> str:
        return f"{_head(text, preview)}{_ELLIPSIS} [{len(text)}{CUT_ARGUMENT_SUFFIX}"

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return cut(value) if len(value) > limit else value
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                new_key = cut(key) if len(key) > limit else key
                if new_key in out or (new_key != key and new_key in value):
                    new_key = key
                out[new_key] = walk(item)
            return out
        if isinstance(value, list):
            items = [walk(item) for item in value]
            if len(json.dumps(items, ensure_ascii=False, separators=(",", ":"))) <= limit:
                return items
            kept: list[Any] = []
            used = 0
            for item in items:
                size = len(json.dumps(item, ensure_ascii=False, separators=(",", ":"))) + 1
                if kept and used + size > preview:
                    break
                kept.append(item)
                used += size
            if len(kept) < len(items):
                kept.append(f"[{len(items) - len(kept)}{CUT_LIST_SUFFIX}")
            return kept
        return value

    text = json.dumps(walk(parsed), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate the original carried escaped: keep the call as it
        # was rather than emit text no request can encode.
        return call
    if len(text) >= len(raw):
        return call
    return call.model_copy(update={"arguments_json": text})


def _strip_reasoning(message: Message) -> Message | None:
    """An assistant message without its reasoning; ``None`` when nothing else is left."""
    blocks = tuple(b for b in message.content_blocks if not isinstance(b, ThinkingBlock))
    if (
        len(blocks) == len(message.content_blocks)
        and message.reasoning_content is None
        and message.reasoning_items is None
    ):
        return message
    if not blocks:
        return None
    return message.model_copy(
        update={"content_blocks": blocks, "reasoning_content": None, "reasoning_items": None}
    )


def _mask(
    turn: Sequence[Message],
    rc: LoopConstants,
    rule: MaskingRule,
    *,
    preview: int,
    whole_turn: bool,
    kept_memo: dict[tuple[int, int], list[str]],
) -> list[tuple[int, Message]]:
    """The masked (``whole_turn``) or cut (only the result text) form of a repaired turn.

    Each message comes with the index of the message of ``turn`` it stands for
    (a message that is only reasoning is left out).
    """
    answered = _pairing(turn)
    out: list[tuple[int, Message]] = []
    for mi, message in enumerate(turn):
        if message.role is MessageRole.assistant and whole_turn:
            stripped = _strip_reasoning(message)
            if stripped is None:
                continue
            blocks_a = tuple(
                _mask_args(b, rc) if isinstance(b, ToolUseBlock) else b for b in stripped.content_blocks
            )
            if any(new is not old for new, old in zip(blocks_a, stripped.content_blocks, strict=True)):
                stripped = stripped.model_copy(update={"content_blocks": blocks_a})
            out.append((mi, stripped))
            continue
        if message.role is not MessageRole.tool:
            out.append((mi, message))
            continue
        blocks: list[Any] = []
        changed = False
        for bi, block in enumerate(message.content_blocks):
            if not isinstance(block, ToolResultBlock):
                blocks.append(block)
                continue
            call_at = answered.get((mi, bi))
            call_block = turn[call_at[0]].content_blocks[call_at[1]] if call_at else None
            call = call_block if isinstance(call_block, ToolUseBlock) else None
            if rule.keep_whole(call, block) or is_compacted_placeholder(block.content):
                blocks.append(block)
                continue
            content = block.content
            if len(content) > preview and len(content) >= rc.session_seed_mask_min_chars:
                kept = kept_memo.get((mi, bi))
                if kept is None:
                    kept = kept_memo[(mi, bi)] = list(rule.kept_lines(call, block))
                candidate = _placeholder(call, block, kept, preview)
                if len(candidate) < len(content):
                    content = candidate
            images = block.images
            if whole_turn and images:
                content = f"{content}\n{_images_note(block)}"
                images = ()
            if content is block.content and images is block.images:
                blocks.append(block)
                continue
            blocks.append(
                ToolResultBlock(
                    tool_call_id=block.tool_call_id,
                    content=content,
                    is_error=block.is_error,
                    metadata=dict(block.metadata),
                    canonical_ref=block.canonical_ref,
                    path=block.path,
                    images=images,
                )
            )
            changed = True
        out.append((mi, message.model_copy(update={"content_blocks": tuple(blocks)}) if changed else message))
    return out


def mask_turn(
    turn: Sequence[Message],
    rc: LoopConstants,
    rule: MaskingRule,
    *,
    preview_chars: int | None = None,
) -> list[Message]:
    """The masked form of a (repaired) turn — PURE. See the module docstring.

    User messages are returned as they are; a tool result keeps its call id,
    so every call still has its result. A result is left whole when it is no
    longer than the preview, shorter than ``session_seed_mask_min_chars``,
    already a compaction placeholder, one the rule keeps whole, or when the
    placeholder would not be shorter. ``preview_chars`` overrides
    ``session_seed_mask_preview_chars``.
    """
    preview = rc.session_seed_mask_preview_chars if preview_chars is None else preview_chars
    return [message for _, message in _mask(turn, rc, rule, preview=preview, whole_turn=True, kept_memo={})]


# ---------------------------------------------------------------------------
# Preparation, shared by the seed and the plan, and its cache
# ---------------------------------------------------------------------------


def _encodable_value(value: Any) -> Any:
    if isinstance(value, str):
        return encodable(value)
    if isinstance(value, dict):
        return {encodable(k) if isinstance(k, str) else k: _encodable_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encodable_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_encodable_value(v) for v in value)
    return value


def _dump(message: Message) -> tuple[Message, str]:
    """``message`` (made encodable when it was not) and its JSON without the timestamp.

    A message holding a lone surrogate cannot be serialised for any request.
    It is rewritten with each such code point as its ``\\udXXX`` escape — the
    same code point, spelled so it encodes — and every later step works on
    the rewritten message.
    """
    try:
        return message, message.model_dump_json(exclude={"created_at"})
    except PydanticSerializationError:
        data = _encodable_value(message.model_dump(mode="python", exclude={"created_at"}))
        fixed = Message.model_validate({**data, "created_at": message.created_at})
        return fixed, fixed.model_dump_json(exclude={"created_at"})


@dataclass(frozen=True, slots=True)
class _Prepared:
    """What masking and estimating one turn produced, independent of message identity.

    ``masked`` lists, per message of the masked form, the index of the
    repaired message it comes from and the replacement (``None`` when the
    message is carried as it is). ``dropped`` are the repaired messages the
    masked form leaves out. Materialising against the turn being seeded keeps
    each message's own timestamp.
    """

    masked: tuple[tuple[int, Message | None], ...]
    dropped: tuple[int, ...]
    full_tokens: int
    masked_tokens: int
    text_tokens: int
    repaired_blocks: int
    longest_result: int


@dataclass(frozen=True, slots=True)
class _Turn:
    full: tuple[Message, ...]
    masked: tuple[Message, ...]
    masked_of: tuple[int, ...]
    """For each message of ``masked``, the index in ``full`` it stands for."""
    dropped: tuple[int, ...]
    text: tuple[Message, ...]
    """The turn reduced to its words: user messages and assistant text."""
    full_tokens: int
    masked_tokens: int
    text_tokens: int
    repaired_blocks: int
    longest_result: int
    user_turns: int


def _materialise(full: Sequence[Message], prepared: _Prepared) -> tuple[tuple[Message, ...], tuple[int, ...]]:
    out: list[Message] = []
    of: list[int] = []
    for index, replacement in prepared.masked:
        source = full[index]
        if replacement is None:
            out.append(source)
        elif replacement.metadata == source.metadata and replacement.created_at == source.created_at:
            out.append(replacement)
        else:
            out.append(replacement.model_copy(update={"metadata": dict(source.metadata), "created_at": source.created_at}))
        of.append(index)
    return tuple(out), tuple(of)


def _text_only(full: Sequence[Message]) -> tuple[Message, ...]:
    """The turn's words: user messages as they are, assistant text without calls or reasoning."""
    out: list[Message] = []
    for message in full:
        if message.role is MessageRole.user and not any(isinstance(b, ToolResultBlock) for b in message.content_blocks):
            out.append(message)
        elif message.role is MessageRole.assistant:
            blocks = tuple(b for b in message.content_blocks if isinstance(b, TextBlock) and b.text)
            if blocks:
                out.append(
                    message.model_copy(update={"content_blocks": blocks, "reasoning_content": None, "reasoning_items": None})
                )
    return tuple(out)


class SeedCache:
    """Prepared turns of one session, kept between calls — owned by the host.

    Keyed by a digest of the turn's whole content — every message with every
    block, its images and its metadata (all of which can change its size or
    its bytes); only the timestamp is left out — by the masking rule's
    :attr:`MaskingRule.key`, and by the constants preparation reads. A
    changed message, rule or relevant setting is simply not found; an
    unrelated setting changes nothing. A rule with an empty key is never
    cached.

    Holding one per session makes a run's seed and fold plan cost
    proportional to the turns added since the previous call. Each entry holds
    the masked copies of one turn's changed messages (placeholders, cut
    arguments — small) and a few numbers; the full messages are not kept.
    Bounded by ``max_entries`` (least recently used dropped first).
    Thread-safe: one session's seed and fold plan may run at the same time on
    different threads.
    """

    def __init__(self, *, max_entries: int = MAX_SESSION_SEED_CACHE_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _Prepared] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _get(self, key: str) -> _Prepared | None:
        with self._lock:
            found = self._entries.get(key)
            if found is not None:
                self._entries.move_to_end(key)
            return found

    def _put(self, key: str, value: _Prepared) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


def _settings_key(rc: LoopConstants, rule: MaskingRule) -> str:
    values = [getattr(rc, name) for name in _PREPARATION_FIELDS]
    return json.dumps([rule.key, values])


def _tokens(messages: Sequence[Message], rc: LoopConstants) -> int:
    return sum(estimate_message_tokens(m, rc) for m in messages)


class _Preparer:
    """Prepares turns for one call, estimating every message at most once."""

    def __init__(self, rc: LoopConstants, rule: MaskingRule, cache: SeedCache | None) -> None:
        self.rc = rc
        self.rule = rule
        self.cache = cache if rule.key else None
        self.settings = _settings_key(rc, rule)

    def prepare(self, turn: Sequence[Message]) -> _Turn:
        digest = hashlib.sha256(self.settings.encode())
        safe: list[Message] = []
        for message in turn:
            fixed, dumped = _dump(message)
            safe.append(fixed)
            digest.update(b"\x00")
            digest.update(dumped.encode("utf-8"))
        full, removed = repair_turn(safe)
        key = digest.hexdigest()
        prepared = self.cache._get(key) if self.cache is not None else None
        if prepared is None:
            prepared = self._prepare(full, removed)
            if self.cache is not None:
                self.cache._put(key, prepared)
        masked, masked_of = _materialise(full, prepared)
        return _Turn(
            full=tuple(full),
            masked=masked,
            masked_of=masked_of,
            dropped=prepared.dropped,
            text=_text_only(full),
            full_tokens=prepared.full_tokens,
            masked_tokens=prepared.masked_tokens,
            text_tokens=prepared.text_tokens,
            repaired_blocks=prepared.repaired_blocks,
            longest_result=prepared.longest_result,
            user_turns=sum(1 for m in full if opens_turn(m)),
        )

    def _prepare(self, full: list[Message], removed: int) -> _Prepared:
        masked = _mask(
            full,
            self.rc,
            self.rule,
            preview=self.rc.session_seed_mask_preview_chars,
            whole_turn=True,
            kept_memo={},
        )
        weights = [estimate_message_tokens(m, self.rc) for m in full]
        entries: list[tuple[int, Message | None]] = []
        masked_tokens = 0
        for index, message in masked:
            if message is full[index]:
                entries.append((index, None))
                masked_tokens += weights[index]
            else:
                entries.append((index, message))
                masked_tokens += estimate_message_tokens(message, self.rc)
        kept = {index for index, _ in masked}
        text_tokens = 0
        for index, message in enumerate(full):
            words = _text_only([message])
            if not words:
                continue
            text_tokens += weights[index] if words[0] is message else estimate_message_tokens(words[0], self.rc)
        longest = max(
            (len(b.content) for m in full for b in m.content_blocks if isinstance(b, ToolResultBlock)),
            default=0,
        )
        return _Prepared(
            masked=tuple(entries),
            dropped=tuple(i for i in range(len(full)) if i not in kept),
            full_tokens=sum(weights),
            masked_tokens=masked_tokens,
            text_tokens=text_tokens,
            repaired_blocks=removed,
            longest_result=longest,
        )


def _tag_seeded(message: Message) -> Message:
    """Tag a carried prior-run message as seeded: finalisation does not persist
    it again, compaction keeps it apart from this run's turns."""
    if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True:
        return message
    return message.model_copy(
        update={"metadata": {**message.metadata, SESSION_HISTORY_SEED_METADATA_KEY: True}}
    )


def _tag_masked_form(turn: _Turn, carried: Sequence[Message]) -> list[Message]:
    """Tag each message of an unmasked newest turn with the form the next seed carries it in."""
    masked_by_index = dict(zip(turn.masked_of, turn.masked, strict=True))
    out: list[Message] = []
    for index, message in enumerate(carried):
        if index in turn.dropped:
            form: dict[str, Any] = {"drop": True, "message": None}
        else:
            replacement = masked_by_index.get(index)
            if replacement is None or replacement is message:
                out.append(message)
                continue
            form = {
                "drop": False,
                "message": replacement.model_dump(mode="json", exclude={"metadata", "created_at"}),
            }
        out.append(
            message.model_copy(update={"metadata": {**message.metadata, SESSION_SEED_MASKED_FORM_METADATA_KEY: form}})
        )
    return out


def _tag_seeded_reference(message: Message) -> Message:
    """Tag a synthetic seed block (summary, notice, registry) as seeded AND as a reference block.

    The seed tag keeps it out of re-persistence and out of Tier-2 collapse;
    the reference tag opens Tier-1's reference-shed path, so a large block can
    be moved to a recoverable snapshot under budget pressure instead of being
    immovable in the window. Carried turns are seed-tagged only.
    """
    return message.model_copy(
        update={
            "metadata": {
                **message.metadata,
                SESSION_HISTORY_SEED_METADATA_KEY: True,
                COMPACTION_REFERENCE_METADATA_KEY: True,
            }
        }
    )


def _block(text: str) -> list[Message]:
    return [_tag_seeded_reference(Message(role=MessageRole.user, content_blocks=[TextBlock(text=encodable(text))]))]


def _summary_body(text: str, pinned: int) -> str:
    note = (
        f"\n({pinned} earlier turn{'s' if pinned != 1 else ''} could not be summarised and "
        f"{'are' if pinned != 1 else 'is'} carried below in short form.)"
        if pinned
        else ""
    )
    return (
        "[Running summary of the earlier turns of this conversation that are "
        f"not carried below]\n{text}{note}\n\n{END_OF_SUMMARY_MARKER}"
    )


_NOTHING_RECORDED: Final[str] = "(the summary holds no entries for those turns)"


def _summary_message(memory: SessionMemory, summarized_turns: int, pinned_turns: int) -> list[Message]:
    text = memory.summary.render()
    if not text and summarized_turns <= 0 and pinned_turns <= 0:
        return []
    return _block(_summary_body(text or _NOTHING_RECORDED, pinned_turns))


def _summary_reserve(memory: SessionMemory, rc: LoopConstants) -> int:
    """What the planner keeps for the summary block: the cap plus its wrapper."""
    cap = rc.session_memory_running_summary_token_cap
    if cap <= 0:
        return _tokens(_summary_message(memory, 1, len(memory.summary.pinned)), rc)
    wrapper = Message(role=MessageRole.user, content_blocks=[TextBlock(text=_summary_body("", len(memory.summary.pinned)))])
    return cap + estimate_message_tokens(wrapper, rc)


def _notice_message(dropped: int, registry: bool, summary: bool, words_only: bool) -> list[Message]:
    parts: list[str] = []
    if dropped:
        parts.append(
            f"{dropped} earlier turn{'s' if dropped != 1 else ''} of this conversation "
            f"{'are' if dropped != 1 else 'is'} not in this context and not in the summary."
        )
    if words_only:
        parts.append("The tool calls and results of the latest earlier turn are not in this context.")
    if summary:
        parts.append("The summary of earlier turns is not in this context.")
    if registry:
        parts.append("The registry of files written in this conversation is not in this context.")
    return _block(f"[{' '.join(parts)}]") if parts else []


def _ledger_message(memory: SessionMemory, rc: LoopConstants, max_tokens: int | None = None) -> list[Message]:
    text = render_ledger(memory.ledger, rc, max_tokens=max_tokens)
    return _block(text) if text else []


def _known_positions(turns: Sequence[list[Message]], position: MessagePosition) -> list[list[int]]:
    """Each turn's known positions — none for a turn that breaks the contract.

    Positions must increase strictly within a turn and lie after every
    position of the turns before it. A turn that breaks that (a reordered
    row, a moved boundary) is treated as having no known position: it is
    carried, never recorded, and never lets a range span what it cannot
    account for.
    """
    out: list[list[int]] = []
    last = -1
    for turn in turns:
        known = [p for m in turn if (p := position(m)) is not None]
        if any(b <= a for a, b in itertools.pairwise(known)) or (known and known[0] <= last):
            out.append([])
            continue
        if known:
            last = known[-1]
        out.append(known)
    return out


@dataclass(frozen=True, slots=True)
class _Layout:
    turns: list[list[Message]]
    fragment: int
    head: int
    known: list[list[int]]
    recorded: list[bool]
    pinned: list[bool]
    pre: list[int]
    """Carried turns placed before the summary: the head, and unrecorded
    turns older than the first recorded one."""
    window: list[int]
    summarized: int
    prepared: dict[int, _Turn]
    head_question: Message | None
    """The session's first user message, protected on its own when the first
    turn is too large to be a protected head turn."""


def _layout(
    history: Sequence[Message], memory: SessionMemory, position: MessagePosition, prep: _Preparer
) -> _Layout:
    turns = split_turns(history)
    fragment = 0
    if turns and not opens_turn(turns[0][0]):
        fragment = len(turns.pop(0))
    rc = prep.rc
    prepared: dict[int, _Turn] = {}
    head = 0
    used = 0
    for index, turn in enumerate(turns[: rc.session_memory_head_protect_turns]):
        prepared[index] = prep.prepare(turn)
        if used + prepared[index].masked_tokens > rc.session_memory_head_max_tokens:
            break
        used += prepared[index].masked_tokens
        head += 1
    known = _known_positions(turns, position)
    summary = memory.summary
    recorded = [bool(k) and all(summary.covers(p) for p in k) for k in known]
    pinned = [bool(k) and not recorded[i] and all(summary.is_pinned(p) for p in k) for i, k in enumerate(known)]
    carried = [i for i in range(len(turns)) if i < head or not recorded[i]]
    first_recorded = next((i for i in range(head, len(turns)) if recorded[i] or pinned[i]), None)
    if first_recorded is None:
        pre = [i for i in carried if i < head]
    else:
        pre = [i for i in carried if i < first_recorded]
    pre_set = set(pre)
    window = [i for i in carried if i not in pre_set]
    for index in carried:
        if index not in prepared:
            prepared[index] = prep.prepare(turns[index])
    summarized = sum(1 for i in range(len(turns)) if recorded[i] and i >= head)
    head_question = turns[0][0] if turns and head == 0 and rc.session_memory_head_protect_turns > 0 else None
    return _Layout(turns, fragment, head, known, recorded, pinned, pre, window, summarized, prepared, head_question)


def mask_boundary(turn_count: int, rc: LoopConstants) -> int:
    """Index of the first turn carried in full; older turns are masked.

    Moves in steps of ``session_seed_mask_batch_turns``, so between
    ``session_seed_full_turns`` and that plus the step minus one turns are
    full.
    """
    keep = rc.session_seed_full_turns
    step = max(1, rc.session_seed_mask_batch_turns)
    if turn_count <= keep:
        return 0
    return ((turn_count - keep) // step) * step


# ---------------------------------------------------------------------------
# The seed budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeedBudget:
    """How a run's context divides between its overhead, the seed and its own work."""

    seed_tokens: int
    """The budget to pass to :func:`build_session_seed` and :func:`plan_session_fold`."""
    run_reserve_tokens: int
    """Room left between the seed and the compaction trigger for the run's
    own work, before what re-masking the seeded newest turn hands back
    (negative when the floor had to take more than the room)."""
    trigger_tokens: int
    overhead_tokens: int
    floor_tokens: int = 0
    """The least the seed is given when an earlier turn exists: the newest
    earlier turn's question and answer, and the notice that its tool work is
    not shown."""


def seed_floor_tokens(history: Sequence[Message], rc: LoopConstants) -> int:
    """The seed size below which a follow-up cannot see the previous answer — PURE.

    The newest earlier turn reduced to its words (its user message and the
    assistant's text) plus the one-line notice that its tool work is not in
    context: the smallest form :func:`build_session_seed` carries it in. 0
    when there is no earlier turn.
    """
    turns = split_turns(history)
    if not turns or not opens_turn(turns[-1][0]):
        return 0
    full, _ = repair_turn(turns[-1])
    words = _text_only([_dump(m)[0] for m in full])
    if not words:
        return 0
    return _tokens(words, rc) + _tokens(_notice_message(0, False, False, True), rc)


def recommend_seed_budget(
    rc: LoopConstants, overhead_tokens: int, *, history: Sequence[Message] = ()
) -> SeedBudget:
    """The seed budget for a run, from the model's window and the run's measured overhead — PURE.

    ``overhead_tokens`` is what the run's request carries besides the seed:
    the system prompt, tool definitions, preludes and the new task (a
    negative value counts as 0). The room up to the compaction trigger
    (``derive_budgets(rc).compaction_trigger_tokens``) is split:
    ``session_seed_budget_ratio`` of it to the seed, the rest — at least
    ``session_seed_run_reserve_min_tokens``, or half the room when the room is
    less than twice that — to the run. The seed's newest full turns give their
    full-minus-masked size back the moment in-run compaction runs (see
    :func:`~protocore.runtime.context.compaction.remask_seeded_turns`), so a
    run that does heavy new tool work has that room too.

    With ``history`` (the session's earlier messages), the seed is never
    smaller than :func:`seed_floor_tokens` — the previous question and answer
    — as long as that fits the window beside the overhead: on a tiny window,
    or when the overhead alone reaches the trigger, the floor is what is
    given, and the run starts over its trigger (in-run compaction then works
    on it). Only when not even the floor fits beside the overhead in the
    window is the seed smaller, down to 0.

    Pass the same number to :func:`plan_session_fold` as the base budget;
    pass a smaller one to :func:`build_session_seed` when a run's overhead is
    larger than the one the plan assumed.
    """
    budgets = derive_budgets(rc)
    trigger = budgets.compaction_trigger_tokens
    overhead = max(0, overhead_tokens)
    room = max(0, trigger - overhead)
    reserve = min(rc.session_seed_run_reserve_min_tokens, room // 2)
    seed = min(int(room * rc.session_seed_budget_ratio), room - reserve)
    floor = seed_floor_tokens(history, rc) if history else 0
    if floor > seed:
        seed = min(floor, max(0, budgets.max_context - overhead))
    return SeedBudget(
        seed_tokens=seed,
        run_reserve_tokens=room - seed,
        trigger_tokens=trigger,
        overhead_tokens=overhead,
        floor_tokens=floor,
    )


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionSeed:
    """A seed, and the counts a host logs about it."""

    messages: list[Message] = field(default_factory=list)
    tokens: int = 0
    head_turns: int = 0
    summarized_turns: int = 0
    """Turns the summary records and the seed does not carry."""
    pinned_turns: int = 0
    """Turns the summariser could not fold, carried masked inside the summary's range."""
    window_turns: int = 0
    """Carried turns after the summary (the head and older unrecorded turns
    before it are not counted here)."""
    full_turns: int = 0
    trimmed_turns: int = 0
    """Newest turns carried with result text cut to fit rather than masked."""
    masked_turns: int = 0
    words_only_turns: int = 0
    """The newest turn reduced to its words (question and answer) because
    nothing else of it fit; the notice says so."""
    dropped_turns: int = 0
    """Unrecorded turns left out to fit the budget; the notice says how many."""
    registry_reduced: bool = False
    """The registry was shown at less than its bound to make room."""
    dropped_summary: bool = False
    dropped_registry: bool = False
    repaired_blocks: int = 0
    """Unpaired tool calls / results taken out of carried turns."""
    fragment_messages: int = 0
    """Messages before the session's first user message, not carried."""
    reclaimable_tokens: int = 0
    """What in-run compaction gets back by re-masking the unmasked newest turns."""
    over_budget_tokens: int = 0
    """Above zero only when the budget cannot hold even the notice."""


def _fit_preview(
    turn: _Turn, room: int, rc: LoopConstants, rule: MaskingRule
) -> tuple[tuple[Message, ...], int] | None:
    """The newest turn with result text cut to the longest head that fits ``room``.

    Only result text is cut: images, reasoning and arguments stay. ``None``
    when not even the shortest head fits.
    """
    low = rc.session_seed_mask_preview_chars
    if turn.longest_result <= low:
        return None
    memo: dict[tuple[int, int], list[str]] = {}
    weights = {id(m): estimate_message_tokens(m, rc) for m in turn.full}

    def weight(messages: Sequence[Message]) -> int:
        return sum(weights[id(m)] if id(m) in weights else estimate_message_tokens(m, rc) for m in messages)

    best: tuple[tuple[Message, ...], int] | None = None
    high = turn.longest_result - 1
    while low <= high:
        mid = (low + high) // 2
        cut = tuple(m for _, m in _mask(turn.full, rc, rule, preview=mid, whole_turn=False, kept_memo=memo))
        cost = weight(cut)
        if cost <= room:
            best = (cut, cost)
            low = mid + 1
        else:
            high = mid - 1
    return best


def _ensure_encodable(messages: Sequence[Message]) -> list[Message]:
    return [_dump(m)[0] for m in messages]


def build_session_seed(
    history: Sequence[Message],
    memory: SessionMemory,
    budget: int,
    rc: LoopConstants,
    *,
    position: MessagePosition,
    rule: MaskingRule | None = None,
    cache: SeedCache | None = None,
) -> SessionSeed:
    """The seed to put in front of a run's new task — PURE, CPU-bound.

    ``history`` is the session's earlier messages as stored, oldest first,
    without the new run's. ``position`` gives each its session position (see
    "Positions" in the module docstring). ``rule`` is the masking rule
    (:func:`default_masking_rule` when ``None``); ``cache`` the session's
    :class:`SeedCache`. ``budget`` is the token budget of the seed
    (:func:`recommend_seed_budget`).

    The seed's estimated size never exceeds ``budget`` (unless the budget
    cannot hold even the notice), and every message of it encodes as UTF-8.
    When the budget cannot hold everything, content gives way in this order:

    1. older full turns are masked, oldest first;
    2. the registry shrinks within its bound (to its paths, then to a count);
    3. the newest turn's result text is cut to the longest head that fits;
    4. the newest turn is masked;
    5. window turns are left out, oldest first;
    6. head turns are left out;
    7. the registry is left out;
    8. the summary is left out;
    8a. the session's first question, carried on its own when the first turn
        is too large to be a protected head turn, is left out;
    9. the newest turn is reduced to its words (question and answer);
    10. the newest turn is left out.

    So the previous answer outranks the registry, the summary and every
    older turn: a follow-up such as "and the second one?" always sees it
    when anything of the conversation fits. One notice says what is missing.
    Messages of an unmasked newest turn carry
    :data:`SESSION_SEED_MASKED_FORM_METADATA_KEY`, so in-run compaction puts
    them in their masked form first. An empty memory and an empty history give
    an empty seed, so a session's first run is exactly a cold start.
    """
    rule = rule or default_masking_rule(rc)
    prep = _Preparer(rc, rule, cache)
    layout = _layout(history, memory, position, prep)
    turns, prepared = layout.turns, layout.prepared
    if not turns and memory.is_empty():
        return SessionSeed(fragment_messages=layout.fragment)

    boundary = mask_boundary(len(turns), rc)
    carried = [*layout.pre, *layout.window]
    forms: dict[int, tuple[Message, ...]] = {}
    weights: dict[int, int] = {}
    state: dict[int, str] = {}

    def mask(index: int) -> None:
        forms[index], weights[index], state[index] = (
            prepared[index].masked,
            prepared[index].masked_tokens,
            "masked",
        )

    for index in carried:
        if index >= boundary and not layout.pinned[index]:
            forms[index], weights[index], state[index] = prepared[index].full, prepared[index].full_tokens, "full"
        else:
            mask(index)

    pinned_count = sum(1 for i in carried if layout.pinned[i])
    summary = _summary_message(memory, layout.summarized, pinned_count)
    ledger = _ledger_message(memory, rc)
    dropped: list[int] = []
    flags = {"summary": False, "registry": False, "words": False, "reduced": False, "question": False}

    def question() -> list[Message]:
        if layout.head_question is None or flags["question"] or 0 in carried:
            return []
        return [layout.head_question]

    block_cost: dict[tuple[str, ...], int] = {}

    def cost(blocks: list[Message]) -> int:
        key = tuple(m.text for m in blocks)
        if key not in block_cost:
            block_cost[key] = _tokens(blocks, rc)
        return block_cost[key]

    def notice() -> list[Message]:
        return _notice_message(len(dropped), flags["registry"], flags["summary"], flags["words"])

    def total() -> int:
        fixed_blocks = (0 if flags["summary"] else cost(summary)) + (0 if flags["registry"] else cost(ledger))
        return fixed_blocks + cost(notice()) + cost(question()) + sum(weights[i] for i in carried)

    full_indices = [i for i in carried if state[i] == "full"]
    for index in full_indices[:-1]:
        if total() <= budget:
            break
        mask(index)
    if ledger and total() > budget:
        room = budget - (total() - cost(ledger))
        shrunk = _ledger_message(memory, rc, max_tokens=max(0, min(room, rc.session_memory_registry_max_tokens)))
        if not shrunk:
            shrunk = _block(render_ledger_count(memory.ledger))
        if shrunk != ledger:
            ledger = shrunk
            flags["reduced"] = True
    newest = max(carried) if carried else None
    if newest is not None and state[newest] == "full" and total() > budget:
        room = budget - (total() - weights[newest])
        fitted = _fit_preview(prepared[newest], room, rc, rule)
        if fitted is not None:
            forms[newest], weights[newest] = fitted
            state[newest] = "trimmed"
        else:
            mask(newest)
    if newest is not None and total() > budget:
        order = [i for i in layout.window if i != newest] + [i for i in layout.pre if i != newest]
        for index in order:
            if total() <= budget:
                break
            carried.remove(index)
            dropped.append(index)
    if total() > budget and ledger:
        flags["registry"] = True
    if total() > budget and summary:
        flags["summary"] = True
    if total() > budget:
        flags["question"] = True
    if newest is not None and total() > budget and prepared[newest].text:
        forms[newest], weights[newest], state[newest] = prepared[newest].text, prepared[newest].text_tokens, "words"
        flags["words"] = True
    if newest is not None and total() > budget:
        carried.remove(newest)
        dropped.append(newest)
        flags["words"] = False

    kept = set(carried)
    reclaimable = 0
    rendered: dict[int, list[Message]] = {}
    for index in carried:
        if state[index] in ("full", "trimmed"):
            rendered[index] = _tag_masked_form(prepared[index], forms[index])
            reclaimable += weights[index] - prepared[index].masked_tokens
        else:
            rendered[index] = list(forms[index])
    messages = _ensure_encodable(
        [_tag_seeded(m) for m in question()]
        + [_tag_seeded(m) for i in layout.pre if i in kept for m in rendered[i]]
        + ([] if flags["summary"] else summary)
        + notice()
        + [_tag_seeded(m) for i in layout.window if i in kept for m in rendered[i]]
        + ([] if flags["registry"] else ledger)
    )
    tokens = total()
    kinds = [state[i] for i in carried]
    return SessionSeed(
        messages=messages,
        tokens=tokens,
        head_turns=layout.head,
        summarized_turns=layout.summarized,
        pinned_turns=sum(1 for i in carried if layout.pinned[i]),
        window_turns=sum(1 for i in layout.window if i in kept),
        full_turns=kinds.count("full"),
        trimmed_turns=kinds.count("trimmed"),
        masked_turns=kinds.count("masked"),
        words_only_turns=kinds.count("words"),
        dropped_turns=len(dropped),
        registry_reduced=flags["reduced"] and not flags["registry"],
        dropped_summary=flags["summary"],
        dropped_registry=flags["registry"],
        repaired_blocks=sum(prepared[i].repaired_blocks for i in carried),
        fragment_messages=layout.fragment,
        reclaimable_tokens=max(0, reclaimable),
        over_budget_tokens=max(0, tokens - budget),
    )


# ---------------------------------------------------------------------------
# The fold decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldPlan:
    """Which turns a summary fold must record now, and the numbers behind it."""

    turns: tuple[tuple[Message, ...], ...] = ()
    """The turns the step records, in session order, as stored."""
    source: tuple[tuple[Message, ...], ...] = ()
    """The same turns in the masked form the seed carries them in: what the
    summariser is shown (:func:`build_summary_delta_request`)."""
    step: CoverageStep | None = None
    """What to hand :func:`fold_run`: the coverage planned against and the
    coverage after. ``None`` when there is nothing to record."""
    needed: bool = False
    """A summariser call is to be made (False for a pin step, which records
    its turn without one)."""
    pin: bool = False
    """The step pins its single turn: the summariser was cut off on it
    ``session_seed_fold_cuts_before_pin`` times in a row."""
    projected_tokens: int = 0
    """The next seed, as the planner sized it, before the fold."""
    remaining_tokens: int = 0
    """The same after the fold lands."""
    fresh_tokens: int = 0
    """What the planner kept for the newest full turns."""
    newest_counted_full: bool = False
    """The newest turns were counted at their full size (False: masked,
    because counting them in full would fold on every run)."""
    trigger_tokens: int = 0
    goal_tokens: int = 0
    """What the fold brings the seed down to."""
    batch_limit: int = 0
    """Most turns this plan may take: halved for every consecutive fold that
    did not land (``stale_fold_count``)."""
    deferred_turns: int = 0
    """Turns the goal asked for that the batch or source bound left to a later fold."""
    backlog_turns: int = 0
    """Unrecorded foldable turns older than the newest full turns, this plan
    included — the catch-up still owed."""
    coverage_conflict: bool = False
    """The planned range would span a turn it does not record (positions
    that break the contract); no step was made."""

    @property
    def messages(self) -> list[Message]:
        """The planned turns' stored messages in order."""
        return [m for turn in self.turns for m in turn]

    @property
    def source_messages(self) -> list[Message]:
        """The planned turns' masked messages in order — the summariser's input."""
        return [m for turn in self.source for m in turn]


def _fresh_hold(fresh: Sequence[_Turn], fixed: int, budget: int) -> int:
    """What the seed would carry for the full candidates with nothing older in front."""
    weights = [p.full_tokens for p in fresh]
    for index in range(len(fresh) - 1):
        if fixed + sum(weights) <= budget:
            break
        weights[index] = fresh[index].masked_tokens
    if fresh and fixed + sum(weights) > budget:
        weights[-1] = fresh[-1].masked_tokens
    return sum(weights)


def plan_session_fold(
    history: Sequence[Message],
    memory: SessionMemory,
    budget: int,
    rc: LoopConstants,
    *,
    position: MessagePosition,
    rule: MaskingRule | None = None,
    cache: SeedCache | None = None,
    max_source_tokens: int = 0,
    summariser: str = "",
) -> FoldPlan:
    """Which turns need folding into the summary now — PURE, CPU-bound.

    Called after a run, with ``history`` the session's stored messages
    INCLUDING that run's, with their positions — exactly what the next seed
    will be built from — and the same ``position``, ``rule``, ``cache`` and
    base ``budget`` the seed uses. See "When a fold is needed" in the module
    docstring for how the next seed is sized and when the newest turn is
    counted in full.

    The turns planned always join the summary's range without a gap: first
    unrecorded turns just before it (a head whose protection lapsed), nearest
    first, then the oldest unrecorded turns after it, in order; never the
    head, the newest full turns, a pinned turn or a turn without a known
    position. The plan's :attr:`~FoldPlan.step` names the range it was
    planned against, so :func:`fold_run` refuses it on any other summary.

    The batch: at most ``max_source_tokens`` (0 = unbounded) of source as it
    will be rendered — always at least one turn — and at most the wanted
    turns halved once per consecutive fold that did not land
    (``memory.stale_fold_count``), so a summariser that keeps being cut off
    reaches one turn per fold. After ``session_seed_fold_cuts_before_pin``
    cut folds of one and the same turn (``memory.cut_turn``) the plan pins
    exactly that turn (no summariser call); a turn that merely comes first
    after a head lapse or a reset is never pinned for another's cuts. Pins
    are released — their turns planned again, first — when ``summariser``
    (the host's name for its summariser model), the summary cap or the fold
    output bound differ from what they were made under, or after
    ``session_seed_pin_release_runs`` runs.

    Convergence: every needed plan records at least one turn, and the turns
    it leaves stay counted, so a backlog is recorded within
    :attr:`~FoldPlan.backlog_turns` successful folds; packing is greedy and
    stops before a turn that would pass the source bound, so with source
    bound S and largest turn source L < S it is also within
    ``ceil(B / (S - L)) + 1`` folds for a backlog of B source tokens.
    """
    rule = rule or default_masking_rule(rc)
    prep = _Preparer(rc, rule, cache)
    layout = _layout(history, memory, position, prep)
    turns, prepared, head = layout.turns, layout.prepared, layout.head
    count = len(turns)
    boundary = max(mask_boundary(count, rc), count - max(1, rc.session_seed_full_turns))
    carried = [*layout.pre, *layout.window]
    fixed = (
        sum(prepared[i].masked_tokens for i in carried if i < head)
        + _summary_reserve(memory, rc)
        + _tokens(_ledger_message(memory, rc), rc)
        + (_tokens([layout.head_question], rc) if layout.head_question is not None else 0)
    )
    released = bool(memory.summary.pinned) and (
        memory.pin_basis != pin_basis(rc, summariser)
        or memory.turn_index - memory.pinned_at_run >= rc.session_seed_pin_release_runs
    )

    def stuck_pin(index: int) -> bool:
        return layout.pinned[index] and not released

    fresh = [i for i in carried if i >= boundary and i >= head and not layout.pinned[i]]
    fresh_set = set(fresh)
    stuck = [i for i in carried if stuck_pin(i) or (i >= head and i not in fresh_set and not layout.known[i])]
    older = [i for i in carried if i >= head and i not in fresh_set and layout.known[i] and not stuck_pin(i)]
    hold_full = _fresh_hold([prepared[i] for i in fresh], fixed, budget)
    hold_masked = sum(prepared[i].masked_tokens for i in fresh)
    trigger = int(budget * rc.session_seed_fold_trigger_ratio)
    target = int(budget * rc.session_seed_fold_target_ratio)
    # Each run turns the newest turn into an older, masked one; room for
    # that many of them below the trigger is what keeps folds apart.
    spacing = rc.session_seed_fold_min_turns_between * hold_masked
    counted_full = bool(fresh) and fixed + hold_full + spacing <= trigger
    stuck_tokens = sum(prepared[i].masked_tokens for i in stuck)
    if counted_full:
        fresh_tokens, goal = hold_full, max(target, fixed + stuck_tokens + hold_full)
    else:
        fresh_tokens, goal = hold_masked, max(target, fixed + stuck_tokens + hold_masked)
    projected = fixed + stuck_tokens + sum(prepared[i].masked_tokens for i in older) + fresh_tokens
    base_plan = FoldPlan(
        projected_tokens=projected,
        remaining_tokens=projected,
        fresh_tokens=fresh_tokens,
        newest_counted_full=counted_full,
        trigger_tokens=trigger,
        goal_tokens=goal,
        backlog_turns=len(older),
    )
    if projected <= trigger or not older:
        return base_plan
    base = memory.summary.coverage
    inside = [i for i in older if layout.pinned[i]]
    before = [i for i in older if not layout.pinned[i] and base is not None and layout.known[i][-1] < base.first]
    after = [i for i in older if not layout.pinned[i] and i not in set(before)]
    candidates = [*inside, *reversed(before), *after]
    wanted: list[int] = []
    remaining = projected
    for index in candidates:
        if remaining <= goal:
            break
        wanted.append(index)
        remaining -= prepared[index].masked_tokens
    limit = max(1, len(wanted) >> min(memory.stale_fold_count, 30))
    source_bound = max_source_tokens >> min(memory.stale_fold_count, 30) if max_source_tokens > 0 else 0
    chosen: list[int] = []
    source_tokens = 0
    for index in wanted:
        if len(chosen) >= limit:
            break
        weight = fold_source_tokens(prepared[index].masked, rc)
        if chosen and source_bound > 0 and source_tokens + weight > source_bound:
            break
        chosen.append(index)
        source_tokens += weight
    if not chosen:
        return base_plan

    def span(index: int) -> Coverage:
        return Coverage(layout.known[index][0], layout.known[index][-1])

    cut_index = next((i for i in candidates if memory.cut_turn is not None and span(i) == memory.cut_turn), None)
    pin = cut_index is not None and memory.cut_count >= rc.session_seed_fold_cuts_before_pin
    if pin and cut_index is not None:
        chosen = [cut_index]
    positions = [p for i in chosen for p in layout.known[i]]
    after_range = Coverage(
        min(positions) if base is None else min(base.first, *positions),
        max(positions) if base is None else max(base.last, *positions),
    )
    chosen_set = set(chosen)
    for index, known in enumerate(layout.known):
        if index in chosen_set or not known or layout.recorded[index] or layout.pinned[index]:
            continue
        if any(after_range.contains(p) for p in known):
            return replace(base_plan, coverage_conflict=True)
    ordered = sorted(chosen)
    step = CoverageStep(
        base=base,
        after=after_range,
        turns=len(ordered),
        user_turns=sum(prepared[i].user_turns for i in ordered),
        pin=pin,
        pinned=tuple(span(i) for i in ordered) if pin else (),
        spans=tuple(span(i) for i in ordered),
        unpin=() if pin else tuple(span(i) for i in ordered if layout.pinned[i]),
    )
    return FoldPlan(
        turns=tuple(tuple(turns[i]) for i in ordered),
        source=tuple(prepared[i].masked for i in ordered),
        step=step,
        needed=not pin,
        pin=pin,
        projected_tokens=projected,
        remaining_tokens=projected - sum(prepared[i].masked_tokens for i in chosen),
        fresh_tokens=fresh_tokens,
        newest_counted_full=counted_full,
        trigger_tokens=trigger,
        goal_tokens=goal,
        batch_limit=limit,
        deferred_turns=len(wanted) - len(chosen),
        backlog_turns=len(older),
    )


__all__ = [
    "CUT_ARGUMENT_SUFFIX",
    "CUT_LIST_SUFFIX",
    "FoldPlan",
    "KeepWhole",
    "KeptLines",
    "MaskingRule",
    "MessagePosition",
    "SeedBudget",
    "SeedCache",
    "SessionSeed",
    "build_session_seed",
    "default_masking_rule",
    "has_cut_marker",
    "mask_boundary",
    "mask_turn",
    "opens_turn",
    "plan_session_fold",
    "recommend_seed_budget",
    "repair_turn",
    "seed_floor_tokens",
    "split_turns",
]
