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
the previous one may use it at the same time.

Positions
---------

``position(message)`` is the host's session position of a stored message: a
non-negative integer, strictly increasing in session order, the same number
for the same message on every call, or ``None`` when the host does not know
it. Coverage (:class:`~protocore.runtime.context.session_memory.Coverage`) is
a range of these positions. A turn is recorded by the summary when it has at
least one known position and every known position lies in the range; a
message without a position rides along with its turn. A turn none of whose
messages has a position can never be recorded: it is carried while the budget
allows and otherwise counted among the turns not in context, and it does not
stop later turns from being folded.

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
    [window]   every later turn the summary does not record, oldest first:
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

The seed always fits its budget. Under pressure, in order: older full turns
are masked; the newest turn's result text is cut to the longest head that
fits (its images stay); it is masked; then window turns are left out oldest
first, then head turns, then the newest turn, then the registry, then the
summary — and one notice says what is not in context. Turns left out this
way are not recorded anywhere yet; the fold plan, which counts every turn the
summary does not record, folds them in bounded batches over the following
runs (see :func:`plan_session_fold`).

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
* **Leaving turns out.** When the budget cannot hold the masked window, the
  front of the window moves on every run that does it, until the folds catch
  up.
* **A changed history.** A turn edited, regenerated or removed changes the
  seed from that turn on.

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
counted masked instead and the seed cuts it when the window has grown. Once the measured size passes
``session_seed_fold_trigger_ratio`` of the budget, the oldest unrecorded
turns are planned until it is at or below ``session_seed_fold_target_ratio``
(or, with the newest counted in full, at or below what the head, summary,
registry and newest turn need). Otherwise the plan is empty and the host
makes no summariser call.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from protocore.constants import MAX_SESSION_SEED_CACHE_ENTRIES
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    Message,
    MessageOrigin,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import estimate_message_tokens
from protocore.runtime.context.ledger import extract_identifiers
from protocore.runtime.context.session_memory import (
    END_OF_SUMMARY_MARKER,
    Coverage,
    SessionMemory,
    SessionSummary,
    fold_source_tokens,
    render_ledger,
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


def _head(text: str, chars: int) -> str:
    """The first ``chars`` characters, never ending on half of a surrogate pair."""
    head = text[:chars]
    if head and "\ud800" <= head[-1] <= "\udbff":
        head = head[:-1]
    return head


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
    are returned unchanged. Output is compact JSON, so number spelling may
    change (``1E2`` becomes ``100.0``) but nothing else does.
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
        return f"{_head(text, preview)}{_ELLIPSIS} [{len(text)} characters, not carried]"

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
                kept.append(f"[{len(items) - len(kept)} more items, not carried]")
            return kept
        return value

    text = json.dumps(walk(parsed), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
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


@dataclass(frozen=True, slots=True)
class _Prepared:
    """What masking and estimating one turn produced, independent of message identity.

    ``masked`` lists, per message of the masked form, the index of the
    repaired message it comes from and the replacement (``None`` when the
    message is carried as it is). Materialising it against the turn being
    seeded keeps each message's own metadata, so two identical turns never
    share one another's.
    """

    masked: tuple[tuple[int, Message | None], ...]
    full_tokens: int
    masked_tokens: int
    repaired_blocks: int
    longest_result: int


@dataclass(frozen=True, slots=True)
class _Turn:
    full: tuple[Message, ...]
    masked: tuple[Message, ...]
    full_tokens: int
    masked_tokens: int
    repaired_blocks: int
    longest_result: int


def _materialise(full: Sequence[Message], prepared: _Prepared) -> tuple[Message, ...]:
    out: list[Message] = []
    for index, replacement in prepared.masked:
        source = full[index]
        if replacement is None:
            out.append(source)
        elif replacement.metadata == source.metadata and replacement.created_at == source.created_at:
            out.append(replacement)
        else:
            out.append(
                replacement.model_copy(update={"metadata": dict(source.metadata), "created_at": source.created_at})
            )
    return tuple(out)


class SeedCache:
    """Prepared turns of one session, kept between calls — owned by the host.

    Keyed by a digest of the turn's whole content (every message, every block;
    not the messages' metadata or timestamps, which are re-applied from the
    messages being seeded), by the masking rule's :attr:`MaskingRule.key`, and
    by the constants preparation reads — so a changed message, rule or
    relevant setting is simply not found, and an unrelated setting changes
    nothing. A rule with an empty key is never cached.

    Holding one per session makes a run's seed and fold plan cost
    proportional to the turns added since the previous call. Each entry holds
    the masked copies of one turn's changed messages (placeholders, cut
    arguments — small) and four numbers; the full messages are not kept.
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


def _turn_digest(turn: Sequence[Message], settings: str) -> str:
    digest = hashlib.sha256(settings.encode())
    for message in turn:
        digest.update(b"\x00")
        digest.update(message.model_dump_json(exclude={"metadata", "created_at"}).encode())
    return digest.hexdigest()


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
        full, removed = repair_turn(turn)
        key = _turn_digest(turn, self.settings) if self.cache is not None else ""
        prepared = self.cache._get(key) if self.cache is not None else None
        if prepared is None:
            prepared = self._prepare(full, removed)
            if self.cache is not None:
                self.cache._put(key, prepared)
        return _Turn(
            full=tuple(full),
            masked=_materialise(full, prepared),
            full_tokens=prepared.full_tokens,
            masked_tokens=prepared.masked_tokens,
            repaired_blocks=prepared.repaired_blocks,
            longest_result=prepared.longest_result,
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
        longest = max(
            (len(b.content) for m in full for b in m.content_blocks if isinstance(b, ToolResultBlock)),
            default=0,
        )
        return _Prepared(
            masked=tuple(entries),
            full_tokens=sum(weights),
            masked_tokens=masked_tokens,
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
    return [_tag_seeded_reference(Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)]))]


def _summary_body(text: str) -> str:
    return (
        "[Running summary of the earlier turns of this conversation that are "
        f"not carried below]\n{text}\n\n{END_OF_SUMMARY_MARKER}"
    )


_NOTHING_RECORDED: Final[str] = "(the summary holds no entries for those turns)"


def _summary_message(memory: SessionMemory, summarized_turns: int) -> list[Message]:
    text = memory.summary.render()
    if not text and summarized_turns <= 0:
        return []
    return _block(_summary_body(text or _NOTHING_RECORDED))


def _summary_reserve(memory: SessionMemory, rc: LoopConstants) -> int:
    """What the planner keeps for the summary block: the cap plus its wrapper."""
    cap = rc.session_memory_running_summary_token_cap
    if cap <= 0:
        return _tokens(_summary_message(memory, 1), rc)
    wrapper = Message(role=MessageRole.user, content_blocks=[TextBlock(text=_summary_body(""))])
    return cap + estimate_message_tokens(wrapper, rc)


def _notice_message(dropped: int, registry: bool, summary: bool) -> list[Message]:
    parts: list[str] = []
    if dropped:
        parts.append(
            f"{dropped} earlier turn{'s' if dropped != 1 else ''} of this conversation "
            f"{'are' if dropped != 1 else 'is'} not in this context and not in the summary."
        )
    if summary:
        parts.append("The summary of earlier turns is not in this context.")
    if registry:
        parts.append("The registry of files written in this conversation is not in this context.")
    return _block(f"[{' '.join(parts)}]") if parts else []


def _ledger_message(memory: SessionMemory) -> list[Message]:
    text = render_ledger(memory.ledger)
    return _block(text) if text else []


def _known(turn: Sequence[Message], position: MessagePosition) -> list[int]:
    return [p for m in turn if (p := position(m)) is not None]


def _recorded(summary: SessionSummary, known: Sequence[int]) -> bool:
    return bool(known) and summary.coverage is not None and all(summary.covers(p) for p in known)


@dataclass(frozen=True, slots=True)
class _Layout:
    turns: list[list[Message]]
    fragment: int
    head: int
    known: list[list[int]]
    recorded: list[bool]
    pre: list[int]
    """Carried turns placed before the summary: the head, and unrecorded
    turns older than the first recorded one."""
    window: list[int]
    summarized: int
    prepared: dict[int, _Turn]


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
    known = [_known(turn, position) for turn in turns]
    recorded = [_recorded(memory.summary, k) for k in known]
    carried = [i for i in range(len(turns)) if i < head or not recorded[i]]
    first_recorded = next((i for i in range(head, len(turns)) if recorded[i]), None)
    if first_recorded is None:
        pre = [i for i in carried if i < head]
    else:
        pre = [i for i in carried if i < first_recorded]
    window = [i for i in carried if i not in set(pre)]
    for index in carried:
        if index not in prepared:
            prepared[index] = prep.prepare(turns[index])
    summarized = sum(1 for i in range(len(turns)) if recorded[i] and i >= head)
    return _Layout(turns, fragment, head, known, recorded, pre, window, summarized, prepared)


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
    window_turns: int = 0
    """Carried turns after the summary (the head and older unrecorded turns
    before it are not counted here)."""
    full_turns: int = 0
    trimmed_turns: int = 0
    """Newest turns carried with result text cut to fit rather than masked."""
    masked_turns: int = 0
    dropped_turns: int = 0
    """Unrecorded turns left out to fit the budget; the notice says how many."""
    dropped_summary: bool = False
    dropped_registry: bool = False
    repaired_blocks: int = 0
    """Unpaired tool calls / results taken out of carried turns."""
    fragment_messages: int = 0
    """Messages before the session's first user message, not carried."""
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
    :class:`SeedCache`. ``budget`` is the token budget of the seed.

    The seed's estimated size never exceeds ``budget`` (unless the budget
    cannot hold even the notice): see the module docstring for the layout,
    the order in which content gives way, and what is stable across runs. An
    empty memory and an empty history give an empty seed, so a session's first
    run is exactly a cold start.
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
        if index >= boundary:
            forms[index], weights[index], state[index] = prepared[index].full, prepared[index].full_tokens, "full"
        else:
            mask(index)

    summary = _summary_message(memory, layout.summarized)
    ledger = _ledger_message(memory)
    dropped: list[int] = []
    drop_summary = drop_registry = False

    def notice() -> list[Message]:
        return _notice_message(len(dropped), drop_registry, drop_summary)

    def total() -> int:
        blocks = ([] if drop_summary else summary) + ([] if drop_registry else ledger) + notice()
        return _tokens(blocks, rc) + sum(weights[i] for i in carried)

    full_indices = [i for i in carried if state[i] == "full"]
    for index in full_indices[:-1]:
        if total() <= budget:
            break
        mask(index)
    if full_indices and total() > budget:
        newest = full_indices[-1]
        room = budget - (total() - weights[newest])
        fitted = _fit_preview(prepared[newest], room, rc, rule)
        if fitted is not None:
            forms[newest], weights[newest] = fitted
            state[newest] = "trimmed"
        else:
            mask(newest)

    if carried and total() > budget:
        newest_index = max(carried)
        order = [i for i in layout.window if i != newest_index] + [i for i in layout.pre if i != newest_index]
        for index in [*order, newest_index]:
            if total() <= budget:
                break
            carried.remove(index)
            dropped.append(index)
    if total() > budget and ledger:
        drop_registry = True
    if total() > budget and summary:
        drop_summary = True

    kept = set(carried)
    messages = (
        [_tag_seeded(m) for i in layout.pre if i in kept for m in forms[i]]
        + ([] if drop_summary else summary)
        + notice()
        + [_tag_seeded(m) for i in layout.window if i in kept for m in forms[i]]
        + ([] if drop_registry else ledger)
    )
    tokens = total()
    kinds = [state[i] for i in carried]
    return SessionSeed(
        messages=messages,
        tokens=tokens,
        head_turns=layout.head,
        summarized_turns=layout.summarized,
        window_turns=sum(1 for i in layout.window if i in kept),
        full_turns=kinds.count("full"),
        trimmed_turns=kinds.count("trimmed"),
        masked_turns=kinds.count("masked"),
        dropped_turns=len(dropped),
        dropped_summary=drop_summary,
        dropped_registry=drop_registry,
        repaired_blocks=sum(prepared[i].repaired_blocks for i in carried),
        fragment_messages=layout.fragment,
        over_budget_tokens=max(0, tokens - budget),
    )


# ---------------------------------------------------------------------------
# The fold decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldPlan:
    """Which turns a summary fold must record now, and the numbers behind it."""

    turns: tuple[tuple[Message, ...], ...] = ()
    """The turns to fold, oldest first, as stored. Empty: make no call."""
    source: tuple[tuple[Message, ...], ...] = ()
    """The same turns in the masked form the seed carries them in: what the
    summariser is shown (:func:`build_summary_delta_request`)."""
    coverage: Coverage | None = None
    """The session positions of the planned turns — what :func:`fold_run`
    records with the delta."""
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
    deferred_turns: int = 0
    """Turns the goal asked for that ``max_source_tokens`` left to a later fold."""
    backlog_turns: int = 0
    """Unrecorded foldable turns older than the newest full turns, this plan
    included — the catch-up still owed."""

    @property
    def needed(self) -> bool:
        return bool(self.turns)

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
) -> FoldPlan:
    """Which turns need folding into the summary now — PURE, CPU-bound.

    Called after a run, with ``history`` the session's stored messages
    INCLUDING that run's — exactly what the next seed will be built from —
    and the same ``position``, ``rule``, ``cache`` and base ``budget`` the
    seed uses. See "When a fold is needed" in the module docstring for how the
    next seed is sized and when the newest turn is counted in full.

    The turns planned are always the oldest unrecorded turns that can be
    recorded (not the head, not the newest full turns, at least one known
    position), in order, so the coverage they add joins the summary's range
    without a gap. ``max_source_tokens`` (0 = unbounded) bounds what one
    summariser call is shown, measured as it will be rendered: the plan stops
    before the turn that would pass it, but always takes at least one turn,
    and always every unrecorded turn older than the summary's range (a lapsed
    head) so the range stays whole.

    Convergence: every plan that is needed records at least one turn, and the
    turns it leaves stay counted, so a backlog of B source tokens (after a
    reset, a run of failed folds, or turns left out of the seed) is recorded
    within ``ceil(B / max_source_tokens)`` successful folds; a host that
    plans again right after a fold lands catches up within one run.
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
        + _tokens(_ledger_message(memory), rc)
    )
    fresh = [i for i in carried if i >= boundary and i >= head]
    older = [i for i in carried if i >= head and i not in set(fresh) and layout.known[i]]
    hold_full = _fresh_hold([prepared[i] for i in fresh], fixed, budget)
    hold_masked = sum(prepared[i].masked_tokens for i in fresh)
    trigger = int(budget * rc.session_seed_fold_trigger_ratio)
    target = int(budget * rc.session_seed_fold_target_ratio)
    # Each run turns the newest turn into an older, masked one; room for
    # that many of them below the trigger is what keeps folds apart.
    spacing = rc.session_seed_fold_min_turns_between * hold_masked
    counted_full = bool(fresh) and fixed + hold_full + spacing <= trigger
    if counted_full:
        fresh_tokens, goal = hold_full, max(target, fixed + hold_full)
    else:
        fresh_tokens, goal = hold_masked, target
    projected = fixed + sum(prepared[i].masked_tokens for i in older) + fresh_tokens
    base = FoldPlan(
        projected_tokens=projected,
        remaining_tokens=projected,
        fresh_tokens=fresh_tokens,
        newest_counted_full=counted_full,
        trigger_tokens=trigger,
        goal_tokens=goal,
        backlog_turns=len(older),
    )
    if projected <= trigger or not older:
        return base
    coverage = memory.summary.coverage
    mandatory = {i for i in older if coverage is not None and max(layout.known[i]) < coverage.first}
    chosen: list[int] = []
    remaining = projected
    source_tokens = 0
    wanted = 0
    for index in older:
        if remaining <= goal and index not in mandatory:
            break
        wanted += 1
        weight = fold_source_tokens(prepared[index].masked, rc)
        over = max_source_tokens > 0 and source_tokens + weight > max_source_tokens
        if chosen and over and index not in mandatory:
            break
        chosen.append(index)
        source_tokens += weight
        remaining -= prepared[index].masked_tokens
    if not chosen:
        return base
    positions = [p for i in chosen for p in layout.known[i]]
    deferred = 0
    if remaining > goal:
        deferred = sum(1 for i in older if i not in set(chosen))
    return FoldPlan(
        turns=tuple(tuple(turns[i]) for i in chosen),
        source=tuple(prepared[i].masked for i in chosen),
        coverage=Coverage(min(positions), max(positions)),
        projected_tokens=projected,
        remaining_tokens=remaining,
        fresh_tokens=fresh_tokens,
        newest_counted_full=counted_full,
        trigger_tokens=trigger,
        goal_tokens=goal,
        deferred_turns=deferred,
        backlog_turns=len(older),
    )


__all__ = [
    "FoldPlan",
    "KeepWhole",
    "KeptLines",
    "MaskingRule",
    "MessagePosition",
    "SeedCache",
    "SessionSeed",
    "build_session_seed",
    "default_masking_rule",
    "mask_boundary",
    "mask_turn",
    "opens_turn",
    "plan_session_fold",
    "repair_turn",
    "split_turns",
]
