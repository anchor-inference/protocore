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
length.

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
* a string argument of a tool call longer than ``session_seed_mask_arg_max_chars``
  is replaced by its head and the count of characters not carried; the
  arguments stay valid JSON, shorter values (paths, ids) stay whole. The
  artifact registry is built from the run as it happened and keeps the files;
* the reasoning of assistant messages (``reasoning_content``, thinking
  blocks, opaque reasoning records) is dropped. That is safe for tool-call
  pairing on every provider the core speaks to: a masked turn is a completed
  turn followed by later user messages, never the tool loop in progress, and
  the core's own adapters already send earlier turns without their reasoning
  (an OpenAI-compatible endpoint that requires the field gets it empty;
  opaque records go back only to the endpoint that wrote them).

The masked form of a turn depends only on that turn and the configuration, so
the same turn is the same bytes in every seed.

Only turns that no longer fit even masked are represented by the running
summary, and the summary is grown by a delta fold of exactly those turns,
shown in their masked form (:func:`plan_session_fold`).

Layout::

    [head]     the first session_memory_head_protect_turns turns, never folded
    [summary]  the running summary, when it covers turns past the head
                                            (dual-tagged reference block)
    [notice]   only with drop_uncovered: how many turns are not available
    [window]   every turn the summary does not cover, oldest first:
               masked … masked | full (newest session_seed_full_turns)
    [registry] the artifact registry, when the session wrote files
                                            (dual-tagged reference block)
    ← the host then appends its preludes and the new task

Which turns are in the window is decided by the summary's coverage
(:attr:`SessionSummary.covered_through`), not by the budget: a turn leaves the
window only when the summary records it. A turn the summary does not cover is
therefore always carried (at worst masked, and over budget) unless the host
asks otherwise (``drop_uncovered``). A summary that is reset covers nothing,
so a reset brings every turn back rather than losing any.

The seed puts user-role blocks next to user turns (summary before the first
window question, registry before the host's task), as the core's own
compaction does inside a run. An endpoint whose chat template insists on
alternating roles must merge adjacent user messages in its adapter, which is
the one place that knows the template; folding the summary into the user's
question here would make the summary unsheddable and the question no longer
the stored message.

Prefix stability
----------------

A provider's prefix cache serves a request up to its first differing byte, so
the seed is built to grow by appending. Between two consecutive runs of one
session, when no fold has landed in between, the next seed starts with the
previous one byte for byte through its last masked turn: the head, the
summary and every masked window turn are unchanged and in the same positions,
and the next seed adds turns after them. What is NOT stable, and when:

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
* **A fold landing.** The summary changes and the turns it now covers leave
  the window: the prefix is stable only through the head. Folds are planned
  with hysteresis (see below), so in a short tool-heavy chat this happens
  rarely or not at all.
* **A changed history.** A turn edited, regenerated or removed changes the
  seed from that turn on.
* **``drop_uncovered``.** Leaving out window turns to fit the budget moves the
  window's front on every run that does it; it is for a session whose summary
  was lost, not for the steady state.

The masking rule must be a pure function of its arguments; one that is not
makes masked turns differ between runs and gives the cache nothing.

When a fold is needed
---------------------

:func:`plan_session_fold`, after a run, measures the seed the next run would
get with this run included: head, the summary at its cap, the registry, every
older window turn masked, and the newest full turns at their full size when
the budget could hold them with nothing older (otherwise masked). Answer
quality comes first, so the newest turn's tool results are what the budget is
kept for: once that sum passes ``session_seed_fold_trigger_ratio`` of the
budget, the oldest window turns are planned until it is at or below
``session_seed_fold_target_ratio``. Otherwise the plan is empty and the host
makes no summariser call. The plan is computed from the same preparation the
seed builder uses, so the two cannot disagree about what the seed carries.
"""
from __future__ import annotations

import hashlib
import json
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
    SessionMemory,
    fold_source_tokens,
    render_ledger,
)
from protocore.runtime.wire_format import is_compacted_placeholder

# Render-shape caps of the placeholder: they bound a string, not a behaviour.
_KEPT_LINE_MAX_CHARS: Final[int] = 400
_KEPT_LINES_MAX: Final[int] = 40
_ELLIPSIS: Final[str] = "…"

MessagePosition = Callable[[Message], int | None]
"""Where a stored message sits in the session (the host's position, the same
numbering as :attr:`SessionSummary.covered_through`); ``None`` when unknown,
which makes the message's turn uncoverable."""

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

    ``key`` names the rule for :class:`SeedCache`: two rules with the same
    non-empty key must mask identically. An empty key keys the cache on the
    two callables' identities, which is safe and simply misses when the host
    builds the rule anew for every call.
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

    return MaskingRule(kept_lines=kept_lines, key=f"default|{'|'.join(prefixes)}|{limit}")


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


def _mask_args(call: ToolUseBlock, rc: LoopConstants) -> ToolUseBlock:
    """``call`` with string arguments past ``session_seed_mask_arg_max_chars`` cut to a head.

    Keys and shorter values stay as they are; the result is valid JSON. A
    call whose arguments are not JSON, or that would not get shorter, is
    returned unchanged.
    """
    limit = rc.session_seed_mask_arg_max_chars
    raw = call.arguments_json
    if len(raw) <= limit:
        return call
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return call
    preview = rc.session_seed_mask_arg_preview_chars
    changed = False

    def walk(value: Any) -> Any:
        nonlocal changed
        if isinstance(value, str) and len(value) > limit:
            changed = True
            return f"{value[:preview]}{_ELLIPSIS} [{len(value)} characters, not carried]"
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    masked = walk(parsed)
    if not changed:
        return call
    text = json.dumps(masked, ensure_ascii=False)
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
) -> list[Message]:
    """The masked (``whole_turn``) or cut (only the result text) form of a repaired turn."""
    answered = _pairing(turn)
    out: list[Message] = []
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
            out.append(stripped)
            continue
        if message.role is not MessageRole.tool:
            out.append(message)
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
        out.append(message.model_copy(update={"content_blocks": tuple(blocks)}) if changed else message)
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
    return _mask(turn, rc, rule, preview=preview, whole_turn=True, kept_memo={})


# ---------------------------------------------------------------------------
# Preparation, shared by the seed and the plan, and its cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Prepared:
    full: tuple[Message, ...]
    masked: tuple[Message, ...]
    full_tokens: int
    masked_tokens: int
    repaired_blocks: int
    longest_result: int


class SeedCache:
    """What preparing a session's turns produced, kept between calls — owned by the host.

    Keyed by a turn's identity (the host's ``identity`` of each message when
    it gives one for all of them, otherwise a digest of the messages' content),
    the masking rule's key and the configuration, so a changed turn, rule or
    setting is simply not found. Holding one per session makes a run's seed
    and fold plan cost proportional to the turns added since the previous
    call. Bounded (``max_entries``, oldest dropped first). Not thread-safe:
    one call at a time per cache, which the one-fold-per-session contract
    gives anyway.
    """

    def __init__(
        self,
        *,
        identity: Callable[[Message], str | None] | None = None,
        max_entries: int = MAX_SESSION_SEED_CACHE_ENTRIES,
    ) -> None:
        self._identity = identity
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _Prepared] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def _turn_key(self, turn: Sequence[Message], settings: str) -> str:
        digest = hashlib.sha256(settings.encode())
        ids = [self._identity(m) for m in turn] if self._identity is not None else []
        if ids and all(ids):
            for value in ids:
                digest.update(b"\x00" + str(value).encode())
        else:
            for message in turn:
                digest.update(b"\x01" + message.model_dump_json(exclude={"metadata", "created_at"}).encode())
        return digest.hexdigest()

    def _get(self, key: str) -> _Prepared | None:
        found = self._entries.get(key)
        if found is not None:
            self._entries.move_to_end(key)
        return found

    def _put(self, key: str, value: _Prepared) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


def _tokens(messages: Sequence[Message], rc: LoopConstants) -> int:
    return sum(estimate_message_tokens(m, rc) for m in messages)


class _Preparer:
    """Prepares turns for one call, estimating every message at most once."""

    def __init__(self, rc: LoopConstants, rule: MaskingRule, cache: SeedCache) -> None:
        self.rc = rc
        self.rule = rule
        self.cache = cache
        rule_key = rule.key or f"id:{id(rule.kept_lines)}:{id(rule.keep_whole)}"
        self.settings = hashlib.sha256((rc.model_dump_json() + "\x00" + rule_key).encode()).hexdigest()

    def prepare(self, turn: Sequence[Message]) -> _Prepared:
        key = self.cache._turn_key(turn, self.settings)
        found = self.cache._get(key)
        if found is not None:
            return found
        full, removed = repair_turn(turn)
        masked = _mask(
            full,
            self.rc,
            self.rule,
            preview=self.rc.session_seed_mask_preview_chars,
            whole_turn=True,
            kept_memo={},
        )
        weights = {id(m): estimate_message_tokens(m, self.rc) for m in full}
        masked_tokens = sum(
            weights[id(m)] if id(m) in weights else estimate_message_tokens(m, self.rc) for m in masked
        )
        longest = max(
            (len(b.content) for m in full for b in m.content_blocks if isinstance(b, ToolResultBlock)),
            default=0,
        )
        prepared = _Prepared(
            full=tuple(full),
            masked=tuple(masked),
            full_tokens=sum(weights.values()),
            masked_tokens=masked_tokens,
            repaired_blocks=removed,
            longest_result=longest,
        )
        self.cache._put(key, prepared)
        return prepared


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


def _summary_body(text: str) -> str:
    return (
        "[Running summary of the earlier turns of this conversation that are "
        f"not carried below]\n{text}\n\n{END_OF_SUMMARY_MARKER}"
    )


_NOTHING_RECORDED: Final[str] = "(those turns added nothing that needed recording)"


def _summary_message(memory: SessionMemory, summarized_turns: int) -> list[Message]:
    text = memory.summary.render()
    if not text and summarized_turns <= 0:
        return []
    body = _summary_body(text or _NOTHING_RECORDED)
    return [_tag_seeded_reference(Message(role=MessageRole.user, content_blocks=[TextBlock(text=body)]))]


def _summary_reserve(memory: SessionMemory, rc: LoopConstants) -> int:
    """What the planner keeps for the summary block: the cap plus its wrapper."""
    cap = rc.session_memory_running_summary_token_cap
    if cap <= 0:
        return _tokens(_summary_message(memory, 1), rc)
    wrapper = Message(role=MessageRole.user, content_blocks=[TextBlock(text=_summary_body(""))])
    return cap + estimate_message_tokens(wrapper, rc)


def _notice_message(dropped: int) -> list[Message]:
    text = (
        f"[{dropped} earlier turn{'s' if dropped != 1 else ''} of this conversation "
        f"{'are' if dropped != 1 else 'is'} not available here and not in the summary.]"
    )
    return [_tag_seeded_reference(Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)]))]


def _ledger_message(memory: SessionMemory) -> list[Message]:
    text = render_ledger(memory.ledger)
    if not text:
        return []
    return [_tag_seeded_reference(Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)]))]


@dataclass(frozen=True, slots=True)
class _Layout:
    turns: list[list[Message]]
    fragment: int
    head: int
    start: int
    prepared: dict[int, _Prepared]


def _head_count(turns: Sequence[list[Message]], prep: _Preparer) -> tuple[int, dict[int, _Prepared]]:
    rc = prep.rc
    count = 0
    used = 0
    prepared: dict[int, _Prepared] = {}
    for index, turn in enumerate(turns[: rc.session_memory_head_protect_turns]):
        prepared[index] = prep.prepare(turn)
        if used + prepared[index].masked_tokens > rc.session_memory_head_max_tokens:
            break
        used += prepared[index].masked_tokens
        count += 1
    return count, {i: p for i, p in prepared.items() if i < count}


def _layout(
    history: Sequence[Message], memory: SessionMemory, position: MessagePosition, prep: _Preparer
) -> _Layout:
    turns = split_turns(history)
    fragment = 0
    if turns and not opens_turn(turns[0][0]):
        fragment = len(turns.pop(0))
    head, prepared = _head_count(turns, prep)
    covered = 0
    for turn in turns:
        if not all(memory.summary.covers(position(m)) for m in turn):
            break
        covered += 1
    start = max(head, covered)
    for index in range(start, len(turns)):
        prepared[index] = prep.prepare(turns[index])
    return _Layout(turns, fragment, head, start, prepared)


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
    """Turns between the head and the window: the summary stands for them."""
    window_turns: int = 0
    full_turns: int = 0
    trimmed_turns: int = 0
    """Newest turns carried with result text cut to fit rather than masked."""
    masked_turns: int = 0
    dropped_turns: int = 0
    """Uncovered turns left out because ``drop_uncovered`` allowed it; a notice says so."""
    repaired_blocks: int = 0
    """Unpaired tool calls / results taken out of carried turns."""
    fragment_messages: int = 0
    """Messages before the session's first user message, not carried."""
    over_budget_tokens: int = 0


def _fit_preview(
    prepared: _Prepared, room: int, rc: LoopConstants, rule: MaskingRule
) -> tuple[tuple[Message, ...], int] | None:
    """The newest turn with result text cut to the longest head that fits ``room``.

    Only result text is cut: images, reasoning and arguments stay. ``None``
    when not even the shortest head fits.
    """
    low = rc.session_seed_mask_preview_chars
    if prepared.longest_result <= low:
        return None
    memo: dict[tuple[int, int], list[str]] = {}
    weights = {id(m): estimate_message_tokens(m, rc) for m in prepared.full}

    def weight(messages: Sequence[Message]) -> int:
        return sum(
            weights[id(m)] if id(m) in weights else estimate_message_tokens(m, rc) for m in messages
        )

    best: tuple[tuple[Message, ...], int] | None = None
    high = prepared.longest_result - 1
    while low <= high:
        mid = (low + high) // 2
        cut = tuple(_mask(prepared.full, rc, rule, preview=mid, whole_turn=False, kept_memo=memo))
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
    drop_uncovered: bool = False,
) -> SessionSeed:
    """The seed to put in front of a run's new task — PURE, CPU-bound.

    ``history`` is the session's earlier messages as stored, oldest first,
    without the new run's. ``position`` gives each its session position: a
    turn is left to the summary only when the summary's
    :attr:`~SessionSummary.covered_through` reaches every message of it, and
    coverage is a prefix, so the first turn it does not reach ends it. ``rule``
    is the masking rule (:func:`default_masking_rule` when ``None``); ``cache``
    the session's :class:`SeedCache`. ``budget`` is the token budget of the
    seed; see the module docstring for the layout and for what is stable
    across runs.

    Under budget pressure, in order: the older full turns are masked, then the
    newest turn's result text is cut to the longest head that fits (its images
    stay), then it is masked. A seed still over budget is returned as it is,
    with ``over_budget_tokens`` set — unless ``drop_uncovered``, when window
    turns are left out oldest first (never the head, never the newest turn)
    and one notice says how many; that is for a session whose summary was
    lost and whose turns would otherwise all be carried far past the budget.

    An empty memory and an empty history give an empty seed, so a session's
    first run is exactly a cold start.
    """
    rule = rule or default_masking_rule(rc)
    prep = _Preparer(rc, rule, cache if cache is not None else SeedCache())
    layout = _layout(history, memory, position, prep)
    turns, head, start, prepared = layout.turns, layout.head, layout.start, layout.prepared
    if not turns and memory.is_empty():
        return SessionSeed(fragment_messages=layout.fragment)

    count = len(turns)
    boundary = mask_boundary(count, rc)
    carried = [*range(head), *range(start, count)]
    forms: dict[int, tuple[Message, ...]] = {}
    weights: dict[int, int] = {}
    state: dict[int, str] = {}
    for index in carried:
        if index >= boundary:
            forms[index], weights[index], state[index] = prepared[index].full, prepared[index].full_tokens, "full"
        else:
            forms[index], weights[index], state[index] = (
                prepared[index].masked,
                prepared[index].masked_tokens,
                "masked",
            )

    summarized = max(0, start - head)
    summary = _summary_message(memory, summarized)
    ledger = _ledger_message(memory)
    fixed = _tokens(summary, rc) + _tokens(ledger, rc)

    def total() -> int:
        return fixed + sum(weights[i] for i in carried)

    def mask(index: int) -> None:
        forms[index], weights[index], state[index] = (
            prepared[index].masked,
            prepared[index].masked_tokens,
            "masked",
        )

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

    dropped = 0
    notice: list[Message] = []
    if drop_uncovered:
        for index in [i for i in carried if i >= start][:-1]:
            if total() + _tokens(notice, rc) <= budget:
                break
            carried.remove(index)
            dropped += 1
            notice = _notice_message(dropped)

    messages = (
        [_tag_seeded(m) for i in carried if i < head for m in forms[i]]
        + summary
        + notice
        + [_tag_seeded(m) for i in carried if i >= start for m in forms[i]]
        + ledger
    )
    tokens = total() + _tokens(notice, rc)
    kinds = [state[i] for i in carried]
    return SessionSeed(
        messages=messages,
        tokens=tokens,
        head_turns=head,
        summarized_turns=summarized,
        window_turns=sum(1 for i in carried if i >= start),
        full_turns=kinds.count("full"),
        trimmed_turns=kinds.count("trimmed"),
        masked_turns=kinds.count("masked"),
        dropped_turns=dropped,
        repaired_blocks=sum(prepared[i].repaired_blocks for i in carried),
        fragment_messages=layout.fragment,
        over_budget_tokens=max(0, tokens - budget),
    )


# ---------------------------------------------------------------------------
# The fold decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldPlan:
    """Which turns a summary fold must cover now, and the numbers behind it."""

    turns: tuple[tuple[Message, ...], ...] = ()
    """The turns to fold, oldest first, as stored. Empty: make no call."""
    source: tuple[tuple[Message, ...], ...] = ()
    """The same turns in the masked form the seed carried them in: what the
    summariser is shown (:func:`build_summary_delta_request`)."""
    covers_through: int | None = None
    """The session position of the last message of the planned turns — the
    coverage :func:`fold_run` records with the delta."""
    projected_tokens: int = 0
    """The next seed, as the planner sized it, before the fold."""
    remaining_tokens: int = 0
    """The same after the fold lands."""
    fresh_tokens: int = 0
    """What the planner kept for the newest full turns (their full size when
    the budget can hold them, otherwise their masked size)."""
    trigger_tokens: int = 0
    target_tokens: int = 0
    deferred_turns: int = 0
    """Turns the target asked for that the source budget, or a message with no
    known position, left to a later fold."""

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


def _fresh_hold(
    fresh: Sequence[_Prepared], fixed: int, budget: int
) -> int:
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
    seed uses. Sizes the next seed: the head (masked), the summary at
    ``session_memory_running_summary_token_cap`` plus its wrapper (its
    current size when the cap is off), the registry, every older window turn
    masked, and the newest full turns at their full size when the budget can
    hold them with nothing older in front (masked otherwise). Above
    ``session_seed_fold_trigger_ratio`` of ``budget`` the oldest older window
    turns are planned until the rest fits ``session_seed_fold_target_ratio``;
    the head and the full turns are never planned.

    A planned turn must have a known position for every message (coverage is
    recorded by position); the plan stops before the first that does not.
    ``max_source_tokens`` (0 = unbounded) bounds what one summariser call is
    shown, measured as it will be rendered; the plan stops before the turn
    that would pass it, but always takes at least one. The turns it leaves
    stay uncovered and are carried; the next plan picks them up.
    """
    rule = rule or default_masking_rule(rc)
    prep = _Preparer(rc, rule, cache if cache is not None else SeedCache())
    layout = _layout(history, memory, position, prep)
    turns, head, start, prepared = layout.turns, layout.head, layout.start, layout.prepared
    count = len(turns)
    boundary = max(mask_boundary(count, rc), count - max(1, rc.session_seed_full_turns), start)
    fixed = (
        sum(prepared[i].masked_tokens for i in range(head))
        + _summary_reserve(memory, rc)
        + _tokens(_ledger_message(memory), rc)
    )
    older = [i for i in range(start, count) if i < boundary]
    fresh = [prepared[i] for i in range(boundary, count)]
    fresh_tokens = _fresh_hold(fresh, fixed, budget)
    projected = fixed + sum(prepared[i].masked_tokens for i in older) + fresh_tokens
    trigger = int(budget * rc.session_seed_fold_trigger_ratio)
    target = int(budget * rc.session_seed_fold_target_ratio)
    if projected <= trigger:
        return FoldPlan(
            projected_tokens=projected,
            remaining_tokens=projected,
            fresh_tokens=fresh_tokens,
            trigger_tokens=trigger,
            target_tokens=target,
        )
    wanted: list[int] = []
    remaining = projected
    for index in older:
        if remaining <= target:
            break
        wanted.append(index)
        remaining -= prepared[index].masked_tokens
    chosen: list[int] = []
    source_tokens = 0
    for index in wanted:
        if any(position(m) is None for m in turns[index]):
            break
        weight = fold_source_tokens(prepared[index].masked, rc)
        if chosen and max_source_tokens > 0 and source_tokens + weight > max_source_tokens:
            break
        chosen.append(index)
        source_tokens += weight
    covers = max((p for i in chosen for m in turns[i] if (p := position(m)) is not None), default=None)
    return FoldPlan(
        turns=tuple(tuple(turns[i]) for i in chosen),
        source=tuple(prepared[i].masked for i in chosen),
        covers_through=covers,
        projected_tokens=projected,
        remaining_tokens=projected - sum(prepared[i].masked_tokens for i in chosen),
        fresh_tokens=fresh_tokens,
        trigger_tokens=trigger,
        target_tokens=target,
        deferred_turns=len(wanted) - len(chosen),
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
