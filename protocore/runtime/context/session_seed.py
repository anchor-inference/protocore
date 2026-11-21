"""Seeding a run from its session's earlier turns: raw, masked, summarised last.

PURE core module: no host, no store, no model client. The host loads the
session's earlier messages and its :class:`~protocore.runtime.context.session_memory.SessionMemory`,
and says which turns the summary already covers; this module decides what the
next run is shown and which turns, if any, need a summary fold now.

What a seed carries
-------------------

The session's earlier messages are split into turns (:func:`split_turns`): a
user message and everything that answered it. A turn is carried whole or not
at all, and a tool call whose result is missing (or a result whose call is) is
taken out of it, so the seed never puts an orphan in front of a provider.

Every user and assistant word of a carried turn is carried verbatim, and so is
every tool call. What changes with age is the tool RESULTS: the newest
``session_seed_full_turns`` turns carry them in full; in older turns each
result is replaced by a short placeholder that names the tool and its
arguments, keeps the lines a :class:`MaskingRule` says must survive (by
default lines starting with a configured citation prefix and lines carrying an
identifier), keeps the head of the result, and says that calling the tool
again gets the rest. The masked form of a turn depends only on that turn and
the configuration, so the same turn is the same bytes in every seed.

Only turns that no longer fit even masked are represented by the running
summary, and the summary is grown by a delta fold of exactly those turns
(:func:`plan_session_fold`).

Layout::

    [head]     the first session_memory_head_protect_turns turns, never folded
    [summary]  the running summary, when there is one — the turns between the
               head and the window          (dual-tagged reference block)
    [window]   every turn after the summary, oldest first:
               masked … masked | full (newest session_seed_full_turns)
    [registry] the artifact registry, when the session wrote files
                                            (dual-tagged reference block)
    ← the host then appends its preludes and the new task

Which turns are in the window is decided by the summary's coverage, not by
the budget: a turn leaves the window only when the summary covers it. A turn
the summary does not cover is therefore always carried (at worst masked, and
over budget) unless the host asks otherwise (``drop_uncovered``).

Prefix stability
----------------

A provider's prefix cache serves a request up to its first differing byte, so
the seed is built to grow by appending. Between two consecutive runs of one
session, when no fold has landed in between, the next seed starts with the
previous one byte for byte up to the end of its last masked turn: the head,
the summary and every masked window turn are unchanged and in the same
positions, and the next seed adds turns after them. What is NOT stable, and
when:

* **The full tail.** A turn that crosses the masking boundary is shown masked
  from then on, so the prefix is invalidated from that turn's first masked
  tool result on (its question and its calls are the same bytes). The boundary
  moves in steps of ``session_seed_mask_batch_turns`` turns, and it is at the
  end of the seed, so the miss costs the re-read of those turns and the new
  ones only. Budget pressure (a long new message) may mask or cut the full
  turns of one run: again the end of the seed only.
* **The registry.** It changes when a file is written, and sits after the
  window for that reason.
* **A fold landing.** The summary changes and the turns it now covers leave
  the window: the prefix is stable only through the head. Folds are planned
  with hysteresis — triggered at ``session_seed_fold_trigger_ratio`` of the
  budget, folding down to ``session_seed_fold_target_ratio`` — so this happens
  once every several turns, not every turn, and in a tool-heavy chat of a few
  turns not at all.
* **A changed history.** A turn edited, regenerated or removed changes the
  seed from that turn on.
* **``drop_uncovered``.** Leaving out the oldest window turns to fit the
  budget moves the window's front on every run that does it; it is for a
  session whose summary was lost, not for the steady state.

The masking rule must be a pure function of its arguments; one that is not
makes masked turns differ between runs and gives the cache nothing.

When a fold is needed
---------------------

:func:`plan_session_fold`, after a run, measures the seed the next run would
get with this run included — head, the summary at its cap, the registry, every
window turn in masked form — and returns the oldest window turns to fold when
that passes the trigger share of the budget, enough of them to come down to
the target share. Otherwise it returns no turns and the host makes no
summariser call. Full turns never cause a fold: they are what gives way first
under pressure. The gate is computed from the same layout the seed builder
uses, so the two cannot disagree about what the seed carries.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    Message,
    MessageOrigin,
    MessageRole,
    TextBlock,
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
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.wire_format import is_compacted_placeholder

# Render-shape caps of the placeholder: they bound a string, not a behaviour.
_PLACEHOLDER_ARGS_MAX_CHARS: Final[int] = 300
_KEPT_LINE_MAX_CHARS: Final[int] = 400
_KEPT_LINES_MAX: Final[int] = 40
_ELLIPSIS: Final[str] = "…"

TurnPredicate = Callable[[Sequence[Message]], bool]
"""A question about one turn, asked of the stored messages (``covered``)."""

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
    results that are never masked. Both are called with the call the result
    answers (``None`` when the turn holds no such call) and must be pure: the
    seed's prefix stability rests on a masked turn being the same bytes every
    time.
    """

    kept_lines: KeptLines
    keep_whole: KeepWhole = _keep_nothing_whole


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
    add (wrap this rule's ``kept_lines``).
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

    return MaskingRule(kept_lines=kept_lines)


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def _use_ids(message: Message) -> list[str]:
    return [b.tool_call_id for b in message.content_blocks if isinstance(b, ToolUseBlock)]


def _result_ids(message: Message) -> list[str]:
    return [b.tool_call_id for b in message.content_blocks if isinstance(b, ToolResultBlock)]


def opens_turn(message: Message) -> bool:
    """A user message of the conversation that is not carrying tool results back.

    Compaction records are user-role and written mid-turn, often right after a
    tool result; the loop's scaffolding (nudges, recovery notes) is user-role
    too. Read as the start of a turn either would split a run in two.
    """
    return (
        message.role is MessageRole.user
        and message.origin is MessageOrigin.conversation
        and not _result_ids(message)
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


def repair_turn(turn: Sequence[Message]) -> tuple[list[Message], int]:
    """The turn without tool calls that lack a result or results that lack a call.

    Returns the repaired messages and how many blocks were taken out. A
    message left with no blocks is dropped. Text is never removed.
    """
    uses = {i for m in turn for i in _use_ids(m)}
    results = {i for m in turn for i in _result_ids(m)}
    paired = uses & results
    out: list[Message] = []
    removed = 0
    for message in turn:
        blocks = [
            b
            for b in message.content_blocks
            if not (isinstance(b, ToolUseBlock | ToolResultBlock) and b.tool_call_id not in paired)
        ]
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


def _placeholder(
    call: ToolUseBlock | None,
    result: ToolResultBlock,
    tokens: int,
    kept: Sequence[str],
    preview_chars: int,
) -> str:
    content = result.content
    name = call.name if call is not None else "a tool"
    args = f"({_clip(call.arguments_json, _PLACEHOLDER_ARGS_MAX_CHARS)})" if call is not None else ""
    lines = [
        f"[Earlier result of {name}{args}: {tokens} tokens, not carried here in full; "
        "call the tool again if its full output is needed.]"
    ]
    if kept:
        lines.append("Kept lines:")
        lines.extend(_clip(line, _KEPT_LINE_MAX_CHARS) for line in kept[:_KEPT_LINES_MAX])
    if preview_chars > 0 and content.strip():
        head = content[:preview_chars]
        lines.append(f"First {len(head)} of {len(content)} characters:")
        lines.append(head + (_ELLIPSIS if len(head) < len(content) else ""))
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _ResultFacts:
    """What masking needs to know of one result, worked out once per build."""

    maskable: bool
    tokens: int
    kept: tuple[str, ...]


def _facts(
    call: ToolUseBlock | None, result: ToolResultBlock, rule: MaskingRule, rc: LoopConstants
) -> _ResultFacts:
    if is_compacted_placeholder(result.content) or rule.keep_whole(call, result):
        return _ResultFacts(False, 0, ())
    tokens = estimate_tokens(result.content, rc)
    if tokens < rc.session_seed_mask_min_tokens:
        return _ResultFacts(False, tokens, ())
    return _ResultFacts(True, tokens, tuple(rule.kept_lines(call, result)))


def _mask(
    turn: Sequence[Message],
    rc: LoopConstants,
    rule: MaskingRule,
    preview: int,
    memo: dict[str, _ResultFacts],
) -> list[Message]:
    calls = {
        b.tool_call_id: b for m in turn for b in m.content_blocks if isinstance(b, ToolUseBlock)
    }
    out: list[Message] = []
    for message in turn:
        if message.role is not MessageRole.tool:
            out.append(message)
            continue
        blocks = []
        changed = False
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock) and len(block.content) > preview:
                call = calls.get(block.tool_call_id)
                facts = memo.get(block.tool_call_id)
                if facts is None:
                    facts = memo[block.tool_call_id] = _facts(call, block, rule, rc)
                if facts.maskable:
                    block = ToolResultBlock(
                        tool_call_id=block.tool_call_id,
                        content=_placeholder(call, block, facts.tokens, facts.kept, preview),
                        is_error=block.is_error,
                        metadata=dict(block.metadata),
                        canonical_ref=block.canonical_ref,
                        path=block.path,
                    )
                    changed = True
            blocks.append(block)
        out.append(message.model_copy(update={"content_blocks": tuple(blocks)}) if changed else message)
    return out


def mask_turn(
    turn: Sequence[Message],
    rc: LoopConstants,
    rule: MaskingRule,
    *,
    preview_chars: int | None = None,
) -> list[Message]:
    """The turn with its bulky tool results replaced by placeholders — PURE.

    User and assistant messages are returned as they are; a tool result keeps
    its call id, so every call still has its result. A result is left whole
    when it is no longer than the preview, smaller than
    ``session_seed_mask_min_tokens``, already a compaction placeholder, or
    one the rule keeps whole. ``preview_chars`` overrides
    ``session_seed_mask_preview_chars`` (the seed uses a larger one to cut the
    newest turn to fit rather than mask it).
    """
    preview = rc.session_seed_mask_preview_chars if preview_chars is None else preview_chars
    return _mask(turn, rc, rule, preview, {})


def _tokens(messages: Sequence[Message], rc: LoopConstants) -> int:
    return sum(estimate_message_tokens(m, rc) for m in messages)


@dataclass(slots=True)
class _Turn:
    original: list[Message]
    full: list[Message]
    masked: list[Message]
    full_tokens: int
    masked_tokens: int
    repaired_blocks: int
    memo: dict[str, _ResultFacts]


def _prepare(
    history: Sequence[Message], rc: LoopConstants, rule: MaskingRule
) -> tuple[list[_Turn], int]:
    parts = split_turns(history)
    fragment = 0
    if parts and not opens_turn(parts[0][0]):
        fragment = len(parts.pop(0))
    turns: list[_Turn] = []
    for part in parts:
        full, removed = repair_turn(part)
        memo: dict[str, _ResultFacts] = {}
        masked = _mask(full, rc, rule, rc.session_seed_mask_preview_chars, memo)
        turns.append(
            _Turn(
                original=list(part),
                full=full,
                masked=masked,
                full_tokens=_tokens(full, rc),
                masked_tokens=_tokens(masked, rc),
                repaired_blocks=removed,
                memo=memo,
            )
        )
    return turns, fragment


def _head_count(turns: Sequence[_Turn], rc: LoopConstants) -> int:
    count = 0
    used = 0
    for turn in turns[: rc.session_memory_head_protect_turns]:
        if used + turn.masked_tokens > rc.session_memory_head_max_tokens:
            break
        used += turn.masked_tokens
        count += 1
    return count


def _covered_count(turns: Sequence[_Turn], covered: TurnPredicate | None) -> int:
    if covered is None:
        return 0
    count = 0
    for turn in turns:
        if not covered(turn.original):
            break
        count += 1
    return count


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


def _tag_seeded(message: Message) -> Message:
    """Tag a carried prior-run message as seeded: finalisation does not persist
    it again, compaction keeps it apart from this run's turns."""
    if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True:
        return message
    return message.model_copy(
        update={"metadata": {**message.metadata, SESSION_HISTORY_SEED_METADATA_KEY: True}}
    )


def _tag_seeded_reference(message: Message) -> Message:
    """Tag a synthetic seed block (summary, registry) as seeded AND as a reference block.

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


def _summary_message(memory: SessionMemory) -> list[Message]:
    text = memory.summary.render()
    if not text:
        return []
    body = (
        "[Running summary of the earlier turns of this conversation that are "
        f"not carried below]\n{text}\n\n{END_OF_SUMMARY_MARKER}"
    )
    return [_tag_seeded_reference(Message(role=MessageRole.user, content_blocks=[TextBlock(text=body)]))]


def _ledger_message(memory: SessionMemory) -> list[Message]:
    text = render_ledger(memory.ledger)
    if not text:
        return []
    return [_tag_seeded_reference(Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)]))]


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
    """Newest turns carried with results cut to fit rather than masked."""
    masked_turns: int = 0
    dropped_turns: int = 0
    """Uncovered turns left out because ``drop_uncovered`` allowed it."""
    repaired_blocks: int = 0
    """Unpaired tool calls / results taken out of carried turns."""
    fragment_messages: int = 0
    """Messages before the session's first user message, not carried."""
    over_budget_tokens: int = 0


def _fit_preview(
    turn: _Turn, room: int, rc: LoopConstants, rule: MaskingRule
) -> tuple[list[Message], int] | None:
    """The largest head that lets ``turn`` fit ``room`` tokens, or ``None``."""
    lo = rc.session_seed_mask_preview_chars
    longest = max(
        (len(b.content) for m in turn.full for b in m.content_blocks if isinstance(b, ToolResultBlock)),
        default=0,
    )
    if longest <= lo or turn.masked_tokens > room:
        return None
    best: tuple[list[Message], int] | None = None
    low, high = lo + 1, longest - 1
    while low <= high:
        mid = (low + high) // 2
        cut = _mask(turn.full, rc, rule, mid, turn.memo)
        weight = _tokens(cut, rc)
        if weight <= room:
            best = (cut, weight)
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
    covered: TurnPredicate | None = None,
    rule: MaskingRule | None = None,
    drop_uncovered: bool = False,
) -> SessionSeed:
    """The seed to put in front of a run's new task — PURE.

    ``history`` is the session's earlier messages as stored, oldest first,
    without the new run's. ``covered`` says whether the summary in ``memory``
    covers a turn (asked of the stored messages; coverage is a prefix, so the
    first turn it denies ends it); without it nothing is covered. ``rule`` is
    the masking rule (:func:`default_masking_rule` when ``None``). ``budget``
    is the token budget of the seed; see the module docstring for the layout
    and for what is stable across runs.

    Under budget pressure, in order: the older full turns are masked, then the
    newest turn's results are cut to the longest head that fits, then masked.
    A seed still over budget is returned as it is, with
    ``over_budget_tokens`` set — unless ``drop_uncovered``, when window turns
    are left out oldest first (never the head, never the newest turn); that is
    for a session whose summary was lost and whose turns would otherwise all
    be carried far past the budget.

    An empty memory and an empty history give an empty seed, so a session's
    first run is exactly a cold start.
    """
    rule = rule or default_masking_rule(rc)
    turns, fragment = _prepare(history, rc, rule)
    if not turns and memory.is_empty():
        return SessionSeed(fragment_messages=fragment)

    count = len(turns)
    head = _head_count(turns, rc)
    start = max(head, _covered_count(turns, covered))
    boundary = mask_boundary(count, rc)
    carried = [*range(head), *range(start, count)]
    forms: dict[int, list[Message]] = {}
    weights: dict[int, int] = {}
    for index in carried:
        full = index >= boundary
        forms[index] = turns[index].full if full else turns[index].masked
        weights[index] = turns[index].full_tokens if full else turns[index].masked_tokens
    state = {index: ("full" if index >= boundary else "masked") for index in carried}

    summary = _summary_message(memory)
    ledger = _ledger_message(memory)
    fixed = _tokens(summary, rc) + _tokens(ledger, rc)

    def total() -> int:
        return fixed + sum(weights[i] for i in carried)

    full_indices = [i for i in carried if state[i] == "full"]
    for index in full_indices[:-1]:
        if total() <= budget:
            break
        forms[index], weights[index] = turns[index].masked, turns[index].masked_tokens
        state[index] = "masked"
    if full_indices and total() > budget:
        newest = full_indices[-1]
        room = budget - (total() - weights[newest])
        fitted = _fit_preview(turns[newest], room, rc, rule)
        if fitted is not None:
            forms[newest], weights[newest] = fitted
            state[newest] = "trimmed"
        else:
            forms[newest], weights[newest] = turns[newest].masked, turns[newest].masked_tokens
            state[newest] = "masked"

    dropped = 0
    if drop_uncovered:
        for index in [i for i in carried if i >= start][:-1]:
            if total() <= budget:
                break
            carried.remove(index)
            dropped += 1

    messages = (
        [_tag_seeded(m) for i in carried if i < head for m in forms[i]]
        + summary
        + [_tag_seeded(m) for i in carried if i >= start for m in forms[i]]
        + ledger
    )
    tokens = total()
    kinds = [state[i] for i in carried]
    return SessionSeed(
        messages=messages,
        tokens=tokens,
        head_turns=head,
        summarized_turns=max(0, start - head),
        window_turns=sum(1 for i in carried if i >= start),
        full_turns=kinds.count("full"),
        trimmed_turns=kinds.count("trimmed"),
        masked_turns=kinds.count("masked"),
        dropped_turns=dropped,
        repaired_blocks=sum(turns[i].repaired_blocks for i in carried),
        fragment_messages=fragment,
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
    projected_tokens: int = 0
    """The next seed, as the planner sized it, before the fold."""
    remaining_tokens: int = 0
    """The same after the fold lands."""
    trigger_tokens: int = 0
    target_tokens: int = 0
    deferred_turns: int = 0
    """Turns the target asked for that the source budget left to a later fold."""

    @property
    def needed(self) -> bool:
        return bool(self.turns)

    @property
    def messages(self) -> list[Message]:
        """The planned turns' messages in order — the summariser's source."""
        return [m for turn in self.turns for m in turn]


def plan_session_fold(
    history: Sequence[Message],
    memory: SessionMemory,
    budget: int,
    rc: LoopConstants,
    *,
    covered: TurnPredicate | None = None,
    rule: MaskingRule | None = None,
    max_source_tokens: int = 0,
) -> FoldPlan:
    """Which turns need folding into the summary now — PURE.

    Called after a run, with ``history`` the session's messages INCLUDING that
    run's — exactly what the next seed will be built from — and the same
    ``covered``, ``rule`` and base ``budget`` the seed uses. Sizes the next
    seed on what it would carry: the head, the summary at
    ``session_memory_running_summary_token_cap`` (its current size when the
    cap is off), the registry, and every window turn masked. At or below
    ``session_seed_fold_trigger_ratio`` of ``budget`` there is nothing to fold.
    Above it the oldest window turns are planned, until the rest fits
    ``session_seed_fold_target_ratio`` of ``budget``; the newest
    ``max(1, session_seed_full_turns)`` turns are never folded.

    ``max_source_tokens`` (0 = unbounded) bounds what one summariser call is
    shown, measured as it will be rendered: the plan stops before the turn
    that would pass it, but always takes at least one. The turns it leaves
    stay uncovered and are carried; the next plan picks them up.
    """
    rule = rule or default_masking_rule(rc)
    turns, _fragment = _prepare(history, rc, rule)
    count = len(turns)
    head = _head_count(turns, rc)
    start = max(head, _covered_count(turns, covered))
    cap = rc.session_memory_running_summary_token_cap
    summary_reserve = cap if cap > 0 else _tokens(_summary_message(memory), rc)
    projected = (
        sum(t.masked_tokens for t in turns[:head])
        + summary_reserve
        + _tokens(_ledger_message(memory), rc)
        + sum(t.masked_tokens for t in turns[start:])
    )
    trigger = int(budget * rc.session_seed_fold_trigger_ratio)
    target = int(budget * rc.session_seed_fold_target_ratio)
    if projected <= trigger:
        return FoldPlan(
            projected_tokens=projected,
            remaining_tokens=projected,
            trigger_tokens=trigger,
            target_tokens=target,
        )
    protect = max(1, rc.session_seed_full_turns)
    wanted: list[_Turn] = []
    remaining = projected
    for turn in turns[start : max(start, count - protect)]:
        if remaining <= target:
            break
        wanted.append(turn)
        remaining -= turn.masked_tokens
    chosen: list[_Turn] = []
    source = 0
    for turn in wanted:
        weight = fold_source_tokens(turn.original, rc)
        if chosen and max_source_tokens > 0 and source + weight > max_source_tokens:
            break
        chosen.append(turn)
        source += weight
    return FoldPlan(
        turns=tuple(tuple(t.original) for t in chosen),
        projected_tokens=projected,
        remaining_tokens=projected - sum(t.masked_tokens for t in chosen),
        trigger_tokens=trigger,
        target_tokens=target,
        deferred_turns=len(wanted) - len(chosen),
    )


__all__ = [
    "FoldPlan",
    "KeepWhole",
    "KeptLines",
    "MaskingRule",
    "SessionSeed",
    "TurnPredicate",
    "build_session_seed",
    "default_masking_rule",
    "mask_boundary",
    "mask_turn",
    "opens_turn",
    "plan_session_fold",
    "repair_turn",
    "split_turns",
]
