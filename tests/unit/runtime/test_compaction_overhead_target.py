"""A pass whose target lies below the prompt's fixed overhead empties the history.

The pass aims at ``compaction_target_ratio * min(trigger, prompt)`` on the whole
prompt. When the system prompt and tool definitions alone are larger than that
target, no amount of history compaction can reach it, so every tier and the
floor run to exhaustion: a forced pass removes every removable span even though
removing a few would have brought the prompt back under the trigger.
"""
from __future__ import annotations

import json

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock, ToolResultBlock, ToolUseBlock
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.context.compaction import CompactionState, estimate_history_tokens
from protocore.runtime.context.manager import ContextManager
from protocore.tests_support.adapters import InMemoryBlobStore


def _history(rounds: int, prefix: str = "call") -> list[Message]:
    history = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="Audit every section.")])]
    for n in range(rounds):
        call = f"{prefix}-{n}"
        history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    TextBlock(text=f"Checking section {n}."),
                    ToolUseBlock(tool_call_id=call, name="Exec", arguments_json=json.dumps({"command": f"inspect {n}"})),
                ],
            )
        )
        history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id=call, content=f"section {n} ok " + "x " * 400)],
            )
        )
    return history


async def test_forced_pass_over_a_large_overhead_keeps_history_it_did_not_need_to_remove() -> None:
    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=2)
    trigger = derive_budgets(rc).compaction_trigger_tokens
    # Overhead above the target (0.6 * trigger), below the trigger.
    overhead = int(trigger * 0.75)
    history = _history(60)
    # Trim until the whole prompt is just over the trigger: a little over.
    while estimate_history_tokens(history[:-2], rc) + overhead > trigger:
        history = history[:-2]
    assert estimate_history_tokens(history, rc) + overhead > trigger
    rounds_before = sum(1 for m in history if m.role is MessageRole.tool)

    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    attempt = await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t",
        model_name="m",
        overhead_tokens=overhead,
    )

    rounds_after = sum(1 for m in history if m.role is MessageRole.tool)
    assert attempt.prompt_after <= trigger
    # A round or two had to go to fit under the trigger. The pass instead ran
    # the floor out of material (it aims at a target below the overhead alone)
    # and left only the kept tail.
    assert attempt.floor is not None
    assert not attempt.floor.reached, (rounds_before, rounds_after, attempt.outcome)
    assert rounds_after > 2, (rounds_before, rounds_after)


async def test_routine_pass_behind_a_session_seed_keeps_most_of_the_runs_own_work() -> None:
    from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY

    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=2)
    trigger = derive_budgets(rc).compaction_trigger_tokens
    # A host that seeds half the trigger of earlier turns and carries a third of
    # it in system prompt and tools, as a large tool surface does.
    overhead = int(trigger * 0.35)
    seed = _history(200, prefix="seed")
    while estimate_history_tokens(seed, rc) > trigger * 0.5:
        seed = seed[:-2]
    seed = [m.model_copy(update={"metadata": {**m.metadata, SESSION_HISTORY_SEED_METADATA_KEY: True}}) for m in seed]
    own = _history(40)
    history = seed + own
    while estimate_history_tokens(history[:-2], rc) + overhead > trigger:
        history = history[:-2]
    assert estimate_history_tokens(history, rc) + overhead > trigger
    own_rounds = sum(1 for m in history[len(seed):] if m.role is MessageRole.tool)
    assert own_rounds >= 4

    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    attempt = await manager.run_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t",
        model_name="m",
        overhead_tokens=overhead,
    )
    kept = sum(
        1
        for m in history
        if m.role is MessageRole.tool and not m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY)
    )
    assert attempt.prompt_after <= trigger
    assert attempt.floor is not None
    assert not attempt.floor.reached, (own_rounds, kept, attempt.outcome)
    assert kept > 2, (own_rounds, kept)


async def test_the_target_leaves_the_ratio_of_the_room_above_what_no_tier_can_remove() -> None:
    from protocore.runtime.context.compaction import compaction_event_payload

    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=2)
    trigger = derive_budgets(rc).compaction_trigger_tokens
    overhead = int(trigger * 0.3)
    history = _history(60)
    while estimate_history_tokens(history[:-2], rc) + overhead > trigger:
        history = history[:-2]
    prompt = estimate_history_tokens(history, rc) + overhead

    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    attempt = await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t",
        model_name="m",
        overhead_tokens=overhead,
    )

    fixed = attempt.fixed_tokens
    # The overhead, the task and the kept tail.
    assert overhead < fixed < overhead + trigger * 0.1
    ceiling = min(trigger, prompt)
    assert attempt.target_tokens == fixed + int((ceiling - fixed) * rc.compaction_target_ratio)
    assert attempt.prompt_after <= attempt.target_tokens
    assert compaction_event_payload(attempt, reason="r")["fixed_tokens"] == fixed


async def test_a_fixed_part_above_the_trigger_takes_everything_removable() -> None:
    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=2)
    trigger = derive_budgets(rc).compaction_trigger_tokens
    history = _history(6)

    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    attempt = await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t",
        model_name="m",
        overhead_tokens=trigger + 1,
    )

    assert attempt.target_tokens == attempt.fixed_tokens
    assert attempt.floor is not None and attempt.floor.reached
    assert attempt.outcome == "at_floor"
