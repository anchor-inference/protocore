"""The compaction trigger is the configured ratio of the window, and a clamp says so.

Requests are fitted to the window: the output cap of each one is cut to what the
window has left after the prompt. The trigger nevertheless kept a quarter of
the window back for the output on top of the turn headroom and the safety
margin, so on a 65k window a trigger configured at 0.8 sat at 0.57 — and nothing
anywhere said the configured value was not the one in force.
"""
from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.context.budgets import derive_budgets


def test_the_configured_ratio_is_the_trigger_on_a_65k_window() -> None:
    rc = LoopConstants(model_context_window=65_536, compaction_trigger_ratio=0.8)
    budgets = derive_budgets(rc)
    assert budgets.compaction_trigger_tokens == int(65_536 * 0.8)
    assert budgets.configured_trigger_tokens == int(65_536 * 0.8)
    assert budgets.trigger_limited_by == ""


def test_a_ratio_the_window_cannot_honour_is_clamped_and_says_why() -> None:
    rc = LoopConstants(model_context_window=65_536, compaction_trigger_ratio=0.9)
    budgets = derive_budgets(rc)
    assert budgets.configured_trigger_tokens == int(65_536 * 0.9)
    assert budgets.compaction_trigger_tokens < budgets.configured_trigger_tokens
    assert budgets.trigger_limited_by == "accept_ceiling"


def test_a_large_output_ratio_does_not_eat_the_trigger() -> None:
    small = derive_budgets(LoopConstants(model_context_window=65_536, llm_output_max_tokens_ratio=0.1))
    large = derive_budgets(LoopConstants(model_context_window=65_536, llm_output_max_tokens_ratio=0.5))
    assert small.compaction_trigger_tokens == large.compaction_trigger_tokens


async def test_compaction_events_carry_the_configured_and_the_effective_trigger() -> None:
    from protocore.contracts.types import Message, MessageRole, TextBlock
    from protocore.runtime.context.compaction import CompactionState, compaction_event_payload
    from protocore.runtime.context.manager import ContextManager
    from protocore.tests_support.adapters import InMemoryBlobStore

    rc = LoopConstants(model_context_window=4_096, compaction_trigger_ratio=0.9)
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        *(
            Message(
                role=MessageRole.assistant if i % 2 else MessageRole.user,
                content_blocks=[TextBlock(text="word " * 400)],
            )
            for i in range(8)
        ),
    ]
    attempt = await manager.force_compaction(
        history=history, compaction_state=CompactionState(), tenant_id="t", model_name="m"
    )
    payload = compaction_event_payload(attempt, reason="test")
    budgets = derive_budgets(rc)
    assert payload["trigger_threshold"] == budgets.compaction_trigger_tokens
    assert payload["configured_trigger_tokens"] == int(4_096 * 0.9)
    assert payload["trigger_limited_by"] == "accept_ceiling"
