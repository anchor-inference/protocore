"""Session-seed sizes are read on the ONE shared token estimate.

The seed builder and the fold planner size turns with
:func:`~protocore.runtime.context.compaction.estimate_message_tokens`, which
costs every block: thinking blocks, re-emitted reasoning and image refs are not
free. A session-memory estimate once priced those at zero and let a seed overrun
the window it was measured against; nothing in an output assertion would show
that coming back, so it is pinned here.
"""
from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from protocore.runtime.context.compaction import estimate_message_tokens
from protocore.runtime.context.session_memory import SessionMemory
from protocore.runtime.context.session_seed import build_session_seed, plan_session_fold

_TEXT = "the quick brown fox " * 20


def _text_only() -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=_TEXT)])


def _with_thinking_and_image() -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            TextBlock(text=_TEXT),
            ThinkingBlock(text=_TEXT),
            ImageRefBlock(blob_ref="blob-1"),
        ],
        reasoning_content=_TEXT,
    )


def _question() -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text="question")])


def test_thinking_and_image_blocks_cost_tokens() -> None:
    rc = LoopConstants()
    text_only = estimate_message_tokens(_text_only(), rc)
    enriched = estimate_message_tokens(_with_thinking_and_image(), rc)
    assert text_only > 0
    assert enriched == text_only * 3 + rc.token_count_image_tokens


def test_the_seed_and_the_plan_size_turns_on_the_shared_scale() -> None:
    rc = LoopConstants(session_memory_running_summary_token_cap=0)
    plain = [_question(), _text_only()]
    rich = [_question(), _with_thinking_and_image()]
    memory = SessionMemory(turn_index=1)

    def position(message: Message) -> int | None:
        return None

    plain_seed = build_session_seed(plain, memory, 100_000, rc, position=position)
    rich_seed = build_session_seed(rich, memory, 100_000, rc, position=position)
    assert rich_seed.tokens == sum(estimate_message_tokens(m, rc) for m in rich_seed.messages)
    assert rich_seed.tokens > plain_seed.tokens

    assert (
        plan_session_fold(rich, memory, 100_000, rc, position=position).projected_tokens
        > plan_session_fold(plain, memory, 100_000, rc, position=position).projected_tokens
    )
