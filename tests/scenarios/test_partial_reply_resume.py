"""A retry after a stream that dropped mid-reply does not end on that reply.

The partial reply stays in the history, so the transcript agrees with what the
reader saw, and the retry is built on that history. A request whose last
message is an assistant message is a prefill to some providers, which refuse
it, and with it the retry, the wind-down and the run; a chat template renders
a second assistant turn after the first. The retry is closed with a short user
note instead, in that request only.
"""
from __future__ import annotations

from protocore.contracts.llm import LLMStreamIdleError
from protocore.contracts.turn_context import VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY
from protocore.contracts.types import (
    PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PARTIAL_REPLY_RESUME,
    Message,
    MessageRole,
    TextBlock,
)
from protocore.runtime.query import _close_on_partial_reply

from .conftest import FailingProvider, ProviderChainDouble, ScenarioFactory, classified, default_rc


def _idle() -> BaseException:
    return classified(LLMStreamIdleError("went quiet"), "timeout")


async def test_the_retry_after_a_dropped_reply_ends_on_a_user_note(scenario: ScenarioFactory) -> None:
    primary = FailingProvider(failures={0: _idle()}, partial_text="the first half")
    fallback = FailingProvider(failures={}, text="unused")
    chain = ProviderChainDouble([("primary-model", primary), ("fallback-model", fallback)])
    run = scenario(
        rc=default_rc(llm_transient_error_retry_backoff_base_seconds=0.0),
        llm_provider=primary,
        provider_chain=chain,
        model_name="primary-model",
    )

    await run.run("ask")

    assert len(primary.calls) == 2
    retry = primary.calls[1]
    *_, partial, note = retry.messages
    assert partial.role is MessageRole.assistant
    assert partial.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
    assert "the first half" in partial.text
    assert note.role is MessageRole.user
    assert note.metadata[SYNTHETIC_RECOVERY_METADATA_KEY] == SYNTHETIC_RECOVERY_PARTIAL_REPLY_RESUME
    assert "cut off" in note.text
    # Request-only: counted as volatile, never written to the history.
    assert retry.extra[VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY] == 1
    assert not any(
        message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY) == SYNTHETIC_RECOVERY_PARTIAL_REPLY_RESUME
        for message in run.engine.history
    )
    # The first request had nothing to close.
    assert primary.calls[0].messages[-1].role is MessageRole.user
    assert VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY not in primary.calls[0].extra


async def test_only_a_cut_reply_is_closed(scenario: ScenarioFactory) -> None:
    run = scenario()
    finished = Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="done")])
    cut = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="half")],
        metadata={PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True},
    )
    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])

    assert _close_on_partial_reply(run.engine, []) == ([], 0)
    assert _close_on_partial_reply(run.engine, [user, finished]) == ([user, finished], 0)
    assert _close_on_partial_reply(run.engine, [cut, user]) == ([cut, user], 0)
    closed, added = _close_on_partial_reply(run.engine, [user, cut])
    assert added == 1
    assert closed[:2] == [user, cut]
    assert closed[2].role is MessageRole.user
