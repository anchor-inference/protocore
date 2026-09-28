"""Compaction's own user-role messages must not start a new round.

``_this_round_messages`` takes the round to begin after the last user-role
message that is not a runtime nudge. The ledger, the summaries and the floor
digest are user-role messages written by the runtime, and they are not tagged
as nudges, so after a pass the "round" is only the kept tail. A run that did
real work, whose request was refused for length (large tool outputs, then
the runtime's continue prompt) and compacted with the emergency keep window
of one message, gets the ledger placed in front of that prompt and then reads
as a run that produced nothing, and the provider-failure policy fails it instead of winding it down.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import CompactionState
from protocore.runtime.context.manager import ContextManager
from protocore.runtime.query import _run_produced_output
from protocore.tests_support.adapters import InMemoryBlobStore


def _history() -> list[Message]:
    history = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="Audit the ledger at /srv/ledger.db.")])]
    for n in range(3):
        call = f"call-{n}"
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
                content_blocks=[ToolResultBlock(tool_call_id=call, content=f"fetched https://ci.example.org/job/{n}/log\n" + "\n".join(f"line {i} of the build log for section {n}" for i in range(1500)))],
            )
        )
    # The runtime's own continue prompt after a reasoning-only round.
    history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="Continue: answer or call a tool.")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: "continue_prompt"},
        )
    )
    return history


async def test_a_run_compacted_after_a_refusal_still_reads_as_having_produced_output() -> None:
    history = _history()
    engine = SimpleNamespace(history=history)
    assert _run_produced_output(engine) is True  # type: ignore[arg-type]

    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=2)
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t",
        model_name="m",
        reactive=True,
    )
    # The ledger now stands between the run's work and the continue prompt.
    assert history[-2].metadata.get("protocore.compaction_ledger") is not None
    # The run's rounds of tool work are still this run's work.
    assert _run_produced_output(engine) is True  # type: ignore[arg-type]


def _engine(*messages: Message) -> SimpleNamespace:
    return SimpleNamespace(history=list(messages))


def _say(role: MessageRole, text: str, **metadata: object) -> Message:
    return Message(role=role, content_blocks=[TextBlock(text=text)], metadata=dict(metadata))


def test_an_answer_written_before_the_ledger_is_still_this_rounds() -> None:
    from protocore.runtime.context.ledger import LEDGER_METADATA_KEY
    from protocore.runtime.query import _this_round_messages

    answer = _say(MessageRole.assistant, "The ledger balances; three sections checked.")
    engine = _engine(
        _say(MessageRole.user, "Audit the ledger."),
        answer,
        _say(MessageRole.user, "<compaction-ledger>", **{LEDGER_METADATA_KEY: {}}),
        _say(MessageRole.user, "Call the finalizing tool.", **{SYNTHETIC_RECOVERY_METADATA_KEY: "terminal_tool_nudge"}),
    )
    assert answer in _this_round_messages(engine)  # type: ignore[arg-type]


def test_a_caller_message_after_a_compaction_record_still_starts_a_round() -> None:
    from protocore.contracts.types import COMPACTION_SUMMARY_METADATA_KEY

    engine = _engine(
        _say(MessageRole.user, "Audit the ledger."),
        _say(MessageRole.assistant, "Done: it balances."),
        _say(MessageRole.user, "<compacted-turn>", **{COMPACTION_SUMMARY_METADATA_KEY: True}),
        _say(MessageRole.user, "And the archive?"),
    )
    assert _run_produced_output(engine) is False  # type: ignore[arg-type]


def test_a_rounds_work_compacted_into_a_summary_is_still_output() -> None:
    from protocore.contracts.types import COMPACTION_SUMMARY_METADATA_KEY

    engine = _engine(
        _say(MessageRole.user, "Audit the ledger."),
        _say(MessageRole.user, "<compacted-turn> read three sections", **{COMPACTION_SUMMARY_METADATA_KEY: True}),
        _say(MessageRole.user, "Continue.", **{SYNTHETIC_RECOVERY_METADATA_KEY: "continue_prompt"}),
    )
    assert _run_produced_output(engine) is True  # type: ignore[arg-type]
