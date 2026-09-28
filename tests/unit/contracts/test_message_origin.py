"""Compaction's records carry an origin of their own, so a host never takes them for the user's turn."""
from __future__ import annotations

import json

from protocore import MessageOrigin
from protocore.contracts.observability import model_visible
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_LEDGER_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.compact_checkpoint import CompactCheckpoint, apply_checkpoint
from protocore.runtime.context.compaction import CompactionState
from protocore.runtime.context.ledger import is_ledger
from protocore.runtime.context.manager import ContextManager
from protocore.tests_support.adapters import InMemoryBlobStore


def _user(text: str, **metadata: object) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)], metadata=dict(metadata))


def test_a_plain_message_is_the_conversations() -> None:
    assert _user("hello").origin is MessageOrigin.conversation
    assert Message(role=MessageRole.assistant).origin is MessageOrigin.conversation


def test_a_summary_and_a_ledger_are_compactions() -> None:
    assert _user("s", **{COMPACTION_SUMMARY_METADATA_KEY: True}).origin is MessageOrigin.compaction
    assert _user("l", **{COMPACTION_LEDGER_METADATA_KEY: {}}).origin is MessageOrigin.compaction
    # The summary tag is a flag: anything but True is not one.
    assert _user("x", **{COMPACTION_SUMMARY_METADATA_KEY: "yes"}).origin is MessageOrigin.conversation


def test_the_origin_is_serialised_but_derived_on_the_way_back() -> None:
    record = _user("s", **{COMPACTION_SUMMARY_METADATA_KEY: True})
    dumped = record.model_dump(mode="json")
    assert dumped["origin"] == "compaction"
    assert Message.model_validate(dumped).origin is MessageOrigin.compaction

    # A claimed origin without the tag is not believed.
    forged = {**_user("hi").model_dump(mode="json"), "origin": "compaction"}
    assert Message.model_validate(forged).origin is MessageOrigin.conversation


def test_the_origin_is_never_shown_to_a_provider() -> None:
    record = _user("s", **{COMPACTION_SUMMARY_METADATA_KEY: True})
    assert "origin" not in model_visible(record)


def test_the_checkpoint_summary_is_a_compaction_record() -> None:
    history = [_user("task"), Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")])]
    checkpoint = CompactCheckpoint(entry_id="c", summary="compacted 1 messages", retained_from_index=1)
    request = apply_checkpoint(history, checkpoint)
    assert request[0].origin is MessageOrigin.compaction
    assert request[1].origin is MessageOrigin.conversation


async def test_everything_a_pass_writes_into_history_is_marked_and_rides_the_snapshot_shape() -> None:
    history = [_user("Audit the ledger at /srv/ledger.db.")]
    for n in range(4):
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
                content_blocks=[ToolResultBlock(tool_call_id=call, content=f"see https://ci.example.org/job/{n}\n" + "row\n" * 3000)],
            )
        )
    history.append(_user("Now write the report."))
    originals = {id(m) for m in history}

    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=2)
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)
    await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t",
        model_name="m",
        reactive=True,
    )

    written = [m for m in history if id(m) not in originals and m.role is MessageRole.user]
    assert any(is_ledger(m) for m in written)
    assert all(m.origin is MessageOrigin.compaction for m in written)
    # The caller's own turns keep theirs.
    assert history[0].origin is MessageOrigin.conversation
    assert history[-1].origin is MessageOrigin.conversation
    dumped = [m.model_dump(mode="json") for m in history]
    assert [d["origin"] for d in dumped] == [m.origin.value for m in history]
