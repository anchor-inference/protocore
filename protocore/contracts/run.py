"""IRunStore Protocol — durable per-run state.

Reference shape: a short summary record paired with a detail blob held
wherever the host keeps large values.

``flush_terminal`` is called once a run reaches a terminal status
(completed/error/cancelled) to write the detail blob and update the summary.

``IRunToolErrorCounter`` is the narrow protocol the core
:class:`~protocore.runtime.tool_dispatch.ToolDispatcher` uses to record
per-run tool dispatch errors. Reference shape: one atomic increment of a
per-run counter. Keeping
the protocol in core lets the dispatcher remain host-agnostic while
still surfacing the per-run counter for the terminal classifier.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from protocore.contracts.types import Run, RunStatus


class RunStoreError(Exception):
    """Base for run-store domain errors."""


class RunNotFoundError(RunStoreError):
    """Requested run_id does not exist."""


@runtime_checkable
class IRunStore(Protocol):
    """Adapter Protocol over the PG ``runs`` table."""

    async def create(self, run: Run) -> None:
        """Create a new run row."""
        ...

    async def get(self, run_id: str, tenant_id: str) -> Run:
        """Fetch run row. Raise :class:`RunNotFoundError`."""
        ...

    async def update_status(self, run_id: str, tenant_id: str, status: RunStatus) -> None:
        """Update run status. Validates legal transitions."""
        ...

    async def list(
        self,
        tenant_id: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> Sequence[Run]:
        """List runs for a tenant (filtered, paginated)."""
        ...

    async def flush_terminal(
        self,
        run_id: str,
        tenant_id: str,
        detail_blob_ref: str,
    ) -> None:
        """Mark run as flushed to durable storage; bind detail blob ref."""
        ...


@runtime_checkable
class IRunToolErrorCounter(Protocol):
    """Adapter Protocol for atomic increments of ``runs.tool_errors_count``.

    Implementations MUST be atomic (single SQL ``UPDATE ... SET col = col + N``
    or equivalent), idempotent under concurrent calls, and non-raising — the
    dispatcher swallows counter failures so a flaky telemetry plane never
    corrupts the agent loop.
    """

    async def increment_tool_errors_count(self, run_id: str, by: int = 1) -> None:
        """Increment ``runs.tool_errors_count`` for ``run_id`` by ``by``.

        ``by`` must be a positive integer; ``ValueError`` if zero or negative.
        Missing ``run_id`` is a no-op (the row may not exist yet during early
        bootstrap or in unit tests with mocked rows).
        """
        ...


@runtime_checkable
class IRunRefusedCallCounter(Protocol):
    """Adapter Protocol for atomic increments of a run's refused-call tally.

    A call the run's wind-down turned away never reached a tool: nothing
    failed, the tool was withdrawn. It is therefore NOT a tool error and is not
    counted in ``tool_errors_count`` — a host that reports a run as partial
    when a tool failed keeps reading that count as it always did. The refusals
    are tallied here instead, so a run whose model kept asking for withdrawn
    tools can still be told apart from one that simply finished.

    Implementations MUST be atomic and non-raising, like
    :class:`IRunToolErrorCounter`; the loop swallows a failure of this
    telemetry rather than let it reach the run.
    """

    async def increment_refused_tool_calls_count(
        self, run_id: str, by: int = 1
    ) -> None:
        """Add ``by`` (a positive integer) to the refused-call tally of ``run_id``.

        A subagent's refusals are attributed to the root run, the id the host
        keeps a row for. A missing ``run_id`` is a no-op.
        """
        ...


__all__ = [
    "IRunRefusedCallCounter",
    "IRunStore",
    "IRunToolErrorCounter",
    "RunNotFoundError",
    "RunStoreError",
]
