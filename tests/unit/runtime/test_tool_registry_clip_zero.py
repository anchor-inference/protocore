"""``tool_retrieval_top_k = 0`` means "no clip" wherever the value is used.

The constant documents 0 as turning the clip off, and it is the default. The
registry takes the same number under the same name, so it reads 0 the same
way: a host that hands it the constant as-is advertises every visible tool,
not the pinned ones alone.
"""

from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tests_support.adapters import InMemoryToolRegistry

from ._tool_fixtures import MockTool


def test_the_default_top_k_leaves_the_surface_unclipped() -> None:
    reg = ToolRegistry([MockTool(tool_name=name) for name in ("Alpha", "Bravo", "Charlie")])
    policy = ToolVisibilityPolicy(pinned={"Alpha"})
    unclipped = [d.name for d in reg.compute_effective_surface("t", policy, query="bravo", top_k=None)]

    top_k = LoopConstants().tool_retrieval_top_k
    assert top_k == 0
    clipped = [d.name for d in reg.compute_effective_surface("t", policy, query="bravo", top_k=top_k)]

    assert clipped == unclipped


def test_in_memory_registry_reads_zero_as_no_clip() -> None:
    reg = InMemoryToolRegistry()
    for name in ("Alpha", "Bravo", "Charlie"):
        reg.register(MockTool(tool_name=name))
    policy = ToolVisibilityPolicy(pinned={"Alpha"})
    surface = [d.name for d in reg.compute_effective_surface("t", policy, top_k=0)]
    assert surface == ["Alpha", "Bravo", "Charlie"]
