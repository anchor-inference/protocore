"""``tool_retrieval_top_k = 0`` means "no clip" wherever the value is used.

The constant documents 0 as turning the clip off, and it is the default. The
loop translates it (``top_k or None``), but the registry, which takes the same
number under the same name, reads 0 as "retrieve no tool": a host that hands
it the constant as-is advertises the pinned tools and nothing else.
"""

from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.runtime.tool_registry import ToolRegistry

from ._tool_fixtures import MockTool


@pytest.mark.xfail(
    strict=True,
    reason="ToolRegistry.compute_effective_surface treats top_k=0 as a clip to the pinned tools",
)
def test_the_default_top_k_leaves_the_surface_unclipped() -> None:
    reg = ToolRegistry([MockTool(tool_name=name) for name in ("Alpha", "Bravo", "Charlie")])
    policy = ToolVisibilityPolicy(pinned={"Alpha"})
    unclipped = [d.name for d in reg.compute_effective_surface("t", policy, query="bravo", top_k=None)]

    top_k = LoopConstants().tool_retrieval_top_k
    assert top_k == 0
    clipped = [d.name for d in reg.compute_effective_surface("t", policy, query="bravo", top_k=top_k)]

    assert clipped == unclipped
