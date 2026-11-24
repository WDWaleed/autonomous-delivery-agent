"""Solvability feasibility certificate engine (§7.4, D8, M4.5).

Produces honest solvability certificates for instances with all twists enabled:
- FEASIBLE: Proven reachable with a concrete plan found by A* or UCS.
- INFEASIBLE: Proven unreachable by exhaustive search of the reachable state space.
- UNKNOWN: Search interrupted by resource limits (time, nodes, memory) without conclusive proof.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from delivery.heuristics.baselines import MaxRelaxedDistanceHeuristic
from delivery.instance import Instance
from delivery.search.astar import astar_search
from delivery.search.core import Limits, SearchResult, SearchStatus
from delivery.search.problem import DeliveryProblem
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld


class FeasibilityStatus(str, enum.Enum):
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class FeasibilityCertificate:
    """Honest solvability certificate for an instance under configured twists."""

    instance_id: str
    status: FeasibilityStatus
    plan: Optional[List[int]] = None
    cost: Optional[float] = None
    trajectory: Optional[List[Tuple[int, int]]] = None
    duration_seconds: float = 0.0
    expanded_nodes: int = 0
    generated_nodes: int = 0
    search_status: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "status": self.status.value,
            "feasible": self.status == FeasibilityStatus.FEASIBLE,
            "cost": self.cost,
            "plan_length": len(self.plan) if self.plan is not None else 0,
            "duration_seconds": self.duration_seconds,
            "expanded_nodes": self.expanded_nodes,
            "generated_nodes": self.generated_nodes,
            "search_status": self.search_status,
            "reason": self.reason,
        }


def certify_feasibility(
    instance: Instance,
    limits: Optional[Limits] = None,
    algo: str = "astar",
) -> FeasibilityCertificate:
    """Evaluates and certifies whether an instance is FEASIBLE, INFEASIBLE, or UNKNOWN."""
    world = GridWorld(instance)
    twist_stack = build_twist_stack(instance.twists, world=world)
    problem = DeliveryProblem(instance, world=world, twist_stack=twist_stack)

    # Use A* with admissible B2 (or UCS if requested)
    active_limits = limits or Limits(max_time_seconds=30.0, max_expanded=500_000)

    if algo == "ucs":
        from delivery.search.ucs import uniform_cost_search
        result = uniform_cost_search(problem, limits=active_limits)
    else:
        h_b2 = MaxRelaxedDistanceHeuristic(problem)
        result = astar_search(problem, h_b2, limits=active_limits)

    if result.status == SearchStatus.SUCCESS:
        return FeasibilityCertificate(
            instance_id=instance.id,
            status=FeasibilityStatus.FEASIBLE,
            plan=result.plan,
            cost=result.cost,
            trajectory=result.trajectory,
            duration_seconds=result.metrics.duration_seconds,
            expanded_nodes=result.metrics.expanded_nodes,
            generated_nodes=result.metrics.generated_nodes,
            search_status=result.status.value,
            reason="Goal state found with valid plan.",
        )
    elif result.status == SearchStatus.FAILED:
        return FeasibilityCertificate(
            instance_id=instance.id,
            status=FeasibilityStatus.INFEASIBLE,
            plan=None,
            cost=None,
            trajectory=None,
            duration_seconds=result.metrics.duration_seconds,
            expanded_nodes=result.metrics.expanded_nodes,
            generated_nodes=result.metrics.generated_nodes,
            search_status=result.status.value,
            reason="Reachable state space exhausted without encountering goal state.",
        )
    else:
        # TIMEOUT, NODELIMIT, MEMORYLIMIT
        return FeasibilityCertificate(
            instance_id=instance.id,
            status=FeasibilityStatus.UNKNOWN,
            plan=None,
            cost=None,
            trajectory=None,
            duration_seconds=result.metrics.duration_seconds,
            expanded_nodes=result.metrics.expanded_nodes,
            generated_nodes=result.metrics.generated_nodes,
            search_status=result.status.value,
            reason=f"Search interrupted by resource bound ({result.status.value}); conclusive proof not reached.",
        )
