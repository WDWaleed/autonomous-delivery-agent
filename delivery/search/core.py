"""Core search nodes, metrics, limit handlers, and canonical optimal planning (§7.4, D7, D13, D15, D18, M4.2).

Governed by decisions D7, D13, D15, and D18.
Provides Node representation, Limit checkers, SearchResult, and the lexicographic
canonical_optimal_plan solver.
"""

from __future__ import annotations

import enum
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from delivery.search.problem import ACTION_NAMES, DeliveryProblem


class SearchStatus(str, enum.Enum):
    """Reason search terminated."""

    SUCCESS = "SUCCESS"
    CUTOFF = "CUTOFF"
    TIMEOUT = "TIMEOUT"
    NODELIMIT = "NODELIMIT"
    MEMORYLIMIT = "MEMORYLIMIT"
    COSTLIMIT = "COSTLIMIT"
    FAILED = "FAILED"


class SearchLimitException(Exception):
    """Raised when a search resource limit is exceeded."""

    def __init__(self, status: SearchStatus, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Limits:
    """Resource bounds governing search execution (D13, D18)."""

    max_expanded: Optional[int] = 2_000_000  # Default expanded node limit (D13)
    max_time_seconds: Optional[float] = 60.0  # Default time budget (D13)
    max_stored_records: Optional[int] = None  # Stored-record budget (D18)
    max_cost: Optional[float] = None  # Optional path-cost bound


@dataclass
class Metrics:
    """Resource accounting and execution metrics (§7.4, D18)."""

    expanded_nodes: int = 0
    generated_nodes: int = 0
    peak_stored: int = 0  # Peak stored records in frontier + explored (D18)
    duration_seconds: float = 0.0
    peak_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expanded_nodes": self.expanded_nodes,
            "generated_nodes": self.generated_nodes,
            "peak_stored": self.peak_stored,
            "duration_seconds": round(self.duration_seconds, 4),
            "peak_bytes": self.peak_bytes,
        }


class Node:
    """Search tree node with memory-efficient slots and lexicographic action sequence tracking."""

    __slots__ = ("state", "parent", "action", "cost", "depth", "action_seq")

    def __init__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        parent: Optional[Node] = None,
        action: Optional[int] = None,
        cost: float = 0.0,
        depth: int = 0,
        action_seq: Tuple[int, ...] = (),
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = depth
        self.action_seq = action_seq

    @classmethod
    def root(cls, initial_state: Tuple[int, int, int, Tuple[Any, ...]]) -> Node:
        return cls(state=initial_state, parent=None, action=None, cost=0.0, depth=0, action_seq=())

    def child(
        self,
        next_state: Tuple[int, int, int, Tuple[Any, ...]],
        action: int,
        step_cost: float,
    ) -> Node:
        if step_cost <= 0 or not np_isfinite(step_cost):
            raise ValueError(f"Step cost must be strictly positive and finite (D15), got {step_cost}")
        return Node(
            state=next_state,
            parent=self,
            action=action,
            cost=self.cost + step_cost,
            depth=self.depth + 1,
            action_seq=self.action_seq + (action,),
        )

    def __lt__(self, other: Node) -> bool:
        # Lexicographic tie-breaking: cost, depth, action_seq
        if abs(self.cost - other.cost) > 1e-9:
            return self.cost < other.cost
        if self.depth != other.depth:
            return self.depth < other.depth
        return self.action_seq < other.action_seq


def np_isfinite(val: float) -> bool:
    import math
    return not math.isnan(val) and not math.isinf(val)


@dataclass
class SearchResult:
    """Standardized search execution outcome (§7.4)."""

    status: SearchStatus
    plan: Optional[List[int]] = None
    trajectory: Optional[List[Tuple[int, int]]] = None
    cost: float = float("inf")
    length: int = 0
    metrics: Metrics = field(default_factory=Metrics)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "success": self.status == SearchStatus.SUCCESS,
            "cost": self.cost if np_isfinite(self.cost) else None,
            "length": self.length,
            "plan": [ACTION_NAMES.get(a, str(a)) for a in self.plan] if self.plan else None,
            "trajectory": [list(p) for p in self.trajectory] if self.trajectory else None,
            "metrics": self.metrics.to_dict(),
            "failure_reason": self.failure_reason,
        }


def reconstruct_path(goal_node: Node) -> Tuple[List[int], List[Tuple[int, int]], float]:
    """Reconstructs action plan, coordinate trajectory, and total cost from goal node."""
    actions: List[int] = []
    trajectory: List[Tuple[int, int]] = []
    curr: Optional[Node] = goal_node

    while curr is not None:
        trajectory.append((curr.state[0], curr.state[1]))
        if curr.action is not None:
            actions.append(curr.action)
        curr = curr.parent

    actions.reverse()
    trajectory.reverse()
    return actions, trajectory, goal_node.cost


def canonical_optimal_plan(
    problem: DeliveryProblem,
    limits: Optional[Limits] = None,
) -> SearchResult:
    """Computes the unique canonical optimal plan per Decision D7.

    Minimises lexicographically: (total_cost, action_count, action_rank_sequence),
    using fixed cardinal action order N, E, S, W, then WAIT.
    Used for generating C* and L* ground-truth baselines.
    """
    import heapq

    active_limits = limits or Limits()
    metrics = Metrics()
    start_time = time.perf_counter()

    root = Node.root(problem.initial_state())
    if problem.is_goal(root.state):
        metrics.duration_seconds = time.perf_counter() - start_time
        return SearchResult(
            status=SearchStatus.SUCCESS,
            plan=[],
            trajectory=[(root.state[0], root.state[1])],
            cost=0.0,
            length=0,
            metrics=metrics,
        )

    # Priority entry: (cost, length, action_seq, node)
    frontier: List[Tuple[float, int, Tuple[int, ...], Node]] = []
    heapq.heappush(frontier, (root.cost, root.depth, root.action_seq, root))
    metrics.generated_nodes = 1

    # best_g: state -> (cost, length, action_seq)
    best_g: Dict[Tuple[int, int, int, Tuple[Any, ...]], Tuple[float, int, Tuple[int, ...]]] = {
        root.state: (root.cost, root.depth, root.action_seq)
    }

    while frontier:
        # Check stored records limit under D18
        current_stored = len(frontier) + len(best_g)
        if current_stored > metrics.peak_stored:
            metrics.peak_stored = current_stored
        if active_limits.max_stored_records and current_stored > active_limits.max_stored_records:
            metrics.duration_seconds = time.perf_counter() - start_time
            return SearchResult(
                status=SearchStatus.MEMORYLIMIT,
                metrics=metrics,
                failure_reason=f"Stored record budget exceeded ({current_stored} > {active_limits.max_stored_records})",
            )

        # Check time limit
        elapsed = time.perf_counter() - start_time
        if active_limits.max_time_seconds and elapsed > active_limits.max_time_seconds:
            metrics.duration_seconds = elapsed
            return SearchResult(
                status=SearchStatus.TIMEOUT,
                metrics=metrics,
                failure_reason=f"Time limit exceeded ({elapsed:.2f}s > {active_limits.max_time_seconds}s)",
            )

        c, l, seq, node = heapq.heappop(frontier)

        # If current tuple is worse than best recorded for state, prune
        if (c, l, seq) > best_g.get(node.state, (float("inf"), 0, ())):
            continue

        # Goal check upon expansion
        if problem.is_goal(node.state):
            plan, trajectory, total_cost = reconstruct_path(node)
            metrics.duration_seconds = time.perf_counter() - start_time
            return SearchResult(
                status=SearchStatus.SUCCESS,
                plan=plan,
                trajectory=trajectory,
                cost=total_cost,
                length=len(plan),
                metrics=metrics,
            )

        # Check node expansion limit
        if active_limits.max_expanded and metrics.expanded_nodes >= active_limits.max_expanded:
            metrics.duration_seconds = time.perf_counter() - start_time
            return SearchResult(
                status=SearchStatus.NODELIMIT,
                metrics=metrics,
                failure_reason=f"Expanded node limit exceeded ({metrics.expanded_nodes} >= {active_limits.max_expanded})",
            )

        metrics.expanded_nodes += 1

        for next_state, action, step_cost in problem.successors(node.state):
            child_node = node.child(next_state, action, step_cost)
            metrics.generated_nodes += 1

            if active_limits.max_cost and child_node.cost > active_limits.max_cost:
                continue

            child_key = (child_node.cost, child_node.depth, child_node.action_seq)
            if next_state not in best_g or child_key < best_g[next_state]:
                best_g[next_state] = child_key
                heapq.heappush(frontier, (child_node.cost, child_node.depth, child_node.action_seq, child_node))

    metrics.duration_seconds = time.perf_counter() - start_time
    return SearchResult(
        status=SearchStatus.FAILED,
        metrics=metrics,
        failure_reason="Search space exhausted without finding a goal.",
    )
