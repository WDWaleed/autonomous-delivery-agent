"""Uniform-Cost Search / Dijkstra on State Space (§7.4, D15, M4.3).

Optimal in accumulated path cost under strictly positive edge costs (D15).
Graph search with min-heap priority queue, lazy deletion, and best-g tracking.
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Optional, Tuple

from delivery.search.core import Limits, Metrics, Node, SearchResult, SearchStatus, reconstruct_path
from delivery.search.problem import DeliveryProblem


def uniform_cost_search(
    problem: DeliveryProblem,
    limits: Optional[Limits] = None,
) -> SearchResult:
    """Executes Uniform-Cost Graph Search.

    Optimal in accumulated step cost under strictly positive edge costs (D15).
    Goal test is performed upon expansion of nodes popped from priority queue.
    """
    active_limits = limits or Limits()
    metrics = Metrics()
    start_time = time.perf_counter()

    root = Node.root(problem.initial_state())
    metrics.generated_nodes = 1

    # Frontier storing (cost, depth, action_seq, node)
    frontier: List[Tuple[float, int, Tuple[int, ...], Node]] = []
    heapq.heappush(frontier, (root.cost, root.depth, root.action_seq, root))

    # best_g: state -> lowest cost reached
    best_g: Dict[Tuple[int, int, int, Tuple[Any, ...]], float] = {
        root.state: root.cost
    }

    while frontier:
        # Resource limit checks under D13 and D18
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

        elapsed = time.perf_counter() - start_time
        if active_limits.max_time_seconds and elapsed > active_limits.max_time_seconds:
            metrics.duration_seconds = elapsed
            return SearchResult(
                status=SearchStatus.TIMEOUT,
                metrics=metrics,
                failure_reason=f"Time limit exceeded ({elapsed:.2f}s > {active_limits.max_time_seconds}s)",
            )

        cost, depth, seq, node = heapq.heappop(frontier)

        # Lazy deletion: if node cost is strictly worse than best recorded for this state, discard
        if cost > best_g.get(node.state, float("inf")):
            continue

        # Goal test upon expansion
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

            if next_state not in best_g or child_node.cost < best_g[next_state]:
                best_g[next_state] = child_node.cost
                heapq.heappush(frontier, (child_node.cost, child_node.depth, child_node.action_seq, child_node))

    metrics.duration_seconds = time.perf_counter() - start_time
    return SearchResult(
        status=SearchStatus.FAILED,
        metrics=metrics,
        failure_reason="Search space exhausted without finding a goal.",
    )
