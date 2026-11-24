"""Greedy Best-First Search (§7.4, M4.4).

Evaluates nodes solely by heuristic estimate h(n).
Fast heuristic exploration without optimality guarantee.
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from delivery.heuristics.base import Heuristic
from delivery.search.core import Limits, Metrics, Node, SearchResult, SearchStatus, reconstruct_path
from delivery.search.problem import DeliveryProblem


def greedy_best_first_search(
    problem: DeliveryProblem,
    heuristic: Heuristic,
    limits: Optional[Limits] = None,
) -> SearchResult:
    """Executes Greedy Best-First Search using the provided heuristic.

    Prioritizes nodes solely by estimated cost to goal h(n).
    Fast search, but does not guarantee cost or depth optimality.
    """
    active_limits = limits or Limits()
    metrics = Metrics()
    start_time = time.perf_counter()

    root = Node.root(problem.initial_state())
    metrics.generated_nodes = 1

    h_root = float(heuristic(root.state, problem))
    # Priority entry: (h_val, depth, action_seq, node)
    frontier: List[Tuple[float, int, Tuple[int, ...], Node]] = []
    heapq.heappush(frontier, (h_root, root.depth, root.action_seq, root))

    explored: Set[Tuple[int, int, int, Tuple[Any, ...]]] = set()

    while frontier:
        current_stored = len(frontier) + len(explored)
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

        h_val, depth, seq, node = heapq.heappop(frontier)

        if node.state in explored:
            continue

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
        explored.add(node.state)

        for next_state, action, step_cost in problem.successors(node.state):
            if next_state in explored:
                continue

            child_node = node.child(next_state, action, step_cost)
            metrics.generated_nodes += 1

            if active_limits.max_cost and child_node.cost > active_limits.max_cost:
                continue

            h_child = float(heuristic(next_state, problem))
            heapq.heappush(frontier, (h_child, child_node.depth, child_node.action_seq, child_node))

    metrics.duration_seconds = time.perf_counter() - start_time
    return SearchResult(
        status=SearchStatus.FAILED,
        metrics=metrics,
        failure_reason="Search space exhausted without finding a goal.",
    )
