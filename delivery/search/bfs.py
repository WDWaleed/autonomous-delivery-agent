"""Breadth-First Graph Search (§7.4, M4.3).

Optimal in the number of actions (shallowest goal depth).
Graph search with FIFO queue, explored set, and resource limit accounting.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from delivery.search.core import Limits, Metrics, Node, SearchResult, SearchStatus, reconstruct_path
from delivery.search.problem import DeliveryProblem


def breadth_first_search(
    problem: DeliveryProblem,
    limits: Optional[Limits] = None,
) -> SearchResult:
    """Executes Breadth-First Graph Search.

    Optimal in action count (solution length), not necessarily path cost.
    Early goal test on generation per standard textbook BFS.
    """
    active_limits = limits or Limits()
    metrics = Metrics()
    start_time = time.perf_counter()

    root = Node.root(problem.initial_state())
    metrics.generated_nodes = 1

    # Early goal test at root
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

    queue = deque([root])
    # Track states in frontier or explored
    seen = {root.state}

    while queue:
        # Resource limit checks under D13 and D18
        current_stored = len(queue) + len(seen)
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

        if active_limits.max_expanded and metrics.expanded_nodes >= active_limits.max_expanded:
            metrics.duration_seconds = time.perf_counter() - start_time
            return SearchResult(
                status=SearchStatus.NODELIMIT,
                metrics=metrics,
                failure_reason=f"Expanded node limit exceeded ({metrics.expanded_nodes} >= {active_limits.max_expanded})",
            )

        node = queue.popleft()
        metrics.expanded_nodes += 1

        for next_state, action, step_cost in problem.successors(node.state):
            if next_state in seen:
                continue

            child_node = node.child(next_state, action, step_cost)
            metrics.generated_nodes += 1

            if active_limits.max_cost and child_node.cost > active_limits.max_cost:
                continue

            # Early goal test on child generation
            if problem.is_goal(next_state):
                plan, trajectory, total_cost = reconstruct_path(child_node)
                metrics.duration_seconds = time.perf_counter() - start_time
                return SearchResult(
                    status=SearchStatus.SUCCESS,
                    plan=plan,
                    trajectory=trajectory,
                    cost=total_cost,
                    length=len(plan),
                    metrics=metrics,
                )

            seen.add(next_state)
            queue.append(child_node)

    metrics.duration_seconds = time.perf_counter() - start_time
    return SearchResult(
        status=SearchStatus.FAILED,
        metrics=metrics,
        failure_reason="Search space exhausted without finding a goal.",
    )
