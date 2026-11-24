"""Depth-Limited Search with cycle avoidance (§7.4, D18, M4.3).

Explicit stack search with on-path cycle avoidance and distinct CUTOFF vs FAILED statuses.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from delivery.search.core import Limits, Metrics, Node, SearchResult, SearchStatus, reconstruct_path
from delivery.search.problem import DeliveryProblem


def depth_limited_search(
    problem: DeliveryProblem,
    depth_limit: int,
    limits: Optional[Limits] = None,
    shared_metrics: Optional[Metrics] = None,
    start_time: Optional[float] = None,
    check_cycle: bool = True,
) -> SearchResult:
    """Executes Depth-Limited Search to a specified depth bound.

    Returns distinctly:
    - SUCCESS: Goal found at or before depth_limit.
    - CUTOFF: Goal not found, but at least one branch was truncated by depth_limit.
    - FAILED: Entire reachable space searched without finding goal and without any cutoff.
    """
    active_limits = limits or Limits()
    metrics = shared_metrics or Metrics()
    t0 = start_time or time.perf_counter()

    root = Node.root(problem.initial_state())
    metrics.generated_nodes += 1

    if problem.is_goal(root.state):
        metrics.duration_seconds = time.perf_counter() - t0
        return SearchResult(
            status=SearchStatus.SUCCESS,
            plan=[],
            trajectory=[(root.state[0], root.state[1])],
            cost=0.0,
            length=0,
            metrics=metrics,
        )

    if depth_limit == 0:
        return SearchResult(
            status=SearchStatus.CUTOFF,
            metrics=metrics,
            failure_reason="Root is not a goal and depth_limit=0",
        )

    # Stack holds tuples: (node, iterator_of_successors)
    # on_path set avoids cycles along current search branch
    stack: List[Tuple[Node, List[Tuple[Tuple[int, int, int, Tuple[Any, ...]], int, float]], int]] = []
    initial_succs = problem.successors(root.state)
    stack.append((root, initial_succs, 0))
    on_path: Set[Tuple[int, int, int, Tuple[Any, ...]]] = {root.state} if check_cycle else set()

    cutoff_occurred = False

    while stack:
        current_stored = len(stack) + len(on_path)
        if current_stored > metrics.peak_stored:
            metrics.peak_stored = current_stored
        if active_limits.max_stored_records and current_stored > active_limits.max_stored_records:
            metrics.duration_seconds = time.perf_counter() - t0
            return SearchResult(
                status=SearchStatus.MEMORYLIMIT,
                metrics=metrics,
                failure_reason=f"Stored record budget exceeded ({current_stored} > {active_limits.max_stored_records})",
            )

        elapsed = time.perf_counter() - t0
        if active_limits.max_time_seconds and elapsed > active_limits.max_time_seconds:
            metrics.duration_seconds = elapsed
            return SearchResult(
                status=SearchStatus.TIMEOUT,
                metrics=metrics,
                failure_reason=f"Time limit exceeded ({elapsed:.2f}s > {active_limits.max_time_seconds}s)",
            )

        curr_node, succs, child_idx = stack[-1]

        if child_idx >= len(succs):
            # All children processed; pop and remove from on_path
            stack.pop()
            if check_cycle:
                on_path.discard(curr_node.state)
            continue

        # Get next child
        next_state, action, step_cost = succs[child_idx]
        # Advance child index for parent
        stack[-1] = (curr_node, succs, child_idx + 1)

        # Cycle avoidance on current path
        if check_cycle and next_state in on_path:
            continue

        child_node = curr_node.child(next_state, action, step_cost)
        metrics.generated_nodes += 1

        if active_limits.max_cost and child_node.cost > active_limits.max_cost:
            continue

        if problem.is_goal(child_node.state):
            plan, trajectory, total_cost = reconstruct_path(child_node)
            metrics.duration_seconds = time.perf_counter() - t0
            return SearchResult(
                status=SearchStatus.SUCCESS,
                plan=plan,
                trajectory=trajectory,
                cost=total_cost,
                length=len(plan),
                metrics=metrics,
            )

        if child_node.depth >= depth_limit:
            cutoff_occurred = True
            continue

        # Expand child if under node expansion budget
        if active_limits.max_expanded and metrics.expanded_nodes >= active_limits.max_expanded:
            metrics.duration_seconds = time.perf_counter() - t0
            return SearchResult(
                status=SearchStatus.NODELIMIT,
                metrics=metrics,
                failure_reason=f"Expanded node limit exceeded ({metrics.expanded_nodes} >= {active_limits.max_expanded})",
            )

        metrics.expanded_nodes += 1
        child_succs = problem.successors(child_node.state)
        if check_cycle:
            on_path.add(child_node.state)
        stack.append((child_node, child_succs, 0))

    metrics.duration_seconds = time.perf_counter() - t0
    final_status = SearchStatus.CUTOFF if cutoff_occurred else SearchStatus.FAILED
    return SearchResult(
        status=final_status,
        metrics=metrics,
        failure_reason="Depth cutoff reached without goal." if cutoff_occurred else "Exhausted reachable tree.",
    )
