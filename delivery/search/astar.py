"""A* Graph Search with Closed-Node Re-opening (§7.4, D15, M4.4).

Optimal in accumulated path cost under admissible heuristics.
Re-opens closed states if a strictly cheaper path is found (best-g improvement),
preserving optimality even for admissible but inconsistent heuristics.
Uses strict lexicographic tie-breaking (f, cost, depth, action_seq).
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from delivery.heuristics.base import Heuristic
from delivery.search.core import Limits, Metrics, Node, SearchResult, SearchStatus, reconstruct_path
from delivery.search.problem import DeliveryProblem


def astar_search(
    problem: DeliveryProblem,
    heuristic: Heuristic,
    limits: Optional[Limits] = None,
) -> SearchResult:
    """Executes A* Graph Search with closed-node re-opening.

    Evaluates nodes by f(n) = g(n) + h(n).
    When an admissible heuristic is supplied and edge costs are strictly positive,
    A* guarantees finding an optimal cost plan. Re-opening closed nodes when a strictly
    cheaper g(n) is found ensures optimality even if the heuristic is inconsistent.
    """
    active_limits = limits or Limits()
    metrics = Metrics()
    start_time = time.perf_counter()

    root = Node.root(problem.initial_state())
    metrics.generated_nodes = 1

    h_root = float(heuristic(root.state, problem))
    f_root = root.cost + h_root

    # Frontier entries: (f_val, cost, depth, action_seq, node)
    # Tie-breaking: lowest f, lowest g (cost), lowest depth, lexicographic action_seq
    frontier: List[Tuple[float, float, int, Tuple[int, ...], Node]] = []
    heapq.heappush(frontier, (f_root, root.cost, root.depth, root.action_seq, root))

    # Track best known g-cost for each state reached: state -> best_g
    best_g: Dict[Tuple[int, int, int, Tuple[Any, ...]], float] = {
        root.state: root.cost
    }

    # Set of states currently expanded (closed)
    closed: Set[Tuple[int, int, int, Tuple[Any, ...]]] = set()

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
            )

        if active_limits.max_time_seconds:
            elapsed = time.perf_counter() - start_time
            if elapsed > active_limits.max_time_seconds:
                metrics.duration_seconds = elapsed
                return SearchResult(
                    status=SearchStatus.TIMEOUT,
                    metrics=metrics,
                )

        f_val, g_val, depth, action_seq, current_node = heapq.heappop(frontier)

        # Lazy deletion: if we already found a strictly better path to this state, skip
        if g_val > best_g[current_node.state]:
            continue

        # Goal test on expansion (crucial for optimality in A* and UCS)
        if problem.is_goal(current_node.state):
            plan, trajectory, total_cost = reconstruct_path(current_node)
            metrics.duration_seconds = time.perf_counter() - start_time
            return SearchResult(
                status=SearchStatus.SUCCESS,
                plan=plan,
                trajectory=trajectory,
                cost=total_cost,
                length=len(plan),
                metrics=metrics,
            )

        closed.add(current_node.state)
        metrics.expanded_nodes += 1

        if active_limits.max_expanded and metrics.expanded_nodes >= active_limits.max_expanded:
            metrics.duration_seconds = time.perf_counter() - start_time
            return SearchResult(
                status=SearchStatus.NODELIMIT,
                metrics=metrics,
                failure_reason=f"Expanded node limit exceeded ({metrics.expanded_nodes} >= {active_limits.max_expanded})",
            )

        # Expand successors
        for next_state, action, step_cost in problem.successors(current_node.state):
            child_g = current_node.cost + step_cost
            prev_best = best_g.get(next_state, float("inf"))

            # Standard A* pruning: if we have seen next_state with <= child_g, prune
            if child_g >= prev_best:
                continue

            # If next_state was in closed set, we are reopening it because child_g < prev_best
            if next_state in closed:
                closed.remove(next_state)

            best_g[next_state] = child_g
            child_node = current_node.child(next_state, action, step_cost)
            metrics.generated_nodes += 1

            h_child = float(heuristic(next_state, problem))
            f_child = child_g + h_child

            heapq.heappush(
                frontier,
                (f_child, child_node.cost, child_node.depth, child_node.action_seq, child_node),
            )

    metrics.duration_seconds = time.perf_counter() - start_time
    return SearchResult(
        status=SearchStatus.FAILED,
        metrics=metrics,
    )
