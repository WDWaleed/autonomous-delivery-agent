"""Iterative Deepening Search (§7.4, D18, M4.3).

Repeated Depth-Limited Search with BFS-minimum action depth optimality,
cycle avoidance, and stored-record bookkeeping under D18.
"""

from __future__ import annotations

import time
from typing import Optional

from delivery.search.core import Limits, Metrics, SearchResult, SearchStatus
from delivery.search.dls import depth_limited_search
from delivery.search.problem import DeliveryProblem


def iterative_deepening_search(
    problem: DeliveryProblem,
    max_depth: int = 100,
    limits: Optional[Limits] = None,
    check_cycle: bool = True,
) -> SearchResult:
    """Executes Iterative Deepening Search (IDS).

    Guarantees BFS-optimal solution depth while using linear stack storage.
    Enforces overall limits across iterations.
    """
    active_limits = limits or Limits()
    metrics = Metrics()
    start_time = time.perf_counter()

    for depth in range(max_depth + 1):
        elapsed = time.perf_counter() - start_time
        if active_limits.max_time_seconds and elapsed > active_limits.max_time_seconds:
            metrics.duration_seconds = elapsed
            return SearchResult(
                status=SearchStatus.TIMEOUT,
                metrics=metrics,
                failure_reason=f"Time limit exceeded during IDS iteration depth={depth}",
            )

        if active_limits.max_expanded and metrics.expanded_nodes >= active_limits.max_expanded:
            metrics.duration_seconds = elapsed
            return SearchResult(
                status=SearchStatus.NODELIMIT,
                metrics=metrics,
                failure_reason=f"Expanded node limit exceeded ({metrics.expanded_nodes} >= {active_limits.max_expanded})",
            )

        res = depth_limited_search(
            problem,
            depth_limit=depth,
            limits=active_limits,
            shared_metrics=metrics,
            start_time=start_time,
            check_cycle=check_cycle,
        )

        if res.status == SearchStatus.SUCCESS:
            return res

        if res.status == SearchStatus.FAILED:
            # Entire tree searched with no cutoffs; goal is unreachable
            return res

        if res.status in (SearchStatus.TIMEOUT, SearchStatus.NODELIMIT, SearchStatus.MEMORYLIMIT):
            return res

    metrics.duration_seconds = time.perf_counter() - start_time
    return SearchResult(
        status=SearchStatus.CUTOFF,
        metrics=metrics,
        failure_reason=f"Reached maximum IDS search depth ({max_depth}) without finding goal.",
    )
