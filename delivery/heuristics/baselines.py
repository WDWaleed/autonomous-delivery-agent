"""Baseline heuristics B1 and B2 for multi-stop delivery planning (§7.4, D15, M4.4).

B1: c_min * Manhattan distance to nearest undelivered stop.
B2: max_{u in U} d_R(pos, u) where d_R is directed shortest-path in relaxed graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from delivery.heuristics.base import Heuristic
from delivery.world.distance import distance_field

if TYPE_CHECKING:
    from delivery.search.problem import DeliveryProblem


class ManhattanNearestStopHeuristic(Heuristic):
    """Baseline B1: c_min * min_{u in U} Manhattan(pos, u).

    Admissible under the condition that all planned edges are cardinal grid steps
    with cost >= c_min.
    """

    name: str = "B1_ManhattanNearest"

    def __init__(self, c_min: float = 3.0):
        self.c_min = float(c_min)

    def __call__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        problem: DeliveryProblem,
    ) -> float:
        r, c, mask, _ = state
        curr_pos = (r, c)

        # Check undelivered stops
        undelivered = [
            stop for idx, stop in enumerate(problem.stops) if not (mask & (1 << idx))
        ]

        if not undelivered:
            if problem.return_to_depot and curr_pos != problem.start:
                sr, sc = problem.start
                return self.c_min * (abs(r - sr) + abs(c - sc))
            return 0.0

        min_manhattan = min(abs(r - sr) + abs(c - sc) for sr, sc in undelivered)
        return self.c_min * float(min_manhattan)


class MaxRelaxedDistanceHeuristic(Heuristic):
    """Baseline B2: max_{u in U} d_R(pos, u).

    Strictly admissible: any route that visits all remaining stops U must visit
    the farthest remaining stop, hence cost is lower-bounded by the relaxed distance
    to that farthest stop.
    Precomputes reverse distance fields from all targets in the relaxed graph for O(1) lookups.
    """

    name: str = "B2_MaxRelaxedDistance"

    def __init__(self, problem: DeliveryProblem):
        self.world = problem.world
        self.relaxed_graph = problem.twists.relaxed_graph(self.world)
        self.stops = list(problem.stops)
        self.start = problem.start
        self.return_to_depot = problem.return_to_depot

        # Precompute reverse distance field from each delivery stop
        # dist_fields[stop_idx][r, c] gives relaxed distance FROM (r, c) TO stop
        self.dist_fields: List[np.ndarray] = []
        for stop in self.stops:
            field = distance_field(
                self.world,
                source=stop,
                reverse=True,
                edge_model=self.relaxed_graph,
            )
            self.dist_fields.append(field)

        # Precompute reverse field from start depot if return_to_depot is active
        self.depot_field: Optional[np.ndarray] = None
        if self.return_to_depot:
            self.depot_field = distance_field(
                self.world,
                source=self.start,
                reverse=True,
                edge_model=self.relaxed_graph,
            )

    def __call__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        problem: DeliveryProblem,
    ) -> float:
        r, c, mask, _ = state

        max_d = 0.0
        for idx in range(problem.num_stops):
            if not (mask & (1 << idx)):
                d_val = float(self.dist_fields[idx][r, c])
                if np.isfinite(d_val) and d_val > max_d:
                    max_d = d_val

        # If all stops delivered and must return to depot
        if max_d == 0.0 and self.return_to_depot and self.depot_field is not None:
            depot_d = float(self.depot_field[r, c])
            if np.isfinite(depot_d):
                max_d = depot_d

        return max_d
