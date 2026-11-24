"""Own Admissible Heuristic: Combined MST and Farthest Reach (§7.4, M5.1).

Computes h_comb(s) = max(h_MST(U(s) U {pos}), max_{u in U(s)} d_R(pos, u)).
Precomputes pairwise relaxed distance matrices between all delivery stops and depot,
and leverages reverse distance fields for O(1) distance lookups from pos.
Uses Kruskal's algorithm with disjoint-set union for the MST calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from delivery.heuristics.base import Heuristic
from delivery.world.distance import distance_field

if TYPE_CHECKING:
    from delivery.search.problem import DeliveryProblem


class DisjointSet:
    """Disjoint Set Union (Union-Find) with path compression and rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return False
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1
        return True


class CombinedMSTHeuristic(Heuristic):
    """Candidate B: h(s) = max(MST(U U {pos}), max_{u in U} d_R(pos, u)).

    Dominates both the pure MST lower bound and baseline B2.
    Admissible under the positive step-cost framework and relaxed graph lower bounds.
    """

    name: str = "Own_CombinedMST"

    def __init__(self, problem: DeliveryProblem):
        self.world = problem.world
        self.relaxed_graph = problem.twists.relaxed_graph(self.world)
        self.stops = list(problem.stops)
        self.num_stops = len(self.stops)
        self.start = problem.start
        self.return_to_depot = problem.return_to_depot

        # 1. Precompute reverse distance field from each delivery stop:
        # stop_fields[i][r, c] = directed relaxed distance from (r, c) to stops[i]
        self.stop_fields: List[np.ndarray] = []
        for stop in self.stops:
            field = distance_field(
                self.world,
                source=stop,
                reverse=True,
                edge_model=self.relaxed_graph,
            )
            self.stop_fields.append(field)

        # Reverse field from start depot
        self.depot_field: Optional[np.ndarray] = None
        if self.return_to_depot:
            self.depot_field = distance_field(
                self.world,
                source=self.start,
                reverse=True,
                edge_model=self.relaxed_graph,
            )

        # 2. Precompute pairwise symmetric distance matrix between all stops:
        # pair_dist[i, j] = min(d_R(i, j), d_R(j, i))
        self.pair_dist = np.zeros((self.num_stops, self.num_stops), dtype=np.float64)
        for i in range(self.num_stops):
            for j in range(i + 1, self.num_stops):
                sr, sc = self.stops[i]
                tr, tc = self.stops[j]
                d_ij = float(self.stop_fields[j][sr, sc])  # from i to j
                d_ji = float(self.stop_fields[i][tr, tc])  # from j to i
                # Symmetric lower bound for undirected MST
                w = min(d_ij, d_ji)
                self.pair_dist[i, j] = w
                self.pair_dist[j, i] = w

    def _compute_mst(self, active_stop_indices: List[int], curr_pos: Tuple[int, int]) -> float:
        """Computes the Minimum Spanning Tree weight over active stops U and curr_pos."""
        k = len(active_stop_indices)
        if k == 0:
            return 0.0

        r, c = curr_pos
        # Node index 0 .. k-1 are active stops; node k is curr_pos
        # If return_to_depot, node k+1 is depot
        total_nodes = k + 1
        edges: List[Tuple[float, int, int]] = []

        # 1. Edges between active stops
        for local_i in range(k):
            stop_i = active_stop_indices[local_i]
            for local_j in range(local_i + 1, k):
                stop_j = active_stop_indices[local_j]
                w = self.pair_dist[stop_i, stop_j]
                if np.isfinite(w):
                    edges.append((w, local_i, local_j))

        # 2. Edges between curr_pos (index k) and active stops
        for local_i in range(k):
            stop_i = active_stop_indices[local_i]
            # Distance from pos to stop_i
            d_pos_to_stop = float(self.stop_fields[stop_i][r, c])
            if np.isfinite(d_pos_to_stop):
                edges.append((d_pos_to_stop, k, local_i))

        # 3. If return to depot, connect to depot as well
        if self.return_to_depot and self.depot_field is not None:
            depot_node = k + 1
            total_nodes += 1
            # Edges from stops to depot
            for local_i in range(k):
                stop_i = active_stop_indices[local_i]
                sr, sc = self.stops[stop_i]
                d_stop_depot = float(self.depot_field[sr, sc])
                if np.isfinite(d_stop_depot):
                    edges.append((d_stop_depot, depot_node, local_i))
            # Edge from pos to depot
            d_pos_depot = float(self.depot_field[r, c])
            if np.isfinite(d_pos_depot):
                edges.append((d_pos_depot, depot_node, k))

        # Kruskal's MST algorithm
        edges.sort(key=lambda x: x[0])
        dsu = DisjointSet(total_nodes)
        mst_weight = 0.0
        edges_count = 0

        for w, u, v in edges:
            if dsu.union(u, v):
                mst_weight += w
                edges_count += 1
                if edges_count == total_nodes - 1:
                    break

        return mst_weight

    def __call__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        problem: DeliveryProblem,
    ) -> float:
        r, c, mask, _ = state
        curr_pos = (r, c)

        # Identify undelivered stop indices
        undelivered_indices = [
            i for i in range(self.num_stops) if not (mask & (1 << i))
        ]

        if not undelivered_indices:
            # All delivered
            if self.return_to_depot and self.depot_field is not None:
                return float(self.depot_field[r, c])
            return 0.0

        # Component 1: Baseline B2 (max directed distance to any remaining stop)
        max_directed_b2 = 0.0
        for i in undelivered_indices:
            d = float(self.stop_fields[i][r, c])
            if d > max_directed_b2:
                max_directed_b2 = d

        # Component 2: Relaxed MST lower bound
        mst_weight = self._compute_mst(undelivered_indices, curr_pos)

        # Pointwise maximum: dominates both B2 and MST
        return max(max_directed_b2, mst_weight)
