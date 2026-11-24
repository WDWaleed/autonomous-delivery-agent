"""Custom Dijkstra distance field algorithm and Bellman-Ford verification oracle.

Governed by decisions D3, D15, and Hard Rule 10 (no library solvers).
Uses only standard library heapq and NumPy.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from delivery.world.grid import GridWorld


def distance_field(
    world: GridWorld,
    source: Union[Tuple[int, int], int],
    reverse: bool = False,
    edge_model: Optional[Any] = None,
) -> np.ndarray:
    """Computes all-targets shortest-path distance field from source using custom Dijkstra.

    Parameters:
    - world: GridWorld instance
    - source: (r, c) tuple or flat integer index
    - reverse: If True, computes distances from all cells TO source (directed reverse-Dijkstra).
               Critical for directed twists like one-way streets.
    - edge_model: Optional custom edge provider with method `get_edges(p, reverse=False)`
                  returning list of (dest_p, cost). If None, uses world.neighbor_table.

    Returns:
    - 2D np.ndarray of shape (n, n) with dtype float64. Unreachable cells hold np.inf.
    """
    n = world.n
    n2 = world.n2

    # Parse source flat index
    if isinstance(source, tuple):
        src_p = world.idx(*source)
    else:
        src_p = int(source)

    # Validate source
    sr, sc = world.rc(src_p)
    if not world.in_bounds(sr, sc) or not world.passable[sr, sc]:
        dist_grid = np.full((n, n), np.inf, dtype=np.float64)
        return dist_grid

    dist = np.full(n2, np.inf, dtype=np.float64)
    dist[src_p] = 0.0

    # Priority queue storing (cost, flat_node)
    pq: List[Tuple[float, int]] = [(0.0, src_p)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        # Get edges
        if edge_model is not None:
            edges = edge_model.get_edges(u, reverse=reverse)
        else:
            # Default world transitions
            edges = []
            for d_idx in range(4):
                v = world.neighbor_table[u, d_idx]
                if v != -1:
                    if reverse:
                        # For reverse Dijkstra from target, an incoming edge is w -> u.
                        # The cost to traverse w -> u is the entry cost of u!
                        cost = float(world.cost_map[world.rc(u)])
                    else:
                        cost = float(world.move_cost_table[u, d_idx])
                    edges.append((v, cost))

        for v, cost in edges:
            if cost <= 0 or np.isinf(cost):
                continue
            new_dist = d + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return dist.reshape((n, n))


def bellman_ford_oracle(
    num_nodes: int,
    edges: List[Tuple[int, int, float]],
    source: int,
    reverse: bool = False,
) -> np.ndarray:
    """Independent brute-force Bellman-Ford reference oracle.

    Used strictly for testing and property verification (§6.3).
    edges: list of (u, v, cost) representing directed edge u -> v with weight cost.
    If reverse=True, computes distance from all nodes TO source.
    """
    dist = np.full(num_nodes, np.inf, dtype=np.float64)
    dist[source] = 0.0

    # Invert edge directions if reverse search is requested
    active_edges = [(v, u, cost) for u, v, cost in edges] if reverse else edges

    # Relax edges up to num_nodes - 1 times
    for _ in range(num_nodes - 1):
        updated = False
        for u, v, cost in active_edges:
            if dist[u] != np.inf and dist[u] + cost < dist[v]:
                dist[v] = dist[u] + cost
                updated = True
        if not updated:
            break

    return dist
