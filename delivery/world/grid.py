"""Grid world model and vectorized cost map.

Governed by decisions D2, D3, D15, and D16.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from delivery.instance import Instance

# Cell type integer constants matching oel_instance.py
ROAD = 0
GRASS = 1
MUD = 2
FORD = 3
BLOCKED = -1

TERRAIN_NAMES = {
    ROAD: "road",
    GRASS: "grass",
    MUD: "mud",
    FORD: "ford",
}

# Fixed cardinal move vectors: North, East, South, West (D2)
CARDINAL_MOVES: List[Tuple[int, int]] = [
    (-1, 0),  # North
    (0, 1),   # East
    (1, 0),   # South
    (0, -1),  # West
]

ACTION_NAMES: List[str] = ["NORTH", "EAST", "SOUTH", "WEST"]


class GridWorld:
    """Represents the static 2D grid city with vectorised cost maps and precomputed lookup tables."""

    # Blocked / off-grid sentinel for int32 tables
    BLOCKED_SENTINEL = -1

    def __init__(self, instance: Instance):
        self.instance = instance
        self.n = instance.n
        self.n2 = self.n * self.n
        self.grid = instance.grid.astype(np.int8)
        self.start = instance.start
        self.delivery_stops = instance.delivery_stops
        self.terrain_costs = instance.terrain_cost

        # Boolean passability mask [n, n]
        self.passable = (self.grid >= 0)

        # Terrain lookup array indexed 0..3 matching [road, grass, mud, ford]
        self.terrain_cost_lookup = np.array([
            self.terrain_costs["road"],
            self.terrain_costs["grass"],
            self.terrain_costs["mud"],
            self.terrain_costs["ford"],
        ], dtype=np.int32)

        # Build vectorized cost map and precomputed neighbour / move-cost tables
        self.cost_map = self._build_cost_map_vectorized()
        self.entering_cost_map = self._build_entering_cost_int32()
        self.neighbor_table, self.move_cost_table = self._build_vectorized_tables()

    def _build_cost_map_vectorized(self) -> np.ndarray:
        """Constructs float64 cost map using vectorized NumPy indexing.

        Blocked cells are represented with sentinel np.inf.
        """
        cost_map = np.full((self.n, self.n), np.inf, dtype=np.float64)
        for terrain_type, name in TERRAIN_NAMES.items():
            cost_map[self.grid == terrain_type] = self.terrain_costs[name]
        return cost_map

    def _build_entering_cost_int32(self) -> np.ndarray:
        """Constructs int32 entering cost map with BLOCKED_SENTINEL (-1) for blocked cells."""
        entering_cost = np.full((self.n, self.n), self.BLOCKED_SENTINEL, dtype=np.int32)
        for terrain_type, name in TERRAIN_NAMES.items():
            entering_cost[self.grid == terrain_type] = self.terrain_costs[name]
        return entering_cost

    def _build_vectorized_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised construction of flat neighbour table [n², 4] and move cost table [n², 4].

        Columns correspond to directions 0:N, 1:E, 2:S, 3:W.
        Entries are -1 if the transition leads off-grid or into a blocked cell.
        """
        r_grid, c_grid = np.meshgrid(
            np.arange(self.n, dtype=np.int32),
            np.arange(self.n, dtype=np.int32),
            indexing="ij",
        )

        neighbor_table = np.full((self.n2, 4), self.BLOCKED_SENTINEL, dtype=np.int32)
        move_cost_table = np.full((self.n2, 4), self.BLOCKED_SENTINEL, dtype=np.int32)

        for d_idx, (dr, dc) in enumerate(CARDINAL_MOVES):
            nr = r_grid + dr
            nc = c_grid + dc

            valid_in_bounds = (nr >= 0) & (nr < self.n) & (nc >= 0) & (nc < self.n)

            # Mask of destinations where both source and destination are passable
            valid_dest = np.zeros((self.n, self.n), dtype=bool)
            src_pass = self.passable[r_grid[valid_in_bounds], c_grid[valid_in_bounds]]
            dest_pass = self.passable[nr[valid_in_bounds], nc[valid_in_bounds]]
            valid_dest[valid_in_bounds] = src_pass & dest_pass

            # Flat destination indices
            dest_flat = np.full((self.n, self.n), self.BLOCKED_SENTINEL, dtype=np.int32)
            dest_flat[valid_dest] = nr[valid_dest] * self.n + nc[valid_dest]

            # Move cost is the terrain cost of destination cell
            dest_cost = np.full((self.n, self.n), self.BLOCKED_SENTINEL, dtype=np.int32)
            dest_cost[valid_dest] = self.entering_cost_map[nr[valid_dest], nc[valid_dest]]

            neighbor_table[:, d_idx] = dest_flat.reshape(-1)
            move_cost_table[:, d_idx] = dest_cost.reshape(-1)

        return neighbor_table, move_cost_table

    def in_bounds(self, r: int, c: int) -> bool:
        """Checks if (r, c) is within the grid boundaries."""
        return 0 <= r < self.n and 0 <= c < self.n

    def is_free(self, r: int, c: int) -> bool:
        """Checks if cell is within bounds and unblocked."""
        return self.in_bounds(r, c) and bool(self.passable[r, c])

    def idx(self, r: int, c: int) -> int:
        """Flat 1D index mapping p = r * n + c (D2)."""
        return r * self.n + c

    def rc(self, p: int) -> Tuple[int, int]:
        """Inverts flat 1D index p back to (r, c)."""
        return divmod(p, self.n)

    def to_flat(self, r: int, c: int) -> int:
        """Alias for idx(r, c)."""
        return self.idx(r, c)

    def from_flat(self, p: int) -> Tuple[int, int]:
        """Alias for rc(p)."""
        return self.rc(p)

    def legal_neighbors(self, r: int, c: int) -> List[Tuple[int, int, float]]:
        """Returns valid 4-connected non-blocked neighbors in deterministic order N, E, S, W (D2).

        Returns list of tuples: (nr, nc, step_cost), where step_cost is the terrain
        cost of entering cell (nr, nc) per decision D3.
        """
        neighbors = []
        for dr, dc in CARDINAL_MOVES:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                cost = self.cost_map[nr, nc]
                neighbors.append((nr, nc, cost))
        return neighbors

    def get_cell_cost(self, r: int, c: int) -> float:
        """Returns the traversal cost of entering cell (r, c)."""
        if not self.in_bounds(r, c) or self.grid[r, c] < 0:
            return np.inf
        return float(self.cost_map[r, c])
