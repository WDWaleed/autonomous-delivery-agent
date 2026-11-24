"""Pure-Python reference implementations using nested loops (Milestone M1.2).

Used for benchmarking and algorithmic verification against NumPy vectorized construction.
"""

from typing import Dict, List, Tuple

import numpy as np

CARDINAL_MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def build_cost_map_loops(
    grid_list: List[List[int]],
    terrain_costs: Dict[str, int],
) -> List[List[float]]:
    """Builds cost map using nested pure-Python loops without NumPy operations."""
    n = len(grid_list)
    cost_map: List[List[float]] = []

    # Mapping integer type to cost
    type_to_cost = {
        0: float(terrain_costs["road"]),
        1: float(terrain_costs["grass"]),
        2: float(terrain_costs["mud"]),
        3: float(terrain_costs["ford"]),
        -1: float("inf"),
    }

    for r in range(n):
        row: List[float] = []
        for c in range(n):
            val = grid_list[r][c]
            row.append(type_to_cost.get(val, float("inf")))
        cost_map.append(row)

    return cost_map


def build_passable_mask_loops(grid_list: List[List[int]]) -> List[List[bool]]:
    """Builds boolean passability mask using nested pure-Python loops."""
    n = len(grid_list)
    mask: List[List[bool]] = []
    for r in range(n):
        row: List[bool] = []
        for c in range(n):
            row.append(grid_list[r][c] >= 0)
        mask.append(row)
    return mask


def build_move_cost_table_loops(
    grid_list: List[List[int]],
    terrain_costs: Dict[str, int],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Builds neighbor and move cost tables using pure-Python nested loops."""
    n = len(grid_list)
    type_to_cost = {
        0: terrain_costs["road"],
        1: terrain_costs["grass"],
        2: terrain_costs["mud"],
        3: terrain_costs["ford"],
    }

    neighbor_table: List[List[int]] = []
    move_cost_table: List[List[int]] = []

    for r in range(n):
        for c in range(n):
            nbr_row: List[int] = []
            cost_row: List[int] = []

            # Blocked cells cannot move anywhere
            if grid_list[r][c] < 0:
                nbr_row = [-1, -1, -1, -1]
                cost_row = [-1, -1, -1, -1]
            else:
                for dr, dc in CARDINAL_MOVES:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and grid_list[nr][nc] >= 0:
                        nbr_row.append(nr * n + nc)
                        cost_row.append(type_to_cost[grid_list[nr][nc]])
                    else:
                        nbr_row.append(-1)
                        cost_row.append(-1)

            neighbor_table.append(nbr_row)
            move_cost_table.append(cost_row)

    return neighbor_table, move_cost_table
