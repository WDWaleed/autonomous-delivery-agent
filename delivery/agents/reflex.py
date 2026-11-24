"""Stateless simple reflex delivery agent baseline (§6.1, §7.3, M3.2).

Governed by decisions D1, D6, and D11.
Greedily steps to the neighbouring cell that most reduces Manhattan distance
to the nearest undelivered stop, with deterministic tie-breaking (N, E, S, W).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from delivery.agents.base import Agent
from delivery.sim.env import (
    ACTION_DELTAS,
    ACTION_NAMES,
    EAST,
    NORTH,
    SOUTH,
    WAIT,
    WEST,
)

CARDINAL_ORDER = [NORTH, EAST, SOUTH, WEST]


class ReflexAgent(Agent):
    """Stateless simple reflex agent that minimizes Manhattan distance to nearest stop."""

    name: str = "SimpleReflexAgent"

    def choose_action(self, percept: Dict[str, Any]) -> Tuple[int, str]:
        """Examines percept and selects the locally distance-minimizing cardinal action."""
        agent_pos: Tuple[int, int] = tuple(percept["agent_pos"])
        undelivered = percept.get("undelivered_stops", [])

        if not undelivered:
            return WAIT, "All delivery stops completed."

        # Select nearest undelivered stop by Manhattan distance (lexicographic tie-break by coordinate)
        def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        target_stop = min(undelivered, key=lambda s: (manhattan(agent_pos, s), s))
        curr_dist = manhattan(agent_pos, target_stop)

        # Check map information from percept
        grid = percept.get("grid")
        cost_map = percept.get("cost_map")
        discovered_mask = percept.get("discovered_mask")
        extra = percept.get("extra", {})
        is_fragile_active = not extra.get("fragile_delivered", True)

        candidates: List[Tuple[int, Tuple[int, int], int]] = []  # (action, pos, new_dist)

        for a in CARDINAL_ORDER:
            dr, dc = ACTION_DELTAS[a]
            nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
            neighbor_pos = (nr, nc)

            # Legality check using percept visibility
            is_legal = True
            if grid is not None:
                n_rows, n_cols = grid.shape
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    is_legal = False
                elif grid[nr, nc] < 0:
                    is_legal = False
                elif is_fragile_active and grid[nr, nc] in (2, 3):
                    # Mud or ford illegal while fragile parcel undelivered
                    is_legal = False

            elif discovered_mask is not None:
                n_rows, n_cols = discovered_mask.shape
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    is_legal = False
                elif discovered_mask[nr, nc]:
                    opt_map = percept.get("optimistic_cost_map")
                    if opt_map is not None and np.isinf(opt_map[nr, nc]):
                        is_legal = False

            if is_legal:
                new_dist = manhattan(neighbor_pos, target_stop)
                candidates.append((a, neighbor_pos, new_dist))

        if not candidates:
            return WAIT, "Completely blocked: no legal cardinal moves available."

        # 1. Strictly distance-reducing moves: new_dist < curr_dist
        reducing_moves = [c for c in candidates if c[2] < curr_dist]
        if reducing_moves:
            # Sort by: minimal distance, then fixed cardinal action order (N, E, S, W)
            best_a, best_pos, best_d = min(reducing_moves, key=lambda c: (c[2], c[0]))
            rationale = (
                f"Step {ACTION_NAMES[best_a]} towards stop {target_stop}: "
                f"Manhattan {curr_dist} -> {best_d}"
            )
            return best_a, rationale

        # 2. Local minimum / barrier fallback: every move fails to strictly reduce distance
        # Minimize new distance (even if >= curr_dist) with deterministic cardinal tie-break
        best_a, best_pos, best_d = min(candidates, key=lambda c: (c[2], c[0]))
        rationale = (
            f"Local minimum / barrier: fallback step {ACTION_NAMES[best_a]} towards stop {target_stop} "
            f"(Manhattan {curr_dist} -> {best_d})"
        )
        return best_a, rationale
