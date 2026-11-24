"""Canonical multi-stop delivery search problem formulation (§7.3, §7.4, D1, D3, D15, M4.1).

Defines the search state (r, c, delivered_mask, extra_state) and deterministic
nominal successor generator. Shared between planners and simulator parity verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from delivery.instance import Instance
from delivery.twists.base import TwistStack
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

# Cardinal action definitions
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
WAIT = 4

ACTION_DELTAS = {
    NORTH: (-1, 0),
    EAST: (0, 1),
    SOUTH: (1, 0),
    WEST: (0, -1),
    WAIT: (0, 0),
}

ACTION_NAMES = {
    NORTH: "NORTH",
    EAST: "EAST",
    SOUTH: "SOUTH",
    WEST: "WEST",
    WAIT: "WAIT",
}


class DeliveryProblem:
    """Deterministic planning problem formulation for multi-stop delivery."""

    def __init__(
        self,
        instance: Instance,
        world: Optional[GridWorld] = None,
        twists: Optional[TwistStack] = None,
        enable_third_twist: bool = False,
    ):
        self.instance = instance
        self.world = world or GridWorld(instance)
        self.twists = twists or build_twist_stack(instance, self.world, enable_third_twist=enable_third_twist)

        self.start = self.instance.start
        self.stops = list(self.instance.delivery_stops)
        self.num_stops = len(self.stops)
        self.target_mask = (1 << self.num_stops) - 1

        self.stop_indices: Dict[Tuple[int, int], int] = {
            pos: idx for idx, pos in enumerate(self.stops)
        }

    def initial_state(self) -> Tuple[int, int, int, Tuple[Any, ...]]:
        """Returns initial state tuple: (row, col, delivered_mask, extra_state)."""
        extra = self.twists.initial_extra_state(self.world)
        # Check if start position is itself a delivery stop
        init_mask = 0
        if self.start in self.stop_indices:
            init_mask |= (1 << self.stop_indices[self.start])
        return (self.start[0], self.start[1], init_mask, extra)

    def is_goal(self, state: Tuple[int, int, int, Tuple[Any, ...]]) -> bool:
        """Goal is achieved when all parcels are delivered."""
        r, c, mask, extra = state
        return mask == self.target_mask

    def legal_actions(self, state: Tuple[int, int, int, Tuple[Any, ...]]) -> List[int]:
        """Returns list of actions that produce legal deterministic transitions from state."""
        r, c, mask, extra = state
        curr_pos = (r, c)
        legal = []

        # Cardinal actions
        for a in [NORTH, EAST, SOUTH, WEST]:
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc

            if not self.world.in_bounds(nr, nc) or not self.world.passable[nr, nc]:
                continue

            next_pos = (nr, nc)
            if self.twists.is_action_legal(curr_pos, a, next_pos, extra, self.world):
                legal.append(a)

        return legal

    def successor(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        action: int,
    ) -> Tuple[Tuple[int, int, int, Tuple[Any, ...]], float]:
        """Computes deterministic nominal successor state and step cost."""
        r, c, mask, extra = state
        curr_pos = (r, c)
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        next_pos = (nr, nc)

        # Base cost from entering cell
        u = self.world.idx(r, c)
        base_cost = float(self.world.move_cost_table[u, action]) if action < 4 else 3.0
        step_cost = self.twists.compute_step_cost(curr_pos, action, next_pos, base_cost, extra, self.world)

        # Update twist extra state
        next_extra = self.twists.update_extra_state(extra, action, curr_pos, next_pos, self.world)

        # Update delivery mask
        next_mask = mask
        if next_pos in self.stop_indices:
            next_mask |= (1 << self.stop_indices[next_pos])

        next_state = (nr, nc, next_mask, next_extra)
        return next_state, step_cost

    def successors(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
    ) -> List[Tuple[Tuple[int, int, int, Tuple[Any, ...]], int, float]]:
        """Returns all legal deterministic successors as (next_state, action, cost) triples."""
        result = []
        for a in self.legal_actions(state):
            nxt_state, cost = self.successor(state, a)
            result.append((nxt_state, a, cost))
        return result
