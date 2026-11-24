"""Canonical multi-stop delivery search problem formulation (§7.3, §7.4, D1, D3, D5, D15, M4.1).

Defines the search state (r, c, delivered_mask, extra_state) and deterministic
nominal successor generator with support for precedence constraints, return-to-depot,
and human-readable state descriptions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from delivery.instance import Instance
from delivery.twists.base import TwistStack
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

# Cardinal actions
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
        world_or_instance: Union[GridWorld, Instance],
        world: Optional[GridWorld] = None,
        twist_stack: Optional[TwistStack] = None,
        twists: Optional[TwistStack] = None,
        constraints: Optional[Dict[int, int]] = None,
        return_to_depot: bool = False,
        enable_third_twist: bool = False,
        max_cost: Optional[float] = None,
    ):
        if isinstance(world_or_instance, GridWorld):
            self.world = world_or_instance
            self.instance = self.world.instance
        elif world is not None:
            self.instance = world_or_instance
            self.world = world
        else:
            self.instance = world_or_instance
            self.world = GridWorld(self.instance)

        self.twists = twist_stack or twists or build_twist_stack(
            self.instance, self.world, enable_third_twist=enable_third_twist
        )
        self.constraints = constraints or {}  # {dependent_stop_idx: prerequisite_stop_idx}
        self.return_to_depot = bool(return_to_depot)
        self.max_cost = max_cost

        self.start = self.instance.start
        self.stops = [tuple(map(int, p)) for p in self.instance.delivery_stops]
        self.num_stops = len(self.stops)
        self.target_mask = (1 << self.num_stops) - 1

        self.stop_indices: Dict[Tuple[int, int], int] = {
            pos: idx for idx, pos in enumerate(self.stops)
        }

    def initial_state(self) -> Tuple[int, int, int, Tuple[Any, ...]]:
        """Returns initial state tuple: (row, col, delivered_mask, extra_state)."""
        extra = self.twists.initial_extra_state(self.world)
        init_mask = 0

        # Auto-delivery if start position is itself a delivery stop
        if self.start in self.stop_indices:
            idx = self.stop_indices[self.start]
            if self._satisfies_precedence(idx, init_mask):
                init_mask |= (1 << idx)

        return (self.start[0], self.start[1], init_mask, extra)

    def _satisfies_precedence(self, stop_idx: int, current_mask: int) -> bool:
        """Checks if prerequisite for stop_idx is met in current delivery mask."""
        if stop_idx in self.constraints:
            prereq_idx = self.constraints[stop_idx]
            return bool(current_mask & (1 << prereq_idx))
        return True

    def is_goal(self, state: Tuple[int, int, int, Tuple[Any, ...]]) -> bool:
        """Goal is achieved when all parcels are delivered (and at depot if return_to_depot=True)."""
        r, c, mask, _ = state
        parcels_delivered = (mask == self.target_mask)
        if not parcels_delivered:
            return False

        if self.return_to_depot:
            return (r, c) == self.start

        return True

    def legal_actions(self, state: Tuple[int, int, int, Tuple[Any, ...]]) -> List[int]:
        """Returns list of actions that produce legal deterministic transitions from state."""
        r, c, mask, extra = state
        curr_pos = (r, c)
        legal: List[int] = []

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

        # Base entry cost
        u = self.world.idx(r, c)
        base_cost = float(self.world.move_cost_table[u, action]) if action < 4 else 3.0
        step_cost = self.twists.compute_step_cost(
            curr_pos, action, next_pos, base_cost, extra, self.world
        )

        # Twist extra state update
        next_extra = self.twists.update_extra_state(
            extra, action, curr_pos, next_pos, self.world
        )

        # Delivery mask update with precedence checking
        next_mask = mask
        if next_pos in self.stop_indices:
            stop_idx = self.stop_indices[next_pos]
            if self._satisfies_precedence(stop_idx, mask):
                next_mask |= (1 << stop_idx)

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

    def describe(self, state: Tuple[int, int, int, Tuple[Any, ...]]) -> str:
        """Returns a human-readable diagnostic description of a search state."""
        r, c, mask, extra = state
        delivered = [idx for idx in range(self.num_stops) if (mask & (1 << idx))]
        remaining = [idx for idx in range(self.num_stops) if not (mask & (1 << idx))]
        goal_status = "GOAL" if self.is_goal(state) else "IN_PROGRESS"
        extra_dict = self.twists.unpack_extra(extra) if self.twists.extra_names else {}

        return (
            f"State(Pos=({r}, {c}), Delivered={delivered}/{self.num_stops}, "
            f"Remaining={remaining}, Status={goal_status}, Extra={extra_dict})"
        )
