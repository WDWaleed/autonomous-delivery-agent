"""Stochastic delivery environment simulation engine (§6.1, §7.3, D6, M3.1).

Governed by decisions D1, D3, D6, D15, and D16.
Simulates nominal and slip transitions, obstacle/one-way rejection, auto-delivery,
percept generation under fog/full observability, and deterministic zero-slip parity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from delivery.instance import Instance
from delivery.twists.base import TwistStack
from delivery.twists.fog import FogTwist
from delivery.twists.protocols import PerceptionModifier
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

PERPENDICULAR_CANDIDATES = {
    NORTH: (EAST, WEST),
    SOUTH: (EAST, WEST),
    EAST: (NORTH, SOUTH),
    WEST: (NORTH, SOUTH),
}


class DeliveryEnv:
    """Stochastic city delivery simulation environment with twist hooks and slip dynamics."""

    def __init__(
        self,
        world: GridWorld,
        twists: Optional[TwistStack] = None,
        slip_p: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        enable_third_twist: bool = False,
        enable_rival: bool = False,
        rival_start: Optional[Tuple[int, int]] = None,
        rival_depth: Optional[int] = None,
    ):
        self.world = world
        self.instance = world.instance
        self.twists = twists or build_twist_stack(
            self.instance, self.world, enable_third_twist=enable_third_twist
        )
        self.slip_p = float(slip_p)

        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.instance.seed)

        self.start = self.instance.start
        self.stops = list(self.instance.delivery_stops)
        self.num_stops = len(self.stops)
        self.target_mask = (1 << self.num_stops) - 1

        self.stop_indices: Dict[Tuple[int, int], int] = {
            pos: idx for idx, pos in enumerate(self.stops)
        }

        # Find active fog twist if any
        self.fog_twist: Optional[FogTwist] = None
        for t in self.twists.twists:
            if t.enabled and isinstance(t, FogTwist):
                self.fog_twist = t
                break

        # State attributes
        self.agent_pos: Tuple[int, int] = self.start
        self.delivered_mask: int = 0
        self.extra_state: Tuple[Any, ...] = ()
        self.enable_rival: bool = bool(enable_rival or rival_start is not None)
        self.rival_start: Optional[Tuple[int, int]] = rival_start
        self.rival_depth: int = int(rival_depth or self.instance.adversary_search_depth or 6)
        self.rival_pos: Optional[Tuple[int, int]] = None
        self.captured: bool = False
        self.step_count: int = 0
        self.total_cost: float = 0.0
        self.done: bool = False
        self.events_log: List[Dict[str, Any]] = []

        self.reset()

    def _default_rival_start(self) -> Tuple[int, int]:
        """Selects a reachable passable cell farthest from agent start."""
        from collections import deque

        n = self.world.n
        visited = np.zeros((n, n), dtype=bool)
        sr, sc = self.start
        visited[sr, sc] = True
        q = deque([(sr, sc)])
        farthest = self.start
        while q:
            r, c = q.popleft()
            farthest = (r, c)
            for a in (NORTH, EAST, SOUTH, WEST):
                dr, dc = ACTION_DELTAS[a]
                nr, nc = r + dr, c + dc
                if self.world.in_bounds(nr, nc) and self.world.passable[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return farthest

    def reset(
        self,
        start_pos: Optional[Tuple[int, int]] = None,
        rival_pos: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Resets the environment to initial state and returns initial percept."""
        self.agent_pos = start_pos or self.start
        self.delivered_mask = 0
        self.step_count = 0
        self.total_cost = 0.0
        self.done = False
        self.captured = False
        self.events_log = []
        if rival_pos is not None:
            self.rival_pos = rival_pos
            self.enable_rival = True
        elif self.enable_rival:
            self.rival_pos = self.rival_start or self._default_rival_start()
        else:
            self.rival_pos = None

        # Reset twist extra state
        self.extra_state = self.twists.initial_extra_state(self.world)

        # Check if start position is a delivery stop
        if self.agent_pos in self.stop_indices:
            stop_idx = self.stop_indices[self.agent_pos]
            self.delivered_mask |= (1 << stop_idx)

        # If Fog is active, re-initialize fog window around agent
        if self.fog_twist is not None:
            self.fog_twist.compute_percept(self.agent_pos, self.world)

        return self.get_percept()

    def is_done(self) -> bool:
        """Returns True if all delivery stops have been completed."""
        return self.delivered_mask == self.target_mask

    def get_percept(self) -> Dict[str, Any]:
        """Constructs the agent percept dictionary."""
        percept: Dict[str, Any] = {
            "agent_pos": self.agent_pos,
            "rival_pos": self.rival_pos,
            "delivered_mask": self.delivered_mask,
            "undelivered_stops": [
                pos for idx, pos in enumerate(self.stops) if not (self.delivered_mask & (1 << idx))
            ],
            "step_count": self.step_count,
            "total_cost": self.total_cost,
            "done": self.done,
            "extra": self.twists.unpack_extra(self.extra_state) if self.twists.extra_names else {},
        }

        # Partial observability under Fog
        if self.fog_twist is not None:
            percept["is_partially_observable"] = True
            percept["fog_radius"] = self.fog_twist.radius
            percept["discovered_mask"] = self.fog_twist.discovered_mask.copy()
            percept["optimistic_cost_map"] = self.fog_twist.optimistic_cost_map.copy()
        else:
            percept["is_partially_observable"] = False
            percept["grid"] = self.world.grid
            percept["cost_map"] = self.world.cost_map

        return percept

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Executes one simulation action step following the canonical transition order.

        Returns:
        - (percept, step_cost, events)
        """
        if self.done:
            return self.get_percept(), 0.0, {"event": "already_done", "done": True}

        intended_action = action
        if intended_action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action index {intended_action}. Expected 0..4.")

        from_pos = self.agent_pos
        u = self.world.idx(*from_pos)

        slipped = False
        sampled_action: Optional[int] = intended_action
        illegal_move = False
        fallback_stay = False
        realised_destination = from_pos
        step_cost = 0.0

        # Case 1: WAIT action (does not slip)
        if intended_action == WAIT:
            sampled_action = WAIT
            realised_destination = from_pos
            step_cost = 3.0  # nominal wait cost
            slipped = False

        # Case 2: Cardinal actions
        else:
            dr, dc = ACTION_DELTAS[intended_action]
            target_r, target_c = from_pos[0] + dr, from_pos[1] + dc
            target_pos = (target_r, target_c)

            # Check legality of intended action per Decision D6
            intended_legal = (
                self.world.in_bounds(target_r, target_c)
                and self.world.passable[target_r, target_c]
                and self.twists.is_action_legal(
                    from_pos, intended_action, target_pos, self.extra_state, self.world
                )
            )

            if not intended_legal:
                # Illegal move rejected before slip sampling per D6
                illegal_move = True
                fallback_stay = True
                sampled_action = intended_action
                realised_destination = from_pos
                step_cost = 3.0  # fallback cost for illegal attempt
            else:
                # Intended move is legal; check slip probability
                if self.slip_p > 0.0 and self.rng.random() < self.slip_p:
                    slipped = True
                    # Perpendicular candidates
                    candidates = PERPENDICULAR_CANDIDATES[intended_action]

                    # Filter candidates to legal traversable moves
                    eligible = []
                    for cand_a in candidates:
                        p_dr, p_dc = ACTION_DELTAS[cand_a]
                        pr, pc = from_pos[0] + p_dr, from_pos[1] + p_dc
                        if (
                            self.world.in_bounds(pr, pc)
                            and self.world.passable[pr, pc]
                            and self.twists.is_action_legal(
                                from_pos, cand_a, (pr, pc), self.extra_state, self.world
                            )
                        ):
                            eligible.append((cand_a, (pr, pc)))

                    if eligible:
                        # Choose uniformly from eligible perpendicular moves
                        choice_idx = int(self.rng.integers(0, len(eligible)))
                        sampled_action, realised_destination = eligible[choice_idx]
                        base_cost = float(self.world.move_cost_table[u, sampled_action])
                        step_cost = self.twists.compute_step_cost(
                            from_pos, sampled_action, realised_destination, base_cost, self.extra_state, self.world
                        )
                        fallback_stay = False
                    else:
                        # Neither perpendicular cell traversable: stay in place per §3
                        sampled_action = None
                        realised_destination = from_pos
                        step_cost = 3.0  # fallback cost
                        fallback_stay = True
                else:
                    # Nominal transition (no slip)
                    sampled_action = intended_action
                    realised_destination = target_pos
                    base_cost = float(self.world.move_cost_table[u, intended_action])
                    step_cost = self.twists.compute_step_cost(
                        from_pos, intended_action, realised_destination, base_cost, self.extra_state, self.world
                    )
                    fallback_stay = False

        # State updates
        effective_action = sampled_action if sampled_action is not None else WAIT
        self.extra_state = self.twists.update_extra_state(
            self.extra_state, effective_action, from_pos, realised_destination, self.world
        )
        self.agent_pos = realised_destination
        self.step_count += 1
        self.total_cost += step_cost

        # D7 Capture precedence before delivery completion
        if self.rival_pos is not None and self.agent_pos == self.rival_pos:
            self.captured = True
            self.done = True
            new_deliveries = []
        else:
            # Auto-delivery check
            new_deliveries = []
            if self.agent_pos in self.stop_indices:
                stop_idx = self.stop_indices[self.agent_pos]
                if not (self.delivered_mask & (1 << stop_idx)):
                    self.delivered_mask |= (1 << stop_idx)
                    new_deliveries.append(stop_idx)

        # Fog perceptual update
        if self.fog_twist is not None:
            self.fog_twist.compute_percept(self.agent_pos, self.world)

        if self.delivered_mask == self.target_mask and not self.captured:
            self.done = True

        event = {
            "step": self.step_count,
            "intended_action": intended_action,
            "sampled_action": sampled_action,
            "slipped": slipped,
            "illegal_move": illegal_move,
            "fallback_stay": fallback_stay,
            "from_pos": from_pos,
            "realised_destination": realised_destination,
            "cost": step_cost,
            "total_cost": self.total_cost,
            "new_deliveries": new_deliveries,
            "delivered_mask": self.delivered_mask,
            "done": self.done,
        }
        self.events_log.append(event)

        return self.get_percept(), step_cost, event

    def step_rival(self, rival_action: Optional[int] = None) -> Optional[int]:
        """Executes one rival adversary transition and checks D7 capture."""
        if self.rival_pos is None or self.done:
            return None

        if rival_action is None:
            from delivery.game.rivals import ChaserRival
            from delivery.game.window import RIVAL, GameState, WindowGame

            game = WindowGame(
                self.instance,
                world=self.world,
                twists=self.twists,
                window_radius=None,
            )
            state = GameState(
                agent_pos=self.agent_pos,
                rival_pos=self.rival_pos,
                delivered_mask=self.delivered_mask,
                extra_state=self.extra_state,
                turn=RIVAL,
                agent_cost=self.total_cost,
                steps=self.step_count,
            )
            rival = ChaserRival(depth=min(4, self.rival_depth))
            rival_action, _ = rival.choose_action(game, state)

        if rival_action != WAIT:
            dr, dc = ACTION_DELTAS[rival_action]
            nr, nc = self.rival_pos[0] + dr, self.rival_pos[1] + dc
            if (
                self.world.in_bounds(nr, nc)
                and self.world.passable[nr, nc]
                and self.twists.is_action_legal(
                    self.rival_pos, rival_action, (nr, nc), (), self.world
                )
            ):
                self.rival_pos = (nr, nc)

        if self.rival_pos == self.agent_pos:
            self.captured = True
            self.done = True

        return rival_action
