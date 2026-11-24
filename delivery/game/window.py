"""Reduced Window Adversarial Game Model (§7.3, §7.5, D7, D12, M6.1).

Implements:
- Complete immutable GameState(agent_pos, rival_pos, delivered_mask, extra_state, turn, agent_cost, steps)
- WindowGame with Chebyshev square window extraction and boundary restriction
- Base actions (N, E, S, W, WAIT) plus enabled twist actions (e.g. TELEPORT)
- D7 terminal precedence: capture -> theft -> completed delivery (kappa threshold) -> excessive cost draw -> step-cap draw
- Same-cell capture and reusable edge-swap capture helper
- Canonical baseline (C*, L*) computation and T_max = 4 * L* + 20
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from delivery.instance import Instance
from delivery.search.core import Limits, SearchStatus, canonical_optimal_plan
from delivery.search.problem import (
    ACTION_DELTAS,
    ACTION_NAMES,
    EAST,
    NORTH,
    SOUTH,
    WAIT,
    WEST,
    DeliveryProblem,
)
from delivery.twists.base import TwistStack
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

AGENT = 0  # MAX player
RIVAL = 1  # MIN player
TELEPORT = 5  # Twist-specific action for teleport_tunnels

UTILITY_M = 10_000.0  # Terminal magnitude >> max |f(s)| (<= 1,000.0)


class GameOutcome(str, enum.Enum):
    """Terminal or non-terminal outcome under D7 precedence."""

    ONGOING = "ONGOING"
    WIN = "WIN"
    LOSS_CAPTURE = "LOSS_CAPTURE"
    LOSS_THEFT = "LOSS_THEFT"
    DRAW_EXCESS_COST = "DRAW_EXCESS_COST"
    DRAW_STEP_CAP = "DRAW_STEP_CAP"


@dataclass(frozen=True)
class GameState:
    """Complete immutable adversarial game state (§7.3, D12)."""

    agent_pos: Tuple[int, int]
    rival_pos: Tuple[int, int]
    delivered_mask: int
    extra_state: Tuple[Any, ...]
    turn: int = AGENT
    agent_cost: float = 0.0
    steps: int = 0  # Agent actions taken
    plies: int = 0  # Total half-moves (agent + rival)

    def key(self) -> Tuple[Any, ...]:
        """Hashable canonical state key."""
        return (
            self.agent_pos,
            self.rival_pos,
            self.delivered_mask,
            self.extra_state,
            self.turn,
            round(self.agent_cost, 6),
            self.steps,
        )


def check_same_cell_capture(
    agent_pos: Tuple[int, int],
    rival_pos: Tuple[int, int],
) -> bool:
    """Checks immediate same-cell capture."""
    return agent_pos == rival_pos


def check_edge_swap_capture(
    prev_agent: Tuple[int, int],
    curr_agent: Tuple[int, int],
    prev_rival: Tuple[int, int],
    curr_rival: Tuple[int, int],
) -> bool:
    """Reusable edge-swap capture helper for paired/joint transitions and replay semantics."""
    return prev_agent == curr_rival and curr_agent == prev_rival


def compute_canonical_baseline(
    instance: Instance,
    world: Optional[GridWorld] = None,
    twists: Optional[TwistStack] = None,
    limits: Optional[Limits] = None,
) -> Tuple[Optional[float], Optional[int], bool]:
    """Computes canonical optimal cost C* and action count L* without rival or slip.

    Minimizes (total_cost, action_count, action-rank sequence) where N < E < S < W < WAIT.
    Returns (C*, L*, certified). If uncertified, returns (None, None, False).
    """
    w = world or GridWorld(instance)
    ts = twists or build_twist_stack(instance, w)
    prob = DeliveryProblem(instance, world=w, twist_stack=ts)
    active_limits = limits or Limits(max_time_seconds=20.0, max_expanded=500_000)

    res = canonical_optimal_plan(prob, limits=active_limits)
    if res.status == SearchStatus.SUCCESS and res.plan is not None:
        return float(res.cost), int(len(res.plan)), True
    return None, None, False


class WindowGame:
    """Reduced Chebyshev-window two-player zero-sum delivery game (D7, D12)."""

    def __init__(
        self,
        instance: Instance,
        world: Optional[GridWorld] = None,
        twists: Optional[TwistStack] = None,
        window_radius: Optional[int] = 4,
        window_center: Optional[Tuple[int, int]] = None,
        kappa: float = 1.5,
        c_star: Optional[float] = None,
        l_star: Optional[int] = None,
        target_mask: Optional[int] = None,
        return_to_depot: bool = False,
        allow_wait: bool = True,
        tunnel_pairs: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
    ):
        self.instance = instance
        self.world = world or GridWorld(instance)
        self.twists = twists or build_twist_stack(instance, self.world)
        self.window_radius = window_radius
        self.window_center = window_center
        self.kappa = float(kappa)
        self.c_star = c_star
        self.l_star = l_star
        self.t_max: Optional[int] = (4 * int(l_star) + 20) if l_star is not None else None
        self.certified: bool = (c_star is not None and l_star is not None)

        self.start = instance.start
        self.stops = list(instance.delivery_stops)
        self.stop_indices: Dict[Tuple[int, int], int] = {
            pos: idx for idx, pos in enumerate(self.stops)
        }
        self.target_mask = (
            target_mask if target_mask is not None else (1 << len(self.stops)) - 1
        )
        self.return_to_depot = return_to_depot
        self.allow_wait = allow_wait

        # Precedence constraints from DeliveryProblem
        self.precedence: Dict[int, List[int]] = getattr(instance, "precedence", {}) or {}

        # Twist awareness
        self.enabled_twist_names = [t.name for t in self.twists.twists if t.enabled]
        self.has_teleport = "teleport_tunnels" in self.enabled_twist_names
        self.has_theft = "rival_steals_parcels" in self.enabled_twist_names
        self.tunnel_pairs: Dict[Tuple[int, int], Tuple[int, int]] = tunnel_pairs or {}

        # If teleport_tunnels twist is active and has portals, extract them
        if self.has_teleport and not self.tunnel_pairs:
            for t in self.twists.twists:
                if t.enabled and t.name == "teleport_tunnels":
                    pairs = getattr(t, "portals", None) or getattr(t, "tunnels", None)
                    if isinstance(pairs, dict):
                        self.tunnel_pairs = dict(pairs)

    def extract_window(
        self,
        center: Tuple[int, int],
        radius: Optional[int] = None,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """Extracts a square Chebyshev window around center, clipped to grid bounds.

        Returns ((r_min, r_max, c_min, c_max), subgrid_view).
        Boundary restriction: actors inside the window cannot step outside [r_min..r_max, c_min..c_max].
        """
        rad = self.window_radius if radius is None else radius
        if rad is None:
            return (0, self.world.n - 1, 0, self.world.n - 1), self.world.grid

        cr, cc = center
        r_min = max(0, cr - rad)
        r_max = min(self.world.n - 1, cr + rad)
        c_min = max(0, cc - rad)
        c_max = min(self.world.n - 1, cc + rad)
        return (r_min, r_max, c_min, c_max), self.world.grid[r_min : r_max + 1, c_min : c_max + 1]

    def in_window(self, pos: Tuple[int, int], center: Optional[Tuple[int, int]] = None) -> bool:
        """Checks whether pos lies within the active Chebyshev window."""
        if self.window_radius is None:
            return self.world.in_bounds(pos[0], pos[1])
        c = center or self.window_center
        if c is None:
            return self.world.in_bounds(pos[0], pos[1])
        return (
            self.world.in_bounds(pos[0], pos[1])
            and max(abs(pos[0] - c[0]), abs(pos[1] - c[1])) <= self.window_radius
        )

    def initial_state(
        self,
        agent_pos: Optional[Tuple[int, int]] = None,
        rival_pos: Optional[Tuple[int, int]] = None,
        delivered_mask: int = 0,
        extra_state: Optional[Tuple[Any, ...]] = None,
        turn: int = AGENT,
        agent_cost: float = 0.0,
        steps: int = 0,
    ) -> GameState:
        """Constructs a complete initial GameState."""
        a_pos = agent_pos or self.start
        r_pos = rival_pos or (self.world.n - 1, self.world.n - 1)
        extra = extra_state if extra_state is not None else self.twists.initial_extra_state(self.world)
        return GameState(
            agent_pos=a_pos,
            rival_pos=r_pos,
            delivered_mask=delivered_mask,
            extra_state=extra,
            turn=turn,
            agent_cost=agent_cost,
            steps=steps,
            plies=0,
        )

    def _satisfies_precedence(self, stop_idx: int, mask: int) -> bool:
        reqs = self.precedence.get(stop_idx, [])
        return all(bool(mask & (1 << req)) for req in reqs)

    def legal_actions(self, state: GameState, player: Optional[int] = None) -> List[int]:
        """Returns legal actions for the specified player in canonical order: N, E, S, W, WAIT, twist actions."""
        actor = state.turn if player is None else player
        pos = state.agent_pos if actor == AGENT else state.rival_pos
        r, c = pos
        legal: List[int] = []

        for a in (NORTH, EAST, SOUTH, WEST):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)
            if not self.world.in_bounds(nr, nc) or not self.world.passable[nr, nc]:
                continue
            if not self.in_window(next_pos):
                continue

            if actor == AGENT:
                if self.twists.is_action_legal(pos, a, next_pos, state.extra_state, self.world):
                    legal.append(a)
            else:
                # Rival respects terrain obstacles and one-way street restrictions
                if self.twists.is_action_legal(pos, a, next_pos, (), self.world):
                    legal.append(a)

        if self.allow_wait:
            legal.append(WAIT)

        # Twist-specific actions (e.g., TELEPORT when teleport_tunnels is enabled)
        if actor == AGENT and self.has_teleport and pos in self.tunnel_pairs:
            dest = self.tunnel_pairs[pos]
            if self.world.in_bounds(dest[0], dest[1]) and self.world.passable[dest[0], dest[1]]:
                if self.in_window(dest):
                    legal.append(TELEPORT)

        return legal

    def step(self, state: GameState, action: int) -> GameState:
        """Applies a single player's action and advances turn to the other player."""
        if state.turn == AGENT:
            r, c = state.agent_pos
            if action == TELEPORT and self.has_teleport and (r, c) in self.tunnel_pairs:
                next_pos = self.tunnel_pairs[(r, c)]
                step_cost = float(self.world.cost_map[next_pos])
                next_extra = state.extra_state
            elif action == WAIT:
                next_pos = (r, c)
                step_cost = 3.0
                next_extra = self.twists.update_extra_state(
                    state.extra_state, action, state.agent_pos, next_pos, self.world
                )
            else:
                dr, dc = ACTION_DELTAS[action]
                next_pos = (r + dr, c + dc)
                u = self.world.idx(r, c)
                base_cost = float(self.world.move_cost_table[u, action])
                step_cost = self.twists.compute_step_cost(
                    state.agent_pos, action, next_pos, base_cost, state.extra_state, self.world
                )
                next_extra = self.twists.update_extra_state(
                    state.extra_state, action, state.agent_pos, next_pos, self.world
                )

            next_mask = state.delivered_mask
            if next_pos in self.stop_indices:
                idx = self.stop_indices[next_pos]
                if self._satisfies_precedence(idx, next_mask):
                    next_mask |= 1 << idx

            return GameState(
                agent_pos=next_pos,
                rival_pos=state.rival_pos,
                delivered_mask=next_mask,
                extra_state=next_extra,
                turn=RIVAL,
                agent_cost=state.agent_cost + step_cost,
                steps=state.steps + 1,
                plies=state.plies + 1,
            )

        # RIVAL turn
        rr, rc = state.rival_pos
        if action == WAIT:
            next_rival = (rr, rc)
        else:
            dr, dc = ACTION_DELTAS[action]
            next_rival = (rr + dr, rc + dc)

        return GameState(
            agent_pos=state.agent_pos,
            rival_pos=next_rival,
            delivered_mask=state.delivered_mask,
            extra_state=state.extra_state,
            turn=AGENT,
            agent_cost=state.agent_cost,
            steps=state.steps,
            plies=state.plies + 1,
        )

    def outcome(
        self,
        state: GameState,
        prev_state: Optional[GameState] = None,
    ) -> GameOutcome:
        """Evaluates D7 terminal precedence on state.

        Precedence:
        1. Capture (same-cell, or edge-swap if prev_state is provided for joint transition)
        2. Theft (if rival_steals_parcels enabled and rival reaches undelivered stop/agent)
        3. Completed delivery -> WIN if agent_cost <= kappa * C* (or C* uncertified), else DRAW_EXCESS_COST
        4. Excessive-cost draw -> DRAW_EXCESS_COST if agent_cost > kappa * C*
        5. Step-cap draw -> DRAW_STEP_CAP if steps >= T_max
        """
        # 1. Capture
        if check_same_cell_capture(state.agent_pos, state.rival_pos):
            return GameOutcome.LOSS_CAPTURE
        if prev_state is not None and check_edge_swap_capture(
            prev_state.agent_pos, state.agent_pos, prev_state.rival_pos, state.rival_pos
        ):
            return GameOutcome.LOSS_CAPTURE

        # 2. Theft (only if rival_steals_parcels twist is active)
        if self.has_theft and state.rival_pos in self.stop_indices:
            s_idx = self.stop_indices[state.rival_pos]
            if not (state.delivered_mask & (1 << s_idx)):
                return GameOutcome.LOSS_THEFT

        # 3. Completed delivery
        all_delivered = (state.delivered_mask & self.target_mask) == self.target_mask
        at_depot_ok = (not self.return_to_depot) or (state.agent_pos == self.start)
        if all_delivered and at_depot_ok:
            if self.c_star is not None and state.agent_cost > self.kappa * self.c_star + 1e-9:
                return GameOutcome.DRAW_EXCESS_COST
            return GameOutcome.WIN

        # 4. Excessive-cost draw
        if self.c_star is not None and state.agent_cost > self.kappa * self.c_star + 1e-9:
            return GameOutcome.DRAW_EXCESS_COST

        # 5. Step-cap draw
        if self.t_max is not None and state.steps >= self.t_max:
            return GameOutcome.DRAW_STEP_CAP

        return GameOutcome.ONGOING

    def is_terminal(self, state: GameState, prev_state: Optional[GameState] = None) -> bool:
        return self.outcome(state, prev_state=prev_state) != GameOutcome.ONGOING

    def terminal_utility(
        self,
        state: GameState,
        prev_state: Optional[GameState] = None,
    ) -> float:
        """Returns terminal utility with M >> max |f(s)| and bounded cost tie-break on WIN."""
        out = self.outcome(state, prev_state=prev_state)
        if out == GameOutcome.WIN:
            # Small cost tie-break strictly bounded below 100.0 so it can never cross into draw/loss
            tie_break = min(99.0, 0.001 * state.agent_cost)
            return UTILITY_M - tie_break
        if out in (GameOutcome.LOSS_CAPTURE, GameOutcome.LOSS_THEFT):
            return -UTILITY_M
        return 0.0
