"""Hybrid Strategic A* + Tactical Alpha-Beta Delivery Agent (§7.3, §7.5, M7.2).

Combines:
1. Strategic Planner: A* with CombinedMSTHeuristic (own) on the known/discovered map.
2. Tactical Adversary Avoidance: When a rival is present inside the configured
   Chebyshev window (radius W), switches to depth-limited Alpha-Beta search
   using WindowGame and own move ordering.
3. Replan Trigger Accounting: Separately counts 'slip', 'rival_block',
   'fog_discovery', 'theft_threat', and 'resource_change'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from delivery.agents.base import Agent
from delivery.game.alphabeta import alphabeta_search
from delivery.game.ordering import make_own_ordering
from delivery.game.stats import EvalFn, placeholder_eval
from delivery.game.window import AGENT, GameState, WindowGame, compute_canonical_baseline
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.instance import Instance
from delivery.search.astar import astar_search
from delivery.search.core import Limits, SearchStatus
from delivery.search.problem import ACTION_DELTAS, NORTH, EAST, SOUTH, WEST, WAIT, DeliveryProblem
from delivery.twists.base import TwistStack
from delivery.twists.registry import build_twist_stack
from delivery.world.distance import distance_field
from delivery.world.grid import GridWorld


@dataclass
class ReplanCounts:
    """Separate counters for replan triggers (M7.2)."""

    initial_plan: int = 0
    slip: int = 0
    rival_block: int = 0
    fog_discovery: int = 0
    theft_threat: int = 0
    resource_change: int = 0

    def total_replans(self) -> int:
        return (
            self.slip
            + self.rival_block
            + self.fog_discovery
            + self.theft_threat
            + self.resource_change
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "initial_plan": self.initial_plan,
            "slip": self.slip,
            "rival_block": self.rival_block,
            "fog_discovery": self.fog_discovery,
            "theft_threat": self.theft_threat,
            "resource_change": self.resource_change,
            "total_replans": self.total_replans(),
        }


class HybridAgent(Agent):
    """Hybrid Strategic A* + Tactical Alpha-Beta delivery agent."""

    name: str = "HybridAgent"

    def __init__(
        self,
        instance: Instance,
        world: Optional[GridWorld] = None,
        twists: Optional[TwistStack] = None,
        window_radius: int = 4,
        tactical_depth: Optional[int] = None,
        eval_fn: Optional[EvalFn] = None,
        kappa: float = 1.5,
    ):
        self.instance = instance
        self.world = world or GridWorld(instance)
        self.twists = twists or build_twist_stack(instance, self.world)
        self.window_radius = int(window_radius)
        self.tactical_depth = int(tactical_depth or instance.adversary_search_depth or 4)
        self.eval_fn = eval_fn or placeholder_eval
        self.kappa = float(kappa)

        # Precompute relaxed stop fields for move ordering
        relaxed = self.twists.relaxed_graph(self.world)
        self.stop_fields = [
            distance_field(self.world, s, reverse=True, edge_model=relaxed)
            for s in instance.delivery_stops
        ]
        self.own_order = make_own_ordering(stop_fields=self.stop_fields)

        # Canonical baseline for WindowGame
        self.c_star, self.l_star, self.certified = compute_canonical_baseline(
            instance, world=self.world, twists=self.twists
        )

        # Episode state
        self.plan_actions: List[int] = []
        self.plan_positions: List[Tuple[int, int]] = []
        self.plan_idx: int = 0
        self.expected_pos: Optional[Tuple[int, int]] = None
        self.last_extra: Optional[Tuple[Any, ...]] = None
        self.last_delivered_mask: int = 0
        self.replan_counts = ReplanCounts()
        self.tactical_moves: int = 0
        self.strategic_moves: int = 0

    def reset(self) -> None:
        self.plan_actions = []
        self.plan_positions = []
        self.plan_idx = 0
        self.expected_pos = None
        self.last_extra = None
        self.last_delivered_mask = 0
        self.replan_counts = ReplanCounts()
        self.tactical_moves = 0
        self.strategic_moves = 0

    def _build_planning_world(self, percept: Dict[str, Any]) -> Tuple[Instance, GridWorld, TwistStack]:
        """Builds a planning GridWorld reflecting Fog discoveries if partially observable."""
        if not percept.get("is_partially_observable", False):
            return self.instance, self.world, self.twists

        opt_cost = percept.get("optimistic_cost_map")
        disc_mask = percept.get("discovered_mask")
        if opt_cost is None or disc_mask is None:
            return self.instance, self.world, self.twists

        # Update grid copy with discovered obstacles (-1 where optimistic_cost_map is inf)
        new_grid = self.instance.grid.copy()
        for r in range(self.instance.n):
            for c in range(self.instance.n):
                if disc_mask[r, c] and np.isinf(opt_cost[r, c]):
                    new_grid[r, c] = -1

        plan_inst = Instance(
            id=self.instance.id,
            n=self.instance.n,
            grid=new_grid,
            terrain_cost=dict(self.instance.terrain_cost),
            obstacle_density=self.instance.obstacle_density,
            start=percept["agent_pos"],
            delivery_stops=list(self.instance.delivery_stops),
            adversary=self.instance.adversary,
            slip_probability=0.0,
            adversary_search_depth=self.instance.adversary_search_depth,
            twists=list(self.instance.twists),
            seed=self.instance.seed,
            cms_ids=list(self.instance.cms_ids),
        )
        plan_world = GridWorld(plan_inst)
        plan_twists = build_twist_stack(plan_inst, plan_world)
        return plan_inst, plan_world, plan_twists

    def _replan_strategic(self, percept: Dict[str, Any]) -> None:
        """Runs A* with CombinedMSTHeuristic from current state."""
        pos = percept["agent_pos"]
        mask = percept["delivered_mask"]

        plan_inst, plan_world, plan_twists = self._build_planning_world(percept)
        problem = DeliveryProblem(plan_inst, world=plan_world, twist_stack=plan_twists)

        # Override initial state to current pos & delivered_mask & extra_state
        extra_tuple = tuple(percept.get("extra", {}).values()) if percept.get("extra") else plan_twists.initial_extra_state(plan_world)
        # Ensure extra_tuple matches TwistStack length
        if len(extra_tuple) != len(plan_twists.extra_names):
            extra_tuple = plan_twists.initial_extra_state(plan_world)

        curr_state = (pos[0], pos[1], mask, extra_tuple)
        problem.initial_state = lambda: curr_state  # type: ignore

        h_own = CombinedMSTHeuristic(problem)
        res = astar_search(problem, h_own, limits=Limits(max_time_seconds=10.0, max_expanded=100_000))
        if res.status == SearchStatus.SUCCESS and res.plan is not None and res.trajectory is not None:
            self.plan_actions = list(res.plan)
            self.plan_positions = list(res.trajectory)
            self.plan_idx = 0
        else:
            self.plan_actions = []
            self.plan_positions = []
            self.plan_idx = 0

    def _check_planned_path_blocked_by_fog(self, percept: Dict[str, Any]) -> bool:
        if not percept.get("is_partially_observable", False):
            return False
        opt_cost = percept.get("optimistic_cost_map")
        disc_mask = percept.get("discovered_mask")
        if opt_cost is None or disc_mask is None:
            return False
        for pos in self.plan_positions[self.plan_idx + 1 :]:
            if disc_mask[pos] and np.isinf(opt_cost[pos]):
                return True
        return False

    def choose_action(self, percept: Dict[str, Any]) -> Tuple[int, str]:
        pos = percept["agent_pos"]
        rival_pos = percept.get("rival_pos")
        extra_tuple = tuple(sorted(percept.get("extra", {}).items()))
        delivered_mask = percept["delivered_mask"]

        # 1. Tactical Alpha-Beta if rival is inside Chebyshev window
        if rival_pos is not None:
            cheb_dist = max(abs(pos[0] - rival_pos[0]), abs(pos[1] - rival_pos[1]))
            if cheb_dist <= self.window_radius:
                self.replan_counts.rival_block += 1
                self.tactical_moves += 1
                game = WindowGame(
                    self.instance,
                    world=self.world,
                    twists=self.twists,
                    window_radius=self.window_radius,
                    window_center=pos,
                    kappa=self.kappa,
                    c_star=self.c_star,
                    l_star=self.l_star,
                )
                extra_state = self.twists.initial_extra_state(self.world)
                if self.twists.extra_names and percept.get("extra"):
                    extra_state = tuple(percept["extra"][k] for k in self.twists.extra_names if k in percept["extra"])
                g_state = GameState(
                    agent_pos=pos,
                    rival_pos=rival_pos,
                    delivered_mask=delivered_mask,
                    extra_state=extra_state,
                    turn=AGENT,
                    agent_cost=float(percept.get("total_cost", 0.0)),
                    steps=int(percept.get("step_count", 0)),
                )
                decision = alphabeta_search(
                    game=game,
                    state=g_state,
                    depth_limit=self.tactical_depth,
                    eval_fn=self.eval_fn,
                    order_fn=self.own_order,
                )
                act = decision.best_action if decision.best_action is not None else WAIT
                # Invalidate expected strategic position since tactical took over
                self.expected_pos = None
                self.plan_actions = []
                return act, f"Tactical Alpha-Beta (rival dist={cheb_dist}, val={decision.value:.1f})"

        # 2. Check strategic replan triggers
        need_replan = False
        trigger_reason = ""

        if not self.plan_actions or self.plan_idx >= len(self.plan_actions):
            if self.replan_counts.initial_plan == 0 and self.strategic_moves == 0 and self.tactical_moves == 0:
                self.replan_counts.initial_plan += 1
                trigger_reason = "initial_plan"
            else:
                self.replan_counts.rival_block += 1
                trigger_reason = "post_tactical_replan"
            need_replan = True
        elif self.expected_pos is not None and pos != self.expected_pos:
            self.replan_counts.slip += 1
            trigger_reason = "slip"
            need_replan = True
        elif self._check_planned_path_blocked_by_fog(percept):
            self.replan_counts.fog_discovery += 1
            trigger_reason = "fog_discovery"
            need_replan = True
        elif self.last_extra is not None and extra_tuple != self.last_extra:
            self.replan_counts.resource_change += 1
            trigger_reason = "resource_change"
            need_replan = True

        if need_replan:
            self._replan_strategic(percept)

        self.last_extra = extra_tuple
        self.last_delivered_mask = delivered_mask

        if self.plan_actions and self.plan_idx < len(self.plan_actions):
            action = self.plan_actions[self.plan_idx]
            if self.plan_idx + 1 < len(self.plan_positions):
                self.expected_pos = self.plan_positions[self.plan_idx + 1]
            else:
                self.expected_pos = pos
            self.plan_idx += 1
            self.strategic_moves += 1
            return action, f"Strategic A* ({trigger_reason or 'on_plan'})"

        # Fallback legal move if no plan found
        self.expected_pos = pos
        return WAIT, "Fallback WAIT (no strategic plan)"
