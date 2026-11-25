"""Deterministic Dispatcher Tools & JSON Schemas (§7.6, M8.2).

Tools:
1. plan_route(algorithm, heuristic, constraints): invokes Part B search planners
   with constrained DeliveryProblem (avoid_terrains, forbidden_cells, target_stops, max_cost, return_to_depot).
2. simulate(plan, slip_p, seed): runs DeliveryEnv simulation and reports outcome & constraint checks.
3. inspect_twists(pos, action): twist-specific inspection for assigned twists
   (one_way_streets, fog_partial_observability, fragile_parcel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from delivery.heuristics.base import ZeroHeuristic
from delivery.heuristics.baselines import ManhattanNearestStopHeuristic, MaxRelaxedDistanceHeuristic
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.instance import Instance, get_canonical_instance
from delivery.search.astar import astar_search
from delivery.search.bfs import breadth_first_search
from delivery.search.core import Limits, SearchStatus
from delivery.search.dls import depth_limited_search
from delivery.search.greedy import greedy_best_first_search
from delivery.search.ids import iterative_deepening_search
from delivery.search.problem import ACTION_DELTAS, ACTION_NAMES, EAST, NORTH, SOUTH, WAIT, WEST, DeliveryProblem
from delivery.search.ucs import uniform_cost_search
from delivery.sim.env import DeliveryEnv
from delivery.twists.base import TwistStack
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

VALID_ALGORITHMS = {"bfs", "ucs", "dls", "ids", "greedy", "astar"}
VALID_HEURISTICS = {"zero", "b1", "b2", "own"}
TERRAIN_ID_BY_NAME = {"road": 0, "grass": 1, "mud": 2, "ford": 3}
ACTION_BY_NAME = {v: k for k, v in ACTION_NAMES.items()}


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "plan_route",
            "description": "Plan a delivery route using a Part B search algorithm and optional heuristic/constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": sorted(VALID_ALGORITHMS),
                        "description": "Search algorithm to run.",
                    },
                    "heuristic": {
                        "type": "string",
                        "enum": sorted(VALID_HEURISTICS),
                        "description": "Heuristic for A* or Greedy (default: own).",
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Optional routing constraints: avoid_terrains, forbidden_cells, target_stops, max_cost, return_to_depot.",
                        "properties": {
                            "avoid_terrains": {"type": "array", "items": {"type": "string"}},
                            "forbidden_cells": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                            "target_stops": {"type": "array", "items": {"type": "integer"}},
                            "max_cost": {"type": "number"},
                            "return_to_depot": {"type": "boolean"},
                        },
                    },
                },
                "required": ["algorithm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate",
            "description": "Simulate a planned sequence of actions (integer codes 0..4 or names NORTH/EAST/SOUTH/WEST/WAIT) in DeliveryEnv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {"type": ["integer", "string"]},
                        "description": "Sequence of actions to simulate.",
                    },
                    "slip_p": {
                        "type": "number",
                        "description": "Optional slip probability override (default 0.0 for deterministic verification).",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Optional RNG seed for stochastic simulation.",
                    },
                },
                "required": ["plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_twists",
            "description": "Inspect assigned twist rules (one_way_streets, fog_partial_observability, fragile_parcel) at a grid position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pos": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[row, col] coordinate to inspect.",
                    },
                    "fragile_delivered": {
                        "type": "boolean",
                        "description": "Whether the fragile parcel stop has already been delivered.",
                    },
                },
                "required": ["pos"],
            },
        },
    },
]


class DispatcherToolbox:
    """Deterministic tool executor bound to an Instance, GridWorld, and TwistStack."""

    def __init__(
        self,
        instance: Optional[Instance] = None,
        world: Optional[GridWorld] = None,
        twists: Optional[TwistStack] = None,
    ):
        self.instance = instance or get_canonical_instance()
        self.world = world or GridWorld(self.instance)
        self.twists = twists or build_twist_stack(self.instance, self.world)
        self.call_history: List[Dict[str, Any]] = []

    def _build_constrained_instance(
        self,
        constraints: Optional[Dict[str, Any]],
    ) -> Tuple[Instance, GridWorld, TwistStack, bool]:
        c = constraints or {}
        grid = self.instance.grid.copy()

        # 1. Avoid terrains
        for t_name in c.get("avoid_terrains", []):
            t_id = TERRAIN_ID_BY_NAME.get(str(t_name).lower())
            if t_id is not None:
                grid[grid == t_id] = -1

        # 2. Forbidden cells
        for cell in c.get("forbidden_cells", []):
            r, col = int(cell[0]), int(cell[1])
            if 0 <= r < self.instance.n and 0 <= col < self.instance.n:
                grid[r, col] = -1

        # 3. Target stops subset
        target_indices = c.get("target_stops")
        if target_indices is not None:
            stops = [self.instance.delivery_stops[int(i)] for i in target_indices]
        else:
            stops = list(self.instance.delivery_stops)

        return_to_depot = bool(c.get("return_to_depot", False))

        # Keep start passable? No—if user forbade start or terrain of start, let search detect infeasibility!
        c_inst = Instance(
            id=f"{self.instance.id}-constrained",
            n=self.instance.n,
            grid=grid,
            terrain_cost=dict(self.instance.terrain_cost),
            obstacle_density=self.instance.obstacle_density,
            start=self.instance.start,
            delivery_stops=stops,
            adversary=self.instance.adversary,
            slip_probability=0.0,
            adversary_search_depth=self.instance.adversary_search_depth,
            twists=list(self.instance.twists),
            seed=self.instance.seed,
            cms_ids=list(self.instance.cms_ids),
        )
        c_world = GridWorld(c_inst)
        c_twists = build_twist_stack(c_inst, c_world)
        return c_inst, c_world, c_twists, return_to_depot

    def plan_route(
        self,
        algorithm: str = "astar",
        heuristic: str = "own",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        algo = str(algorithm).lower()
        heur_name = str(heuristic).lower()
        c = constraints or {}

        c_inst, c_world, c_twists, return_to_depot = self._build_constrained_instance(c)
        problem = DeliveryProblem(
            c_inst,
            world=c_world,
            twist_stack=c_twists,
            return_to_depot=return_to_depot,
        )
        max_cost = c.get("max_cost")
        limits = Limits(
            max_time_seconds=15.0,
            max_expanded=100_000,
            max_cost=float(max_cost) if max_cost is not None else None,
        )

        # Heuristic
        h_obj = None
        if algo in ("astar", "greedy"):
            if heur_name == "zero":
                h_obj = ZeroHeuristic()
            elif heur_name == "b1":
                h_obj = ManhattanNearestStopHeuristic()
            elif heur_name == "b2":
                h_obj = MaxRelaxedDistanceHeuristic(problem)
            elif heur_name == "own":
                h_obj = CombinedMSTHeuristic(problem)
            else:
                return {
                    "status": "INVALID_ARG",
                    "reason_codes": [f"UNKNOWN_HEURISTIC:{heur_name}"],
                    "constraints_satisfied": [],
                    "constraints_violated": ["heuristic"],
                }

        if algo == "bfs":
            res = breadth_first_search(problem, limits=limits)
        elif algo == "ucs":
            res = uniform_cost_search(problem, limits=limits)
        elif algo == "dls":
            res = depth_limited_search(problem, depth_limit=80, limits=limits)
        elif algo == "ids":
            res = iterative_deepening_search(problem, max_depth=80, limits=limits)
        elif algo == "greedy":
            res = greedy_best_first_search(problem, heuristic=h_obj, limits=limits)
        elif algo == "astar":
            res = astar_search(problem, heuristic=h_obj, limits=limits)
        else:
            return {
                "status": "INVALID_ARG",
                "reason_codes": [f"UNKNOWN_ALGORITHM:{algo}"],
                "constraints_satisfied": [],
                "constraints_violated": ["algorithm"],
            }

        satisfied: List[str] = []
        violated: List[str] = []
        reason_codes: List[str] = [f"SEARCH_{res.status.value}"]

        if res.status == SearchStatus.SUCCESS:
            satisfied.extend(["reach_all_stops"])
            if max_cost is not None:
                if res.cost <= float(max_cost) + 1e-9:
                    satisfied.append("max_cost")
                else:
                    violated.append("max_cost")
                    reason_codes.append("EXCEEDED_MAX_COST")
            if c.get("avoid_terrains"):
                satisfied.append("avoid_terrains")
            if c.get("forbidden_cells"):
                satisfied.append("forbidden_cells")
            if return_to_depot:
                satisfied.append("return_to_depot")
        else:
            violated.append("feasible_route")
            if res.status == SearchStatus.FAILED:
                reason_codes.append("PROVEN_INFEASIBLE")
            else:
                reason_codes.append("RESOURCE_LIMIT")

        final_status = res.status.value
        if violated and final_status == "SUCCESS":
            final_status = "INFEASIBLE"

        return {
            "status": final_status,
            "cost": float(res.cost) if np.isfinite(res.cost) else None,
            "plan": list(res.plan) if res.plan is not None else None,
            "route_summary": {
                "algorithm": algo,
                "heuristic": heur_name if algo in ("astar", "greedy") else "none",
                "num_actions": len(res.plan) if res.plan is not None else 0,
                "expanded_nodes": res.metrics.expanded_nodes,
                "stops_count": len(c_inst.delivery_stops),
            },
            "constraints_satisfied": satisfied,
            "constraints_violated": violated,
            "reason_codes": reason_codes,
        }

    def simulate(
        self,
        plan: Sequence[Any],
        slip_p: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        parsed_actions: List[int] = []
        for a in plan:
            if isinstance(a, int):
                parsed_actions.append(a)
            elif isinstance(a, str) and a.upper() in ACTION_BY_NAME:
                parsed_actions.append(ACTION_BY_NAME[a.upper()])
            else:
                return {
                    "status": "INVALID_ARG",
                    "reason_codes": [f"INVALID_ACTION:{a}"],
                    "constraints_satisfied": [],
                    "constraints_violated": ["valid_actions"],
                }

        env = DeliveryEnv(
            world=self.world,
            twists=self.twists,
            slip_p=float(slip_p),
            seed=seed if seed is not None else self.instance.seed,
        )
        illegal_steps = 0
        for act in parsed_actions:
            _, _, ev = env.step(act)
            if ev.get("illegal_move"):
                illegal_steps += 1
            if env.is_done():
                break

        delivered = [
            idx for idx in range(len(self.instance.delivery_stops))
            if (env.delivered_mask & (1 << idx))
        ]
        all_done = env.is_done()
        satisfied = ["legal_execution"] if illegal_steps == 0 else []
        violated = ["legal_execution"] if illegal_steps > 0 else []
        if all_done:
            satisfied.append("all_stops_delivered")
        else:
            violated.append("all_stops_delivered")

        return {
            "status": "SUCCESS" if all_done else "INCOMPLETE",
            "cost": float(env.total_cost),
            "route_summary": {
                "steps_executed": env.step_count,
                "final_pos": list(env.agent_pos),
                "delivered_stops": delivered,
                "illegal_steps": illegal_steps,
            },
            "constraints_satisfied": satisfied,
            "constraints_violated": violated,
            "reason_codes": ["SIM_COMPLETED" if all_done else "SIM_INCOMPLETE"],
        }

    def inspect_twists(
        self,
        pos: Sequence[int],
        fragile_delivered: bool = False,
    ) -> Dict[str, Any]:
        r, c = int(pos[0]), int(pos[1])
        if not self.world.in_bounds(r, c):
            return {
                "status": "INVALID_ARG",
                "reason_codes": ["OUT_OF_BOUNDS"],
                "constraints_satisfied": [],
                "constraints_violated": ["in_bounds"],
            }

        extra_state = (bool(fragile_delivered),) if self.twists.extra_names else ()
        allowed_dirs = []
        blocked_dirs = []
        for a in (NORTH, EAST, SOUTH, WEST):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if not self.world.in_bounds(nr, nc) or not self.world.passable[nr, nc]:
                blocked_dirs.append(ACTION_NAMES[a])
            elif self.twists.is_action_legal((r, c), a, (nr, nc), extra_state, self.world):
                allowed_dirs.append(ACTION_NAMES[a])
            else:
                blocked_dirs.append(ACTION_NAMES[a])

        return {
            "status": "SUCCESS",
            "cost": 0.0,
            "route_summary": {
                "pos": [r, c],
                "terrain_id": int(self.world.grid[r, c]),
                "allowed_directions": allowed_dirs,
                "blocked_directions": blocked_dirs,
                "active_twists": [t.name for t in self.twists.twists if t.enabled],
            },
            "constraints_satisfied": ["twist_inspection"],
            "constraints_violated": [],
            "reason_codes": ["INSPECT_OK"],
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.call_history.append({"tool": tool_name, "arguments": dict(arguments)})
        if tool_name == "plan_route":
            return self.plan_route(**arguments)
        if tool_name == "simulate":
            return self.simulate(**arguments)
        if tool_name == "inspect_twists":
            return self.inspect_twists(**arguments)
        return {
            "status": "UNKNOWN_TOOL",
            "reason_codes": [f"UNKNOWN_TOOL:{tool_name}"],
            "constraints_satisfied": [],
            "constraints_violated": ["known_tool"],
        }
