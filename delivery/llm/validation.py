"""Schema and Semantic Argument Validation Layer (§7.6, M8.5).

Blocks invalid, out-of-bounds, unknown, or contradictory tool arguments before
they can reach DispatcherToolbox (planner/simulator side effects), returning
machine-readable error feedback for bounded self-repair (<= 2 attempts).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from delivery.instance import Instance, get_canonical_instance
from delivery.llm.tools import (
    ACTION_BY_NAME,
    TERRAIN_ID_BY_NAME,
    VALID_ALGORITHMS,
    VALID_HEURISTICS,
)


class ArgumentValidator:
    """Validates LLM tool calls against schema and instance semantics."""

    def __init__(self, instance: Optional[Instance] = None):
        self.instance = instance or get_canonical_instance()
        self.interventions: int = 0
        self.intervention_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.interventions = 0
        self.intervention_log.clear()

    def _reject(self, tool_name: str, code: str, message: str) -> str:
        self.interventions += 1
        self.intervention_log.append({
            "tool": tool_name,
            "code": code,
            "message": message,
        })
        return json.dumps({"code": code, "message": message, "tool": tool_name})

    def validate(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Returns None if valid, or a machine-readable JSON error string if invalid."""
        if not isinstance(args, dict):
            return self._reject(tool_name, "INVALID_SCHEMA", "Tool arguments must be a JSON object.")

        if tool_name == "plan_route":
            return self._validate_plan_route(args)
        if tool_name == "simulate":
            return self._validate_simulate(args)
        if tool_name == "inspect_twists":
            return self._validate_inspect_twists(args)
        return self._reject(tool_name, "UNKNOWN_TOOL", f"Unknown tool '{tool_name}'.")

    def _validate_plan_route(self, args: Dict[str, Any]) -> Optional[str]:
        algo = str(args.get("algorithm", "")).lower()
        if algo not in VALID_ALGORITHMS:
            return self._reject(
                "plan_route",
                "INVALID_ALGORITHM",
                f"Unknown algorithm '{algo}'. Allowed: {sorted(VALID_ALGORITHMS)}.",
            )

        heur = str(args.get("heuristic", "own")).lower()
        if heur not in VALID_HEURISTICS:
            return self._reject(
                "plan_route",
                "INVALID_HEURISTIC",
                f"Unknown heuristic '{heur}'. Allowed: {sorted(VALID_HEURISTICS)}.",
            )

        c = args.get("constraints") or {}
        if not isinstance(c, dict):
            return self._reject("plan_route", "INVALID_CONSTRAINTS", "'constraints' must be an object.")

        # avoid_terrains
        avoid = c.get("avoid_terrains") or []
        if not isinstance(avoid, list):
            return self._reject("plan_route", "INVALID_TERRAIN", "'avoid_terrains' must be a list.")
        avoid_norm = set()
        for t in avoid:
            t_low = str(t).lower()
            if t_low not in TERRAIN_ID_BY_NAME:
                return self._reject(
                    "plan_route",
                    "UNKNOWN_TERRAIN",
                    f"Unknown terrain '{t}'. Valid terrains: {sorted(TERRAIN_ID_BY_NAME.keys())}.",
                )
            avoid_norm.add(t_low)
        if len(avoid_norm) == len(TERRAIN_ID_BY_NAME):
            return self._reject(
                "plan_route",
                "CONTRADICTORY_TERRAINS",
                "Cannot avoid all 4 passable terrains simultaneously.",
            )

        # target_stops
        num_stops = len(self.instance.delivery_stops)
        target_stops = c.get("target_stops")
        if target_stops is not None:
            if not isinstance(target_stops, list) or len(target_stops) == 0:
                return self._reject("plan_route", "INVALID_STOPS", "'target_stops' must be a non-empty list.")
            for idx in target_stops:
                if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < num_stops):
                    return self._reject(
                        "plan_route",
                        "UNKNOWN_STOP_ID",
                        f"Stop ID {idx!r} out of range [0, {num_stops - 1}].",
                    )

        # forbidden_cells
        forbidden = c.get("forbidden_cells") or []
        if not isinstance(forbidden, list):
            return self._reject("plan_route", "INVALID_CELLS", "'forbidden_cells' must be a list.")
        forbidden_set = set()
        for cell in forbidden:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                return self._reject("plan_route", "INVALID_CELL_FORMAT", f"Invalid cell {cell!r}; expected [r, c].")
            r, col = cell[0], cell[1]
            if not (isinstance(r, int) and isinstance(col, int)) or isinstance(r, bool) or isinstance(col, bool):
                return self._reject("plan_route", "INVALID_CELL_TYPE", f"Cell coordinates {cell!r} must be integers.")
            if not (0 <= r < self.instance.n and 0 <= col < self.instance.n):
                return self._reject(
                    "plan_route",
                    "CELL_OUT_OF_BOUNDS",
                    f"Cell {(r, col)} out of bounds for {self.instance.n}x{self.instance.n} grid.",
                )
            forbidden_set.add((r, col))

        # Contradiction check: forbidding start depot or requested stop
        if self.instance.start in forbidden_set:
            return self._reject(
                "plan_route",
                "CONTRADICTORY_START_FORBIDDEN",
                f"Cannot forbid start depot {self.instance.start}.",
            )
        check_indices = target_stops if target_stops is not None else list(range(num_stops))
        for idx in check_indices:
            if self.instance.delivery_stops[idx] in forbidden_set:
                return self._reject(
                    "plan_route",
                    "CONTRADICTORY_STOP_FORBIDDEN",
                    f"Cannot forbid requested Stop {idx} at {self.instance.delivery_stops[idx]}.",
                )

        # max_cost
        if "max_cost" in c and c["max_cost"] is not None:
            try:
                mc = float(c["max_cost"])
            except (TypeError, ValueError):
                return self._reject("plan_route", "INVALID_MAX_COST", "'max_cost' must be numeric.")
            if mc <= 0.0:
                return self._reject("plan_route", "NONPOSITIVE_MAX_COST", f"'max_cost' must be > 0, got {mc}.")

        return None

    def _validate_simulate(self, args: Dict[str, Any]) -> Optional[str]:
        plan = args.get("plan")
        if not isinstance(plan, list) or len(plan) == 0:
            return self._reject("simulate", "INVALID_PLAN", "'plan' must be a non-empty list of actions.")
        for act in plan:
            if isinstance(act, int) and not isinstance(act, bool):
                if not (0 <= act <= 4):
                    return self._reject("simulate", "INVALID_ACTION_CODE", f"Action code {act} out of range 0..4.")
            elif isinstance(act, str):
                if act.upper() not in ACTION_BY_NAME:
                    return self._reject("simulate", "INVALID_ACTION_NAME", f"Unknown action '{act}'.")
            else:
                return self._reject("simulate", "INVALID_ACTION_TYPE", f"Invalid action type {type(act).__name__}.")

        if "slip_p" in args and args["slip_p"] is not None:
            try:
                sp = float(args["slip_p"])
            except (TypeError, ValueError):
                return self._reject("simulate", "INVALID_SLIP_P", "'slip_p' must be numeric.")
            if not (0.0 <= sp <= 1.0):
                return self._reject("simulate", "SLIP_OUT_OF_RANGE", f"'slip_p' must be in [0.0, 1.0], got {sp}.")

        return None

    def _validate_inspect_twists(self, args: Dict[str, Any]) -> Optional[str]:
        pos = args.get("pos")
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            return self._reject("inspect_twists", "INVALID_POS", "'pos' must be [r, c].")
        r, c = pos[0], pos[1]
        if not (isinstance(r, int) and isinstance(c, int)) or isinstance(r, bool) or isinstance(c, bool):
            return self._reject("inspect_twists", "INVALID_POS_TYPE", "'pos' coordinates must be integers.")
        if not (0 <= r < self.instance.n and 0 <= c < self.instance.n):
            return self._reject(
                "inspect_twists",
                "POS_OUT_OF_BOUNDS",
                f"Position {(r, c)} out of bounds for {self.instance.n}x{self.instance.n} grid.",
            )
        return None

    def __call__(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        return self.validate(tool_name, args)
