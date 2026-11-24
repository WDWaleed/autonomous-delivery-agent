"""Runtime and experiment parameter overrides parser and validator (§6.1, M2.6).

Governed by decisions D1, D3, D8, D15, and D20.
Enables command-line, configuration, or viva live parameter modifications.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from delivery.instance import Instance


VALID_ADVERSARIES = {"chaser", "random", "patrol", "intercept"}


def validate_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Validates override dictionary against domain constraints and invariants.

    Raises ValueError if any override violates game or search rules (e.g. negative costs).
    """
    cleaned: Dict[str, Any] = {}

    for k, v in overrides.items():
        if k == "grid_size" or k == "n":
            n_val = int(v)
            if n_val < 5 or n_val > 100:
                raise ValueError(f"grid_size must be in [5, 100], got {n_val}")
            cleaned["n"] = n_val

        elif k == "slip_probability":
            p_val = float(v)
            if not (0.0 <= p_val <= 1.0):
                raise ValueError(f"slip_probability must be in [0.0, 1.0], got {p_val}")
            cleaned["slip_probability"] = p_val

        elif k == "obstacle_density":
            d_val = float(v)
            if not (0.0 <= d_val < 0.8):
                raise ValueError(f"obstacle_density must be in [0.0, 0.8), got {d_val}")
            cleaned["obstacle_density"] = d_val

        elif k == "terrain_cost":
            if not isinstance(v, dict):
                raise ValueError(f"terrain_cost must be a dict, got {type(v)}")
            for t_name, cost in v.items():
                if int(cost) <= 0:
                    raise ValueError(f"Terrain cost for {t_name} must be strictly positive (> 0 per D3/D15), got {cost}")
            cleaned["terrain_cost"] = {str(t): int(c) for t, c in v.items()}

        elif k == "adversary":
            adv_str = str(v).strip().lower()
            if adv_str not in VALID_ADVERSARIES:
                raise ValueError(f"Unknown adversary '{v}'. Expected one of {VALID_ADVERSARIES}")
            cleaned["adversary"] = adv_str

        elif k == "adversary_search_depth":
            depth = int(v)
            if depth < 0 or depth > 20:
                raise ValueError(f"adversary_search_depth must be in [0, 20], got {depth}")
            cleaned["adversary_search_depth"] = depth

        elif k == "start":
            r, c = int(v[0]), int(v[1])
            cleaned["start"] = (r, c)

        elif k == "delivery_stops":
            stops = [tuple(map(int, p)) for p in v]
            if len(stops) < 1:
                raise ValueError("delivery_stops must contain at least 1 stop.")
            cleaned["delivery_stops"] = stops

        elif k == "twists":
            if not isinstance(v, (list, tuple)):
                raise ValueError(f"twists must be a list of strings, got {type(v)}")
            cleaned["twists"] = [str(t) for t in v]

        elif k == "seed":
            cleaned["seed"] = int(v)

        elif k == "cms_ids":
            cleaned["cms_ids"] = list(v)

        else:
            # Pass through any other specific experiment flags
            cleaned[k] = v

    return cleaned


def parse_cli_overrides(args: Sequence[str]) -> Dict[str, Any]:
    """Parses command-line style override arguments such as --slip=0.20 or k=v or --grid-size 25."""
    raw: Dict[str, Any] = {}
    it = iter(args)
    for token in it:
        clean = token[2:] if token.startswith("--") else token
        if "=" in clean:
            k, v = clean.split("=", 1)
        else:
            k = clean
            try:
                v = next(it)
            except StopIteration:
                v = "true"
        k = k.replace("-", "_")
        # Parse primitives
        if v.lower() == "true":
            raw[k] = True
        elif v.lower() == "false":
            raw[k] = False
        else:
            try:
                if "." in v:
                    raw[k] = float(v)
                else:
                    raw[k] = int(v)
            except ValueError:
                raw[k] = v
    return validate_overrides(raw)


def apply_overrides(instance: Instance, overrides: Optional[Dict[str, Any]] = None) -> Instance:
    """Creates a new Instance copy with validated overrides applied."""
    if not overrides:
        return copy.deepcopy(instance)

    valid = validate_overrides(overrides)
    inst_copy = copy.deepcopy(instance)

    for k, v in valid.items():
        if hasattr(inst_copy, k):
            setattr(inst_copy, k, v)

    # If start is overridden, ensure valid bounds
    sr, sc = inst_copy.start
    if not (0 <= sr < inst_copy.n and 0 <= sc < inst_copy.n):
        raise ValueError(f"Start coordinates {inst_copy.start} out of bounds for grid {inst_copy.n}x{inst_copy.n}")

    # If delivery stops overridden, ensure valid bounds
    for dr, dc in inst_copy.delivery_stops:
        if not (0 <= dr < inst_copy.n and 0 <= dc < inst_copy.n):
            raise ValueError(f"Stop coordinates ({dr}, {dc}) out of bounds for grid {inst_copy.n}x{inst_copy.n}")

    return inst_copy
