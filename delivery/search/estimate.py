"""State space size estimator and combinatorial upper bound calculator (§7.4, M4.2).

Computes a conservative combinatorial upper bound on the number of planning states:
Upper Bound = N_passable * (2^k) * product(extension_cardinalities).
Clearly documented as a loose combinatorial upper bound, not exact reachable count.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from delivery.search.problem import DeliveryProblem


def estimate_state_space_bound(problem: DeliveryProblem) -> Dict[str, Any]:
    """Computes combinatorial upper bound on state space for DeliveryProblem (§7.4, M4.2).

    Returns a structured dictionary detailing factors and the total bound.
    """
    world = problem.world
    n_passable = int(np.sum(world.passable))
    k_stops = problem.num_stops
    n_subsets = 1 << k_stops  # 2^k

    # Extension cardinalities from twists
    ext_cardinalities: Dict[str, int] = {}
    total_ext_factor = 1

    for twist in problem.twists.twists:
        if not twist.enabled:
            continue
        if twist.name == "fragile_parcel":
            ext_cardinalities["fragile_delivered"] = 2
            total_ext_factor *= 2
        elif twist.name == "fuel_limit_with_charging_stations":
            cap = getattr(twist, "capacity", 50)
            ext_cardinalities["fuel"] = cap + 1
            total_ext_factor *= (cap + 1)
        elif twist.name == "toll_gates_budget":
            budget = getattr(twist, "budget", 20)
            ext_cardinalities["budget"] = budget + 1
            total_ext_factor *= (budget + 1)

    total_bound = n_passable * n_subsets * total_ext_factor

    return {
        "passable_cells": n_passable,
        "delivery_stops": k_stops,
        "delivery_subsets": n_subsets,
        "extension_cardinalities": ext_cardinalities,
        "total_extension_factor": total_ext_factor,
        "total_upper_bound": total_bound,
        "log10_upper_bound": round(math.log10(max(1, total_bound)), 2),
        "disclaimer": (
            "Loose combinatorial upper bound representing the Cartesian product of "
            "independent dimensions. True reachable state space is significantly smaller "
            "due to topological reachability and prerequisite transition constraints."
        ),
    }
