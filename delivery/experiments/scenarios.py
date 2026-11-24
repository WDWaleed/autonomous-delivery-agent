"""Scenario generator and feasibility certification (§6.1, D8, M2.6).

Generates independently seeded scenarios for experimental suites using deterministic
scenario tokens with oel_instance.make_instance(), enforcing base reachability certificates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import oel_instance
from delivery.certificate import FeasibilityStatus, certify_feasibility
from delivery.instance import Instance, derive_seed_from_ids
from delivery.overrides import apply_overrides
from delivery.stamp import load_members


@dataclass
class Scenario:
    """Certified experimental scenario specification."""

    k: int
    key: str
    instance: Instance
    is_direct: bool
    generation_type: str
    retry_count: int
    feasible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "key": self.key,
            "is_direct": self.is_direct,
            "generation_type": self.generation_type,
            "retry_count": self.retry_count,
            "feasible": self.feasible,
            "instance": self.instance.to_dict(),
        }


def check_base_reachability(grid: np.ndarray, start: Tuple[int, int], stops: List[Tuple[int, int]]) -> bool:
    """Verifies that all delivery stops are reachable from the start depot via passable cells."""
    n = grid.shape[0]
    sr, sc = start

    if not (0 <= sr < n and 0 <= sc < n) or grid[sr, sc] < 0:
        return False

    visited = np.zeros((n, n), dtype=bool)
    visited[sr, sc] = True
    queue = deque([(sr, sc)])

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not visited[nr, nc] and grid[nr, nc] >= 0:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # All stops must be visited
    for dr, dc in stops:
        if not visited[dr, dc]:
            return False

    return True


def scenario(
    k: int,
    base_ids: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> Scenario:
    """Generates scenario k deterministically following Decision D8.

    Attempts generation with primary token 'SCN-{k:03d}'. If base reachability fails,
    retries up to max_retries with 'SCN-{k:03d}-r{retry}'.
    """
    if base_ids is None:
        members = load_members()
        ids = [cms for _, cms in members]
    else:
        ids = list(base_ids)

    retry_count = 0
    token = f"SCN-{k:03d}"
    last_inst: Optional[Instance] = None

    while retry_count <= max_retries:
        scenario_token = token if retry_count == 0 else f"{token}-r{retry_count}"
        combined_ids = ids + [scenario_token]

        raw_inst, grid = oel_instance.make_instance(combined_ids)
        grid_int8 = grid.astype(np.int8)
        _, seed = derive_seed_from_ids(combined_ids)

        inst = Instance(
            id=f"{raw_inst['instance_id']}-{scenario_token}",
            n=raw_inst["grid_size"],
            grid=grid_int8,
            terrain_cost=dict(raw_inst["terrain_cost"]),
            obstacle_density=float(raw_inst["obstacle_density"]),
            start=tuple(map(int, raw_inst["start"])),
            delivery_stops=[tuple(map(int, p)) for p in raw_inst["delivery_stops"]],
            adversary=str(raw_inst["adversary"]),
            slip_probability=float(raw_inst["slip_probability"]),
            adversary_search_depth=int(raw_inst["adversary_search_depth"]),
            twists=list(raw_inst["twists"]),
            seed=seed,
            cms_ids=sorted(r.strip().upper() for r in combined_ids),
        )

        if overrides:
            inst = apply_overrides(inst, overrides)

        is_feasible = check_base_reachability(inst.grid, inst.start, inst.delivery_stops)

        # Full planner feasibility certification with active twists
        if is_feasible and inst.twists:
            cert = certify_feasibility(inst)
            if cert.status == FeasibilityStatus.INFEASIBLE:
                is_feasible = False

        if is_feasible:
            return Scenario(
                k=k,
                key=scenario_token,
                instance=inst,
                is_direct=True,
                generation_type="direct",
                retry_count=retry_count,
                feasible=True,
            )

        last_inst = inst
        retry_count += 1

    # If retries exhausted, report shortfall truthfully per D8
    return Scenario(
        k=k,
        key=token,
        instance=last_inst,  # type: ignore
        is_direct=True,
        generation_type="direct",
        retry_count=retry_count,
        feasible=False,
    )


def generate_scenario_suite(
    n_scenarios: int = 30,
    base_ids: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Scenario]:
    """Generates an evaluation suite of N independently seeded scenarios (default 30)."""
    suite: List[Scenario] = []
    for k in range(n_scenarios):
        sc = scenario(k, base_ids=base_ids, overrides=overrides)
        suite.append(sc)
    return suite
