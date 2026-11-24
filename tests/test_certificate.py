"""Tests for Feasibility Certificate Engine (§7.4, D8, M4.5)."""

import numpy as np
import pytest

from delivery.certificate import FeasibilityCertificate, FeasibilityStatus, certify_feasibility
from delivery.instance import Instance
from delivery.search.core import Limits


def make_test_instance(seed: int = 42, solvable: bool = True) -> Instance:
    """Creates a deterministic test instance with controlled solvability."""
    grid = np.zeros((5, 5), dtype=np.int8)
    if not solvable:
        # Enclose the stop (4, 4) with impassable obstacles (-1)
        grid[3, 4] = -1
        grid[4, 3] = -1

    return Instance(
        id=f"CERT-TEST-{seed}-{solvable}",
        seed=seed,
        cms_ids=["513303", "522356", "514549"],
        n=5,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.0,
        start=(0, 0),
        delivery_stops=[(4, 4)],
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=1,
        twists=[],
    )


def test_certify_feasible_instance():
    inst = make_test_instance(solvable=True)
    cert = certify_feasibility(inst)

    assert cert.status == FeasibilityStatus.FEASIBLE
    assert cert.cost is not None
    assert cert.cost > 0
    assert cert.plan is not None
    assert len(cert.plan) == 8  # Manhattan distance 4+4=8 steps of cost 3 = 24
    assert cert.cost == pytest.approx(24.0)


def test_certify_infeasible_instance():
    inst = make_test_instance(solvable=False)
    cert = certify_feasibility(inst)

    assert cert.status == FeasibilityStatus.INFEASIBLE
    assert cert.plan is None
    assert cert.cost is None
    assert "exhausted" in cert.reason.lower()


def test_certify_unknown_on_resource_limit():
    inst = make_test_instance(solvable=True)
    # Severe limit of 1 node expansion
    limits = Limits(max_expanded=1)
    cert = certify_feasibility(inst, limits=limits)

    assert cert.status == FeasibilityStatus.UNKNOWN
    assert cert.plan is None
    assert "interrupted" in cert.reason.lower()
