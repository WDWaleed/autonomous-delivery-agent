"""Unit tests for Uninformed Search Algorithms: BFS, UCS, DLS, and IDS (§7.4, M4.3)."""

import numpy as np
import pytest

from delivery.instance import Instance
from delivery.search.bfs import breadth_first_search
from delivery.search.core import Limits, SearchStatus, canonical_optimal_plan
from delivery.search.dls import depth_limited_search
from delivery.search.ids import iterative_deepening_search
from delivery.search.problem import EAST, NORTH, SOUTH, WAIT, WEST, DeliveryProblem
from delivery.search.ucs import uniform_cost_search


def make_fixture_3x3() -> Instance:
    """3x3 road-only grid, start (0, 0), stops (0, 2) and (2, 2)."""
    grid = np.zeros((3, 3), dtype=np.int8)
    return Instance(
        id="fixture_3x3",
        n=3,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.0,
        start=(0, 0),
        delivery_stops=[(0, 2), (2, 2)],
        adversary="chaser",
        slip_probability=0.0,
        adversary_search_depth=1,
        twists=[],
        seed=1,
        cms_ids=["TEST01"],
    )


def make_cost_vs_depth_fixture() -> Instance:
    """3x3 grid where direct 2-step move is expensive (ford cost 10), detour of 4 steps is cheap (road cost 1)."""
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 1] = 3  # ford (cost 10)
    # Target is (0, 2) [road, cost 1]

    return Instance(
        id="cost_vs_depth",
        n=3,
        grid=grid,
        terrain_cost={"road": 1, "grass": 4, "mud": 5, "ford": 10},
        obstacle_density=0.0,
        start=(0, 0),
        delivery_stops=[(0, 2)],
        adversary="chaser",
        slip_probability=0.0,
        adversary_search_depth=1,
        twists=[],
        seed=2,
        cms_ids=["TEST02"],
    )


def test_bfs_action_depth_optimality():
    inst = make_cost_vs_depth_fixture()
    prob = DeliveryProblem(inst)

    res_bfs = breadth_first_search(prob)
    assert res_bfs.status == SearchStatus.SUCCESS
    # BFS prioritizes fewest actions: length 2 (direct through ford)
    assert res_bfs.length == 2
    assert res_bfs.cost == 11.0  # 10.0 (ford) + 1.0 (road)
    assert res_bfs.plan == [EAST, EAST]


def test_ucs_cost_optimality():
    inst = make_cost_vs_depth_fixture()
    prob = DeliveryProblem(inst)

    res_ucs = uniform_cost_search(prob)
    assert res_ucs.status == SearchStatus.SUCCESS
    # UCS prioritizes lowest accumulated cost: cost 4.0 (length 4 detour through road)
    assert res_ucs.cost == 4.0
    assert res_ucs.length == 4
    assert res_ucs.cost < breadth_first_search(prob).cost

    # UCS matches canonical_optimal_plan cost
    res_canon = canonical_optimal_plan(prob)
    assert res_ucs.cost == pytest.approx(res_canon.cost)


def test_dls_cutoff_and_success():
    inst = make_fixture_3x3()
    prob = DeliveryProblem(inst)
    # Minimum depth for 3x3 fixture is 4 steps

    # Depth limit 2: goal cannot be reached -> CUTOFF
    res_dls_cutoff = depth_limited_search(prob, depth_limit=2)
    assert res_dls_cutoff.status == SearchStatus.CUTOFF

    # Depth limit 4: goal is reached -> SUCCESS
    res_dls_success = depth_limited_search(prob, depth_limit=4)
    assert res_dls_success.status == SearchStatus.SUCCESS
    assert res_dls_success.length == 4


def test_dls_failure_on_unreachable_goal():
    # Grid where goal is surrounded by obstacles
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 1] = -1
    grid[1, 0] = -1
    grid[1, 2] = -1
    grid[2, 1] = -1
    inst = Instance(
        id="unreachable",
        n=3,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.4,
        start=(0, 0),
        delivery_stops=[(2, 2)],  # completely isolated
        adversary="chaser",
        slip_probability=0.0,
        adversary_search_depth=1,
        twists=[],
        seed=3,
        cms_ids=["TEST03"],
    )
    prob = DeliveryProblem(inst)
    res_dls = depth_limited_search(prob, depth_limit=10)
    assert res_dls.status == SearchStatus.FAILED


def test_ids_bfs_depth_parity_and_cycle_avoidance():
    inst = make_cost_vs_depth_fixture()
    prob = DeliveryProblem(inst)

    res_ids = iterative_deepening_search(prob, max_depth=10)
    res_bfs = breadth_first_search(prob)

    assert res_ids.status == SearchStatus.SUCCESS
    # IDS must have BFS-minimum depth optimality
    assert res_ids.length == res_bfs.length
    assert res_ids.length == 2
    assert res_ids.metrics.expanded_nodes > 0
