"""Tests for Combined MST and Farthest Reach Heuristic (§7.4, M5.1)."""

import numpy as np
import pytest

from delivery.heuristics.baselines import MaxRelaxedDistanceHeuristic
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.instance import Instance
from delivery.search.astar import astar_search
from delivery.search.core import SearchStatus
from delivery.search.problem import DeliveryProblem
from delivery.search.ucs import uniform_cost_search


def make_test_instance(seed: int = 42, size: int = 7) -> Instance:
    """Creates a small solvable instance for testing heuristic properties."""
    rng = np.random.default_rng(seed)
    start = (1, 1)
    stops = [(size - 2, size - 2), (1, size - 2), (size - 2, 1)]
    cells = rng.choice([0, 1], size=(size, size), p=[0.7, 0.3]).astype(np.int8)
    cells[start] = 0
    for s in stops:
        cells[s] = 0

    return Instance(
        id=f"HEUR-TEST-{seed}",
        seed=seed,
        cms_ids=["513303", "522356", "514549"],
        n=size,
        grid=cells,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.0,
        start=start,
        delivery_stops=stops,
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=1,
        twists=[],
    )


def test_combined_mst_heuristic_dominates_b2():
    """Validates that Combined MST heuristic h_own(s) >= h_B2(s) everywhere."""
    inst = make_test_instance(seed=77)
    problem = DeliveryProblem(inst)
    h_b2 = MaxRelaxedDistanceHeuristic(problem)
    h_own = CombinedMSTHeuristic(problem)

    init_state = problem.initial_state()
    val_b2 = h_b2(init_state, problem)
    val_own = h_own(init_state, problem)

    assert val_own >= val_b2, f"Dominance violated: h_own ({val_own}) < h_b2 ({val_b2})"

    # Check across reachable successor states
    for succ_state, _, _ in problem.successors(init_state):
        v_b2 = h_b2(succ_state, problem)
        v_own = h_own(succ_state, problem)
        assert v_own >= v_b2


def test_astar_with_combined_mst_finds_optimal_cost_matching_ucs():
    """A* with Combined MST finds the exact optimal cost matching UCS."""
    inst = make_test_instance(seed=123)
    problem = DeliveryProblem(inst)
    h_own = CombinedMSTHeuristic(problem)

    res_ucs = uniform_cost_search(problem)
    res_astar = astar_search(problem, h_own)

    assert res_ucs.status == SearchStatus.SUCCESS
    assert res_astar.status == SearchStatus.SUCCESS
    assert res_astar.cost == pytest.approx(res_ucs.cost)


def test_combined_mst_expands_fewer_or_equal_nodes_than_b2():
    """Combined MST should expand <= B2 nodes due to dominance."""
    inst = make_test_instance(seed=99)
    problem = DeliveryProblem(inst)
    h_b2 = MaxRelaxedDistanceHeuristic(problem)
    h_own = CombinedMSTHeuristic(problem)

    res_b2 = astar_search(problem, h_b2)
    res_own = astar_search(problem, h_own)

    assert res_b2.status == SearchStatus.SUCCESS
    assert res_own.status == SearchStatus.SUCCESS
    assert res_own.cost == pytest.approx(res_b2.cost)
    assert res_own.metrics.expanded_nodes <= res_b2.metrics.expanded_nodes
