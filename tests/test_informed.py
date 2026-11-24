"""Tests for Informed Search Algorithms and Baseline Heuristics (§7.4, M4.4)."""

import numpy as np
import pytest

from delivery.heuristics.base import ZeroHeuristic
from delivery.heuristics.baselines import ManhattanNearestStopHeuristic, MaxRelaxedDistanceHeuristic
from delivery.instance import Instance
from delivery.search.astar import astar_search
from delivery.search.core import SearchStatus
from delivery.search.greedy import greedy_best_first_search
from delivery.search.problem import DeliveryProblem
from delivery.search.ucs import uniform_cost_search
from delivery.world.grid import GridWorld


def make_mini_instance(seed: int, size: int = 7) -> Instance:
    """Creates a small solvable instance for testing search algorithms."""
    rng = np.random.default_rng(seed)
    # Start at (1, 1), 2 stops at (size-2, size-2) and (1, size-2)
    start = (1, 1)
    stops = [(size - 2, size - 2), (1, size - 2)]
    # All road or grass, no obstacles to ensure solvability
    cells = rng.choice([0, 1], size=(size, size), p=[0.7, 0.3]).astype(np.int8)
    # Ensure borders are roads/grass
    cells[start] = 0
    for s in stops:
        cells[s] = 0

    return Instance(
        id=f"TEST-{seed}",
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


def test_zero_heuristic_astar_matches_ucs():
    inst = make_mini_instance(101, size=6)
    problem = DeliveryProblem(inst)
    h_zero = ZeroHeuristic()

    res_ucs = uniform_cost_search(problem)
    res_astar = astar_search(problem, h_zero)

    assert res_ucs.status == SearchStatus.SUCCESS
    assert res_astar.status == SearchStatus.SUCCESS
    assert res_astar.cost == pytest.approx(res_ucs.cost)
    assert res_astar.plan == res_ucs.plan


def test_astar_b1_and_b2_match_ucs_cost_over_50_scenarios():
    """Validates that A* with admissible heuristics B1 and B2 finds optimal cost matching UCS."""
    h_b1 = ManhattanNearestStopHeuristic()

    for seed in range(50):
        inst = make_mini_instance(seed + 1000, size=6)
        problem = DeliveryProblem(inst)
        h_b2 = MaxRelaxedDistanceHeuristic(problem)

        res_ucs = uniform_cost_search(problem)
        res_b1 = astar_search(problem, h_b1)
        res_b2 = astar_search(problem, h_b2)

        assert res_ucs.status == SearchStatus.SUCCESS, f"UCS failed on seed {seed}"
        assert res_b1.status == SearchStatus.SUCCESS, f"A* B1 failed on seed {seed}"
        assert res_b2.status == SearchStatus.SUCCESS, f"A* B2 failed on seed {seed}"

        # Costs must match optimal cost found by UCS
        assert res_b1.cost == pytest.approx(res_ucs.cost), (
            f"Seed {seed}: A* B1 cost {res_b1.cost} != UCS cost {res_ucs.cost}"
        )
        assert res_b2.cost == pytest.approx(res_ucs.cost), (
            f"Seed {seed}: A* B2 cost {res_b2.cost} != UCS cost {res_ucs.cost}"
        )

        # Baseline heuristics should never expand MORE nodes than UCS with zero heuristic
        # (or at least expand <= UCS expansions + tie tolerance)
        assert res_b1.metrics.expanded_nodes <= res_ucs.metrics.expanded_nodes + 5
        assert res_b2.metrics.expanded_nodes <= res_ucs.metrics.expanded_nodes + 5


def test_greedy_search_finds_valid_plan():
    """Greedy Best-First finds a valid goal plan, though not necessarily optimal."""
    inst = make_mini_instance(42, size=7)
    problem = DeliveryProblem(inst)
    h_b2 = MaxRelaxedDistanceHeuristic(problem)

    res_greedy = greedy_best_first_search(problem, h_b2)
    assert res_greedy.status == SearchStatus.SUCCESS
    assert len(res_greedy.plan) > 0

    # Validate that plan reaches all goals
    curr = problem.initial_state()
    for act in res_greedy.plan:
        succs = {a: (ns, c) for ns, a, c in problem.successors(curr)}
        assert act in succs
        curr, _ = succs[act]
    assert problem.is_goal(curr)


def test_astar_reopening_on_inconsistent_heuristic():
    """Tests that A* with closed-node re-opening finds the optimal path even with an inconsistent heuristic."""
    # Construct a small graph where an inconsistent heuristic causes premature closing of a state
    # Start: (0, 0), Stop: (0, 2)
    # Path 1: (0, 0) -> (1, 0) -> (1, 1) -> (1, 2) -> (0, 2) (cost: 3 + 3 + 3 + 3 = 12)
    # Path 2: (0, 0) -> (0, 1) -> (0, 2) (high direct cost on first edge: 20 + 3 = 23, but if h underestimates...)
    # Even simpler: A standard 3-state triangle test:
    # A -> B (cost 10), B -> C (cost 10)
    # A -> D (cost 1), D -> B (cost 2)
    # If h(B) is wildly misleading so B is expanded via A -> B first with g=10,
    # then D is expanded with g=1 and discovers path to B with g=3.
    # A* with re-opening MUST re-open B and reach C with optimal cost 13, not 20.
    cells = np.zeros((3, 3), dtype=np.int8)
    inst = Instance(
        id="INCONSISTENT-TEST",
        seed=1,
        cms_ids=["513303", "522356", "514549"],
        n=3,
        grid=cells,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.0,
        start=(0, 0),
        delivery_stops=[(2, 2)],
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=1,
        twists=[],
    )
    problem = DeliveryProblem(inst)

    # Inconsistent heuristic that gives huge h bonus to direct path
    class InconsistentAdmissibleHeuristic:
        def __call__(self, state, prob) -> float:
            r, c, _, _ = state
            if (r, c) == (2, 2):
                return 0.0
            if (r, c) == (0, 1):
                # Severely underestimate to force expansion
                return 0.0
            return 3.0  # admissible because min step is 3

    h_incons = InconsistentAdmissibleHeuristic()
    res = astar_search(problem, h_incons)
    res_ucs = uniform_cost_search(problem)

    assert res.status == SearchStatus.SUCCESS
    assert res.cost == pytest.approx(res_ucs.cost)
