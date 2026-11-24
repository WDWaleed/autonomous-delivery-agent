"""Exhaustive Admissibility and Empirical Consistency Checks (§7.4, M5.2).

1. Exhaustive Admissibility:
   Computes exact h*(s) by backward Dijkstra / reverse state-space search over
   compact scenarios, verifying h(s) <= h*(s) for all reachable states.
2. Empirical Consistency:
   Traverses states across 30 scenarios, testing monotonicity h(s) <= c(s, a, s') + h(s')
   on all visited transitions and reporting violations.
3. A* Reopening:
   Verifies that closed-node reopening preserves optimality under inconsistent conditions.
"""

from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

from delivery.experiments.scenarios import scenario
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.instance import Instance
from delivery.search.astar import astar_search
from delivery.search.core import SearchStatus
from delivery.search.problem import DeliveryProblem
from delivery.search.ucs import uniform_cost_search


def make_compact_instance(seed: int = 50, n: int = 5) -> Instance:
    """Creates a small, fully searchable grid instance."""
    rng = np.random.default_rng(seed)
    start = (0, 0)
    stops = [(n - 1, n - 1), (0, n - 1)]
    cells = rng.choice([0, 1], size=(n, n), p=[0.7, 0.3]).astype(np.int8)
    cells[start] = 0
    for s in stops:
        cells[s] = 0

    return Instance(
        id=f"COMPACT-{seed}",
        seed=seed,
        cms_ids=["513303", "522356", "514549"],
        n=n,
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


def compute_exact_h_star(problem: DeliveryProblem) -> Dict[Tuple, float]:
    """Computes exact h*(s) for all reachable states in a compact problem by forward UCS from each state."""
    # BFS to discover all reachable states
    init_state = problem.initial_state()
    reachable: Set[Tuple] = set()
    queue = deque([init_state])
    reachable.add(init_state)

    while queue:
        s = queue.popleft()
        for next_s, _, _ in problem.successors(s):
            if next_s not in reachable:
                reachable.add(next_s)
                queue.append(next_s)

    # For each state, run UCS to find true optimal cost to goal
    h_star: Dict[Tuple, float] = {}
    for s in reachable:
        if problem.is_goal(s):
            h_star[s] = 0.0
            continue

        # Temporarily create subproblem from state s
        class SubProblem:
            def __init__(self, prob, start_s):
                self.prob = prob
                self.start_s = start_s

            def initial_state(self):
                return self.start_s

            def is_goal(self, state):
                return self.prob.is_goal(state)

            def successors(self, state):
                return self.prob.successors(state)

        sub = SubProblem(problem, s)
        res = uniform_cost_search(sub)
        if res.status == SearchStatus.SUCCESS:
            h_star[s] = res.cost
        else:
            h_star[s] = float("inf")

    return h_star


def test_exhaustive_admissibility_on_compact_scenario():
    """Exhaustively asserts h(s) <= h*(s) across all reachable states in a compact scenario."""
    inst = make_compact_instance(seed=42, n=5)
    problem = DeliveryProblem(inst)
    h_own = CombinedMSTHeuristic(problem)

    h_star_map = compute_exact_h_star(problem)
    assert len(h_star_map) > 0, "No reachable states found"

    admissibility_violations = 0
    for state, h_star in h_star_map.items():
        h_val = h_own(state, problem)
        if h_star < float("inf"):
            assert h_val <= h_star + 1e-6, (
                f"Admissibility violated at state {state}: h={h_val:.2f} > h*={h_star:.2f}"
            )
            if h_val > h_star:
                admissibility_violations += 1

    assert admissibility_violations == 0


def test_empirical_consistency_across_30_scenarios():
    """Checks monotonicity h(s) <= c(s, a, s') + h(s') on visited transitions across 30 scenarios."""
    violations = 0
    total_edges_checked = 0
    max_violation = 0.0

    for k in range(30):
        sc = scenario(k)
        if not sc.feasible:
            continue
        problem = DeliveryProblem(sc.instance)
        h_own = CombinedMSTHeuristic(problem)

        # Explore a sample of states from the initial state
        curr = problem.initial_state()
        frontier = [curr]
        visited = {curr}

        for _ in range(50):
            if not frontier:
                break
            s = frontier.pop(0)
            h_s = h_own(s, problem)

            for next_s, action, step_cost in problem.successors(s):
                total_edges_checked += 1
                h_next = h_own(next_s, problem)

                diff = h_s - (step_cost + h_next)
                if diff > 1e-6:
                    violations += 1
                    if diff > max_violation:
                        max_violation = diff

                if next_s not in visited and len(visited) < 100:
                    visited.add(next_s)
                    frontier.append(next_s)

    # Print empirical report for diagnostics
    print(f"\n[Consistency Check] Checked {total_edges_checked} edges across 30 scenarios.")
    print(f"[Consistency Check] Violations: {violations}, Max violation: {max_violation:.4f}")

    # Monotonicity violations occur under asymmetric one-way constraints and delivery dropoffs
    # (e.g. delivering a stop causes a discrete step in the MST/max bound).
    # Task M5.2 requires reporting the violation count and max violation honestly,
    # and verifying A* closed-node reopening preserves optimality.
    assert violations > 0, "Expected non-zero empirical violations under one-way street asymmetry"
    assert max_violation <= 30.0


def test_astar_reopening_preserves_optimality_under_monotonicity_gap():
    """Verifies that A* with closed-node re-opening finds the exact optimal solution matching UCS."""
    for seed in range(5):
        inst = make_compact_instance(seed=seed + 200, n=5)
        problem = DeliveryProblem(inst)
        h_own = CombinedMSTHeuristic(problem)

        res_ucs = uniform_cost_search(problem)
        res_astar = astar_search(problem, h_own)

        assert res_ucs.status == SearchStatus.SUCCESS
        assert res_astar.status == SearchStatus.SUCCESS
        assert res_astar.cost == pytest.approx(res_ucs.cost)
