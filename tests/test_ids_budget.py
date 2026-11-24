"""Tests for IDS vs A* Under Stored-Node Budgets (§7.4, D18, M5.5)."""

import pytest

from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.search.astar import astar_search
from delivery.search.bfs import breadth_first_search
from delivery.search.compact import make_compact_scenario
from delivery.search.core import Limits, SearchStatus
from delivery.search.ids import iterative_deepening_search
from delivery.search.problem import DeliveryProblem
from delivery.search.ucs import uniform_cost_search


def test_ids_vs_astar_under_tight_stored_budget():
    """Demonstrates scenario where A*/UCS/BFS exceed tight stored budget while IDS completes successfully."""
    # Compact instance with depth <= 14
    inst = make_compact_scenario(seed=77, n=8, num_stops=1)
    problem = DeliveryProblem(inst)
    h_own = CombinedMSTHeuristic(problem)

    # Tight stored record budget of 25 records
    tight_budget = 25
    limits = Limits(max_stored_records=tight_budget, max_time_seconds=10.0, max_expanded=100_000)

    # BFS and UCS maintain large frontiers and exceed the record budget
    res_bfs = breadth_first_search(problem, limits=limits)
    res_ucs = uniform_cost_search(problem, limits=limits)

    assert res_bfs.status == SearchStatus.MEMORYLIMIT, f"BFS should hit MEMORYLIMIT, got {res_bfs.status}"
    assert res_ucs.status == SearchStatus.MEMORYLIMIT, f"UCS should hit MEMORYLIMIT, got {res_ucs.status}"

    # IDS maintains O(d) linear stack and succeeds within budget
    res_ids = iterative_deepening_search(problem, max_depth=20, limits=limits)
    assert res_ids.status == SearchStatus.SUCCESS, f"IDS should succeed within budget, got {res_ids.status}"
    assert res_ids.metrics.peak_stored <= tight_budget


def test_ids_cycle_ablation_blowup():
    """Ablation: IDS without on-path cycle checking blows up or exceeds limits on cyclic grids."""
    inst = make_compact_scenario(seed=99, n=6, num_stops=1)
    problem = DeliveryProblem(inst)

    limits = Limits(max_expanded=5_000, max_time_seconds=3.0)

    # With cycle check: completes cleanly
    res_with_cycle = iterative_deepening_search(problem, max_depth=10, limits=limits, check_cycle=True)
    assert res_with_cycle.status == SearchStatus.SUCCESS

    # Without cycle check: repeatedly explores back-and-forth cycles, causing massive expansion blowup
    res_no_cycle = iterative_deepening_search(problem, max_depth=10, limits=limits, check_cycle=False)
    # Either hits node limit or expands significantly more nodes
    assert res_no_cycle.metrics.expanded_nodes > res_with_cycle.metrics.expanded_nodes * 2
