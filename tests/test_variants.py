"""Regression tests and formal suboptimality demonstration for Inadmissible Variants (§7.4, M5.4).

Verifies:
1. The exact mandated hand-built fixture:
   - S -> A (cost 5), A -> G (cost 10) [Total: 15]
   - S -> B (cost 1), B -> G (cost 8)  [Total: 9 - Optimal]
   - Admissible estimates: h(A) = 0, h(B) = 8, h(G) = 0, h(S) = 0.
   - For w = 1.0 (Standard A* / UCS): discovers optimal path S -> B -> G (cost 9).
   - For w = 2.0 (Weighted A*): discovers suboptimal path S -> A -> G (cost 15),
     expanding strictly fewer non-goal nodes (only expanding S and A before reaching G).
"""

from typing import Any, List, Tuple

import pytest

from delivery.heuristics.variants import WeightedHeuristic
from delivery.search.astar import astar_search
from delivery.search.core import SearchResult, SearchStatus
from delivery.search.ucs import uniform_cost_search


class MandatedFixtureProblem:
    """Explicitly encodes the M5.4 mandated graph fixture:

    Nodes: 'S' (start), 'A', 'B', 'G' (goal)
    Edges:
      S -> A : cost 5
      A -> G : cost 10
      S -> B : cost 1
      B -> G : cost 8
    """

    def __init__(self):
        self.start = (0, 0, 0, ())  # maps to 'S'
        self.node_names = {
            (0, 0, 0, ()): "S",
            (1, 0, 0, ()): "A",
            (2, 0, 0, ()): "B",
            (3, 0, 0, ()): "G",
        }

    def initial_state(self):
        return (0, 0, 0, ())

    def is_goal(self, state: Tuple[int, int, int, Tuple[Any, ...]]) -> bool:
        return state == (3, 0, 0, ())

    def successors(self, state: Tuple[int, int, int, Tuple[Any, ...]]) -> List[Tuple[Tuple[int, int, int, Tuple[Any, ...]], int, float]]:
        name = self.node_names.get(state)
        # We assign cardinal actions 0: North, 1: East to differentiate
        if name == "S":
            # S -> A (action 0, cost 5), S -> B (action 1, cost 1)
            # To test tie-breaking independently, list B first or S->A first
            return [
                ((1, 0, 0, ()), 0, 5.0),  # S -> A
                ((2, 0, 0, ()), 1, 1.0),  # S -> B
            ]
        elif name == "A":
            return [((3, 0, 0, ()), 0, 10.0)]  # A -> G
        elif name == "B":
            return [((3, 0, 0, ()), 1, 8.0)]   # B -> G
        return []


class MandatedAdmissibleHeuristic:
    """Mandated admissible estimates: h(A)=0, h(B)=8, h(G)=0, h(S)=0."""

    name = "MandatedAdmissible"

    def __call__(self, state, prob) -> float:
        # State mapping
        name = prob.node_names.get(state)
        if name == "A":
            return 0.0
        elif name == "B":
            return 8.0
        elif name == "G":
            return 0.0
        elif name == "S":
            return 0.0
        return 0.0


def test_mandated_inadmissible_fixture_demonstrates_suboptimality_and_speedup():
    """Proves that weighted A* (w=2.0) expands fewer non-goal nodes and returns a strictly worse cost (15 > 9)."""
    prob = MandatedFixtureProblem()
    h_base = MandatedAdmissibleHeuristic()

    # 1. Standard A* with w=1.0 (Admissible)
    res_opt = astar_search(prob, h_base)
    res_ucs = uniform_cost_search(prob)

    assert res_opt.status == SearchStatus.SUCCESS
    assert res_opt.cost == 9.0, f"Optimal cost should be 9.0, got {res_opt.cost}"
    assert res_ucs.cost == 9.0

    # 2. Weighted A* with w=2.0 (Inadmissible)
    h_weighted = WeightedHeuristic(h_base, weight=2.0)
    res_subopt = astar_search(prob, h_weighted)

    assert res_subopt.status == SearchStatus.SUCCESS
    # Weighted A* returns the suboptimal path S -> A -> G with cost 15.0!
    # Why:
    # At S:
    # Child A: g=5, h=0 -> f = 5 + 2*0 = 5
    # Child B: g=1, h=8 -> f = 1 + 2*8 = 17
    # Frontier pops A (f=5 < 17).
    # Expanding A generates G: g = 5 + 10 = 15, h = 0 -> f = 15 + 2*0 = 15.
    # Frontier pops G (f=15 < 17), achieving goal with cost 15.0 without ever expanding B!
    assert res_subopt.cost == 15.0, (
        f"Weighted A* should find suboptimal cost 15.0, got {res_subopt.cost}"
    )
    assert res_subopt.cost > res_opt.cost, "Weighted A* cost must be strictly worse"

    # Expanded non-goal nodes count
    # Standard A* expands S, then B (g=1, f=9), then A (g=5, f=5? wait: at S, A has f=5, B has f=9.
    # A is popped first, expands G (f=15). Then B is popped (f=9 < 15), expands G with cost 9!
    # So Standard A* expanded BOTH A and B.
    # Weighted A* only expands S and A, skipping B entirely!
    assert res_subopt.metrics.expanded_nodes < res_opt.metrics.expanded_nodes, (
        f"Weighted A* expansions ({res_subopt.metrics.expanded_nodes}) must be < Standard A* ({res_opt.metrics.expanded_nodes})"
    )
