"""Inadmissible Heuristic Variants (§7.4, M5.4).

Provides:
- WeightedHeuristic: w * h(s) with w in {1.2, 1.5, 2.0, 3.0}
  Multiplies an underlying admissible heuristic by w > 1.0, trading off
  guaranteed cost optimality for potentially faster search (fewer expansions).
- StructurallyInadmissibleHeuristic:
  Overestimates remaining cost (e.g. Euclidean * max_terrain_cost or inflated MST)
  to aggressively bias exploration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Tuple

from delivery.heuristics.base import Heuristic

if TYPE_CHECKING:
    from delivery.search.problem import DeliveryProblem


class WeightedHeuristic(Heuristic):
    """Inadmissible weighted variant: h_w(s) = weight * base_heuristic(s).

    For weight > 1.0, admissibility is mathematically forfeited unless h*(s) = 0.
    In return, priority evaluation f(n) = g(n) + w * h(n) strongly favors nodes
    closer to the goal according to the heuristic estimate.
    """

    def __init__(self, base_heuristic: Heuristic, weight: float = 1.5):
        self.base_heuristic = base_heuristic
        self.weight = float(weight)
        self.name = f"Weighted_{self.weight:.1f}x_{getattr(base_heuristic, 'name', 'Heuristic')}"

    def __call__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        problem: DeliveryProblem,
    ) -> float:
        base_val = self.base_heuristic(state, problem)
        return self.weight * base_val


class OverestimatingHeuristic(Heuristic):
    """Structurally inadmissible variant that inflates distances by max possible step cost."""

    def __init__(self, base_heuristic: Heuristic, inflation_factor: float = 2.5):
        self.base_heuristic = base_heuristic
        self.inflation_factor = inflation_factor
        self.name = f"Inflated_{self.inflation_factor:.1f}x_{getattr(base_heuristic, 'name', 'Heuristic')}"

    def __call__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        problem: DeliveryProblem,
    ) -> float:
        return self.inflation_factor * self.base_heuristic(state, problem)
