"""Heuristic protocol and zero-heuristic baseline (§7.4, M4.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from delivery.search.problem import DeliveryProblem


@runtime_checkable
class Heuristic(Protocol):
    """Protocol for state heuristic evaluators."""

    name: str

    def __call__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        problem: DeliveryProblem,
    ) -> float:
        """Returns non-negative estimated remaining cost to achieve goal."""
        ...


class ZeroHeuristic:
    """Trivial admissible and consistent heuristic returning h(s) = 0.0."""

    name: str = "ZeroHeuristic"

    def __call__(
        self,
        state: Tuple[int, int, int, Tuple[Any, ...]],
        problem: DeliveryProblem,
    ) -> float:
        return 0.0
