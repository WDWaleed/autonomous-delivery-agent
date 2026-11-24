"""Base classes and TwistStack composition engine (§7.2).

Composes enabled twists in deterministic order and delegates hooks selectively
to twists implementing the corresponding capability protocols.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from delivery.twists.protocols import (
    CostModifier,
    LLMToolProvider,
    PerceptionModifier,
    RelaxationProvider,
    Renderer,
    StateExtension,
    TransitionModifier,
    Twist,
)


class BaseTwist(abc.ABC):
    """Abstract base class for all city twists."""

    def __init__(self, name: str, title: str, enabled: bool = True):
        self.name = name
        self.title = title
        self.enabled = enabled

    @abc.abstractmethod
    def initialize(self, instance: Any, world: Any) -> None:
        """Initializes twist parameters from instance seed and world geometry."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Default serialization of twist state."""
        return {
            "name": self.name,
            "title": self.title,
            "enabled": self.enabled,
        }


class RelaxedGraph:
    """Provides relaxed directed edges for admissible heuristic calculations."""

    def __init__(self, world: Any, stack: "TwistStack"):
        self.world = world
        self.stack = stack

    def get_edges(self, u: int, reverse: bool = False) -> List[Tuple[int, float]]:
        """Returns relaxed transitions from/to node u with lower-bound transition costs."""
        # Check if any enabled twist provides custom relaxed edges
        for twist in self.stack.twists:
            if twist.enabled and isinstance(twist, RelaxationProvider):
                return twist.get_relaxed_edges(u, self.world, reverse=reverse)

        # Default relaxed edges: base 4-connected grid transitions
        edges: List[Tuple[int, float]] = []
        for d in range(4):
            v = self.world.neighbor_table[u, d]
            if v != -1:
                if reverse:
                    # Reverse Dijkstra: incoming edge is w -> u with entry cost of u
                    cost = float(self.world.cost_map[self.world.rc(u)])
                else:
                    cost = float(self.world.move_cost_table[u, d])
                edges.append((v, cost))
        return edges


class TwistStack:
    """Manages and composes active twists in fixed registry order."""

    def __init__(self, twists: Sequence[Twist]):
        self.twists: List[Twist] = list(twists)
        self._validate_conflicts()
        self._cache_state_layout()

    def _validate_conflicts(self) -> None:
        """Validates that enabled twists do not violate compatibility rules."""
        enabled_names = {t.name for t in self.twists if t.enabled}
        # Example validation: checks are extensible
        if len(enabled_names) < len([t for t in self.twists if t.enabled]):
            raise ValueError("Duplicate twists detected in TwistStack!")

    def _cache_state_layout(self) -> None:
        """Determines the ordered layout of extra state variables across all twists."""
        self._extra_names: List[str] = []
        for twist in self.twists:
            if twist.enabled and isinstance(twist, StateExtension):
                self._extra_names.extend(twist.extra_state_names())
        self.extra_names: Tuple[str, ...] = tuple(self._extra_names)

    def initial_extra_state(self, world: Any) -> Tuple[Any, ...]:
        """Builds the combined initial extra state tuple."""
        extra: List[Any] = []
        for twist in self.twists:
            if twist.enabled and isinstance(twist, StateExtension):
                extra.extend(twist.initial_extra_state(world))
        return tuple(extra)

    def unpack_extra(self, extra_tuple: Tuple[Any, ...]) -> Dict[str, Any]:
        """Unpacks flat extra state tuple into named dictionary."""
        return dict(zip(self.extra_names, extra_tuple))

    def pack_extra(self, extra_dict: Dict[str, Any]) -> Tuple[Any, ...]:
        """Packs named extra dictionary back into ordered tuple."""
        return tuple(extra_dict[name] for name in self.extra_names)

    def is_action_legal(
        self,
        from_pos: Tuple[int, int],
        action_idx: int,
        to_pos: Tuple[int, int],
        extra_tuple: Tuple[Any, ...],
        world: Any,
    ) -> bool:
        """Checks if transition is permitted across all enabled transition modifiers."""
        extra_dict = self.unpack_extra(extra_tuple) if self.extra_names else {}
        for twist in self.twists:
            if twist.enabled and isinstance(twist, TransitionModifier):
                if not twist.is_action_legal(from_pos, action_idx, to_pos, extra_dict, world):
                    return False
        return True

    def compute_step_cost(
        self,
        from_pos: Tuple[int, int],
        action_idx: int,
        to_pos: Tuple[int, int],
        base_cost: float,
        extra_tuple: Tuple[Any, ...],
        world: Any,
    ) -> float:
        """Computes step cost after applying all active cost modifiers (must be > 0)."""
        extra_dict = self.unpack_extra(extra_tuple) if self.extra_names else {}
        cost = base_cost
        for twist in self.twists:
            if twist.enabled and isinstance(twist, CostModifier):
                cost = twist.modify_transition_cost(from_pos, action_idx, to_pos, cost, extra_dict, world)
        if cost <= 0:
            raise ValueError(f"Twist modified cost to non-positive value ({cost})! Violates D3/D15.")
        return cost

    def update_extra_state(
        self,
        extra_tuple: Tuple[Any, ...],
        action_idx: int,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        world: Any,
    ) -> Tuple[Any, ...]:
        """Computes updated extra state variables across all stateful twists."""
        if not self.extra_names:
            return ()
        extra_dict = self.unpack_extra(extra_tuple)
        for twist in self.twists:
            if twist.enabled and isinstance(twist, StateExtension):
                extra_dict = twist.update_extra_state(extra_dict, action_idx, from_pos, to_pos, world)
        return self.pack_extra(extra_dict)

    def relaxed_graph(self, world: Any) -> RelaxedGraph:
        """Returns the composed relaxed graph for admissible distance/heuristic computation."""
        return RelaxedGraph(world, self)

    def render_overlays(self, ax: plt.Axes, world: Any, state: Optional[Any] = None) -> None:
        """Renders visual overlays for all active twists implementing Renderer."""
        for twist in self.twists:
            if twist.enabled and isinstance(twist, Renderer):
                twist.render_overlay(ax, world, state=state)

    def get_llm_tool_specs(self) -> List[Dict[str, Any]]:
        """Collects LLM tool specifications across all active twists."""
        specs = []
        for twist in self.twists:
            if twist.enabled and isinstance(twist, LLMToolProvider):
                specs.append(twist.get_tool_spec())
        return specs
