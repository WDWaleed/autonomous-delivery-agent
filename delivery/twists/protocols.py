"""Capability protocols for the twist plugin framework (§7.2).

Twists implement only the capabilities they require, avoiding dummy hooks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np


@runtime_checkable
class Twist(Protocol):
    """Core identity protocol implemented by every twist."""

    name: str
    title: str
    enabled: bool

    def initialize(self, instance: Any, world: Any) -> None:
        """Initializes twist parameters and deterministic data from instance seed."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serializes twist parameters to dictionary for saving in twists.json."""
        ...


@runtime_checkable
class StateExtension(Protocol):
    """Capability for twists that extend the search/simulation state tuple."""

    def extra_state_names(self) -> Tuple[str, ...]:
        """Returns names of state components added by this twist in fixed order."""
        ...

    def initial_extra_state(self, world: Any) -> Tuple[Any, ...]:
        """Returns initial values for this twist's state variables."""
        ...

    def update_extra_state(
        self,
        current_extra: Dict[str, Any],
        action: int,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        world: Any,
    ) -> Dict[str, Any]:
        """Computes updated values for this twist's state variables."""
        ...


@runtime_checkable
class TransitionModifier(Protocol):
    """Capability for filtering legal actions or modifying successor positions."""

    def is_action_legal(
        self,
        from_pos: Tuple[int, int],
        action_idx: int,
        to_pos: Tuple[int, int],
        state_extra: Dict[str, Any],
        world: Any,
    ) -> bool:
        """Checks if a transition from from_pos to to_pos via action_idx is allowed."""
        ...


@runtime_checkable
class CostModifier(Protocol):
    """Capability for altering deterministic transition costs."""

    def modify_transition_cost(
        self,
        from_pos: Tuple[int, int],
        action_idx: int,
        to_pos: Tuple[int, int],
        base_cost: float,
        state_extra: Dict[str, Any],
        world: Any,
    ) -> float:
        """Returns modified step cost (must remain strictly positive > 0 per D3/D15)."""
        ...


@runtime_checkable
class PerceptionModifier(Protocol):
    """Capability for partially observable environments (e.g. fog)."""

    def compute_percept(
        self,
        agent_pos: Tuple[int, int],
        world: Any,
        current_belief: Any,
    ) -> Tuple[Any, bool]:
        """Updates agent's belief map based on current position.

        Returns (updated_belief, has_new_discovery).
        """
        ...


@runtime_checkable
class RelaxationProvider(Protocol):
    """Capability for supplying relaxed transitions/costs for admissible heuristics."""

    def get_relaxed_edges(
        self,
        node_flat: int,
        world: Any,
        reverse: bool = False,
    ) -> List[Tuple[int, float]]:
        """Returns relaxed outgoing/incoming edges (dest_flat, cost) for admissible lower bounds."""
        ...


@runtime_checkable
class Renderer(Protocol):
    """Capability for rendering visual overlays on the grid plot."""

    def render_overlay(self, ax: plt.Axes, world: Any, state: Optional[Any] = None) -> None:
        """Draws twist-specific visual indicators on the matplotlib axes."""
        ...


@runtime_checkable
class LLMToolProvider(Protocol):
    """Capability for providing an LLM tool tied to this twist."""

    def get_tool_spec(self) -> Dict[str, Any]:
        """Returns JSON schema definition of the LLM tool."""
        ...

    def execute_tool(self, args: Dict[str, Any], world: Any, state: Any) -> Dict[str, Any]:
        """Executes the tool deterministically and returns structured result."""
        ...
