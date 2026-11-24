"""Base agent protocol and interface definitions (§7.3, M3.2).

Governed by decisions D1, D6, and D11.
Defines the Sense-Reason-Act interface: choose_action(percept) -> (action, rationale).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Tuple


class Agent(abc.ABC):
    """Abstract base class for all delivery agents."""

    name: str = "BaseAgent"

    @abc.abstractmethod
    def choose_action(self, percept: Dict[str, Any]) -> Tuple[int, str]:
        """Examines the environment percept and returns (action_idx, concise_rationale).

        The concise rationale is recorded in execution traces without requiring
        opaque or private internal chain-of-thought tokens.
        """
        pass

    def reset(self) -> None:
        """Resets any internal agent state between episodes (stateless agents no-op)."""
        pass
