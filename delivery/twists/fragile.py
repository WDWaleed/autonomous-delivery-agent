"""Approved Third Twist: Fragile Parcel (§7.1, §7.2, config/third_twist.yaml).

Governed by decisions D1, D3, D15, and the capability protocol architecture.
While the fragile parcel is undelivered, mud and ford cells are illegal to traverse.
Once delivered to the designated fragile stop, standard terrain rules apply.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from delivery.twists.base import BaseTwist
from delivery.twists.protocols import LLMToolProvider, Renderer, StateExtension, TransitionModifier


class FragileParcelTwist(BaseTwist):
    """Third Twist: fragile_parcel."""

    def __init__(self, fragile_stop_idx: int = 0, enabled: bool = True):
        super().__init__(
            name="fragile_parcel",
            title="Fragile Parcel",
            enabled=enabled,
        )
        self.fragile_stop_idx = fragile_stop_idx
        self.fragile_stop_pos: Tuple[int, int] = (0, 0)

    def initialize(self, instance: Any, world: Any) -> None:
        """Designates the fragile delivery stop from instance delivery stops."""
        stops = instance.delivery_stops
        idx = min(self.fragile_stop_idx, len(stops) - 1)
        self.fragile_stop_pos = stops[idx]

    def extra_state_names(self) -> Tuple[str, ...]:
        return ("fragile_delivered",)

    def initial_extra_state(self, world: Any) -> Tuple[Any, ...]:
        return (False,)

    def update_extra_state(
        self,
        current_extra: Dict[str, Any],
        action: int,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        world: Any,
    ) -> Dict[str, Any]:
        delivered = bool(current_extra.get("fragile_delivered", False))
        if not delivered and to_pos == self.fragile_stop_pos:
            delivered = True
        updated = dict(current_extra)
        updated["fragile_delivered"] = delivered
        return updated

    def is_action_legal(
        self,
        from_pos: Tuple[int, int],
        action_idx: int,
        to_pos: Tuple[int, int],
        state_extra: Dict[str, Any],
        world: Any,
    ) -> bool:
        """Restricts mud (2) and ford (3) traversal while fragile parcel is undelivered."""
        is_delivered = bool(state_extra.get("fragile_delivered", False))
        if not is_delivered:
            tr, tc = to_pos
            if world.in_bounds(tr, tc):
                terrain_type = world.grid[tr, tc]
                # Mud (2) or Ford (3)
                if terrain_type in (2, 3):
                    return False
        return True

    def render_overlay(self, ax: plt.Axes, world: Any, state: Optional[Any] = None) -> None:
        """Renders fragile parcel indicator on its destination stop."""
        fr, fc = self.fragile_stop_pos
        ax.plot(
            fc,
            fr,
            marker="*",
            markersize=14,
            color="#fbbf24",
            markeredgecolor="#78350f",
            markeredgewidth=1.5,
            zorder=6,
            label="Fragile Parcel Stop",
        )

    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "name": "check_fragile_parcel_status",
            "description": "Checks whether the fragile parcel has been delivered and identifies restricted terrain types.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    def execute_tool(self, args: Dict[str, Any], world: Any, state: Any) -> Dict[str, Any]:
        delivered = False
        if state is not None and hasattr(state, "extra"):
            delivered = bool(state.extra.get("fragile_delivered", False))
        return {
            "fragile_stop": list(self.fragile_stop_pos),
            "fragile_delivered": delivered,
            "restricted_terrains": ["mud", "ford"] if not delivered else [],
            "status": "Safe: Normal navigation permitted" if delivered else "Restricted: Avoid Mud and Ford!",
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["fragile_stop"] = list(self.fragile_stop_pos)
        return data
