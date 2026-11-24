"""Assigned Twist A: Fog / Partial Observability (§7.1, §7.2).

Governed by decisions D1, D3, D15, and the capability protocol architecture.
Agent only observes terrain and obstacles within Chebyshev radius rho = 4.
Unobserved cells use an optimistic belief cost c_min = 3 (road cost).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from delivery.twists.base import BaseTwist
from delivery.twists.protocols import LLMToolProvider, PerceptionModifier, Renderer


class FogTwist(BaseTwist):
    """Twist 4: fog_partial_observability."""

    def __init__(self, radius: int = 4, enabled: bool = True):
        super().__init__(
            name="fog_partial_observability",
            title="Fog of War / Partial Observability",
            enabled=enabled,
        )
        self.radius = radius
        self.n = 0
        self.discovered_mask: np.ndarray = np.empty((0, 0), dtype=bool)
        self.optimistic_cost_map: np.ndarray = np.empty((0, 0), dtype=np.float64)

    def initialize(self, instance: Any, world: Any) -> None:
        """Initializes fog state for the world instance."""
        self.n = world.n
        self.discovered_mask = np.zeros((self.n, self.n), dtype=bool)

        # Agent starts with initial perception at start position
        sr, sc = instance.start
        self._reveal_window(sr, sc, world)

        # Also know locations of delivery stops as targets, but not intermediate terrain
        # per standard navigation benchmarks (destination coordinates are known in advance)
        self.optimistic_cost_map = np.full((self.n, self.n), 3.0, dtype=np.float64)
        # Copy observed cells from ground truth
        self._sync_optimistic_costs(world)

    def _reveal_window(self, r: int, c: int, world: Any) -> int:
        """Reveals all cells within Chebyshev radius rho around (r, c)."""
        r_min = max(0, r - self.radius)
        r_max = min(self.n - 1, r + self.radius)
        c_min = max(0, c - self.radius)
        c_max = min(self.n - 1, c + self.radius)

        sub_mask = self.discovered_mask[r_min : r_max + 1, c_min : c_max + 1]
        new_count = int(np.sum(~sub_mask))
        self.discovered_mask[r_min : r_max + 1, c_min : c_max + 1] = True
        return new_count

    def _sync_optimistic_costs(self, world: Any) -> None:
        """Syncs known cells into the optimistic cost map."""
        for r in range(self.n):
            for c in range(self.n):
                if self.discovered_mask[r, c]:
                    self.optimistic_cost_map[r, c] = world.cost_map[r, c]

    def compute_percept(
        self,
        agent_pos: Tuple[int, int],
        world: Any,
        current_belief: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Updates agent's belief mask based on current position."""
        r, c = agent_pos
        new_cells = self._reveal_window(r, c, world)
        self._sync_optimistic_costs(world)
        has_new = new_cells > 0
        return self.discovered_mask.copy(), has_new

    def is_cell_discovered(self, r: int, c: int) -> bool:
        """Returns True if (r, c) has been revealed by fog exploration."""
        if 0 <= r < self.n and 0 <= c < self.n:
            return bool(self.discovered_mask[r, c])
        return False

    def render_overlay(self, ax: plt.Axes, world: Any, state: Optional[Any] = None) -> None:
        """Renders dark fog overlay on unobserved cells and vision contour."""
        # Unobserved cells get shaded overlay
        unobserved = ~self.discovered_mask
        if np.any(unobserved):
            fog_patch = np.zeros((self.n, self.n, 4), dtype=np.float32)
            # Semi-transparent dark slate fog
            fog_patch[unobserved] = [0.12, 0.16, 0.22, 0.65]
            ax.imshow(fog_patch, origin="upper", zorder=3)

        # If agent position is provided in state, draw sight perimeter
        agent_pos = None
        if state is not None:
            if isinstance(state, tuple) and len(state) >= 2:
                agent_pos = state[:2]
            elif hasattr(state, "pos"):
                agent_pos = state.pos

        if agent_pos is not None:
            ar, ac = agent_pos
            rect = patches.Rectangle(
                (ac - self.radius - 0.5, ar - self.radius - 0.5),
                2 * self.radius + 1,
                2 * self.radius + 1,
                linewidth=1.8,
                edgecolor="#ffc107",
                linestyle="--",
                facecolor="none",
                zorder=4,
                label=f"Fog Sight ($\rho={self.radius}$)",
            )
            ax.add_patch(rect)

    def get_tool_spec(self) -> Dict[str, Any]:
        """Returns LLM tool specification for scanning surroundings."""
        return {
            "name": "scan_surroundings",
            "description": (
                "Scans the environment within observation radius (rho=4) around a given grid position, "
                "revealing discovered terrain types and impassable obstacles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {"type": "integer", "description": "Row index of scan center"},
                    "c": {"type": "integer", "description": "Column index of scan center"},
                },
                "required": ["r", "c"],
            },
        }

    def execute_tool(self, args: Dict[str, Any], world: Any, state: Any) -> Dict[str, Any]:
        """Executes sensor scan and returns structured survey of surroundings."""
        r = int(args.get("r", 0))
        c = int(args.get("c", 0))

        if not world.in_bounds(r, c):
            return {"error": f"Coordinates ({r}, {c}) are out of grid bounds [0, {world.n - 1}]."}

        new_cells = self._reveal_window(r, c, world)
        self._sync_optimistic_costs(world)

        # Summarize visible window
        r_min = max(0, r - self.radius)
        r_max = min(self.n - 1, r + self.radius)
        c_min = max(0, c - self.radius)
        c_max = min(self.n - 1, c + self.radius)

        window_terrains = world.grid[r_min : r_max + 1, c_min : c_max + 1]
        obstacle_count = int(np.sum(window_terrains == -1))
        road_count = int(np.sum(window_terrains == 0))
        grass_count = int(np.sum(window_terrains == 1))
        mud_count = int(np.sum(window_terrains == 2))
        ford_count = int(np.sum(window_terrains == 3))

        total_discovered = int(np.sum(self.discovered_mask))
        pct_discovered = round(100.0 * total_discovered / (self.n * self.n), 1)

        return {
            "center": [r, c],
            "radius": self.radius,
            "window_bounds": {"r_min": r_min, "r_max": r_max, "c_min": c_min, "c_max": c_max},
            "newly_revealed_cells": new_cells,
            "total_explored_percentage": pct_discovered,
            "surrounding_summary": {
                "obstacles": obstacle_count,
                "road": road_count,
                "grass": grass_count,
                "mud": mud_count,
                "ford": ford_count,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "radius": self.radius,
            "explored_cells": int(np.sum(self.discovered_mask)) if self.discovered_mask.size else 0,
        })
        return data
