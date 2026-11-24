"""Assigned Twist B: One-Way Streets (§7.1, §7.2).

Governed by decisions D1, D2, D3, D15, and the capability protocol architecture.
Deterministically directs ~25% of traversable grid edges while preserving
mutual reachability among start depot and all delivery stops.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from delivery.twists.base import BaseTwist
from delivery.twists.protocols import LLMToolProvider, RelaxationProvider, Renderer, TransitionModifier

OPPOSITE = [2, 3, 0, 1]  # N<->S, E<->W
DIR_NAMES = ["N", "E", "S", "W"]
DIR_MAP = {"N": 0, "E": 1, "S": 2, "W": 3, "NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}


class OneWayTwist(BaseTwist):
    """Twist 2: one_way_streets."""

    def __init__(self, directed_ratio: float = 0.25, enabled: bool = True):
        super().__init__(
            name="one_way_streets",
            title="One-Way Streets",
            enabled=enabled,
        )
        self.directed_ratio = directed_ratio
        self.n = 0
        self.n2 = 0
        self.allowed: np.ndarray = np.empty((0, 0), dtype=bool)  # shape (n2, 4)
        self.directed_edges: List[Tuple[int, int, int]] = []  # (u, v, action_idx u->v)

    def initialize(self, instance: Any, world: Any) -> None:
        """Deterministically generates directed edges while guaranteeing reachability."""
        self.n = world.n
        self.n2 = world.n2
        self.allowed = np.zeros((self.n2, 4), dtype=bool)

        # Initially allow all passable transitions from world.neighbor_table
        for u in range(self.n2):
            for d in range(4):
                v = world.neighbor_table[u, d]
                if v != -1:
                    self.allowed[u, d] = True

        # Collect unique undirected edges (u < v)
        undirected_edges: List[Tuple[int, int, int]] = []
        for u in range(self.n2):
            for d in range(4):
                v = world.neighbor_table[u, d]
                if v != -1 and u < v:
                    undirected_edges.append((u, v, d))

        # Target nodes that must remain strongly connected
        targets = [world.idx(*instance.start)] + [world.idx(*p) for p in instance.delivery_stops]

        # Deterministic RNG derived from instance seed
        seed = int(instance.seed) ^ 0x01E_04A
        rng = np.random.default_rng(seed)

        perm = rng.permutation(len(undirected_edges))
        target_directed_count = int(len(undirected_edges) * self.directed_ratio)
        directed_count = 0
        self.directed_edges = []

        for idx in perm:
            if directed_count >= target_directed_count:
                break

            u, v, d_u = undirected_edges[idx]
            d_v = OPPOSITE[d_u]

            # Choose direction randomly: True means u -> v only; False means v -> u only
            direction_forward = bool(rng.choice([True, False]))

            if direction_forward:
                # Disallow v -> u
                self.allowed[v, d_v] = False
                if self._verify_reachability(targets, world):
                    self.directed_edges.append((u, v, d_u))
                    directed_count += 1
                else:
                    # Revert
                    self.allowed[v, d_v] = True
            else:
                # Disallow u -> v
                self.allowed[u, d_u] = False
                if self._verify_reachability(targets, world):
                    self.directed_edges.append((v, u, d_v))
                    directed_count += 1
                else:
                    # Revert
                    self.allowed[u, d_u] = True

    def _verify_reachability(self, targets: List[int], world: Any) -> bool:
        """Verifies all target nodes can reach each other in directed graph."""
        for src in targets:
            visited = set()
            queue = deque([src])
            visited.add(src)

            while queue:
                curr = queue.popleft()
                for d in range(4):
                    if self.allowed[curr, d]:
                        nxt = world.neighbor_table[curr, d]
                        if nxt != -1 and nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)

            # Check if all other targets were reached
            for tgt in targets:
                if tgt not in visited:
                    return False
        return True

    def is_action_legal(
        self,
        from_pos: Tuple[int, int],
        action_idx: int,
        to_pos: Tuple[int, int],
        state_extra: Dict[str, Any],
        world: Any,
    ) -> bool:
        """Checks if transition conforms to one-way street constraints."""
        u = world.idx(*from_pos)
        if 0 <= action_idx < 4:
            return bool(self.allowed[u, action_idx])
        return True

    def get_relaxed_edges(
        self,
        node_flat: int,
        world: Any,
        reverse: bool = False,
    ) -> List[Tuple[int, float]]:
        """Returns relaxed transitions where one-way restrictions are relaxed."""
        edges: List[Tuple[int, float]] = []
        for d in range(4):
            v = world.neighbor_table[node_flat, d]
            if v != -1:
                cost = float(world.move_cost_table[node_flat, d])
                edges.append((v, cost))
        return edges

    def render_overlay(self, ax: plt.Axes, world: Any, state: Optional[Any] = None) -> None:
        """Renders directional indicator arrows on all one-way street segments."""
        for u, v, d in self.directed_edges:
            r1, c1 = world.rc(u)
            r2, c2 = world.rc(v)

            # Direction vector
            dr = r2 - r1
            dc = c2 - c1

            # Arrow midpoints
            mid_r = (r1 + r2) / 2.0
            mid_c = (c1 + c2) / 2.0

            ax.annotate(
                "",
                xy=(mid_c + 0.25 * dc, mid_r + 0.25 * dr),
                xytext=(mid_c - 0.25 * dc, mid_r - 0.25 * dr),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.35,head_length=0.45",
                    color="#f59e0b",
                    lw=2.2,
                ),
                zorder=5,
            )

    def get_tool_spec(self) -> Dict[str, Any]:
        """Returns LLM tool specification for checking street direction."""
        return {
            "name": "check_street_direction",
            "description": (
                "Checks whether movement from a given cell in a cardinal direction (N, E, S, W) "
                "is permitted under one-way street traffic regulations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_r": {"type": "integer", "description": "Origin row"},
                    "from_c": {"type": "integer", "description": "Origin column"},
                    "direction": {"type": "string", "description": "Cardinal action (N, E, S, W)"},
                },
                "required": ["from_r", "from_c", "direction"],
            },
        }

    def execute_tool(self, args: Dict[str, Any], world: Any, state: Any) -> Dict[str, Any]:
        """Executes one-way direction inquiry."""
        fr = int(args.get("from_r", 0))
        fc = int(args.get("from_c", 0))
        d_str = str(args.get("direction", "")).strip().upper()

        if d_str not in DIR_MAP:
            return {"error": f"Invalid direction '{d_str}'. Must be one of N, E, S, W."}

        action_idx = DIR_MAP[d_str]

        if not world.in_bounds(fr, fc):
            return {"error": f"Coordinates ({fr}, {fc}) are out of grid bounds."}

        u = world.idx(fr, fc)
        v = world.neighbor_table[u, action_idx]

        if v == -1:
            return {
                "from": [fr, fc],
                "direction": d_str,
                "status": "blocked",
                "allowed": False,
                "reason": "Transition leads into obstacle or out of bounds.",
            }

        to_r, to_c = world.rc(v)
        is_legal = bool(self.allowed[u, action_idx])
        opp_legal = bool(self.allowed[v, OPPOSITE[action_idx]])

        if is_legal and opp_legal:
            reg = "two_way"
        elif is_legal and not opp_legal:
            reg = "one_way_forward"
        elif not is_legal and opp_legal:
            reg = "one_way_wrong_way"
        else:
            reg = "closed"

        return {
            "from": [fr, fc],
            "to": [to_r, to_c],
            "direction": d_str,
            "status": reg,
            "allowed": is_legal,
            "reason": (
                "Passable" if is_legal else "Illegal move against one-way street regulation!"
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "directed_ratio": self.directed_ratio,
            "total_directed_edges": len(self.directed_edges),
        })
        return data
