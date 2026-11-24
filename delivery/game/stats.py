"""Shared adversarial search data structures and placeholder evaluation (§7.5, M6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from delivery.game.window import AGENT, UTILITY_M, GameState, WindowGame


@dataclass
class SearchStats:
    """Accounting metrics for adversarial game-tree search."""

    nodes_expanded: int = 0
    evaluations: int = 0
    cutoffs: int = 0
    depth_reached: int = 0


@dataclass
class SearchDecision:
    """Outcome of a depth-limited adversarial search."""

    best_action: Optional[int]
    value: float
    principal_variation: List[int] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)


EvalFn = Callable[[GameState, WindowGame], float]
OrderFn = Callable[[GameState, List[int], WindowGame, int, Optional[int]], List[int]]


def placeholder_eval(state: GameState, game: WindowGame) -> float:
    """Bounded placeholder evaluation using route progress and rival distance.

    Strictly bounded within [-1000.0, +1000.0] << UTILITY_M (10,000.0).
    """
    # 1. Delivered stops progress
    num_stops = max(1, len(game.stops))
    delivered_count = bin(state.delivered_mask & game.target_mask).count("1")
    progress_score = 400.0 * (delivered_count / num_stops)

    # 2. Manhattan distance to nearest undelivered stop
    ar, ac = state.agent_pos
    undelivered = [
        stop for idx, stop in enumerate(game.stops)
        if not (state.delivered_mask & (1 << idx))
    ]
    if undelivered:
        min_dist = min(abs(ar - sr) + abs(ac - sc) for sr, sc in undelivered)
        max_grid_dist = max(1, 2 * game.world.n)
        dist_score = -200.0 * min(1.0, min_dist / max_grid_dist)
    else:
        dist_score = 200.0

    # 3. Saturated rival Manhattan distance (farther rival is safer, saturated at 8)
    rr, rc = state.rival_pos
    rival_dist = abs(ar - rr) + abs(ac - rc)
    rival_score = 200.0 * (min(rival_dist, 8) / 8.0) - 100.0

    val = progress_score + dist_score + rival_score
    return float(max(-999.0, min(999.0, val)))
