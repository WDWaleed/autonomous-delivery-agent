"""Assigned Rival Adversary Strategy (§7.5, M7.1).

Implements the mandatory assigned 'chaser' rival strategy using Alpha-Beta search
to adversary_search_depth (default 6 plies) with chaser-specific evaluation
(minimizing relaxed/Manhattan distance to the agent and maximizing capture pressure).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from delivery.game.alphabeta import alphabeta_search
from delivery.game.ordering import make_own_ordering
from delivery.game.stats import EvalFn, OrderFn, SearchDecision
from delivery.game.window import RIVAL, UTILITY_M, GameState, WindowGame
from delivery.search.problem import ACTION_DELTAS, EAST, NORTH, SOUTH, WEST

# Lightweight cache for reverse BFS step-distance fields: (id(game.world), agent_pos) -> np.ndarray
_BFS_CACHE: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}


def _get_rival_dist_field(game: WindowGame, agent_pos: Tuple[int, int]) -> np.ndarray:
    """Computes exact minimum legal rival steps from every cell TO agent_pos via reverse BFS."""
    key = (id(game.world), agent_pos)
    if key in _BFS_CACHE:
        return _BFS_CACHE[key]

    n = game.world.n
    dist = np.full((n, n), -1, dtype=np.int32)
    ar, ac = agent_pos
    if not game.world.in_bounds(ar, ac) or not game.world.passable[ar, ac]:
        return dist

    dist[ar, ac] = 0
    q = deque([(ar, ac)])
    # Reverse edges: a move from (nr, nc) -> (cr, cc) in direction a is legal if twists allow (nr, nc) -> (cr, cc)
    while q:
        cr, cc = q.popleft()
        cd = dist[cr, cc]
        for a in (NORTH, EAST, SOUTH, WEST):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = cr - dr, cc - dc  # predecessor cell that steps via 'a' into (cr, cc)
            if (
                game.world.in_bounds(nr, nc)
                and game.world.passable[nr, nc]
                and dist[nr, nc] == -1
                and game.twists.is_action_legal((nr, nc), a, (cr, cc), (), game.world)
            ):
                dist[nr, nc] = cd + 1
                q.append((nr, nc))

    if len(_BFS_CACHE) > 256:
        _BFS_CACHE.clear()
    _BFS_CACHE[key] = dist
    return dist


def chaser_eval(state: GameState, game: WindowGame) -> float:
    """Chaser evaluation from MAX's (AGENT's) perspective (MIN minimizes this).

    Uses exact obstacle- and one-way-aware reverse BFS distance to agent_pos so
    the Chaser navigates around walls instead of getting trapped in local minima.
    Strictly bounded within [-999.0, +999.0] << UTILITY_M (10,000.0).
    """
    ar, ac = state.agent_pos
    rr, rc = state.rival_pos
    dist_field = _get_rival_dist_field(game, state.agent_pos)
    bfs_d = int(dist_field[rr, rc])
    manhattan_d = abs(ar - rr) + abs(ac - rc)
    dist = float(bfs_d) if bfs_d >= 0 else float(2 * game.world.n + manhattan_d)
    max_d = float(max(1, game.world.n * game.world.n // 2))

    # Closer rival -> more negative for MAX (better for MIN chaser)
    chase_term = 600.0 * (min(dist, max_d) / max_d) - 300.0

    # Delivered progress
    num_stops = max(1, len(game.stops))
    delivered = bin(state.delivered_mask & game.target_mask).count("1")
    progress_term = 300.0 * (delivered / num_stops)

    val = chase_term + progress_term
    return float(max(-999.0, min(999.0, val)))


class ChaserRival:
    """Mandatory assigned 'chaser' adversary using Alpha-Beta search."""

    kind: str = "chaser"

    def __init__(
        self,
        depth: int = 6,
        eval_fn: Optional[EvalFn] = None,
        order_fn: Optional[OrderFn] = None,
    ):
        self.depth = int(depth)
        self.eval_fn = eval_fn or chaser_eval
        self.order_fn = order_fn or make_own_ordering()

    def choose_action(
        self,
        game: WindowGame,
        state: GameState,
    ) -> Tuple[int, SearchDecision]:
        """Selects the chaser rival's legal deterministic action via Alpha-Beta."""
        if state.turn != RIVAL:
            state = GameState(
                agent_pos=state.agent_pos,
                rival_pos=state.rival_pos,
                delivered_mask=state.delivered_mask,
                extra_state=state.extra_state,
                turn=RIVAL,
                agent_cost=state.agent_cost,
                steps=state.steps,
                plies=state.plies,
            )

        legal = game.legal_actions(state, RIVAL)
        if not legal:
            raise RuntimeError("No legal moves available for rival.")

        decision = alphabeta_search(
            game=game,
            state=state,
            depth_limit=self.depth,
            eval_fn=self.eval_fn,
            order_fn=self.order_fn,
        )
        action = decision.best_action if decision.best_action is not None else legal[0]
        return action, decision
