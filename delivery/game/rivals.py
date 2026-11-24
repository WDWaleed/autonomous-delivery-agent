"""Assigned Rival Adversary Strategy (§7.5, M7.1).

Implements the mandatory assigned 'chaser' rival strategy using Alpha-Beta search
to adversary_search_depth (default 6 plies) with chaser-specific evaluation
(minimizing relaxed/Manhattan distance to the agent and maximizing capture pressure).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from delivery.game.alphabeta import alphabeta_search
from delivery.game.ordering import make_own_ordering
from delivery.game.stats import EvalFn, OrderFn, SearchDecision
from delivery.game.window import RIVAL, UTILITY_M, GameState, WindowGame
from delivery.world.distance import distance_field


def chaser_eval(state: GameState, game: WindowGame) -> float:
    """Chaser evaluation from MAX's (AGENT's) perspective (MIN minimizes this).

    Lower values are better for the chaser:
    - Primary term: Manhattan distance between agent and rival (closer = lower utility for MAX).
    - Secondary term: agent progress toward undelivered stops (so chaser prefers states
      where agent is further from completing deliveries).
    Strictly bounded within [-999.0, +999.0] << UTILITY_M (10,000.0).
    """
    ar, ac = state.agent_pos
    rr, rc = state.rival_pos
    dist = abs(ar - rr) + abs(ac - rc)
    max_d = max(1, 2 * game.world.n)

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
