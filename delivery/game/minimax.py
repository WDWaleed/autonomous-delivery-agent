"""Depth-Limited Minimax Search from scratch (§7.5, M6.2)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from delivery.game.stats import EvalFn, OrderFn, SearchDecision, SearchStats, placeholder_eval
from delivery.game.window import AGENT, GameState, WindowGame


def minimax_search(
    game: Any,
    state: GameState,
    depth_limit: int,
    eval_fn: Optional[EvalFn] = None,
    order_fn: Optional[OrderFn] = None,
) -> SearchDecision:
    """Executes depth-limited Minimax search with principal variation tracking.

    MAX (AGENT=0) maximizes utility; MIN (RIVAL=1) minimizes utility.
    Deterministic tie-breaking: strict '>' / '<' preserves the first action
    encountered in the ordered action list.
    """
    evaluator = eval_fn or placeholder_eval
    stats = SearchStats()

    def _search(
        curr_state: GameState,
        depth: int,
        ply: int,
        prev_state: Optional[GameState],
    ) -> Tuple[float, List[int]]:
        if ply > stats.depth_reached:
            stats.depth_reached = ply

        if game.is_terminal(curr_state, prev_state=prev_state):
            stats.evaluations += 1
            return game.terminal_utility(curr_state, prev_state=prev_state), []

        if depth <= 0:
            stats.evaluations += 1
            return evaluator(curr_state, game), []

        actions = game.legal_actions(curr_state, curr_state.turn)
        if not actions:
            stats.evaluations += 1
            return evaluator(curr_state, game), []

        if order_fn is not None:
            actions = order_fn(curr_state, actions, game, depth, None)

        stats.nodes_expanded += 1
        is_max = curr_state.turn == AGENT

        best_val = float("-inf") if is_max else float("inf")
        best_pv: List[int] = []

        for action in actions:
            next_state = game.step(curr_state, action)
            val, child_pv = _search(next_state, depth - 1, ply + 1, curr_state)

            if is_max:
                if val > best_val:
                    best_val = val
                    best_pv = [action] + child_pv
            else:
                if val < best_val:
                    best_val = val
                    best_pv = [action] + child_pv

        return best_val, best_pv

    val, pv = _search(state, depth_limit, 0, None)
    best_action = pv[0] if pv else None
    return SearchDecision(
        best_action=best_action,
        value=val,
        principal_variation=pv,
        stats=stats,
    )
