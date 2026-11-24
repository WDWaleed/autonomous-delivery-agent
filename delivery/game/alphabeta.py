"""Depth-Limited Alpha-Beta Pruning Search from scratch (§7.5, M6.2)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from delivery.game.stats import EvalFn, OrderFn, SearchDecision, SearchStats, placeholder_eval
from delivery.game.window import AGENT, GameState


def alphabeta_search(
    game: Any,
    state: GameState,
    depth_limit: int,
    eval_fn: Optional[EvalFn] = None,
    order_fn: Optional[OrderFn] = None,
    pv_hint: Optional[List[int]] = None,
) -> SearchDecision:
    """Executes depth-limited Alpha-Beta search with principal variation tracking.

    Guarantees identical minimax value to minimax_search while pruning branches
    where alpha >= beta. Tracks nodes_expanded, evaluations, cutoffs, and depth_reached.
    """
    evaluator = eval_fn or placeholder_eval
    stats = SearchStats()

    def _search(
        curr_state: GameState,
        depth: int,
        ply: int,
        alpha: float,
        beta: float,
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

        hint_action = pv_hint[ply] if (pv_hint is not None and ply < len(pv_hint)) else None
        if order_fn is not None:
            actions = order_fn(curr_state, actions, game, depth, hint_action)

        stats.nodes_expanded += 1
        is_max = curr_state.turn == AGENT

        if is_max:
            best_val = float("-inf")
            best_pv: List[int] = []
            for action in actions:
                next_state = game.step(curr_state, action)
                val, child_pv = _search(next_state, depth - 1, ply + 1, alpha, beta, curr_state)
                if val > best_val:
                    best_val = val
                    best_pv = [action] + child_pv
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
            return best_val, best_pv

        best_val = float("inf")
        best_pv = []
        for action in actions:
            next_state = game.step(curr_state, action)
            val, child_pv = _search(next_state, depth - 1, ply + 1, alpha, beta, curr_state)
            if val < best_val:
                best_val = val
                best_pv = [action] + child_pv
            if best_val < beta:
                beta = best_val
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return best_val, best_pv

    val, pv = _search(state, depth_limit, 0, float("-inf"), float("inf"), None)
    best_action = pv[0] if pv else None
    return SearchDecision(
        best_action=best_action,
        value=val,
        principal_variation=pv,
        stats=stats,
    )
