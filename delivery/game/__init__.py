"""Adversarial game modeling, Minimax, Alpha-Beta pruning, and move ordering (§7.3, §7.5, M6)."""

from delivery.game.alphabeta import alphabeta_search
from delivery.game.evaluation import EvalWeights, MultiFeatureEvaluator, make_default_evaluator
from delivery.game.minimax import minimax_search
from delivery.game.ordering import make_own_ordering, make_random_ordering, natural_ordering
from delivery.game.rivals import ChaserRival, chaser_eval
from delivery.game.stats import EvalFn, OrderFn, SearchDecision, SearchStats, placeholder_eval
from delivery.game.window import (
    AGENT,
    RIVAL,
    TELEPORT,
    UTILITY_M,
    GameOutcome,
    GameState,
    WindowGame,
    check_edge_swap_capture,
    check_same_cell_capture,
    compute_canonical_baseline,
)

__all__ = [
    "AGENT",
    "RIVAL",
    "TELEPORT",
    "UTILITY_M",
    "GameOutcome",
    "GameState",
    "WindowGame",
    "check_same_cell_capture",
    "check_edge_swap_capture",
    "compute_canonical_baseline",
    "SearchStats",
    "SearchDecision",
    "EvalFn",
    "OrderFn",
    "placeholder_eval",
    "minimax_search",
    "alphabeta_search",
    "natural_ordering",
    "make_random_ordering",
    "make_own_ordering",
    "ChaserRival",
    "chaser_eval",
    "EvalWeights",
    "MultiFeatureEvaluator",
    "make_default_evaluator",
]
