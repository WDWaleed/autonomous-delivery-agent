"""Normalized Multi-Feature Adversarial Evaluation Function (§7.5, M7.3).

Combines five bounded, normalized features in [0, 1] (or [-1, 1]):
1. delivered_fraction   (+): [0.0, 1.0] fraction of stops completed
2. remaining_route_norm (-): [0.0, 1.0] normalized remaining cost via CombinedMSTHeuristic
3. rival_distance_sat   (+): [0.0, 1.0] saturated distance from rival (R_sat = 6)
4. escape_mobility      (+): [0.0, 1.0] legal cardinal exit fraction (deg / 4.0)
5. cost_margin          (+): [0.0, 1.0] remaining budget before kappa * C* draw

Total evaluation is strictly scaled within [-999.0, +999.0] << UTILITY_M (10,000.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from delivery.game.stats import EvalFn
from delivery.game.window import AGENT, UTILITY_M, GameState, WindowGame
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.search.problem import WAIT, DeliveryProblem


@dataclass(frozen=True)
class EvalWeights:
    """Feature weights for MultiFeatureEvaluator (sum of |w| <= 950 << UTILITY_M)."""

    w_delivered: float = 350.0
    w_remaining_route: float = -300.0
    w_rival_dist: float = 180.0
    w_mobility: float = 70.0
    w_cost_margin: float = 50.0


class MultiFeatureEvaluator:
    """Normalized, bounded multi-feature state evaluation for WindowGame."""

    def __init__(
        self,
        problem: Optional[DeliveryProblem] = None,
        own_heuristic: Optional[CombinedMSTHeuristic] = None,
        weights: Optional[EvalWeights] = None,
        rival_sat_radius: int = 6,
    ):
        self.problem = problem
        self.own_heuristic = own_heuristic
        self.weights = weights or EvalWeights()
        self.rival_sat_radius = int(rival_sat_radius)

    def extract_features(self, state: GameState, game: WindowGame) -> Dict[str, float]:
        """Extracts all five normalized features in [0.0, 1.0]."""
        # 1. Delivered fraction
        num_stops = max(1, len(game.stops))
        delivered_cnt = bin(state.delivered_mask & game.target_mask).count("1")
        f_delivered = delivered_cnt / float(num_stops)

        # 2. Remaining route cost via CombinedMSTHeuristic (or Manhattan fallback)
        if self.own_heuristic is not None and self.problem is not None:
            h_rem = float(
                self.own_heuristic(
                    (state.agent_pos[0], state.agent_pos[1], state.delivered_mask, state.extra_state),
                    self.problem,
                )
            )
        else:
            ar, ac = state.agent_pos
            undel = [s for i, s in enumerate(game.stops) if not (state.delivered_mask & (1 << i))]
            h_rem = float(3 * min(abs(ar - sr) + abs(ac - sc) for sr, sc in undel)) if undel else 0.0

        max_route_scale = float(max(1.0, game.c_star if game.c_star else (7.0 * game.world.n * num_stops)))
        f_remaining = min(1.0, max(0.0, h_rem / max_route_scale))

        # 3. Saturated rival distance (Manhattan)
        ar, ac = state.agent_pos
        rr, rc = state.rival_pos
        d_rival = abs(ar - rr) + abs(ac - rc)
        f_rival_dist = min(1.0, d_rival / float(max(1, self.rival_sat_radius)))

        # 4. Escape mobility (legal cardinal exits / 4.0)
        legal_moves = [a for a in game.legal_actions(state, AGENT) if a != WAIT]
        f_mobility = min(1.0, len(legal_moves) / 4.0)

        # 5. Cost margin relative to kappa * C*
        if game.c_star is not None and game.c_star > 0:
            budget = game.kappa * game.c_star
            f_cost_margin = max(0.0, min(1.0, (budget - state.agent_cost) / budget))
        else:
            f_cost_margin = 1.0

        return {
            "delivered_fraction": f_delivered,
            "remaining_route_norm": f_remaining,
            "rival_distance_sat": f_rival_dist,
            "escape_mobility": f_mobility,
            "cost_margin": f_cost_margin,
        }

    def __call__(self, state: GameState, game: WindowGame) -> float:
        feats = self.extract_features(state, game)
        w = self.weights
        raw_score = (
            w.w_delivered * feats["delivered_fraction"]
            + w.w_remaining_route * feats["remaining_route_norm"]
            + w.w_rival_dist * feats["rival_distance_sat"]
            + w.w_mobility * feats["escape_mobility"]
            + w.w_cost_margin * feats["cost_margin"]
        )
        # Strict clamp well inside terminal utility magnitude (-10,000, +10,000)
        return float(max(-999.0, min(999.0, raw_score)))


def make_default_evaluator(
    game: WindowGame,
    weights: Optional[EvalWeights] = None,
) -> MultiFeatureEvaluator:
    """Factory creating a MultiFeatureEvaluator wired to game's DeliveryProblem & CombinedMSTHeuristic."""
    prob = DeliveryProblem(game.instance, world=game.world, twist_stack=game.twists)
    h_own = CombinedMSTHeuristic(prob)
    return MultiFeatureEvaluator(problem=prob, own_heuristic=h_own, weights=weights)
