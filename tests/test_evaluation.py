"""Tests for Normalized Multi-Feature Evaluation Function (§7.5, M7.3)."""

import numpy as np
import pytest

from delivery.game.evaluation import EvalWeights, MultiFeatureEvaluator, make_default_evaluator
from delivery.game.window import AGENT, UTILITY_M, GameState, WindowGame
from delivery.instance import Instance


def make_eval_game() -> WindowGame:
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[3, 3] = -1
    return WindowGame(
        Instance(
            id="EVAL-TEST",
            seed=11,
            cms_ids=["513303", "522356", "514549"],
            n=8,
            grid=grid,
            terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
            obstacle_density=0.02,
            start=(0, 0),
            delivery_stops=[(0, 7), (7, 7)],
            slip_probability=0.0,
            adversary="chaser",
            adversary_search_depth=6,
            twists=[],
        ),
        window_radius=None,
        c_star=42.0,
        l_star=14,
        kappa=1.5,
    )


def test_evaluation_strictly_within_terminal_bounds():
    game = make_eval_game()
    evaluator = make_default_evaluator(game)

    for r in range(8):
        for c in range(8):
            if not game.world.passable[r, c]:
                continue
            s = game.initial_state(agent_pos=(r, c), rival_pos=(7, 0))
            val = evaluator(s, game)
            assert abs(val) < 1000.0
            assert abs(val) < UTILITY_M


def test_controlled_monotonicity_properties():
    game = make_eval_game()
    evaluator = make_default_evaluator(game)

    # 1. Increasing delivered_fraction (holding pos, rival_pos, cost fixed) increases score
    s0 = GameState(agent_pos=(4, 4), rival_pos=(0, 0), delivered_mask=0, extra_state=(), agent_cost=10.0)
    s1 = GameState(agent_pos=(4, 4), rival_pos=(0, 0), delivered_mask=1, extra_state=(), agent_cost=10.0)
    assert evaluator(s1, game) > evaluator(s0, game)

    # 2. Increasing rival distance (from 1 to 5, holding agent_pos=(4,4), mask=0, cost=10 fixed) increases score
    s_close = GameState(agent_pos=(4, 4), rival_pos=(4, 5), delivered_mask=0, extra_state=(), agent_cost=10.0)
    s_far = GameState(agent_pos=(4, 4), rival_pos=(0, 3), delivered_mask=0, extra_state=(), agent_cost=10.0)
    assert evaluator(s_far, game) > evaluator(s_close, game)

    # 3. Lower cost_margin (higher spent cost, holding pos/rival/mask fixed) decreases score
    s_cheap = GameState(agent_pos=(4, 4), rival_pos=(0, 0), delivered_mask=0, extra_state=(), agent_cost=10.0)
    s_expensive = GameState(agent_pos=(4, 4), rival_pos=(0, 0), delivered_mask=0, extra_state=(), agent_cost=50.0)
    assert evaluator(s_cheap, game) > evaluator(s_expensive, game)

    # 4. Higher escape mobility (interior vs corner, holding delivered_mask=3 and rival far) increases score
    s_corner = GameState(agent_pos=(0, 0), rival_pos=(7, 7), delivered_mask=3, extra_state=(), agent_cost=10.0)
    s_open = GameState(agent_pos=(4, 4), rival_pos=(7, 7), delivered_mask=3, extra_state=(), agent_cost=10.0)
    assert evaluator(s_open, game) > evaluator(s_corner, game)
