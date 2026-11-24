"""Tests for Move Ordering and Alpha-Beta Value Invariance (§7.5, M6.3)."""

import numpy as np
import pytest

from delivery.game.alphabeta import alphabeta_search
from delivery.game.minimax import minimax_search
from delivery.game.ordering import make_own_ordering, make_random_ordering, natural_ordering
from delivery.game.window import AGENT, RIVAL, WindowGame
from delivery.instance import Instance
from delivery.world.distance import distance_field
from delivery.world.grid import GridWorld


def make_test_game() -> WindowGame:
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[2, 2] = -1
    inst = Instance(
        id="ORDER-TEST",
        seed=7,
        cms_ids=["513303", "522356", "514549"],
        n=6,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.03,
        start=(0, 0),
        delivery_stops=[(0, 5), (5, 5)],
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=6,
        twists=[],
    )
    return WindowGame(inst, window_radius=4, window_center=(2, 2))


def test_random_ordering_is_reproducible_with_seed():
    game = make_test_game()
    s = game.initial_state(agent_pos=(1, 1), rival_pos=(4, 4))
    actions = game.legal_actions(s, AGENT)

    ord1 = make_random_ordering(12345)
    ord2 = make_random_ordering(12345)
    assert ord1(s, actions, game, 3, None) == ord2(s, actions, game, 3, None)


def test_all_orderings_agree_on_minimax_value():
    game = make_test_game()
    world = game.world
    stop_fields = [distance_field(world, s, reverse=True) for s in game.stops]
    own_order = make_own_ordering(stop_fields=stop_fields)
    rand_order = make_random_ordering(99)

    positions = [
        game.initial_state(agent_pos=(1, 1), rival_pos=(3, 3), turn=AGENT),
        game.initial_state(agent_pos=(1, 4), rival_pos=(2, 4), turn=RIVAL),
        game.initial_state(agent_pos=(4, 1), rival_pos=(4, 4), delivered_mask=1, turn=AGENT),
    ]

    for state in positions:
        for depth in (2, 3, 4):
            mm_nat = minimax_search(game, state, depth_limit=depth, order_fn=natural_ordering)
            ab_nat = alphabeta_search(game, state, depth_limit=depth, order_fn=natural_ordering)
            ab_rnd = alphabeta_search(game, state, depth_limit=depth, order_fn=rand_order)
            ab_own = alphabeta_search(game, state, depth_limit=depth, order_fn=own_order)

            assert ab_nat.value == pytest.approx(mm_nat.value)
            assert ab_rnd.value == pytest.approx(mm_nat.value)
            assert ab_own.value == pytest.approx(mm_nat.value)
            assert ab_own.stats.evaluations <= mm_nat.stats.evaluations
