"""Tests for Assigned Chaser Rival Strategy (§7.5, M7.1)."""

import numpy as np
import pytest

from delivery.game.rivals import ChaserRival, chaser_eval
from delivery.game.window import AGENT, RIVAL, UTILITY_M, WindowGame
from delivery.instance import Instance
from delivery.search.problem import EAST, NORTH, SOUTH, WAIT, WEST
from delivery.twists.base import TwistStack
from delivery.twists.one_way import OneWayTwist
from delivery.world.grid import GridWorld


def make_rival_test_instance(n: int = 7, twists=None) -> Instance:
    grid = np.zeros((n, n), dtype=np.int8)
    grid[3, 3] = -1  # central obstacle
    return Instance(
        id="RIVAL-TEST",
        seed=17,
        cms_ids=["513303", "522356", "514549"],
        n=n,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.02,
        start=(0, 0),
        delivery_stops=[(0, n - 1), (n - 1, n - 1)],
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=6,
        twists=list(twists or []),
    )


def test_chaser_rival_moves_toward_agent_and_captures():
    inst = make_rival_test_instance(n=6)
    game = WindowGame(inst, window_radius=4, window_center=(2, 2))
    rival = ChaserRival(depth=4)

    # Rival at (2, 3), agent at (2, 2)? (3,3) is obstacle; put agent at (2, 1), rival at (2, 2)
    s = game.initial_state(agent_pos=(2, 1), rival_pos=(2, 2), turn=RIVAL)
    action, decision = rival.choose_action(game, s)

    # Chaser should step WEST to (2, 1) for immediate capture (-UTILITY_M)
    assert action == WEST
    assert decision.value == -UTILITY_M


def test_chaser_rival_deterministic_and_respects_one_way_and_obstacles():
    inst = make_rival_test_instance(n=6, twists=["one_way_streets"])
    world = GridWorld(inst)
    ts = TwistStack([OneWayTwist()])
    ts.twists[0].initialize(inst, world)
    game = WindowGame(inst, world=world, twists=ts, window_radius=None)
    rival = ChaserRival(depth=4)

    s = game.initial_state(agent_pos=(1, 1), rival_pos=(4, 4), turn=RIVAL)
    act1, dec1 = rival.choose_action(game, s)
    act2, dec2 = rival.choose_action(game, s)

    # Deterministic
    assert act1 == act2
    assert dec1.value == pytest.approx(dec2.value)

    # Legal under obstacles and one-way restrictions
    legal = game.legal_actions(s, RIVAL)
    assert act1 in legal


def test_chaser_routes_around_obstacle_wall_instead_of_getting_stuck():
    """Verifies Chaser does not get stuck in a Manhattan local minimum behind a wall."""
    grid = np.zeros((6, 6), dtype=np.int8)
    # Wall blocking direct Manhattan path at col 2, rows 1..4
    for r in range(1, 5):
        grid[r, 2] = -1
    inst = Instance(
        id="WALL-TEST",
        seed=1,
        cms_ids=["513303"],
        n=6,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.1,
        start=(2, 0),
        delivery_stops=[(5, 5)],
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=4,
        twists=[],
    )
    game = WindowGame(inst, window_radius=None)
    rival = ChaserRival(depth=2)
    # Agent at (2, 1), Rival at (2, 3) separated by wall at col 2
    s = game.initial_state(agent_pos=(2, 1), rival_pos=(2, 3), turn=RIVAL)
    act, _ = rival.choose_action(game, s)
    # Chaser must move NORTH or SOUTH around the wall, NOT WAIT!
    assert act in (NORTH, SOUTH)
