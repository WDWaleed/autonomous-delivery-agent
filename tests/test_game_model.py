"""Tests for Reduced Window Adversarial Game Model (§7.3, D7, D12, M6.1)."""

import numpy as np
import pytest

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
from delivery.instance import Instance
from delivery.search.problem import EAST, NORTH, SOUTH, WAIT, WEST
from delivery.twists.base import TwistStack
from delivery.twists.one_way import OneWayTwist
from delivery.world.grid import GridWorld


def make_game_instance(n: int = 7, twists=None) -> Instance:
    grid = np.zeros((n, n), dtype=np.int8)
    grid[2, 2] = -1  # obstacle
    return Instance(
        id="GAME-TEST",
        seed=42,
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


def test_legal_moves_base_and_blocked():
    inst = make_game_instance(n=5)
    game = WindowGame(inst, window_radius=None)
    s = game.initial_state(agent_pos=(2, 1), rival_pos=(4, 4))
    actions = game.legal_actions(s, AGENT)
    # (2, 2) is blocked (-1), so EAST is illegal
    assert EAST not in actions
    assert NORTH in actions
    assert SOUTH in actions
    assert WEST in actions
    assert WAIT in actions


def test_legal_moves_with_window_boundary():
    inst = make_game_instance(n=7)
    game = WindowGame(inst, window_radius=1, window_center=(3, 3))
    # Inside window [2..4, 2..4], agent at (2, 3) cannot step NORTH to (1, 3)
    s = game.initial_state(agent_pos=(2, 3), rival_pos=(4, 4))
    actions = game.legal_actions(s, AGENT)
    assert NORTH not in actions
    assert EAST in actions
    assert SOUTH in actions
    assert WEST not in actions  # (2, 2) is an obstacle


def test_legal_moves_one_way_twist():
    inst = make_game_instance(n=6, twists=["one_way_streets"])
    world = GridWorld(inst)
    ts = TwistStack([OneWayTwist()])
    ts.twists[0].initialize(inst, world)
    game = WindowGame(inst, world=world, twists=ts, window_radius=None)

    # Ensure legal_actions respects one-way twist legality on every cell
    for r in range(6):
        for c in range(6):
            if not world.passable[r, c]:
                continue
            s = game.initial_state(agent_pos=(r, c), rival_pos=(5, 5))
            legal = game.legal_actions(s, AGENT)
            for a in (NORTH, EAST, SOUTH, WEST):
                if a in legal:
                    dr, dc = {NORTH: (-1, 0), EAST: (0, 1), SOUTH: (1, 0), WEST: (0, -1)}[a]
                    assert ts.is_action_legal((r, c), a, (r + dr, c + dc), s.extra_state, world)


def test_legal_moves_teleport_when_enabled():
    inst = make_game_instance(n=5, twists=["teleport_tunnels"])
    game = WindowGame(
        inst,
        window_radius=None,
        tunnel_pairs={(0, 0): (4, 4)},
    )
    s = game.initial_state(agent_pos=(0, 0), rival_pos=(4, 0))
    actions = game.legal_actions(s, AGENT)
    assert TELEPORT in actions
    s_next = game.step(s, TELEPORT)
    assert s_next.agent_pos == (4, 4)
    assert s_next.turn == RIVAL


def test_same_cell_and_edge_swap_capture_helpers():
    assert check_same_cell_capture((2, 2), (2, 2))
    assert not check_same_cell_capture((2, 2), (2, 3))
    assert check_edge_swap_capture((1, 1), (1, 2), (1, 2), (1, 1))
    assert not check_edge_swap_capture((1, 1), (1, 2), (1, 3), (1, 2))


def test_d7_terminal_precedence_and_kappa_boundary():
    inst = make_game_instance(n=5)
    c_star, l_star, cert = compute_canonical_baseline(inst)
    assert cert is True
    assert c_star is not None and l_star is not None

    game = WindowGame(
        inst,
        window_radius=None,
        kappa=1.5,
        c_star=c_star,
        l_star=l_star,
    )
    all_mask = (1 << len(inst.delivery_stops)) - 1

    # 1. Capture overrides completed delivery and step cap (D7 precedence)
    s_cap = GameState(
        agent_pos=(4, 4),
        rival_pos=(4, 4),
        delivered_mask=all_mask,
        extra_state=(),
        turn=AGENT,
        agent_cost=c_star,
        steps=game.t_max,
    )
    assert game.outcome(s_cap) == GameOutcome.LOSS_CAPTURE
    assert game.terminal_utility(s_cap) == -UTILITY_M

    # 2. Win at exact kappa * C* boundary
    s_win_boundary = GameState(
        agent_pos=(4, 4),
        rival_pos=(0, 0),
        delivered_mask=all_mask,
        extra_state=(),
        turn=AGENT,
        agent_cost=1.5 * c_star,
        steps=l_star,
    )
    assert game.outcome(s_win_boundary) == GameOutcome.WIN
    assert game.terminal_utility(s_win_boundary) > UTILITY_M - 100.0

    # 3. Excessive cost draw just above kappa * C*
    s_excess = GameState(
        agent_pos=(4, 4),
        rival_pos=(0, 0),
        delivered_mask=all_mask,
        extra_state=(),
        turn=AGENT,
        agent_cost=1.5 * c_star + 1.0,
        steps=l_star,
    )
    assert game.outcome(s_excess) == GameOutcome.DRAW_EXCESS_COST
    assert game.terminal_utility(s_excess) == 0.0

    # 4. Step-cap draw at T_max = 4 * L* + 20
    s_step_cap = GameState(
        agent_pos=(1, 1),
        rival_pos=(0, 0),
        delivered_mask=0,
        extra_state=(),
        turn=AGENT,
        agent_cost=10.0,
        steps=4 * l_star + 20,
    )
    assert game.outcome(s_step_cap) == GameOutcome.DRAW_STEP_CAP
    assert game.terminal_utility(s_step_cap) == 0.0
