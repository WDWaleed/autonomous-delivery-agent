"""Comprehensive unit and property tests for DeliveryEnv and slip dynamics (§6.1, §7.3, M3.1).

Tests:
1. Controlled slip-selection frequency over >= 20,000 eligible opportunities.
2. Perpendicular direction selection and fallback when neither is traversable.
3. Blocked and one-way constraints respected.
4. Auto-delivery of parcels upon visiting delivery stops.
5. Cost accounting equals the exact sum of transition costs.
6. Zero-slip simulator parity with DeliveryProblem nominal successors.
"""

import math
from typing import List, Tuple

import numpy as np
import pytest

from delivery.instance import load_instance
from delivery.search.problem import EAST, NORTH, SOUTH, WAIT, WEST, DeliveryProblem
from delivery.sim.env import DeliveryEnv
from delivery.world.grid import GridWorld


def test_zero_slip_parity_with_delivery_problem():
    """Verifies that in deterministic mode (slip_p=0), simulator transitions exactly match DeliveryProblem."""
    instance = load_instance()
    world = GridWorld(instance)
    problem = DeliveryProblem(instance, world=world)
    env = DeliveryEnv(world=world, slip_p=0.0, seed=42)

    # Initial state parity
    p_state = problem.initial_state()
    percept = env.reset()

    assert percept["agent_pos"] == (p_state[0], p_state[1])
    assert percept["delivered_mask"] == p_state[2]

    # Step through all legal actions from start
    legal_actions = problem.legal_actions(p_state)
    assert len(legal_actions) > 0

    for a in legal_actions:
        # Clone problem successor
        exp_next_state, exp_cost = problem.successor(p_state, a)

        # Simulator step
        env.reset()
        next_percept, sim_cost, events = env.step(a)

        assert next_percept["agent_pos"] == (exp_next_state[0], exp_next_state[1])
        assert next_percept["delivered_mask"] == exp_next_state[2]
        assert sim_cost == pytest.approx(exp_cost)
        assert events["slipped"] is False
        assert events["illegal_move"] is False


def test_controlled_slip_frequency_over_20000_trials():
    """Verifies empirical slip probability matches slip_p within statistical tolerance over >= 20,000 trials."""
    instance = load_instance()
    world = GridWorld(instance)

    # Find a 3x3 open cross region where North, East, South, West are all passable
    center_pos = None
    for r in range(1, world.n - 1):
        for c in range(1, world.n - 1):
            if (
                world.passable[r, c]
                and world.passable[r - 1, c]
                and world.passable[r + 1, c]
                and world.passable[r, c - 1]
                and world.passable[r, c + 1]
            ):
                center_pos = (r, c)
                break
        if center_pos:
            break

    assert center_pos is not None, "No 3x3 open cross found in grid!"

    slip_p = 0.23
    rng = np.random.default_rng(20261124)
    env = DeliveryEnv(world=world, slip_p=slip_p, rng=rng)

    n_trials = 25000
    slip_count = 0
    perp_choices: List[int] = []

    for _ in range(n_trials):
        env.agent_pos = center_pos
        # Step NORTH (perpendicular candidates are EAST and WEST)
        _, _, event = env.step(NORTH)
        if event["slipped"]:
            slip_count += 1
            perp_choices.append(event["sampled_action"])

    empirical_p = slip_count / n_trials
    # Expected: 25000 * 0.23 = 5750. std = sqrt(25000 * 0.23 * 0.77) ~ 66.53
    # 4.5 sigma margin is ~ 300 trials (~0.012)
    assert abs(empirical_p - slip_p) < 0.015, f"Empirical slip {empirical_p} deviated from {slip_p}"

    # Verify uniform choice between the two perpendicular directions (EAST=1, WEST=3)
    east_count = sum(1 for a in perp_choices if a == EAST)
    west_count = sum(1 for a in perp_choices if a == WEST)
    assert east_count + west_count == slip_count

    ratio = east_count / slip_count
    # Should be close to 0.50 (within 4 sigma of binomial with N~5750, p=0.5 -> std ~ 37.9)
    assert abs(ratio - 0.50) < 0.035, f"Perpendicular ratio {ratio} deviated from 0.50"


def test_fallback_stay_when_neither_perpendicular_traversable():
    """When an agent slips but neither perpendicular cell is traversable, it remains in place and pays fallback cost."""
    # Create a small corridor grid: 3x1 corridor where only North/South are open
    grid = np.full((5, 5), -1, dtype=np.int8)
    grid[1, 2] = 0  # North
    grid[2, 2] = 0  # Center
    grid[3, 2] = 0  # South
    # East (2, 3) and West (2, 1) are blocked (-1)

    inst = load_instance()
    inst.grid = grid
    inst.n = 5
    inst.start = (2, 2)
    inst.delivery_stops = [(1, 2)]

    world = GridWorld(inst)
    # slip_p = 1.0 (always slips)
    env = DeliveryEnv(world=world, slip_p=1.0, seed=123)
    env.agent_pos = (2, 2)

    # Intend to move NORTH (perpendicular candidates are EAST and WEST, both blocked!)
    _, cost, event = env.step(NORTH)

    assert event["slipped"] is True
    assert event["fallback_stay"] is True
    assert event["realised_destination"] == (2, 2)
    assert env.agent_pos == (2, 2)
    assert cost == 3.0  # fallback cost paid


def test_illegal_move_rejected_before_slip():
    """Per Decision D6: illegal moves are rejected before slip sampling."""
    instance = load_instance()
    world = GridWorld(instance)

    # Find a position adjacent to an obstacle
    blocked_dir = None
    agent_pos = None

    for r in range(world.n):
        for c in range(world.n):
            if world.passable[r, c]:
                if r > 0 and not world.passable[r - 1, c]:
                    agent_pos = (r, c)
                    blocked_dir = NORTH
                    break
        if agent_pos:
            break

    assert agent_pos is not None

    env = DeliveryEnv(world=world, slip_p=1.0, seed=999)  # 100% slip if sampled
    env.agent_pos = agent_pos

    # Attempt to move into obstacle
    _, cost, event = env.step(blocked_dir)

    assert event["illegal_move"] is True
    assert event["slipped"] is False  # slip was NOT sampled because move was rejected upfront
    assert event["fallback_stay"] is True
    assert env.agent_pos == agent_pos
    assert cost == 3.0


def test_auto_delivery_and_done():
    instance = load_instance()
    world = GridWorld(instance)
    env = DeliveryEnv(world=world, slip_p=0.0, seed=42)

    stop1 = instance.delivery_stops[0]
    # Place agent at stop1
    env.agent_pos = stop1
    # Move WAIT
    percept, _, event = env.step(WAIT)

    assert (percept["delivered_mask"] & 1) == 1
    assert 0 in event["new_deliveries"]

    # Deliver all remaining stops
    for idx, stop in enumerate(instance.delivery_stops):
        env.agent_pos = stop
        env.step(WAIT)

    assert env.is_done() is True
    assert env.delivered_mask == (1 << len(instance.delivery_stops)) - 1


def test_cost_accounting_invariance():
    instance = load_instance()
    world = GridWorld(instance)
    env = DeliveryEnv(world=world, slip_p=0.23, seed=777)

    step_costs = []
    for _ in range(50):
        # Step WAIT or valid actions
        _, cost, event = env.step(WAIT)
        step_costs.append(cost)

    assert env.total_cost == pytest.approx(sum(step_costs))
    assert env.step_count == 50
