"""Unit tests and hand-made fixtures for DeliveryProblem formulation (§7.3, §7.4, M4.1)."""

import numpy as np
import pytest

from delivery.instance import Instance, load_instance
from delivery.search.problem import EAST, NORTH, SOUTH, WAIT, WEST, DeliveryProblem
from delivery.sim.env import DeliveryEnv
from delivery.twists.base import TwistStack
from delivery.world.grid import GridWorld


def make_tiny_3x3_instance() -> Instance:
    """Creates a deterministic 3x3 road-only grid with 2 delivery stops."""
    grid = np.zeros((3, 3), dtype=np.int8)  # all road (cost 3)
    return Instance(
        id="tiny_3x3",
        n=3,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.0,
        start=(0, 0),
        delivery_stops=[(0, 2), (2, 2)],
        adversary="chaser",
        slip_probability=0.0,
        adversary_search_depth=3,
        twists=[],
        seed=12345,
        cms_ids=["TEST01"],
    )


def test_3x3_initial_and_goal_states():
    inst = make_tiny_3x3_instance()
    prob = DeliveryProblem(inst, return_to_depot=False)

    s0 = prob.initial_state()
    assert s0[0] == 0 and s0[1] == 0
    assert s0[2] == 0  # delivered_mask
    assert prob.is_goal(s0) is False

    # Legal actions from (0, 0): EAST (0, 1) and SOUTH (1, 0)
    actions = prob.legal_actions(s0)
    assert sorted(actions) == [EAST, SOUTH]


def test_3x3_successors_and_costs():
    inst = make_tiny_3x3_instance()
    prob = DeliveryProblem(inst, return_to_depot=False)

    s0 = prob.initial_state()
    succs = prob.successors(s0)
    assert len(succs) == 2

    # Stepping EAST to (0, 1)
    east_succ = next(s for s in succs if s[1] == EAST)
    assert east_succ[0][0] == 0 and east_succ[0][1] == 1
    assert east_succ[2] == 3.0  # entry cost of road

    # From (0, 1), step EAST to (0, 2) which is stop 0
    s1, cost = prob.successor(east_succ[0], EAST)
    assert s1[0] == 0 and s1[1] == 2
    assert s1[2] == 1  # stop 0 delivered (bit 0 set)
    assert prob.is_goal(s1) is False

    # From (0, 2), step SOUTH twice to (2, 2) which is stop 1
    s2, _ = prob.successor(s1, SOUTH)
    s3, _ = prob.successor(s2, SOUTH)
    assert s3[0] == 2 and s3[1] == 2
    assert s3[2] == 3  # both stops 0 and 1 delivered (bits 0 and 1 set)
    assert prob.is_goal(s3) is True


def test_return_to_depot_requirement():
    inst = make_tiny_3x3_instance()
    # return_to_depot = True
    prob = DeliveryProblem(inst, return_to_depot=True)

    # State with all deliveries completed at (2, 2)
    s_delivered = (2, 2, 3, ())
    assert prob.is_goal(s_delivered) is False  # not at depot yet

    # State with all deliveries completed back at depot (0, 0)
    s_depot = (0, 0, 3, ())
    assert prob.is_goal(s_depot) is True


def test_precedence_constraints():
    inst = make_tiny_3x3_instance()
    # Constraint: stop 1 (2, 2) requires stop 0 (0, 2) to be delivered first
    prob = DeliveryProblem(inst, constraints={1: 0})

    s0 = prob.initial_state()

    # Move straight to stop 1 via SOUTH twice, then EAST twice:
    # (0, 0) -> (1, 0) -> (2, 0) -> (2, 1) -> (2, 2)
    s = s0
    for a in [SOUTH, SOUTH, EAST, EAST]:
        s, _ = prob.successor(s, a)

    # At (2, 2), stop 1 was visited, BUT stop 0 was not yet delivered!
    assert s[0] == 2 and s[1] == 2
    assert (s[2] & (1 << 1)) == 0  # bit 1 was NOT set
    # Cell remains fully traversable
    assert prob.legal_actions(s) != []

    # Now step to stop 0 (0, 2), then back to stop 1 (2, 2)
    for a in [NORTH, NORTH]:  # reach (0, 2)
        s, _ = prob.successor(s, a)
    assert (s[2] & (1 << 0)) != 0  # stop 0 delivered!

    # Step back to stop 1: (0, 2) -> (1, 2) -> (2, 2)
    for a in [SOUTH, SOUTH]:
        s, _ = prob.successor(s, a)
    assert (s[2] & (1 << 1)) != 0  # stop 1 now delivered!
    assert prob.is_goal(s) is True


def test_5x5_obstacle_filtering():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1, 1] = -1  # obstacle
    inst = Instance(
        id="test_5x5",
        n=5,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.04,
        start=(1, 0),
        delivery_stops=[(4, 4)],
        adversary="chaser",
        slip_probability=0.0,
        adversary_search_depth=3,
        twists=[],
        seed=111,
        cms_ids=["TEST02"],
    )
    prob = DeliveryProblem(inst)
    s = prob.initial_state()  # at (1, 0)

    # EAST leads to (1, 1) which is blocked
    legal = prob.legal_actions(s)
    assert EAST not in legal
    assert NORTH in legal
    assert SOUTH in legal


def test_describe_state():
    inst = make_tiny_3x3_instance()
    prob = DeliveryProblem(inst)
    s = (0, 2, 1, ())
    desc = prob.describe(s)
    assert "Pos=(0, 2)" in desc
    assert "Delivered=[0]/2" in desc
    assert "Remaining=[1]" in desc


def test_deterministic_env_parity():
    """Confirms 1:1 parity between DeliveryProblem and DeliveryEnv with slip disabled."""
    instance = load_instance()
    world = GridWorld(instance)
    prob = DeliveryProblem(world)
    env = DeliveryEnv(world=world, slip_p=0.0, seed=42)

    s_prob = prob.initial_state()
    p_env = env.reset()

    assert p_env["agent_pos"] == (s_prob[0], s_prob[1])
    assert p_env["delivered_mask"] == s_prob[2]

    # Execute a sequence of legal actions
    curr_s = s_prob
    for _ in range(10):
        actions = prob.legal_actions(curr_s)
        if not actions:
            break
        a = actions[0]
        next_s, cost_prob = prob.successor(curr_s, a)
        percept, cost_env, event = env.step(a)

        assert percept["agent_pos"] == (next_s[0], next_s[1])
        assert percept["delivered_mask"] == next_s[2]
        assert cost_env == pytest.approx(cost_prob)
        curr_s = next_s
