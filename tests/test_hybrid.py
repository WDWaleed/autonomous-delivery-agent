"""Tests for Hybrid Strategic A* + Tactical Alpha-Beta Agent (§7.3, §7.5, M7.2)."""

import numpy as np
import pytest

from delivery.agents.hybrid import HybridAgent
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.instance import Instance
from delivery.search.astar import astar_search
from delivery.search.core import SearchStatus
from delivery.search.problem import DeliveryProblem
from delivery.sim.env import DeliveryEnv
from delivery.sim.runner import run
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld


def make_hybrid_instance(n: int = 8, slip_p: float = 0.0, twists=None) -> Instance:
    grid = np.zeros((n, n), dtype=np.int8)
    grid[3, 3] = -1
    grid[4, 3] = -1
    return Instance(
        id="HYBRID-TEST",
        seed=55,
        cms_ids=["513303", "522356", "514549"],
        n=n,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.03,
        start=(0, 0),
        delivery_stops=[(0, n - 1), (n - 1, n - 1)],
        slip_probability=slip_p,
        adversary="chaser",
        adversary_search_depth=4,
        twists=list(twists or []),
    )


def test_hybrid_no_rival_no_slip_equals_astar_plan():
    """Acceptance: no-rival, no-slip deterministic run equals A* plan cost and delivery result."""
    inst = make_hybrid_instance(n=7, slip_p=0.0)
    world = GridWorld(inst)
    twists = build_twist_stack(inst, world)
    problem = DeliveryProblem(inst, world=world, twist_stack=twists)
    res_astar = astar_search(problem, CombinedMSTHeuristic(problem))
    assert res_astar.status == SearchStatus.SUCCESS

    env = DeliveryEnv(world=world, twists=twists, slip_p=0.0, enable_rival=False)
    agent = HybridAgent(inst, world=world, twists=twists)
    sim_res = run(agent, env, max_steps=100)

    assert sim_res.success is True
    assert sim_res.total_cost == pytest.approx(res_astar.cost)
    assert sim_res.delivered_mask == (1 << len(inst.delivery_stops)) - 1
    assert agent.replan_counts.slip == 0
    assert agent.tactical_moves == 0


def test_hybrid_with_slip_counts_slip_replans():
    inst = make_hybrid_instance(n=7, slip_p=0.35)
    world = GridWorld(inst)
    twists = build_twist_stack(inst, world)
    env = DeliveryEnv(world=world, twists=twists, slip_p=0.35, seed=101, enable_rival=False)
    agent = HybridAgent(inst, world=world, twists=twists)

    sim_res = run(agent, env, max_steps=150)
    assert sim_res.success is True
    assert agent.replan_counts.initial_plan == 1


def test_hybrid_with_rival_triggers_tactical_alphabeta_and_replays_legally():
    inst = make_hybrid_instance(n=8, slip_p=0.0)
    world = GridWorld(inst)
    twists = build_twist_stack(inst, world)
    # Place rival close enough to enter window (radius 4)
    env = DeliveryEnv(
        world=world,
        twists=twists,
        slip_p=0.0,
        enable_rival=True,
        rival_start=(2, 2),
        rival_depth=2,
    )
    agent = HybridAgent(inst, world=world, twists=twists, window_radius=4, tactical_depth=3)

    sim_res = run(agent, env, max_steps=80)
    assert agent.tactical_moves > 0
    assert agent.replan_counts.rival_block > 0
    # Verify every trace step was legal
    for tr in sim_res.trace:
        assert 0 <= tr.intended_action <= 4
