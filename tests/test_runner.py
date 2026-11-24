"""Unit tests for Sense-Reason-Act runner and trace recording (§6.1, §7.3, M3.2)."""

import numpy as np
import pytest

from delivery.agents.reflex import ReflexAgent
from delivery.instance import load_instance
from delivery.sim.env import DeliveryEnv
from delivery.sim.runner import RunnerResult, StepTrace, run
from delivery.world.grid import GridWorld


def test_runner_executes_episode():
    instance = load_instance()
    world = GridWorld(instance)
    env = DeliveryEnv(world=world, slip_p=0.0, seed=42)
    agent = ReflexAgent()

    result = run(agent, env, max_steps=20)
    assert isinstance(result, RunnerResult)
    assert len(result.trajectory) > 1
    assert len(result.trace) > 0
    assert result.total_steps <= 20
    assert result.total_cost >= 0.0

    # Verify first trace record
    first_step = result.trace[0]
    assert isinstance(first_step, StepTrace)
    assert first_step.step == 1
    assert first_step.from_pos == instance.start
    assert first_step.decision_rationale != ""
    assert first_step.cost > 0.0


def test_runner_detects_local_minimum_oscillation():
    # Construct a U-shaped dead end where reflex agent gets trapped oscillating
    # 5x5 grid with U-shaped wall
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1, 1:4] = -1  # wall across row 1
    grid[2, 1] = -1    # left wall
    grid[2, 3] = -1    # right wall
    # agent at (2, 2) inside U trap, target at (0, 2) directly on other side of wall
    inst = load_instance()
    inst.n = 5
    inst.grid = grid
    inst.start = (2, 2)
    inst.delivery_stops = [(0, 2)]

    world = GridWorld(inst)
    env = DeliveryEnv(world=world, slip_p=0.0, seed=1)
    agent = ReflexAgent()

    result = run(agent, env, max_steps=30)
    assert result.success is False
    assert result.failure_reason in ("LOCAL_MINIMUM_OSCILLATION", "STEP_CAP_EXCEEDED")
