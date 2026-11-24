"""Unit tests for ReflexAgent baseline (§6.1, §7.3, M3.2)."""

import numpy as np
import pytest

from delivery.agents.base import Agent
from delivery.agents.reflex import ReflexAgent
from delivery.sim.env import EAST, NORTH, SOUTH, WAIT, WEST


def test_reflex_agent_is_agent_instance():
    agent = ReflexAgent()
    assert isinstance(agent, Agent)
    assert agent.name == "SimpleReflexAgent"


def test_reflex_agent_steps_towards_target():
    agent = ReflexAgent()
    # 5x5 open grid, agent at (2, 2), stop at (0, 2) [directly North]
    grid = np.zeros((5, 5), dtype=np.int8)
    percept = {
        "agent_pos": (2, 2),
        "undelivered_stops": [(0, 2)],
        "grid": grid,
    }

    action, rationale = agent.choose_action(percept)
    assert action == NORTH
    assert "NORTH" in rationale
    assert "2 -> 1" in rationale


def test_reflex_agent_deterministic_cardinal_tie_breaking():
    agent = ReflexAgent()
    # Agent at (2, 2), stop at (0, 4) [North-East diagonal: dist = 2 + 2 = 4]
    # Moving NORTH (1, 2) -> dist = 1 + 2 = 3
    # Moving EAST (2, 3) -> dist = 2 + 1 = 3
    # Tie between NORTH and EAST. Cardinal order is N, E, S, W, so NORTH should win!
    grid = np.zeros((5, 5), dtype=np.int8)
    percept = {
        "agent_pos": (2, 2),
        "undelivered_stops": [(0, 4)],
        "grid": grid,
    }

    action, _ = agent.choose_action(percept)
    assert action == NORTH


def test_reflex_agent_avoids_obstacles():
    agent = ReflexAgent()
    # Agent at (2, 2), stop at (0, 2) directly North
    # Cell (1, 2) is blocked (-1)
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1, 2] = -1

    percept = {
        "agent_pos": (2, 2),
        "undelivered_stops": [(0, 2)],
        "grid": grid,
    }

    action, rationale = agent.choose_action(percept)
    # NORTH is blocked, so it must choose between EAST or WEST (or SOUTH)
    assert action != NORTH
    assert action in (EAST, WEST)


def test_reflex_agent_respects_fragile_parcel_restriction():
    agent = ReflexAgent()
    # Agent at (2, 2), target at (1, 2) directly North
    # Cell (1, 2) is mud (terrain 2)
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1, 2] = 2  # mud

    # 1. Fragile undelivered: mud is illegal
    percept_fragile = {
        "agent_pos": (2, 2),
        "undelivered_stops": [(1, 2)],
        "grid": grid,
        "extra": {"fragile_delivered": False},
    }
    action, _ = agent.choose_action(percept_fragile)
    assert action != NORTH

    # 2. Fragile delivered: mud is legal
    percept_delivered = {
        "agent_pos": (2, 2),
        "undelivered_stops": [(1, 2)],
        "grid": grid,
        "extra": {"fragile_delivered": True},
    }
    action_deliv, _ = agent.choose_action(percept_delivered)
    assert action_deliv == NORTH


def test_reflex_agent_waits_when_no_stops():
    agent = ReflexAgent()
    percept = {
        "agent_pos": (2, 2),
        "undelivered_stops": [],
    }
    action, rationale = agent.choose_action(percept)
    assert action == WAIT
    assert "completed" in rationale
