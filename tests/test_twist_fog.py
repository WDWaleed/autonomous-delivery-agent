"""Unit and property tests for Fog / Partial Observability Twist (M2.2)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from delivery.instance import load_instance
from delivery.twists.fog import FogTwist
from delivery.world.grid import GridWorld


def test_fog_initialization_and_window():
    instance = load_instance()
    world = GridWorld(instance)
    fog = FogTwist(radius=4)
    fog.initialize(instance, world)

    sr, sc = instance.start
    assert fog.is_cell_discovered(sr, sc) is True

    # Check that cells within radius 4 are discovered
    for r in range(max(0, sr - 4), min(world.n, sr + 5)):
        for c in range(max(0, sc - 4), min(world.n, sc + 5)):
            assert fog.is_cell_discovered(r, c) is True

    # Check a distant corner is not discovered
    corner_r, corner_c = (0, 0) if (sr > 5 or sc > 5) else (world.n - 1, world.n - 1)
    if max(abs(corner_r - sr), abs(corner_c - sc)) > 4:
        assert fog.is_cell_discovered(corner_r, corner_c) is False


def test_fog_compute_percept():
    instance = load_instance()
    world = GridWorld(instance)
    fog = FogTwist(radius=4)
    fog.initialize(instance, world)

    # Initial explored count
    count_init = np.sum(fog.discovered_mask)

    # Move to a new position far enough to reveal new cells
    new_pos = (instance.delivery_stops[0][0], instance.delivery_stops[0][1])
    mask, has_new = fog.compute_percept(new_pos, world)

    count_after = np.sum(fog.discovered_mask)
    assert count_after >= count_init
    assert mask.shape == (world.n, world.n)


def test_fog_optimistic_costs():
    instance = load_instance()
    world = GridWorld(instance)
    fog = FogTwist(radius=4)
    fog.initialize(instance, world)

    # For any discovered cell, cost in optimistic_cost_map should match world.cost_map
    for r in range(world.n):
        for c in range(world.n):
            if fog.discovered_mask[r, c]:
                assert fog.optimistic_cost_map[r, c] == world.cost_map[r, c]
            else:
                assert fog.optimistic_cost_map[r, c] == 3.0  # optimistic road cost


def test_fog_llm_tool():
    instance = load_instance()
    world = GridWorld(instance)
    fog = FogTwist(radius=4)
    fog.initialize(instance, world)

    spec = fog.get_tool_spec()
    assert spec["name"] == "scan_surroundings"
    assert "r" in spec["parameters"]["required"]

    result = fog.execute_tool({"r": 4, "c": 20}, world, None)
    assert "surrounding_summary" in result
    assert "obstacles" in result["surrounding_summary"]
    assert result["radius"] == 4


def test_fog_render_overlay():
    instance = load_instance()
    world = GridWorld(instance)
    fog = FogTwist(radius=4)
    fog.initialize(instance, world)

    fig, ax = plt.subplots(figsize=(6, 6))
    fog.render_overlay(ax, world, state=instance.start)
    plt.close(fig)
