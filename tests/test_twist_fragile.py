"""Unit and property tests for Third Twist: Fragile Parcel (M2.5)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from delivery.instance import load_instance
from delivery.twists.fragile import FragileParcelTwist
from delivery.world.grid import GridWorld


def test_fragile_parcel_initialization():
    instance = load_instance()
    world = GridWorld(instance)
    twist = FragileParcelTwist()
    twist.initialize(instance, world)

    assert twist.fragile_stop_pos == instance.delivery_stops[0]
    assert twist.extra_state_names() == ("fragile_delivered",)
    assert twist.initial_extra_state(world) == (False,)


def test_fragile_parcel_terrain_restrictions():
    instance = load_instance()
    world = GridWorld(instance)
    twist = FragileParcelTwist()
    twist.initialize(instance, world)

    # Find a mud cell and a ford cell, and road/grass cells
    mud_cells = np.argwhere(world.grid == 2)
    ford_cells = np.argwhere(world.grid == 3)
    road_cells = np.argwhere(world.grid == 0)
    grass_cells = np.argwhere(world.grid == 1)

    assert len(mud_cells) > 0, "No mud cells in grid!"
    assert len(ford_cells) > 0, "No ford cells in grid!"

    mud_pos = tuple(map(int, mud_cells[0]))
    ford_pos = tuple(map(int, ford_cells[0]))
    road_pos = tuple(map(int, road_cells[0]))
    grass_pos = tuple(map(int, grass_cells[0]))

    from_pos = instance.start

    # 1. While fragile parcel is undelivered (False)
    state_undelivered = {"fragile_delivered": False}
    assert twist.is_action_legal(from_pos, 0, mud_pos, state_undelivered, world) is False
    assert twist.is_action_legal(from_pos, 0, ford_pos, state_undelivered, world) is False
    assert twist.is_action_legal(from_pos, 0, road_pos, state_undelivered, world) is True
    assert twist.is_action_legal(from_pos, 0, grass_pos, state_undelivered, world) is True

    # 2. After fragile parcel is delivered (True)
    state_delivered = {"fragile_delivered": True}
    assert twist.is_action_legal(from_pos, 0, mud_pos, state_delivered, world) is True
    assert twist.is_action_legal(from_pos, 0, ford_pos, state_delivered, world) is True
    assert twist.is_action_legal(from_pos, 0, road_pos, state_delivered, world) is True
    assert twist.is_action_legal(from_pos, 0, grass_pos, state_delivered, world) is True


def test_fragile_parcel_state_update():
    instance = load_instance()
    world = GridWorld(instance)
    twist = FragileParcelTwist()
    twist.initialize(instance, world)

    target_pos = twist.fragile_stop_pos
    other_pos = instance.start

    current_state = {"fragile_delivered": False}

    # Moving to other position: stays False
    s1 = twist.update_extra_state(current_state, 0, (0, 0), other_pos, world)
    assert s1["fragile_delivered"] is False

    # Moving to fragile stop: transitions to True
    s2 = twist.update_extra_state(current_state, 0, (0, 0), target_pos, world)
    assert s2["fragile_delivered"] is True

    # Moving away from fragile stop: stays True
    s3 = twist.update_extra_state(s2, 0, target_pos, other_pos, world)
    assert s3["fragile_delivered"] is True


def test_fragile_parcel_llm_tool():
    instance = load_instance()
    world = GridWorld(instance)
    twist = FragileParcelTwist()
    twist.initialize(instance, world)

    spec = twist.get_tool_spec()
    assert spec["name"] == "check_fragile_parcel_status"

    class MockState:
        extra = {"fragile_delivered": False}

    res_undelivered = twist.execute_tool({}, world, MockState())
    assert res_undelivered["fragile_delivered"] is False
    assert "mud" in res_undelivered["restricted_terrains"]

    MockState.extra = {"fragile_delivered": True}
    res_delivered = twist.execute_tool({}, world, MockState())
    assert res_delivered["fragile_delivered"] is True
    assert res_delivered["restricted_terrains"] == []


def test_fragile_parcel_render_overlay():
    instance = load_instance()
    world = GridWorld(instance)
    twist = FragileParcelTwist()
    twist.initialize(instance, world)

    fig, ax = plt.subplots(figsize=(6, 6))
    twist.render_overlay(ax, world)
    plt.close(fig)
