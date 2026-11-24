"""Unit and property tests for One-Way Streets Twist (M2.3)."""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pytest

from delivery.instance import load_instance
from delivery.twists.one_way import OPPOSITE, OneWayTwist
from delivery.world.grid import GridWorld


def test_one_way_initialization_and_directionality():
    instance = load_instance()
    world = GridWorld(instance)
    one_way = OneWayTwist(directed_ratio=0.25)
    one_way.initialize(instance, world)

    assert len(one_way.directed_edges) > 0
    # For every directed edge (u -> v via d), u->v is allowed and v->u is NOT allowed
    for u, v, d in one_way.directed_edges:
        assert bool(one_way.allowed[u, d]) is True
        d_opp = OPPOSITE[d]
        assert bool(one_way.allowed[v, d_opp]) is False


def test_one_way_reachability_of_all_targets():
    """Verifies all delivery targets remain mutually reachable in the directed graph."""
    instance = load_instance()
    world = GridWorld(instance)
    one_way = OneWayTwist(directed_ratio=0.25)
    one_way.initialize(instance, world)

    targets = [world.idx(*instance.start)] + [world.idx(*p) for p in instance.delivery_stops]

    for src in targets:
        visited = set()
        queue = deque([src])
        visited.add(src)

        while queue:
            curr = queue.popleft()
            for d in range(4):
                if one_way.allowed[curr, d]:
                    nxt = world.neighbor_table[curr, d]
                    if nxt != -1 and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)

        for tgt in targets:
            assert tgt in visited, f"Target {world.rc(tgt)} unreachable from {world.rc(src)}!"


def test_one_way_is_action_legal():
    instance = load_instance()
    world = GridWorld(instance)
    one_way = OneWayTwist(directed_ratio=0.25)
    one_way.initialize(instance, world)

    # Pick a directed edge u -> v
    u, v, d = one_way.directed_edges[0]
    pos_u = world.rc(u)
    pos_v = world.rc(v)

    # Moving u -> v is legal
    assert one_way.is_action_legal(pos_u, d, pos_v, {}, world) is True

    # Moving v -> u is illegal
    d_opp = OPPOSITE[d]
    assert one_way.is_action_legal(pos_v, d_opp, pos_u, {}, world) is False


def test_one_way_llm_tool():
    instance = load_instance()
    world = GridWorld(instance)
    one_way = OneWayTwist(directed_ratio=0.25)
    one_way.initialize(instance, world)

    spec = one_way.get_tool_spec()
    assert spec["name"] == "check_street_direction"

    # Query directed edge
    u, v, d = one_way.directed_edges[0]
    pos_u = world.rc(u)
    dir_name = ["N", "E", "S", "W"][d]

    res = one_way.execute_tool(
        {"from_r": pos_u[0], "from_c": pos_u[1], "direction": dir_name},
        world,
        None,
    )
    assert res["status"] == "one_way_forward"
    assert res["allowed"] is True

    # Query opposite direction
    pos_v = world.rc(v)
    opp_name = ["N", "E", "S", "W"][OPPOSITE[d]]
    res_opp = one_way.execute_tool(
        {"from_r": pos_v[0], "from_c": pos_v[1], "direction": opp_name},
        world,
        None,
    )
    assert res_opp["status"] == "one_way_wrong_way"
    assert res_opp["allowed"] is False


def test_one_way_render_overlay():
    instance = load_instance()
    world = GridWorld(instance)
    one_way = OneWayTwist(directed_ratio=0.25)
    one_way.initialize(instance, world)

    fig, ax = plt.subplots(figsize=(6, 6))
    one_way.render_overlay(ax, world)
    plt.close(fig)
