"""Unit and property tests for Twist Interplay and Relaxed Graph Lower Bounds (M2.4)."""

from typing import List, Tuple

import numpy as np
import pytest

from delivery.instance import load_instance
from delivery.twists.one_way import OneWayTwist
from delivery.twists.registry import build_twist_stack
from delivery.world.distance import distance_field
from delivery.world.grid import GridWorld


class DirectedEdgeModel:
    """Actual directed edge model obeying one-way streets."""

    def __init__(self, world: GridWorld, one_way: OneWayTwist):
        self.world = world
        self.one_way = one_way

    def get_edges(self, u: int, reverse: bool = False) -> List[Tuple[int, float]]:
        edges = []
        for d in range(4):
            if self.one_way.allowed[u, d]:
                v = self.world.neighbor_table[u, d]
                if v != -1:
                    cost = float(self.world.move_cost_table[u, d])
                    edges.append((v, cost))
        return edges


def test_twist_interplay_stack_composition():
    instance = load_instance()
    world = GridWorld(instance)
    stack = build_twist_stack(instance, world)

    assert len(stack.twists) >= 2
    # Check that tools from both twists are registered
    tool_names = [s["name"] for s in stack.get_llm_tool_specs()]
    assert "scan_surroundings" in tool_names
    assert "check_street_direction" in tool_names


def test_relaxed_graph_lower_bound_property():
    """Property test verifying that d_relaxed(u, v) <= d_actual(u, v) for all reachable pairs."""
    instance = load_instance()
    world = GridWorld(instance)
    stack = build_twist_stack(instance, world)

    one_way_twist = next(t for t in stack.twists if t.name == "one_way_streets")
    directed_model = DirectedEdgeModel(world, one_way_twist)
    relaxed_graph = stack.relaxed_graph(world)

    # Test from multiple sources: start and delivery stops
    test_sources = [instance.start] + instance.delivery_stops

    for src in test_sources:
        d_actual = distance_field(world, src, edge_model=directed_model)
        d_relaxed = distance_field(world, src, edge_model=relaxed_graph)

        # For every cell reachable in the actual directed world:
        reachable_mask = np.isfinite(d_actual)
        assert np.any(reachable_mask), f"Source {src} has no reachable nodes!"

        # Lower bound invariant: d_relaxed <= d_actual
        diff = d_relaxed[reachable_mask] - d_actual[reachable_mask]
        max_violation = np.max(diff)
        assert (
            max_violation <= 1e-9
        ), f"Relaxed distance exceeds actual distance by {max_violation} at source {src}!"


def test_symmetric_mst_lower_bound():
    """Verifies that min(d(u, v), d(v, u)) on relaxed graph is symmetric and lower bounds directed tour costs."""
    instance = load_instance()
    world = GridWorld(instance)
    stack = build_twist_stack(instance, world)

    one_way_twist = next(t for t in stack.twists if t.name == "one_way_streets")
    directed_model = DirectedEdgeModel(world, one_way_twist)
    relaxed_graph = stack.relaxed_graph(world)

    targets = [instance.start] + instance.delivery_stops[:3]

    for i, u in enumerate(targets):
        d_actual_u = distance_field(world, u, edge_model=directed_model)
        d_relaxed_u = distance_field(world, u, edge_model=relaxed_graph)

        for j, v in enumerate(targets):
            if i >= j:
                continue

            vr, vc = v
            actual_forward = d_actual_u[vr, vc]

            d_actual_v = distance_field(world, v, edge_model=directed_model)
            d_relaxed_v = distance_field(world, v, edge_model=relaxed_graph)
            actual_backward = d_actual_v[u[0], u[1]]

            relaxed_forward = d_relaxed_u[vr, vc]
            relaxed_backward = d_relaxed_v[u[0], u[1]]

            # Relaxed symmetric edge weight
            w_relaxed = min(relaxed_forward, relaxed_backward)
            w_actual = min(actual_forward, actual_backward)

            assert (
                w_relaxed <= w_actual + 1e-9
            ), f"Symmetric relaxation violated between {u} and {v}: {w_relaxed} > {w_actual}"
