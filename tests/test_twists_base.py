"""Unit tests for Twist framework, base stack, and registry (§7.2, M2.1)."""

import pytest

from delivery.instance import load_instance
from delivery.twists.base import BaseTwist, RelaxedGraph, TwistStack
from delivery.twists.protocols import (
    CostModifier,
    LLMToolProvider,
    PerceptionModifier,
    RelaxationProvider,
    Renderer,
    StateExtension,
    TransitionModifier,
    Twist,
)
from delivery.twists.registry import (
    TWIST_REGISTRY,
    build_twist_stack,
    get_twist_class,
    list_available_twists,
)
from delivery.world.grid import GridWorld


def test_registry_contains_all_canonical_twists():
    available = list_available_twists()
    assert len(available) == 9
    assert "fog_partial_observability" in available
    assert "one_way_streets" in available
    assert "fragile_parcel" in available
    assert "fuel_limit_with_charging_stations" in available


def test_empty_twist_stack():
    stack = TwistStack([])
    assert stack.extra_names == ()
    assert stack.initial_extra_state(None) == ()
    assert stack.unpack_extra(()) == {}
    assert stack.pack_extra({}) == ()

    # Actions allowed by default
    assert stack.is_action_legal((0, 0), 1, (0, 1), (), None) is True
    assert stack.compute_step_cost((0, 0), 1, (0, 1), 5.0, (), None) == 5.0
    assert stack.get_llm_tool_specs() == []


def test_build_twist_stack_from_instance():
    instance = load_instance()
    world = GridWorld(instance)
    stack = build_twist_stack(instance, world)

    # Instance 432382d9 has fog and one_way_streets
    twist_names = [t.name for t in stack.twists]
    assert "fog_partial_observability" in twist_names
    assert "one_way_streets" in twist_names

    # Check tools collected
    tools = stack.get_llm_tool_specs()
    tool_names = [t["name"] for t in tools]
    assert "scan_surroundings" in tool_names
    assert "check_street_direction" in tool_names


def test_twist_stack_with_third_twist():
    instance = load_instance()
    world = GridWorld(instance)
    stack = build_twist_stack(instance, world, enable_third_twist=True)

    twist_names = [t.name for t in stack.twists]
    assert "fragile_parcel" in twist_names
    assert "fragile_delivered" in stack.extra_names

    init_state = stack.initial_extra_state(world)
    assert init_state == (False,)
    unpacked = stack.unpack_extra(init_state)
    assert unpacked == {"fragile_delivered": False}
    repacked = stack.pack_extra(unpacked)
    assert repacked == init_state
