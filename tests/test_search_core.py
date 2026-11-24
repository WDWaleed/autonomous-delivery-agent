"""Unit and property tests for Search Core, Nodes, Limits, and Canonical Plan (M4.2)."""

import numpy as np
import pytest

from delivery.instance import Instance, load_instance
from delivery.search.core import (
    Limits,
    Metrics,
    Node,
    SearchResult,
    SearchStatus,
    canonical_optimal_plan,
    reconstruct_path,
)
from delivery.search.estimate import estimate_state_space_bound
from delivery.search.problem import EAST, NORTH, SOUTH, WAIT, WEST, DeliveryProblem


def make_test_instance_3x3() -> Instance:
    grid = np.zeros((3, 3), dtype=np.int8)  # all road
    return Instance(
        id="core_test_3x3",
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
        seed=42,
        cms_ids=["TEST01"],
    )


def test_node_slots_and_child():
    root = Node.root((0, 0, 0, ()))
    assert root.cost == 0.0
    assert root.depth == 0
    assert root.parent is None
    assert root.action is None

    child = root.child((0, 1, 0, ()), action=EAST, step_cost=3.0)
    assert child.cost == 3.0
    assert child.depth == 1
    assert child.action == EAST
    assert child.parent is root
    assert child.action_seq == (EAST,)

    # Positive cost enforcement (D15)
    with pytest.raises(ValueError, match="strictly positive"):
        root.child((0, 1, 0, ()), action=EAST, step_cost=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        root.child((0, 1, 0, ()), action=EAST, step_cost=-2.0)


def test_state_space_bound_estimator():
    inst = make_test_instance_3x3()
    prob = DeliveryProblem(inst)
    res = estimate_state_space_bound(prob)

    assert res["passable_cells"] == 9
    assert res["delivery_stops"] == 2
    assert res["delivery_subsets"] == 4
    # Upper bound = 9 * 4 = 36
    assert res["total_upper_bound"] == 36
    assert "disclaimer" in res


def test_canonical_optimal_plan_success():
    inst = make_test_instance_3x3()
    prob = DeliveryProblem(inst)
    res = canonical_optimal_plan(prob)

    assert res.status == SearchStatus.SUCCESS
    assert res.length == 4  # (0,0) -> (0,1) -> (0,2) [stop 0] -> (1,2) -> (2,2) [stop 1]
    assert res.cost == 12.0  # 4 steps * 3.0
    assert res.plan == [EAST, EAST, SOUTH, SOUTH]
    assert len(res.trajectory) == 5
    assert res.metrics.expanded_nodes > 0


def test_limits_nodelimit():
    inst = make_test_instance_3x3()
    prob = DeliveryProblem(inst)
    # Exceedingly small node expansion limit
    limits = Limits(max_expanded=2)
    res = canonical_optimal_plan(prob, limits=limits)

    assert res.status == SearchStatus.NODELIMIT
    assert res.status != SearchStatus.FAILED
    assert "Expanded node limit" in (res.failure_reason or "")


def test_limits_stored_records_memorylimit():
    inst = make_test_instance_3x3()
    prob = DeliveryProblem(inst)
    # Exceedingly small stored record budget
    limits = Limits(max_stored_records=3)
    res = canonical_optimal_plan(prob, limits=limits)

    assert res.status == SearchStatus.MEMORYLIMIT
    assert res.status != SearchStatus.FAILED
    assert "Stored record budget" in (res.failure_reason or "")


def test_lexicographic_action_sequence_tie_breaking():
    """Verifies that between two paths with equal cost and equal length, fixed cardinal action order breaks tie."""
    # From (0,0) to reach (1,1):
    # Path 1: EAST then SOUTH: action_seq = (1, 2)
    # Path 2: SOUTH then EAST: action_seq = (2, 1)
    # Since EAST (1) < SOUTH (2), Path 1 must be strictly preferred!
    grid = np.zeros((2, 2), dtype=np.int8)
    inst = Instance(
        id="tie_break_2x2",
        n=2,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.0,
        start=(0, 0),
        delivery_stops=[(1, 1)],
        adversary="chaser",
        slip_probability=0.0,
        adversary_search_depth=1,
        twists=[],
        seed=999,
        cms_ids=["TEST03"],
    )
    prob = DeliveryProblem(inst)
    res = canonical_optimal_plan(prob)

    assert res.status == SearchStatus.SUCCESS
    assert res.length == 2
    # In cardinal order: NORTH(0), EAST(1), SOUTH(2), WEST(3)
    # From (0,0), NORTH and WEST are invalid, so only EAST(1) and SOUTH(2) are available
    # EAST then SOUTH has seq (1, 2); SOUTH then EAST has seq (2, 1)
    # Lexicographically, (1, 2) < (2, 1), so [EAST, SOUTH] must win!
    assert res.plan == [EAST, SOUTH]
