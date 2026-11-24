"""Tests for GridWorld representation and vectorized cost map (Milestone M1.1)."""

import numpy as np
import pytest

from delivery.instance import get_canonical_instance
from delivery.world.grid import (
    BLOCKED,
    CARDINAL_MOVES,
    FORD,
    GRASS,
    MUD,
    ROAD,
    TERRAIN_NAMES,
    GridWorld,
)


@pytest.fixture
def canonical_world():
    inst = get_canonical_instance()
    return GridWorld(inst)


def test_grid_world_shapes_and_costs(canonical_world):
    world = canonical_world
    n = world.n

    assert world.grid.shape == (n, n)
    assert world.cost_map.shape == (n, n)
    assert world.entering_cost_map.shape == (n, n)
    assert world.passable.shape == (n, n)
    assert world.neighbor_table.shape == (n * n, 4)
    assert world.move_cost_table.shape == (n * n, 4)

    # Check terrain costs
    costs = world.terrain_costs
    for r in range(n):
        for c in range(n):
            cell_type = world.grid[r, c]
            cost_val = world.cost_map[r, c]
            int_cost = world.entering_cost_map[r, c]
            if cell_type == BLOCKED:
                assert np.isinf(cost_val)
                assert int_cost == -1
                assert not world.passable[r, c]
            else:
                expected_name = TERRAIN_NAMES[cell_type]
                assert cost_val == costs[expected_name]
                assert int_cost == costs[expected_name]
                assert world.passable[r, c]


def test_idx_and_rc_helpers(canonical_world):
    world = canonical_world
    n = world.n

    for r in range(n):
        for c in range(n):
            p = world.idx(r, c)
            assert 0 <= p < n * n
            r_rec, c_rec = world.rc(p)
            assert (r_rec, c_rec) == (r, c)


def test_neighbor_and_move_cost_tables(canonical_world):
    world = canonical_world
    n = world.n
    opp_dir = {0: 2, 1: 3, 2: 0, 3: 1}  # N <-> S, E <-> W

    for r in range(n):
        for c in range(n):
            p = world.idx(r, c)
            for d, (dr, dc) in enumerate(CARDINAL_MOVES):
                dest_p = world.neighbor_table[p, d]
                dest_cost = world.move_cost_table[p, d]

                nr, nc = r + dr, c + dc
                if (
                    not world.passable[r, c]
                    or not (0 <= nr < n and 0 <= nc < n)
                    or not world.passable[nr, nc]
                ):
                    assert dest_p == -1
                    assert dest_cost == -1
                else:
                    assert dest_p == world.idx(nr, nc)
                    assert dest_cost == world.entering_cost_map[nr, nc]
                    # Symmetry on base undirected grid
                    assert world.neighbor_table[dest_p, opp_dir[d]] == p


def test_hand_constructed_reference_comparison():
    """Validates GridWorld arrays against an explicit 3x3 synthetic instance."""
    from delivery.instance import Instance

    grid = np.array([
        [0, 1, -1],
        [2, 3,  0],
        [-1, 0, 1]
    ], dtype=np.int8)

    inst = Instance(
        id="test3x3",
        n=3,
        grid=grid,
        terrain_cost={"road": 2, "grass": 4, "mud": 6, "ford": 8},
        obstacle_density=2 / 9,
        start=(0, 0),
        delivery_stops=[(1, 2)],
        adversary="chaser",
        slip_probability=0.2,
        adversary_search_depth=3,
        twists=["fog_partial_observability", "one_way_streets"],
        seed=12345,
        cms_ids=["TEST1"],
    )

    world = GridWorld(inst)

    # Expected entering cost map:
    # [ 2,  4, -1]
    # [ 6,  8,  2]
    # [-1,  2,  4]
    expected_entering = np.array([
        [ 2,  4, -1],
        [ 6,  8,  2],
        [-1,  2,  4]
    ], dtype=np.int32)

    assert np.array_equal(world.entering_cost_map, expected_entering)

    # Cell (0,0): East to (0,1) cost 4; South to (1,0) cost 6; North and West off-grid (-1)
    p00 = world.idx(0, 0)
    assert world.neighbor_table[p00, 0] == -1  # North
    assert world.neighbor_table[p00, 1] == world.idx(0, 1)  # East
    assert world.neighbor_table[p00, 2] == world.idx(1, 0)  # South
    assert world.neighbor_table[p00, 3] == -1  # West

    assert world.move_cost_table[p00, 1] == 4
    assert world.move_cost_table[p00, 2] == 6
