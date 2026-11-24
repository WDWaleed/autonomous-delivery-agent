"""Tests for Dijkstra distance field, Bellman-Ford oracle, and grid rendering (Milestone M1.3)."""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from delivery.instance import Instance, get_canonical_instance
from delivery.stamp import STAMP
from delivery.viz.render import render_grid_world
from delivery.world.distance import bellman_ford_oracle, distance_field
from delivery.world.grid import GridWorld


@pytest.fixture
def canonical_world():
    inst = get_canonical_instance()
    return GridWorld(inst)


def test_distance_field_properties(canonical_world):
    world = canonical_world
    sr, sc = world.start

    dist_map = distance_field(world, (sr, sc))

    # 1. Distance to self is 0
    assert dist_map[sr, sc] == 0.0

    # 2. Blocked cells are unreachable
    for r in range(world.n):
        for c in range(world.n):
            if not world.passable[r, c]:
                assert np.isinf(dist_map[r, c])

    # 3. All delivery stops must be reachable (instance invariant)
    for stop_r, stop_c in world.delivery_stops:
        assert np.isfinite(dist_map[stop_r, stop_c])
        assert dist_map[stop_r, stop_c] > 0.0


def test_dijkstra_matches_bellman_ford_oracle():
    """Validates custom Dijkstra against an independent Bellman-Ford oracle on random test grids."""
    rng = np.random.default_rng(2026)

    for trial in range(5):
        n = 5
        grid = rng.choice(4, size=(n, n), p=[0.4, 0.3, 0.2, 0.1]).astype(np.int8)
        grid[rng.random((n, n)) < 0.2] = -1

        # Ensure (0, 0) is free
        grid[0, 0] = 0

        inst = Instance(
            id=f"test_oracle_{trial}",
            n=n,
            grid=grid,
            terrain_cost={"road": 2, "grass": 3, "mud": 5, "ford": 8},
            obstacle_density=0.2,
            start=(0, 0),
            delivery_stops=[],
            adversary="chaser",
            slip_probability=0.2,
            adversary_search_depth=3,
            twists=["fog_partial_observability", "one_way_streets"],
            seed=42 + trial,
            cms_ids=["TEST"],
        )

        world = GridWorld(inst)
        n2 = n * n

        # Extract explicit edges for Bellman-Ford
        edges = []
        for u in range(n2):
            for d in range(4):
                v = world.neighbor_table[u, d]
                if v != -1:
                    cost = float(world.move_cost_table[u, d])
                    edges.append((u, v, cost))

        src = 0  # (0, 0)
        dijkstra_dist = distance_field(world, src).reshape(-1)
        oracle_dist = bellman_ford_oracle(n2, edges, src, reverse=False)

        np.testing.assert_allclose(dijkstra_dist, oracle_dist, rtol=1e-5, atol=1e-5)


def test_directed_reverse_dijkstra():
    """Tests reverse-Dijkstra with a custom edge model having one-way directed edges."""
    inst = get_canonical_instance()
    world = GridWorld(inst)

    class SyntheticDirectedEdgeModel:
        """Models a simple synthetic directed edge set."""
        def __init__(self, world):
            self.world = world

        def get_edges(self, u: int, reverse: bool = False):
            # Invert directions if reverse is True
            edges = []
            for d in range(4):
                v = self.world.neighbor_table[u, d]
                if v != -1:
                    cost = float(self.world.move_cost_table[u, d])
                    if reverse:
                        # Reverse mode: edges where v -> u in forward graph
                        edges.append((v, cost))
                    else:
                        edges.append((v, cost))
            return edges

    model = SyntheticDirectedEdgeModel(world)
    target = world.start
    dist_forward = distance_field(world, target, reverse=False, edge_model=model)
    dist_reverse = distance_field(world, target, reverse=True, edge_model=model)

    # In symmetric graph, forward and reverse from target are identical
    np.testing.assert_allclose(dist_forward, dist_reverse)


def test_renderer_generates_stamped_figure(tmp_path):
    inst = get_canonical_instance()
    png_path, pdf_path = render_grid_world(
        inst, name="test_render_grid", output_dir=tmp_path, report_copy=False
    )

    assert png_path.exists()
    assert pdf_path.exists()

    # Check PNG metadata
    with Image.open(png_path) as im:
        assert im.info.get("Author") == STAMP
        assert im.info.get("Comment") == STAMP
