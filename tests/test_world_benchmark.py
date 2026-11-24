"""Tests for pure-Python loop reference and Stage 1 vectorization speedup benchmark (Milestone M1.2)."""

from pathlib import Path
import numpy as np
import pytest

from delivery.experiments.stages.stage1_world_benchmark import run_stage1
from delivery.instance import get_canonical_instance
from delivery.world.grid import GridWorld
from delivery.world.reference_loops import (
    build_cost_map_loops,
    build_move_cost_table_loops,
    build_passable_mask_loops,
)


def test_exact_equality_loop_vs_vectorized():
    """Asserts that pure-Python loop reference outputs match vectorized NumPy outputs identically."""
    inst = get_canonical_instance()
    world = GridWorld(inst)
    grid_list = inst.grid.tolist()

    # 1. Compare cost map
    loop_costs = build_cost_map_loops(grid_list, inst.terrain_cost)
    loop_costs_arr = np.array(loop_costs, dtype=np.float64)
    # Both should have inf for blocked cells, exact floats elsewhere
    np.testing.assert_allclose(loop_costs_arr, world.cost_map)

    # 2. Compare passable mask
    loop_passable = build_passable_mask_loops(grid_list)
    loop_passable_arr = np.array(loop_passable, dtype=bool)
    assert np.array_equal(loop_passable_arr, world.passable)

    # 3. Compare neighbor and move cost tables
    loop_nbr, loop_move_cost = build_move_cost_table_loops(grid_list, inst.terrain_cost)
    loop_nbr_arr = np.array(loop_nbr, dtype=np.int32)
    loop_cost_arr = np.array(loop_move_cost, dtype=np.int32)

    assert np.array_equal(loop_nbr_arr, world.neighbor_table)
    assert np.array_equal(loop_cost_arr, world.move_cost_table)


def test_stage1_benchmark_execution_and_speedup():
    """Tests Stage 1 execution, speedup (>1.0), and artifact generation."""
    res = run_stage1(quick=True)

    assert res["status"] == "COMPLETED"
    assert res["max_speedup"] > 1.0  # Vectorized is strictly faster than pure Python

    # Verify generated outputs exist
    table_csv = Path("results/tables/stage1_world_speedup.csv")
    fig_png = Path("results/figures/stage1_world_speedup.png")
    fig_pdf = Path("results/figures/stage1_world_speedup.pdf")

    assert table_csv.exists()
    assert fig_png.exists()
    assert fig_pdf.exists()

    # Check stamp in CSV header
    with open(table_csv, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        assert first_line.startswith("# ")
        assert "513303" in first_line
