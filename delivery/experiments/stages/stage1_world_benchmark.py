"""Stage 1 Experiment: Vectorized vs Pure-Python Loop Cost Map Benchmark (Milestone M1.2).

Compares pure-Python nested loops against NumPy vectorized construction on grids
n in {20, 30, 40, 60, 80}. Reports runtime, speedup, and 95% confidence intervals.
Governed by decisions D9 and D19.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from delivery.stamp import banner, save_figure, save_table
from delivery.world.grid import GridWorld
from delivery.world.reference_loops import (
    build_cost_map_loops,
    build_move_cost_table_loops,
    build_passable_mask_loops,
)


def generate_benchmark_grid(n: int, seed: int = 42) -> Tuple[np.ndarray, Dict[str, int]]:
    """Generates a representative grid of size n with terrain distribution matching the lab."""
    rng = np.random.default_rng(seed)
    costs = {"road": 3, "grass": 4, "mud": 5, "ford": 7}
    grid = rng.choice(4, size=(n, n), p=[0.4, 0.3, 0.2, 0.1]).astype(np.int8)
    grid[rng.random((n, n)) < 0.25] = -1
    return grid, costs


def time_loop_construction(grid_list: List[List[int]], costs: Dict[str, int], iterations: int) -> List[float]:
    """Measures runtimes of pure-Python loop construction per iteration."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = build_cost_map_loops(grid_list, costs)
        _ = build_passable_mask_loops(grid_list)
        _ = build_move_cost_table_loops(grid_list, costs)
        times.append(time.perf_counter() - t0)
    return times


def time_vectorized_construction(instance: Any, iterations: int) -> List[float]:
    """Measures runtimes of vectorized NumPy construction per iteration."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = GridWorld(instance)
        times.append(time.perf_counter() - t0)
    return times


def compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Computes sample mean and half-width of Student-t confidence interval (D9)."""
    a = np.array(data)
    n = len(a)
    mean = float(np.mean(a))
    if n < 2:
        return mean, 0.0
    se = stats.sem(a)
    h = float(se * stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, h


def run_stage1(
    quick: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes the Stage 1 benchmark and saves stamped figures and tables."""
    print(banner("STAGE 1: WORLD BENCHMARK (VECTORIZATION SPEEDUP)"))

    # Sizes: [20, 30, instance n (22), 200, 1000]
    grid_sizes = [20, 22, 30] if quick else [20, 22, 30, 200, 1000]

    print(f"Running benchmark across grid sizes {grid_sizes} (quick={quick})...")

    records = []

    for idx, n in enumerate(grid_sizes):
        # Ordinary sizes: 30-50 repeats; large size 1000: 5 repeats to keep runtime reasonable
        if n >= 1000:
            iterations = 3 if quick else 5
        elif n >= 200:
            iterations = 10 if quick else 20
        else:
            iterations = 20 if quick else 50

        grid_np, costs = generate_benchmark_grid(n, seed=1000 + idx)
        grid_list = grid_np.tolist()

        class SyntheticInstance:
            def __init__(self, n, grid, costs):
                self.n = n
                self.grid = grid
                self.start = (0, 0)
                self.delivery_stops = []
                self.terrain_cost = costs

        syn_inst = SyntheticInstance(n, grid_np, costs)

        # Warm-up run
        _ = build_cost_map_loops(grid_list, costs)
        _ = GridWorld(syn_inst)

        # 1. Measure pure-Python loops
        loop_times = time_loop_construction(grid_list, costs, iterations)
        loop_mean, loop_ci = compute_ci(loop_times)

        # 2. Measure NumPy vectorized
        vec_times = time_vectorized_construction(syn_inst, iterations)
        vec_mean, vec_ci = compute_ci(vec_times)

        # 3. Speedup calculation
        speedup = loop_mean / vec_mean if vec_mean > 0 else 1.0

        records.append({
            "Grid Size (N)": n,
            "Total Cells": n * n,
            "Loop Mean (ms)": round(loop_mean * 1000, 3),
            "Loop 95% CI (ms)": round(loop_ci * 1000, 3),
            "Vectorized Mean (ms)": round(vec_mean * 1000, 3),
            "Vec 95% CI (ms)": round(vec_ci * 1000, 3),
            "Speedup (x)": round(speedup, 2),
        })

        print(f"  N={n:4d} ({n*n:7d} cells): Loop={loop_mean*1000:8.2f}ms | Vec={vec_mean*1000:7.2f}ms | Speedup={speedup:5.1f}x")

    df = pd.DataFrame(records)

    # Save stamped table under both stage1_world_speedup and partA_speedup names
    caption = "Runtime Comparison and Speedup Factor: Pure-Python Loop vs NumPy Vectorized World Construction"
    table_files = save_table(df, "stage1_world_speedup", caption=caption)
    save_table(df, "partA_speedup", caption=caption)
    print(f"Saved tables: {table_files['csv'].name} and partA_speedup.csv")

    # Generate and save stamped figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ns = df["Grid Size (N)"]
    # Plot 1: Runtimes in ms
    ax1.errorbar(
        ns,
        df["Loop Mean (ms)"],
        yerr=df["Loop 95% CI (ms)"],
        fmt="r-o",
        label="Pure-Python Loops",
        capsize=4,
        linewidth=2,
    )
    ax1.errorbar(
        ns,
        df["Vectorized Mean (ms)"],
        yerr=df["Vec 95% CI (ms)"],
        fmt="b-s",
        label="NumPy Vectorized",
        capsize=4,
        linewidth=2,
    )
    ax1.set_xlabel("Grid Dimension N (N x N cells)")
    ax1.set_ylabel("Execution Time (milliseconds)")
    ax1.set_title("Construction Runtime vs Grid Size (Log Scale)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2: Speedup Factor
    ax2.plot(ns, df["Speedup (x)"], "g-^", linewidth=2.5, markersize=8)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Parity (1.0x)")
    for _, row in df.iterrows():
        ax2.annotate(
            f"{row['Speedup (x)']:.1f}x",
            (row["Grid Size (N)"], row["Speedup (x)"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            fontsize=8,
        )
    ax2.set_xlabel("Grid Dimension N (N x N cells)")
    ax2.set_ylabel("Speedup Multiplier (Loop / Vectorized)")
    ax2.set_title("Vectorization Speedup Factor (Log-X)")
    ax2.set_xscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    png_path, pdf_path = save_figure(fig, "stage1_world_speedup")
    save_figure(fig, "partA_speedup")
    plt.close(fig)
    print(f"Saved figures: {png_path.name} and partA_speedup.png")

    return {
        "status": "COMPLETED",
        "grid_sizes": grid_sizes,
        "results": records,
        "max_speedup": float(df["Speedup (x)"].max()),
    }


if __name__ == "__main__":
    run_stage1(quick=True)
