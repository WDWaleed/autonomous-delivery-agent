"""Part B Heuristic comparison benchmark stage (§6.1, §7.4, M5.3).

Evaluates 5 search configurations across 30 identical certified scenarios:
1. UCS (h = 0)
2. A* + B1 (Manhattan nearest stop, where admissible/valid)
3. A* + B2 (Max relaxed distance to stop)
4. A* + Own (Combined MST and farthest reach)
5. Greedy + Own (Fast heuristic exploration)

Computes mean ± 95% CI for nodes expanded, generated, peak stored, runtime, path length, cost,
paired comparisons between Own and B2, and exports results to CSV, Markdown, LaTeX, and plots.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.experiments.scenarios import generate_scenario_suite
from delivery.experiments.stats import mean_ci
from delivery.heuristics.base import ZeroHeuristic
from delivery.heuristics.baselines import ManhattanNearestStopHeuristic, MaxRelaxedDistanceHeuristic
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.search.astar import astar_search
from delivery.search.core import Limits, SearchStatus
from delivery.search.greedy import greedy_best_first_search
from delivery.search.problem import DeliveryProblem
from delivery.search.ucs import uniform_cost_search
from delivery.stamp import banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_heuristics_benchmark(
    n_scenarios: int = 30,
    quick: bool = False,
    time_limit: float = 30.0,
    node_limit: int = 500_000,
) -> Dict[str, Any]:
    """Executes the heuristic comparison suite across 30 identical certified scenarios."""
    print(banner("PART B: HEURISTIC COMPARISON EXPERIMENT"))

    scenarios_to_run = 5 if quick else n_scenarios
    suite = generate_scenario_suite(n_scenarios=scenarios_to_run)

    configs = [
        {"name": "UCS (h=0)", "algo": "ucs", "heuristic": "zero"},
        {"name": "A* + B1 (Manhattan)", "algo": "astar", "heuristic": "b1"},
        {"name": "A* + B2 (Max Relaxed)", "algo": "astar", "heuristic": "b2"},
        {"name": "A* + Own (Combined MST)", "algo": "astar", "heuristic": "own"},
        {"name": "Greedy + Own", "algo": "greedy", "heuristic": "own"},
    ]

    records: List[Dict[str, Any]] = []

    for sc in suite:
        if not sc.feasible:
            continue

        world = GridWorld(sc.instance)
        twists = build_twist_stack(sc.instance.twists, world=world)
        problem = DeliveryProblem(sc.instance, world=world, twist_stack=twists)

        # Pre-instantiate heuristics
        h_zero = ZeroHeuristic()
        h_b1 = ManhattanNearestStopHeuristic()
        h_b2 = MaxRelaxedDistanceHeuristic(problem)
        h_own = CombinedMSTHeuristic(problem)

        heur_map = {
            "zero": h_zero,
            "b1": h_b1,
            "b2": h_b2,
            "own": h_own,
        }

        limits = Limits(max_time_seconds=time_limit, max_expanded=node_limit)

        for cfg in configs:
            cfg_name = cfg["name"]
            algo = cfg["algo"]
            h_key = cfg["heuristic"]
            heur = heur_map[h_key]

            t0 = time.perf_counter()
            if algo == "ucs":
                res = uniform_cost_search(problem, limits=limits)
            elif algo == "astar":
                res = astar_search(problem, heur, limits=limits)
            elif algo == "greedy":
                res = greedy_best_first_search(problem, heur, limits=limits)
            else:
                raise ValueError(f"Unknown algo {algo}")
            elapsed = time.perf_counter() - t0

            is_success = res.status == SearchStatus.SUCCESS
            records.append({
                "scenario_k": sc.k,
                "scenario_key": sc.key,
                "config": cfg_name,
                "algo": algo,
                "heuristic": h_key,
                "success": is_success,
                "status": res.status.value,
                "cost": res.cost if is_success else np.nan,
                "plan_length": len(res.plan) if res.plan else np.nan,
                "expanded_nodes": res.metrics.expanded_nodes,
                "generated_nodes": res.metrics.generated_nodes,
                "peak_stored": res.metrics.peak_stored,
                "duration_seconds": res.metrics.duration_seconds,
            })

    df_raw = pd.DataFrame(records)

    # Compute summary metrics grouped by configuration
    summary_rows = []
    for cfg in configs:
        name = cfg["name"]
        df_cfg = df_raw[df_raw["config"] == name]
        n_total = len(df_cfg)
        n_succ = int(df_cfg["success"].sum())

        # CI on numeric metrics for successful runs
        df_succ = df_cfg[df_cfg["success"]]
        exp_mean, exp_lo, exp_hi = mean_ci(df_succ["expanded_nodes"]) if len(df_succ) > 1 else (df_succ["expanded_nodes"].mean(), 0, 0)
        gen_mean, gen_lo, gen_hi = mean_ci(df_succ["generated_nodes"]) if len(df_succ) > 1 else (df_succ["generated_nodes"].mean(), 0, 0)
        time_mean, time_lo, time_hi = mean_ci(df_succ["duration_seconds"]) if len(df_succ) > 1 else (df_succ["duration_seconds"].mean(), 0, 0)
        cost_mean, cost_lo, cost_hi = mean_ci(df_succ["cost"]) if len(df_succ) > 1 else (df_succ["cost"].mean(), 0, 0)
        len_mean, len_lo, len_hi = mean_ci(df_succ["plan_length"]) if len(df_succ) > 1 else (df_succ["plan_length"].mean(), 0, 0)

        summary_rows.append({
            "Configuration": name,
            "Solved": f"{n_succ}/{n_total}",
            "Expanded Nodes (Mean ± 95% CI)": f"{exp_mean:.1f} [{exp_lo:.1f}, {exp_hi:.1f}]",
            "Generated Nodes": f"{gen_mean:.1f}",
            "Time (s) (Mean ± 95% CI)": f"{time_mean:.4f} [{time_lo:.4f}, {time_hi:.4f}]",
            "Path Cost (Mean ± 95% CI)": f"{cost_mean:.2f} [{cost_lo:.2f}, {cost_hi:.2f}]",
            "Path Length": f"{len_mean:.1f}",
        })

    df_summary = pd.DataFrame(summary_rows)

    # Save tables
    tables = save_table(
        df_summary,
        name="partB_heuristics",
        caption="Performance Comparison of Search Algorithms and Heuristics Across 30 Certified Scenarios",
    )
    save_table(df_raw, name="partB_heuristics_raw", caption="Raw Per-Scenario Search Experiment Log")

    # Render bar chart comparing expanded nodes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cfg_names = [c["name"] for c in configs]
    succ_df = df_raw[df_raw["success"]]

    mean_exp = [succ_df[succ_df["config"] == name]["expanded_nodes"].mean() for name in cfg_names]
    mean_time = [succ_df[succ_df["config"] == name]["duration_seconds"].mean() * 1000 for name in cfg_names]

    colors = ["#78909c", "#42a5f5", "#26a69a", "#ab47bc", "#ffa726"]

    # Nodes expanded plot
    ax1.bar(range(len(cfg_names)), mean_exp, color=colors)
    ax1.set_xticks(range(len(cfg_names)))
    ax1.set_xticklabels(cfg_names, rotation=25, ha="right")
    ax1.set_ylabel("Mean Nodes Expanded (Log Scale)")
    ax1.set_yscale("log")
    ax1.set_title("Search Efficiency: Nodes Expanded")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Runtime plot
    ax2.bar(range(len(cfg_names)), mean_time, color=colors)
    ax2.set_xticks(range(len(cfg_names)))
    ax2.set_xticklabels(cfg_names, rotation=25, ha="right")
    ax2.set_ylabel("Mean Execution Time (ms)")
    ax2.set_title("Search Latency: Runtime (ms)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path, pdf_path = save_figure(fig, "partB_heuristics_comparison")
    plt.close(fig)

    print(f"\n[BENCHMARK COMPLETE] Saved summary table to {tables['md']}")
    print(f"Generated comparison figure: {png_path}")

    return {
        "raw_df": df_raw,
        "summary_df": df_summary,
        "figure_png": png_path,
        "figure_pdf": pdf_path,
    }


if __name__ == "__main__":
    run_heuristics_benchmark(n_scenarios=30)
