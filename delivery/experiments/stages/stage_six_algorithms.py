"""Part B Full Six-Algorithm Benchmark Stage (§6.1, §7.4, M5.6).

Evaluates all six canonical search algorithms across 30 certified scenarios:
1. BFS (Graph search, step-count optimal)
2. UCS (Graph search, path-cost optimal)
3. DLS (Depth-limited below optimal depth: d = 10)
4. DLS (Depth-limited at/above optimal depth: d = 25)
5. IDS (Iterative deepening search, step-count optimal, linear stack)
6. Greedy (Greedy best-first with Own Combined MST)
7. A* (A* graph search with Own Combined MST, path-cost optimal)

Accurately measures:
- Solved ratio & DNF/limit breakdown
- Expanded nodes (Mean +- 95% CI)
- Generated nodes
- Peak stored records
- Execution duration seconds (Mean +- 95% CI)
- Solution path length (step-count)
- Solution path cost
- Discussion of optimality objectives (cost-optimality vs move-optimality)
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
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.search.astar import astar_search
from delivery.search.bfs import breadth_first_search
from delivery.search.core import Limits, SearchStatus
from delivery.search.dls import depth_limited_search
from delivery.search.greedy import greedy_best_first_search
from delivery.search.ids import iterative_deepening_search
from delivery.search.problem import DeliveryProblem
from delivery.search.ucs import uniform_cost_search
from delivery.stamp import banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_full_algorithms_benchmark(
    n_scenarios: int = 30,
    quick: bool = False,
    time_limit: float = 15.0,
    node_limit: int = 200_000,
) -> Dict[str, Any]:
    """Runs the six-algorithm evaluation suite across certified scenarios."""
    print(banner("PART B: SIX SEARCH ALGORITHMS COMPARISON BENCHMARK (M5.6)"))

    scenarios_to_run = 5 if quick else n_scenarios
    suite = generate_scenario_suite(n_scenarios=scenarios_to_run)

    configs = [
        {"name": "BFS", "algo": "bfs", "depth": None, "objective": "Move-Optimal (min steps)"},
        {"name": "UCS", "algo": "ucs", "depth": None, "objective": "Cost-Optimal (min cost)"},
        {"name": "DLS (d=10, Below)", "algo": "dls_low", "depth": 10, "objective": "Bounded Depth Search"},
        {"name": "DLS (d=25, Above)", "algo": "dls_high", "depth": 25, "objective": "Bounded Depth Search"},
        {"name": "IDS", "algo": "ids", "depth": 25, "objective": "Move-Optimal (min steps, linear space)"},
        {"name": "Greedy Best-First", "algo": "greedy", "depth": None, "objective": "Heuristic Speed (no guarantee)"},
        {"name": "A* (Own Heuristic)", "algo": "astar", "depth": None, "objective": "Cost-Optimal (min cost)"},
    ]

    records: List[Dict[str, Any]] = []

    for sc in suite:
        if not sc.feasible:
            continue

        world = GridWorld(sc.instance)
        twists = build_twist_stack(sc.instance.twists, world=world)
        problem = DeliveryProblem(sc.instance, world=world, twist_stack=twists)

        h_own = CombinedMSTHeuristic(problem)
        limits = Limits(max_time_seconds=time_limit, max_expanded=node_limit)

        for cfg in configs:
            name = cfg["name"]
            algo = cfg["algo"]
            d_limit = cfg["depth"]

            t0 = time.perf_counter()
            if algo == "bfs":
                res = breadth_first_search(problem, limits=limits)
            elif algo == "ucs":
                res = uniform_cost_search(problem, limits=limits)
            elif algo == "dls_low":
                res = depth_limited_search(problem, depth_limit=10, limits=limits)
            elif algo == "dls_high":
                res = depth_limited_search(problem, depth_limit=25, limits=limits)
            elif algo == "ids":
                res = iterative_deepening_search(problem, max_depth=25, limits=limits)
            elif algo == "greedy":
                res = greedy_best_first_search(problem, heuristic=h_own, limits=limits)
            elif algo == "astar":
                res = astar_search(problem, heuristic=h_own, limits=limits)
            else:
                raise ValueError(f"Unknown algo {algo}")

            elapsed = time.perf_counter() - t0
            is_success = res.status == SearchStatus.SUCCESS

            records.append({
                "scenario_k": sc.k,
                "scenario_key": sc.key,
                "config": name,
                "algo": algo,
                "objective": cfg["objective"],
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

    # Summary table
    summary_rows = []
    for cfg in configs:
        name = cfg["name"]
        df_cfg = df_raw[df_raw["config"] == name]
        n_tot = len(df_cfg)
        n_suc = int(df_cfg["success"].sum())

        df_succ = df_cfg[df_cfg["success"]]
        if len(df_succ) > 0:
            exp_mean, exp_lo, exp_hi = mean_ci(df_succ["expanded_nodes"]) if len(df_succ) > 1 else (df_succ["expanded_nodes"].mean(), 0, 0)
            time_mean, time_lo, time_hi = mean_ci(df_succ["duration_seconds"]) if len(df_succ) > 1 else (df_succ["duration_seconds"].mean(), 0, 0)
            cost_mean, cost_lo, cost_hi = mean_ci(df_succ["cost"]) if len(df_succ) > 1 else (df_succ["cost"].mean(), 0, 0)
            len_mean, len_lo, len_hi = mean_ci(df_succ["plan_length"]) if len(df_succ) > 1 else (df_succ["plan_length"].mean(), 0, 0)
            stored_mean = df_succ["peak_stored"].mean()
        else:
            exp_mean, exp_lo, exp_hi = np.nan, 0, 0
            time_mean, time_lo, time_hi = np.nan, 0, 0
            cost_mean, cost_lo, cost_hi = np.nan, 0, 0
            len_mean, len_lo, len_hi = np.nan, 0, 0
            stored_mean = df_cfg["peak_stored"].mean()

        # DNF breakdown string
        dnf_counts = df_cfg[~df_cfg["success"]]["status"].value_counts().to_dict()
        dnf_str = ", ".join(f"{k}:{v}" for k, v in dnf_counts.items()) if dnf_counts else "None"

        summary_rows.append({
            "Algorithm": name,
            "Optimality Guarantee": cfg["objective"],
            "Solved Ratio": f"{n_suc}/{n_tot}",
            "DNF / Limits": dnf_str,
            "Mean Expansions (95% CI)": f"{exp_mean:.1f} [{exp_lo:.1f}, {exp_hi:.1f}]" if not np.isnan(exp_mean) else "DNF",
            "Mean Runtime (s)": f"{time_mean:.4f}" if not np.isnan(time_mean) else "DNF",
            "Path Length (Moves)": f"{len_mean:.1f}" if not np.isnan(len_mean) else "DNF",
            "Path Cost (Total)": f"{cost_mean:.2f}" if not np.isnan(cost_mean) else "DNF",
            "Peak Stored Records": f"{stored_mean:.1f}",
        })

    df_summary = pd.DataFrame(summary_rows)

    tables = save_table(
        df_summary,
        name="partB_six_algorithms",
        caption="Comprehensive Evaluation of Six Search Algorithms Across 30 Certified Scenarios",
    )
    save_table(df_raw, name="partB_six_algorithms_raw", caption="Raw Six Search Algorithms Log")

    # Render summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    solved_configs = [c["name"] for c in configs if df_raw[(df_raw["config"] == c["name"]) & (df_raw["success"])].shape[0] > 0]
    succ_df = df_raw[df_raw["success"]]

    exp_vals = [succ_df[succ_df["config"] == c]["expanded_nodes"].mean() for c in solved_configs]
    cost_vals = [succ_df[succ_df["config"] == c]["cost"].mean() for c in solved_configs]

    colors = ["#1976d2", "#0288d1", "#7b1fa2", "#ef6c00", "#388e3c", "#fbc02d", "#c2185b"][:len(solved_configs)]

    ax1.bar(range(len(solved_configs)), exp_vals, color=colors)
    ax1.set_xticks(range(len(solved_configs)))
    ax1.set_xticklabels(solved_configs, rotation=30, ha="right")
    ax1.set_ylabel("Mean Node Expansions (Log Scale)")
    ax1.set_yscale("log")
    ax1.set_title("Search Efficiency: Node Expansions Across Algorithms")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.bar(range(len(solved_configs)), cost_vals, color=colors)
    ax2.set_xticks(range(len(solved_configs)))
    ax2.set_xticklabels(solved_configs, rotation=30, ha="right")
    ax2.set_ylabel("Mean Path Cost")
    ax2.set_title("Solution Quality: Path Cost Across Algorithms")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path, pdf_path = save_figure(fig, "partB_six_algorithms_comparison")
    plt.close(fig)

    print(f"\n[BENCHMARK COMPLETE] Saved six-algorithm table to {tables['md']}")
    print(f"Generated comparison figure: {png_path}")

    return {
        "raw_df": df_raw,
        "summary_df": df_summary,
        "figure_png": png_path,
    }


if __name__ == "__main__":
    run_full_algorithms_benchmark(n_scenarios=30)
