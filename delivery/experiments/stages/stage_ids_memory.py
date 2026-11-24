"""Part B IDS vs A* Under Stored-Node Budgets stage (§6.1, §7.4, D18, M5.5).

Sweeps stored-record budgets:
- 25 records (tight)
- 100 records
- 1,000 records
- 5,000 records
- 20,000 records
- 100,000 records

Across deterministic compact delivery scenarios with solution depth <= 16.
Compares BFS, UCS, A*+Own, and IDS:
- Documents scenarios where A*/UCS/BFS exceed the record budget (MEMORYLIMIT) while IDS completes.
- Separately records peak stored records, peak memory bytes (tracemalloc), and expansions.
- Includes ablation of IDS with vs without on-path cycle checking.
- Renders depth-vs-stored-records and depth-vs-expansions scaling curves.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.search.astar import astar_search
from delivery.search.bfs import breadth_first_search
from delivery.search.compact import make_compact_scenario
from delivery.search.core import Limits, SearchStatus
from delivery.search.ids import iterative_deepening_search
from delivery.search.problem import DeliveryProblem
from delivery.search.ucs import uniform_cost_search
from delivery.stamp import banner, save_figure, save_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_ids_memory_budget_experiment(
    n_seeds: int = 5,
    quick: bool = False,
) -> Dict[str, Any]:
    """Executes the stored-record budget sweep comparing IDS against A*, UCS, and BFS."""
    print(banner("PART B: IDS VS A* UNDER STORED-NODE BUDGETS (M5.5)"))

    budgets = [25, 100, 1_000, 5_000, 20_000] if not quick else [25, 500, 2_000]
    seeds = range(101, 101 + (2 if quick else n_seeds))

    records: List[Dict[str, Any]] = []

    # 1. Budget Sweep Evaluation
    for seed in seeds:
        inst = make_compact_scenario(seed=seed, n=8, num_stops=1)
        problem = DeliveryProblem(inst)
        h_own = CombinedMSTHeuristic(problem)

        for budget in budgets:
            limits = Limits(max_stored_records=budget, max_time_seconds=5.0, max_expanded=100_000)

            # A) BFS
            tracemalloc.start()
            t0 = time.perf_counter()
            res_bfs = breadth_first_search(problem, limits=limits)
            t_bfs = time.perf_counter() - t0
            _, peak_bytes_bfs = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            records.append({
                "seed": seed,
                "budget": budget,
                "algo": "BFS",
                "status": res_bfs.status.value,
                "success": res_bfs.status == SearchStatus.SUCCESS,
                "expanded": res_bfs.metrics.expanded_nodes,
                "peak_stored": res_bfs.metrics.peak_stored,
                "peak_bytes": peak_bytes_bfs,
                "time_s": t_bfs,
            })

            # B) UCS
            tracemalloc.start()
            t0 = time.perf_counter()
            res_ucs = uniform_cost_search(problem, limits=limits)
            t_ucs = time.perf_counter() - t0
            _, peak_bytes_ucs = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            records.append({
                "seed": seed,
                "budget": budget,
                "algo": "UCS",
                "status": res_ucs.status.value,
                "success": res_ucs.status == SearchStatus.SUCCESS,
                "expanded": res_ucs.metrics.expanded_nodes,
                "peak_stored": res_ucs.metrics.peak_stored,
                "peak_bytes": peak_bytes_ucs,
                "time_s": t_ucs,
            })

            # C) A* + Own
            tracemalloc.start()
            t0 = time.perf_counter()
            res_astar = astar_search(problem, h_own, limits=limits)
            t_astar = time.perf_counter() - t0
            _, peak_bytes_astar = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            records.append({
                "seed": seed,
                "budget": budget,
                "algo": "A* (Own)",
                "status": res_astar.status.value,
                "success": res_astar.status == SearchStatus.SUCCESS,
                "expanded": res_astar.metrics.expanded_nodes,
                "peak_stored": res_astar.metrics.peak_stored,
                "peak_bytes": peak_bytes_astar,
                "time_s": t_astar,
            })

            # D) IDS
            tracemalloc.start()
            t0 = time.perf_counter()
            res_ids = iterative_deepening_search(problem, max_depth=16, limits=limits, check_cycle=True)
            t_ids = time.perf_counter() - t0
            _, peak_bytes_ids = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            records.append({
                "seed": seed,
                "budget": budget,
                "algo": "IDS",
                "status": res_ids.status.value,
                "success": res_ids.status == SearchStatus.SUCCESS,
                "expanded": res_ids.metrics.expanded_nodes,
                "peak_stored": res_ids.metrics.peak_stored,
                "peak_bytes": peak_bytes_ids,
                "time_s": t_ids,
            })

    df_raw = pd.DataFrame(records)

    # 2. Cycle Ablation on Cyclic Scenario
    ablation_inst = make_compact_scenario(seed=202, n=7, num_stops=1)
    ablation_prob = DeliveryProblem(ablation_inst)
    abl_limits = Limits(max_expanded=10_000, max_time_seconds=3.0)

    res_cycle_on = iterative_deepening_search(ablation_prob, max_depth=12, limits=abl_limits, check_cycle=True)
    res_cycle_off = iterative_deepening_search(ablation_prob, max_depth=12, limits=abl_limits, check_cycle=False)

    print(f"\n[CYCLE ABLATION] IDS with cycle check: {res_cycle_on.metrics.expanded_nodes} expansions (Status: {res_cycle_on.status.value})")
    print(f"[CYCLE ABLATION] IDS without cycle check: {res_cycle_off.metrics.expanded_nodes} expansions (Status: {res_cycle_off.status.value})")

    # 3. Summary Aggregations
    summary_rows = []
    for budget in budgets:
        sub = df_raw[df_raw["budget"] == budget]
        for algo in ["BFS", "UCS", "A* (Own)", "IDS"]:
            algo_sub = sub[sub["algo"] == algo]
            n_tot = len(algo_sub)
            n_suc = int(algo_sub["success"].sum())
            succ_only = algo_sub[algo_sub["success"]]
            mean_exp = succ_only["expanded"].mean() if len(succ_only) > 0 else np.nan
            mean_stored = algo_sub["peak_stored"].mean()
            mean_kb = algo_sub["peak_bytes"].mean() / 1024.0

            summary_rows.append({
                "Budget (Records)": budget,
                "Algorithm": algo,
                "Solved Ratio": f"{n_suc}/{n_tot}",
                "Mean Peak Stored": f"{mean_stored:.1f}",
                "Mean Peak Memory (KB)": f"{mean_kb:.1f}",
                "Mean Expansions (if Solved)": f"{mean_exp:.1f}" if not np.isnan(mean_exp) else "DNF",
            })

    df_summary = pd.DataFrame(summary_rows)
    tables = save_table(
        df_summary,
        name="partB_ids_budget_sweep",
        caption="Comparison of Algorithms Across Stored-Record Budgets",
    )

    # 4. Render Scaling Curves (Peak Stored Records vs Depth & Expansions vs Depth)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Evaluate scaling on depth 2 to 14
    depths = list(range(2, 15, 2))
    sample_inst = make_compact_scenario(seed=101, n=8, num_stops=1)
    sample_prob = DeliveryProblem(sample_inst)
    sample_h = CombinedMSTHeuristic(sample_prob)

    ids_stored, ids_exp = [], []
    astar_stored, astar_exp = [], []
    ucs_stored, ucs_exp = [], []

    for d in depths:
        lim = Limits(max_time_seconds=2.0)
        # IDS up to depth d
        r_ids = iterative_deepening_search(sample_prob, max_depth=d, limits=lim)
        ids_stored.append(r_ids.metrics.peak_stored)
        ids_exp.append(r_ids.metrics.expanded_nodes)

        # A* and UCS
        r_ast = astar_search(sample_prob, sample_h, limits=lim)
        astar_stored.append(r_ast.metrics.peak_stored)
        astar_exp.append(r_ast.metrics.expanded_nodes)

        r_ucs = uniform_cost_search(sample_prob, limits=lim)
        ucs_stored.append(r_ucs.metrics.peak_stored)
        ucs_exp.append(r_ucs.metrics.expanded_nodes)

    ax1.plot(depths, ids_stored, "o-", label="IDS (Linear O(d))", color="#2e7d32", lw=2.5)
    ax1.plot(depths, astar_stored, "s--", label="A* + Own (Frontier + Explored)", color="#7b1fa2", lw=2)
    ax1.plot(depths, ucs_stored, "^:", label="UCS (Exponential Frontier)", color="#1565c0", lw=2)
    ax1.set_xlabel("Search Depth Bound (d)")
    ax1.set_ylabel("Peak Stored Records")
    ax1.set_title("Memory Complexity: Peak Stored Records vs Depth")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(depths, ids_exp, "o-", label="IDS (Re-expansions)", color="#2e7d32", lw=2.5)
    ax2.plot(depths, astar_exp, "s--", label="A* + Own (Targeted)", color="#7b1fa2", lw=2)
    ax2.plot(depths, ucs_exp, "^:", label="UCS (Uniform)", color="#1565c0", lw=2)
    ax2.set_xlabel("Search Depth Bound (d)")
    ax2.set_ylabel("Total Node Expansions (Log Scale)")
    ax2.set_yscale("log")
    ax2.set_title("Time Complexity: Expansions vs Depth")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path, pdf_path = save_figure(fig, "partB_ids_scaling_curves")
    plt.close(fig)

    print(f"\n[BENCHMARK COMPLETE] Saved summary table to {tables['md']}")
    print(f"Generated scaling curve figure: {png_path}")

    return {
        "raw_df": df_raw,
        "summary_df": df_summary,
        "figure_png": png_path,
    }


if __name__ == "__main__":
    run_ids_memory_budget_experiment()
