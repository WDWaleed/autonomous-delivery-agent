"""Part B/M9.1 Heuristic Component Ablation Study (§6.1, §7.4, M9.1).

Evaluates 4 paired heuristic configurations across 30 identical certified scenarios:
1. Baseline B1 (Manhattan * c_min)
2. Baseline B2 (Max directed relaxed distance)
3. MST-Only (Symmetric relaxed MST without B2 max term)
4. Full Own (CombinedMSTHeuristic: max(MST, B2))

Reports nodes expanded, generated, peak stored records, runtime, and path cost (Mean ± 95% CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.experiments.scenarios import generate_scenario_suite
from delivery.experiments.stats import mean_ci
from delivery.heuristics.base import Heuristic
from delivery.heuristics.baselines import ManhattanNearestStopHeuristic, MaxRelaxedDistanceHeuristic
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.search.astar import astar_search
from delivery.search.core import Limits, SearchStatus
from delivery.search.problem import DeliveryProblem
from delivery.stamp import banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld


class MSTOnlyHeuristic(CombinedMSTHeuristic):
    """Ablated Candidate A: pure symmetric relaxed MST without max(..., B2) directed term."""

    name: str = "Ablation_MST_Only"

    def __call__(self, state, problem: DeliveryProblem) -> float:
        r, c, mask, _ = state
        undelivered = [i for i in range(self.num_stops) if not (mask & (1 << i))]
        if not undelivered:
            if self.return_to_depot and self.depot_field is not None:
                return float(self.depot_field[r, c])
            return 0.0
        return self._compute_mst(undelivered, (r, c))


def run_ablation_study(
    n_scenarios: int = 30,
    quick: bool = False,
) -> Dict[str, Any]:
    """Runs the M9.1 4-configuration heuristic ablation across 30 paired scenarios."""
    print(banner("M9.1: HEURISTIC COMPONENT ABLATION STUDY"))

    n_scn = 5 if quick else n_scenarios
    suite = [sc for sc in generate_scenario_suite(n_scenarios=n_scn) if sc.feasible]
    limits = Limits(max_time_seconds=30.0, max_expanded=500_000)

    comp_names = [
        "1. B1 (Manhattan)",
        "2. B2 (Max Directed)",
        "3. MST-Only (Symm.)",
        "4. Full Own (MST+B2)",
    ]

    records: List[Dict[str, Any]] = []

    for sc in suite:
        world = GridWorld(sc.instance)
        twists = build_twist_stack(sc.instance, world)
        problem = DeliveryProblem(sc.instance, world=world, twist_stack=twists)

        heuristics: List[Tuple[str, Heuristic]] = [
            (comp_names[0], ManhattanNearestStopHeuristic()),
            (comp_names[1], MaxRelaxedDistanceHeuristic(problem)),
            (comp_names[2], MSTOnlyHeuristic(problem)),
            (comp_names[3], CombinedMSTHeuristic(problem)),
        ]

        for label, h_fn in heuristics:
            res = astar_search(problem, h_fn, limits=limits)
            ok = res.status == SearchStatus.SUCCESS
            records.append({
                "scenario_k": sc.k,
                "component": label,
                "success": ok,
                "expanded": res.metrics.expanded_nodes,
                "generated": res.metrics.generated_nodes,
                "peak_stored": res.metrics.peak_stored,
                "runtime_s": res.metrics.duration_seconds,
                "cost": res.cost if ok else np.nan,
            })

    df_raw = pd.DataFrame(records)
    summary_rows: List[Dict[str, Any]] = []

    for label in comp_names:
        sub = df_raw[(df_raw["component"] == label) & df_raw["success"]]
        n_tot = len(df_raw[df_raw["component"] == label])
        n_ok = len(sub)
        exp_m, exp_lo, exp_hi = mean_ci(sub["expanded"]) if n_ok > 1 else (sub["expanded"].mean(), 0, 0)
        sto_m, sto_lo, sto_hi = mean_ci(sub["peak_stored"]) if n_ok > 1 else (sub["peak_stored"].mean(), 0, 0)
        rt_m, rt_lo, rt_hi = mean_ci(sub["runtime_s"]) if n_ok > 1 else (sub["runtime_s"].mean(), 0, 0)
        cost_m, cost_lo, cost_hi = mean_ci(sub["cost"]) if n_ok > 1 else (sub["cost"].mean(), 0, 0)

        summary_rows.append({
            "Configuration": label,
            "Solved": f"{n_ok}/{n_tot}",
            "Expanded Nodes (Mean ± 95% CI)": f"{exp_m:.1f} [{exp_lo:.1f}, {exp_hi:.1f}]",
            "Peak Stored (Mean ± 95% CI)": f"{sto_m:.1f} [{sto_lo:.1f}, {sto_hi:.1f}]",
            "Runtime (s) (Mean ± 95% CI)": f"{rt_m:.4f} [{rt_lo:.4f}, {rt_hi:.4f}]",
            "Path Cost (Mean ± 95% CI)": f"{cost_m:.2f} [{cost_lo:.2f}, {cost_hi:.2f}]",
        })

    df_summary = pd.DataFrame(summary_rows)
    tables = save_table(
        df_summary,
        name="ablation",
        caption=f"Ablation Study of Heuristic Components Across {len(suite)} Paired Scenarios",
    )
    save_table(df_raw, name="ablation_raw", caption="Raw Heuristic Component Ablation Log")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    means = [df_raw[(df_raw["component"] == c) & df_raw["success"]]["expanded"].mean() for c in comp_names]
    colors = ["#42a5f5", "#26a69a", "#ffa726", "#2e7d32"]
    ax.bar(range(len(comp_names)), means, color=colors)
    ax.set_xticks(range(len(comp_names)))
    ax.set_xticklabels(comp_names, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Expanded Nodes (Log Scale)")
    ax.set_title(f"Heuristic Component Ablation ({len(suite)} Paired Scenarios)")
    for i, m in enumerate(means):
        ax.text(i, m * 1.12, f"{m:.1f}", ha="center", weight="bold", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    png_path, _ = save_figure(fig, "ablation_components")
    plt.close(fig)

    print(f"[ABLATION COMPLETE] Saved table to {tables['md']}")
    return {
        "summary_df": df_summary,
        "raw_df": df_raw,
        "figure_png": png_path,
    }


if __name__ == "__main__":
    run_ablation_study(n_scenarios=30)
