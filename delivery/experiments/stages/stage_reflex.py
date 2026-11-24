"""Part A Reflex Agent benchmark stage (§6.1, §7.3, M3.2).

Evaluates the stateless SimpleReflexAgent across 30 certified scenarios.
Computes Wilson score CI for success rate and t-based CI for cost.
Renders stamped qualitative failure visualizations demonstrating local minima/wall traps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.agents.reflex import ReflexAgent
from delivery.experiments.scenarios import generate_scenario_suite
from delivery.experiments.stats import mean_ci, wilson_ci
from delivery.sim.env import DeliveryEnv
from delivery.sim.runner import RunnerResult, run
from delivery.stamp import banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.viz.render import NORM, TERRAIN_CMAP, TERRAIN_COLORS
from delivery.world.grid import GridWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_reflex_benchmark(
    n_scenarios: int = 30,
    max_steps: int = 100,
    quick: bool = False,
) -> Dict[str, Any]:
    """Runs the 30-scenario reflex benchmark and generates tables and failure figures."""
    scenarios_to_run = 5 if quick else n_scenarios
    suite = generate_scenario_suite(n_scenarios=scenarios_to_run)

    records: List[Dict[str, Any]] = []
    failure_cases: List[Tuple[Any, RunnerResult]] = []

    for sc in suite:
        world = GridWorld(sc.instance)
        twists = build_twist_stack(sc.instance, world)
        # Seeded environment using scenario seed and instance slip
        env = DeliveryEnv(
            world=world,
            twists=twists,
            slip_p=sc.instance.slip_probability,
            seed=sc.instance.seed,
        )
        agent = ReflexAgent()
        result = run(agent, env, max_steps=max_steps)

        record = {
            "Scenario": sc.k,
            "Key": sc.key,
            "Grid Size": sc.instance.n,
            "Stops": len(sc.instance.delivery_stops),
            "Slip Prob": sc.instance.slip_probability,
            "Success": result.success,
            "Steps": result.total_steps,
            "Total Cost": round(result.total_cost, 2),
            "Outcome": "Delivered" if result.success else (result.failure_reason or "FAILED"),
        }
        records.append(record)

        if not result.success:
            failure_cases.append((sc, result))

    df_details = pd.DataFrame(records)

    # Compute aggregate statistics
    n_total = len(df_details)
    n_success = int(df_details["Success"].sum())
    success_rate, s_low, s_high = wilson_ci(n_success, n_total)

    successful_costs = df_details[df_details["Success"]]["Total Cost"].tolist()
    if successful_costs:
        mean_c, c_low, c_high = mean_ci(successful_costs)
    else:
        mean_c, c_low, c_high = 0.0, 0.0, 0.0

    summary_records = [
        {
            "Metric": "Total Scenarios Evaluated",
            "Value": str(n_total),
            "95% Confidence Interval": "-",
        },
        {
            "Metric": "Successful Deliveries",
            "Value": f"{n_success} / {n_total}",
            "95% Confidence Interval": f"[{s_low*100:.1f}%, {s_high*100:.1f}%] (Wilson)",
        },
        {
            "Metric": "Success Rate (%)",
            "Value": f"{success_rate * 100:.1f}%",
            "95% Confidence Interval": f"[{s_low*100:.1f}%, {s_high*100:.1f}%]",
        },
        {
            "Metric": "Mean Delivery Cost (Successful)",
            "Value": f"{mean_c:.2f}" if successful_costs else "N/A (0 successes)",
            "95% Confidence Interval": f"[{c_low:.2f}, {c_high:.2f}]" if successful_costs else "N/A",
        },
    ]
    df_summary = pd.DataFrame(summary_records)

    # Save stamped tables
    save_table(df_details, "partA_reflex_scenarios", caption="Simple Reflex Agent 30-Scenario Evaluation Details")
    save_table(df_summary, "partA_reflex", caption="Simple Reflex Agent Benchmark Summary (§7.3, Part A)")

    # Render stamped failure picture if any failures occurred
    fig_png, fig_pdf = None, None
    if failure_cases:
        chosen_sc, chosen_res = failure_cases[0]
        fig_png, fig_pdf = render_reflex_failure_case(chosen_sc, chosen_res)

    return {
        "status": "COMPLETED",
        "n_scenarios": n_total,
        "n_success": n_success,
        "success_rate": success_rate,
        "wilson_ci": (s_low, s_high),
        "mean_cost": mean_c,
        "cost_ci": (c_low, c_high),
        "failure_figure_png": str(fig_png) if fig_png else None,
    }


def render_reflex_failure_case(
    scenario: Any,
    result: RunnerResult,
    name: str = "partA_reflex_failure",
) -> Tuple[Path, Path]:
    """Renders qualitative stamped failure picture showing wall traps, oscillation, or local minima."""
    inst = scenario.instance
    n = inst.n
    grid = inst.grid
    trajectory = result.trajectory

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # Base grid
    ax.imshow(
        grid,
        cmap=TERRAIN_CMAP,
        norm=NORM,
        origin="upper",
        extent=(-0.5, n - 0.5, n - 0.5, -0.5),
        zorder=1,
    )

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5, alpha=0.3, zorder=2)
    ax.set_xticks(np.arange(0, n, 5))
    ax.set_yticks(np.arange(0, n, 5))

    # Depot Start
    sr, sc = inst.start
    ax.plot(
        sc,
        sr,
        marker="s",
        markersize=13,
        color="#ffb300",
        markeredgecolor="#5d4037",
        markeredgewidth=1.8,
        zorder=6,
        label=f"Depot Start {inst.start}",
    )

    # Delivery stops
    for idx, (dr, dc) in enumerate(inst.delivery_stops, start=1):
        ax.plot(
            dc,
            dr,
            marker="o",
            markersize=11,
            color="#e53935",
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=6,
        )
        ax.text(
            dc,
            dr,
            str(idx),
            color="#ffffff",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )

    # Plot reflex trajectory path
    if len(trajectory) > 1:
        traj_cols = [p[1] for p in trajectory]
        traj_rows = [p[0] for p in trajectory]

        # Draw trajectory line
        ax.plot(
            traj_cols,
            traj_rows,
            color="#d97706",
            linewidth=2.5,
            alpha=0.75,
            linestyle="-",
            zorder=4,
            label="Reflex Trajectory",
        )

        # Draw final trapped / oscillating location with hazard marker
        final_r, final_c = trajectory[-1]
        ax.plot(
            final_c,
            final_r,
            marker="X",
            markersize=14,
            color="#dc2626",
            markeredgecolor="#000000",
            markeredgewidth=1.8,
            zorder=8,
            label=f"Trapped: {result.failure_reason}",
        )

    reason_desc = (
        "Local Minimum / Oscillation"
        if result.failure_reason == "LOCAL_MINIMUM_OSCILLATION"
        else "Obstacle Barrier / Step Cap Exceeded"
    )
    title_str = (
        f"Simple Reflex Failure: {reason_desc}\n"
        f"Scenario {scenario.key} ({n}x{n}) — Trapped after {result.total_steps} steps (Cost: {result.total_cost:.1f})"
    )
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Column", fontsize=10)
    ax.set_ylabel("Grid Row", fontsize=10)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize=9)

    png_path, pdf_path = save_figure(fig, name, report_copy=True)
    plt.close(fig)
    return png_path, pdf_path


def main():
    print(banner("PART A REFLEX BENCHMARK (M3.2)"))
    res = run_reflex_benchmark(n_scenarios=30)
    print("\nBenchmark Results Summary:")
    print(f"  Scenarios: {res['n_scenarios']}")
    print(f"  Successes: {res['n_success']} / {res['n_scenarios']} ({res['success_rate']*100:.1f}%)")
    print(f"  Wilson 95% CI: [{res['wilson_ci'][0]*100:.1f}%, {res['wilson_ci'][1]*100:.1f}%]")
    if res['failure_figure_png']:
        print(f"  Stamped Failure Figure: {res['failure_figure_png']}")


if __name__ == "__main__":
    main()
