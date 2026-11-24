"""Part C Slip Probability and Replanning Analysis (§6.1, §7.5, M7.4).

Sweeps slip_probability in {0.0, 0.1, 0.2, 0.23 (instance), 0.3, 0.5} on paired
certified scenarios with the assigned Chaser rival active.
Tracks:
- Slip displacements, replan triggers (count & per 100 agent moves)
- Extra cost paired against slip=0.0
- Predicted deterministic cost (C*) vs realised cost
- Capture rate & Win rate with Wilson 95% CI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.agents.hybrid import HybridAgent
from delivery.experiments.scenarios import generate_scenario_suite
from delivery.experiments.stats import mean_ci, wilson_ci
from delivery.game.evaluation import make_default_evaluator
from delivery.game.window import WindowGame, compute_canonical_baseline
from delivery.sim.env import DeliveryEnv
from delivery.sim.runner import run
from delivery.stamp import banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_slip_experiment(
    n_scenarios: int = 15,
    quick: bool = False,
) -> Dict[str, Any]:
    """Runs the M7.4 slip probability sweep on paired scenario seeds."""
    print(banner("PART C: SLIP PROBABILITY & REPLANNING ANALYSIS (M7.4)"))

    n_scn = 4 if quick else n_scenarios
    suite = [sc for sc in generate_scenario_suite(n_scenarios=n_scn) if sc.feasible]
    slip_values = [0.0, 0.1, 0.2, 0.23, 0.3, 0.5]

    # Precompute deterministic baseline (C*, L*) per scenario once
    baselines: Dict[int, Dict[str, Any]] = {}
    for sc in suite:
        world = GridWorld(sc.instance)
        twists = build_twist_stack(sc.instance, world)
        c_star, l_star, cert = compute_canonical_baseline(sc.instance, world=world, twists=twists)
        baselines[sc.k] = {
            "world": world,
            "twists": twists,
            "c_star": c_star or 0.0,
            "l_star": l_star or 60,
        }

    records: List[Dict[str, Any]] = []

    for slip_p in slip_values:
        print(f"  Running slip_p={slip_p} across {len(suite)} scenarios...", flush=True)
        for sc in suite:
            base = baselines[sc.k]
            world = base["world"]
            twists = base["twists"]
            c_star = base["c_star"]
            l_star = base["l_star"]
            max_steps = min(200, 4 * l_star + 20)

            env = DeliveryEnv(
                world=world,
                twists=twists,
                slip_p=slip_p,
                seed=sc.instance.seed + 7400,
                enable_rival=True,
                rival_depth=2,
            )
            game = WindowGame(
                sc.instance,
                world=world,
                twists=twists,
                window_radius=4,
                c_star=c_star,
                l_star=l_star,
            )
            evaluator = make_default_evaluator(game)
            agent = HybridAgent(
                sc.instance,
                world=world,
                twists=twists,
                window_radius=4,
                tactical_depth=2,
                eval_fn=evaluator,
                c_star=c_star,
                l_star=l_star,
            )

            res = run(agent, env, max_steps=max_steps)
            actual_slips = sum(1 for tr in res.trace if tr.slipped)
            total_replans = agent.replan_counts.total_replans()
            replans_per_100 = (total_replans / max(1, res.total_steps)) * 100.0
            slips_per_100 = (actual_slips / max(1, res.total_steps)) * 100.0

            records.append({
                "scenario_k": sc.k,
                "slip_p": slip_p,
                "is_instance_slip": abs(slip_p - 0.23) < 1e-6,
                "win": res.success,
                "captured": res.failure_reason == "CAPTURED_BY_RIVAL",
                "steps": res.total_steps,
                "realised_cost": res.total_cost,
                "predicted_c_star": c_star,
                "actual_slips": actual_slips,
                "slips_per_100": slips_per_100,
                "slip_replans": agent.replan_counts.slip,
                "total_replans": total_replans,
                "replans_per_100": replans_per_100,
                "tactical_moves": agent.tactical_moves,
            })

    df_raw = pd.DataFrame(records)

    # Paired extra cost vs slip=0.0
    base_costs = df_raw[df_raw["slip_p"] == 0.0].set_index("scenario_k")["realised_cost"].to_dict()
    df_raw["extra_cost_vs_0"] = df_raw.apply(
        lambda row: row["realised_cost"] - base_costs.get(row["scenario_k"], row["realised_cost"]),
        axis=1,
    )

    summary_rows: List[Dict[str, Any]] = []
    for sp in slip_values:
        sub = df_raw[df_raw["slip_p"] == sp]
        n_tot = len(sub)
        n_win = int(sub["win"].sum())
        n_cap = int(sub["captured"].sum())
        w_rate, w_lo, w_hi = wilson_ci(n_win, n_tot)
        c_rate, c_lo, c_hi = wilson_ci(n_cap, n_tot)

        cost_m, cost_lo, cost_hi = mean_ci(sub["realised_cost"]) if n_tot > 1 else (sub["realised_cost"].mean(), 0, 0)
        ext_m, ext_lo, ext_hi = mean_ci(sub["extra_cost_vs_0"]) if n_tot > 1 else (sub["extra_cost_vs_0"].mean(), 0, 0)
        rep_m, rep_lo, rep_hi = mean_ci(sub["replans_per_100"]) if n_tot > 1 else (sub["replans_per_100"].mean(), 0, 0)

        label = f"{sp:.2f} (instance)" if abs(sp - 0.23) < 1e-6 else f"{sp:.1f}"
        summary_rows.append({
            "Slip Prob (p)": label,
            "Win Rate (95% Wilson CI)": f"{w_rate*100:.1f}% [{w_lo*100:.1f}, {w_hi*100:.1f}]",
            "Capture Rate (95% CI)": f"{c_rate*100:.1f}% [{c_lo*100:.1f}, {c_hi*100:.1f}]",
            "Mean Slips": f"{sub['actual_slips'].mean():.1f}",
            "Replans / 100 Moves": f"{rep_m:.1f} [{rep_lo:.1f}, {rep_hi:.1f}]",
            "Realised Cost (Mean ± CI)": f"{cost_m:.1f} [{cost_lo:.1f}, {cost_hi:.1f}]",
            "Extra Cost vs p=0": f"{ext_m:+.1f} [{ext_lo:+.1f}, {ext_hi:+.1f}]",
        })

    df_summary = pd.DataFrame(summary_rows)
    tables = save_table(
        df_summary,
        name="partC_slip",
        caption=f"Slip Probability Sweep & Replanning Analysis ({n_scn} Paired Scenarios)",
    )
    save_table(df_raw, name="partC_slip_raw", caption="Raw Slip Experiment Log")

    # Plot Win/Capture Rate & Replans vs Slip Probability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    win_pcts = [df_raw[df_raw["slip_p"] == sp]["win"].mean() * 100.0 for sp in slip_values]
    cap_pcts = [df_raw[df_raw["slip_p"] == sp]["captured"].mean() * 100.0 for sp in slip_values]
    rep_means = [df_raw[df_raw["slip_p"] == sp]["replans_per_100"].mean() for sp in slip_values]
    cost_means = [df_raw[df_raw["slip_p"] == sp]["realised_cost"].mean() for sp in slip_values]

    ax1.plot(slip_values, win_pcts, "o-", color="#2e7d32", lw=2.2, label="Win Rate (%)")
    ax1.plot(slip_values, cap_pcts, "s--", color="#c62828", lw=2.2, label="Capture Rate (%)")
    ax1.axvline(0.23, color="#555555", linestyle=":", label="Instance p=0.23")
    ax1.set_xlabel("Slip Probability (p)")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Win vs Capture Rate Across Slip Probabilities")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(slip_values, rep_means, "^-", color="#1565c0", lw=2.2, label="Replans / 100 Moves")
    ax2_r = ax2.twinx()
    ax2_r.plot(slip_values, cost_means, "d--", color="#ef6c00", lw=2.0, label="Realised Cost")
    ax2.set_xlabel("Slip Probability (p)")
    ax2.set_ylabel("Replans per 100 Moves", color="#1565c0")
    ax2_r.set_ylabel("Mean Realised Cost", color="#ef6c00")
    ax2.set_title("Replanning Frequency & Realised Cost vs Slip")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path, _ = save_figure(fig, "partC_slip_analysis")
    plt.close(fig)

    print(f"[SLIP COMPLETE] Saved summary table to {tables['md']}")
    print(f"Saved slip analysis figure to {png_path}")

    return {
        "summary_df": df_summary,
        "raw_df": df_raw,
        "figure_png": png_path,
    }


if __name__ == "__main__":
    run_slip_experiment(n_scenarios=15)
