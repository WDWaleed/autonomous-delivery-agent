"""Part C Final 50-Game Evaluation Against Assigned Rival (§6.1, §7.5, M7.6).

Evaluates HybridAgent with Default Weights vs Tuned Weights (from config/tuned_weights.json)
across n_games=50 frozen evaluation scenarios (k=0..49, strictly disjoint from tuning seeds 100..107).
Reports:
- Win / Draw / Loss rates with Wilson 95% CI
- Mean realised cost and replan metrics
- Replay strips (multi-frame PNG sequence) of one actual Win and one actual Loss (if observed)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.agents.hybrid import HybridAgent
from delivery.experiments.scenarios import generate_scenario_suite
from delivery.experiments.stats import mean_ci, wilson_ci
from delivery.game.evaluation import EvalWeights, make_default_evaluator
from delivery.game.window import WindowGame, compute_canonical_baseline
from delivery.sim.env import DeliveryEnv
from delivery.sim.runner import RunnerResult, run
from delivery.stamp import banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.viz.render import NORM, TERRAIN_CMAP
from delivery.world.grid import GridWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TUNED_WEIGHTS_PATH = PROJECT_ROOT / "config" / "tuned_weights.json"


def _load_tuned_weights() -> EvalWeights:
    if TUNED_WEIGHTS_PATH.exists():
        with open(TUNED_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        w = data.get("weights", {})
        return EvalWeights(
            w_delivered=float(w.get("w_delivered", 350.0)),
            w_remaining_route=float(w.get("w_remaining_route", -300.0)),
            w_rival_dist=float(w.get("w_rival_dist", 180.0)),
            w_mobility=float(w.get("w_mobility", 70.0)),
            w_cost_margin=float(w.get("w_cost_margin", 50.0)),
        )
    return EvalWeights()


def _render_replay_strip(
    instance: Any,
    res: RunnerResult,
    name: str,
    title_prefix: str,
) -> Tuple[Path, Path]:
    """Renders a 4-frame replay strip (Start, 33%, 66%, End) of an actual game trajectory."""
    traj = res.trajectory
    total = len(traj)
    indices = [
        0,
        min(total - 1, total // 3),
        min(total - 1, (2 * total) // 3),
        total - 1,
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for panel_idx, (ax, step_idx) in enumerate(zip(axes, indices)):
        ax.imshow(instance.grid, cmap=TERRAIN_CMAP, norm=NORM, origin="upper")
        sr, sc = instance.start
        ax.plot(sc, sr, marker="s", color="#e91e63", markersize=7)
        for dr, dc in instance.delivery_stops:
            ax.plot(dc, dr, marker="*", color="#ffd700", markersize=9, markeredgecolor="black")
        sub_traj = traj[: step_idx + 1]
        ax.plot([p[1] for p in sub_traj], [p[0] for p in sub_traj], color="#00e676", lw=2.0)
        ar, ac = traj[step_idx]
        ax.plot(ac, ar, marker="o", color="#00bcd4", markersize=9, markeredgecolor="black", label="Agent")
        ax.set_title(f"Step {step_idx}/{total - 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"{title_prefix}: {instance.id} | Cost={res.total_cost:.1f} | Steps={res.total_steps} | {res.failure_reason or 'WIN'}",
        fontsize=11,
        weight="bold",
    )
    plt.tight_layout()
    png_path, pdf_path = save_figure(fig, name)
    plt.close(fig)
    return png_path, pdf_path


def run_final_games(
    n_games: int = 50,
    quick: bool = False,
) -> Dict[str, Any]:
    """Runs the M7.6 50-game evaluation comparing Default vs Tuned weights."""
    print(banner("PART C: FINAL 50-GAME EVALUATION AGAINST CHASER RIVAL (M7.6)"))

    target_n = 4 if quick else max(30, n_games)
    suite = [sc for sc in generate_scenario_suite(n_scenarios=target_n) if sc.feasible]

    default_weights = EvalWeights()
    tuned_weights = _load_tuned_weights()
    configs = [
        ("Default Weights", default_weights),
        ("Tuned Weights", tuned_weights),
    ]

    # Precompute baselines for evaluation scenarios
    baselines: Dict[int, Dict[str, Any]] = {}
    for sc in suite:
        world = GridWorld(sc.instance)
        twists = build_twist_stack(sc.instance, world)
        c_star, l_star, _ = compute_canonical_baseline(sc.instance, world=world, twists=twists)
        baselines[sc.k] = {
            "world": world,
            "twists": twists,
            "c_star": c_star or 200.0,
            "l_star": l_star or 60,
        }

    records: List[Dict[str, Any]] = []
    win_showcase: Optional[Tuple[Any, RunnerResult]] = None
    loss_showcase: Optional[Tuple[Any, RunnerResult]] = None
    alt_showcase: Optional[Tuple[Any, RunnerResult]] = None

    for cfg_name, weights in configs:
        print(f"  Evaluating {cfg_name} across {len(suite)} games...", flush=True)
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
                slip_p=sc.instance.slip_probability,
                seed=sc.instance.seed + 7600,
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
            evaluator = make_default_evaluator(game, weights=weights)
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

            if res.success:
                outcome = "WIN"
                if cfg_name == "Tuned Weights" and win_showcase is None:
                    win_showcase = (sc.instance, res)
            elif res.failure_reason == "CAPTURED_BY_RIVAL":
                outcome = "LOSS"
                if cfg_name == "Tuned Weights" and loss_showcase is None:
                    loss_showcase = (sc.instance, res)
            else:
                outcome = "DRAW"
                if cfg_name == "Tuned Weights" and alt_showcase is None:
                    alt_showcase = (sc.instance, res)

            records.append({
                "config": cfg_name,
                "scenario_k": sc.k,
                "outcome": outcome,
                "win": outcome == "WIN",
                "draw": outcome == "DRAW",
                "loss": outcome == "LOSS",
                "cost": res.total_cost,
                "steps": res.total_steps,
                "total_replans": agent.replan_counts.total_replans(),
                "tactical_moves": agent.tactical_moves,
            })

    df_raw = pd.DataFrame(records)

    summary_rows: List[Dict[str, Any]] = []
    for cfg_name, _ in configs:
        sub = df_raw[df_raw["config"] == cfg_name]
        n_tot = len(sub)
        n_w = int(sub["win"].sum())
        n_d = int(sub["draw"].sum())
        n_l = int(sub["loss"].sum())

        wr, w_lo, w_hi = wilson_ci(n_w, n_tot)
        dr, d_lo, d_hi = wilson_ci(n_d, n_tot)
        lr, l_lo, l_hi = wilson_ci(n_l, n_tot)
        cm, c_lo, c_hi = mean_ci(sub["cost"]) if n_tot > 1 else (sub["cost"].mean(), 0, 0)

        summary_rows.append({
            "Configuration": cfg_name,
            "Games": n_tot,
            "Win Rate (95% Wilson CI)": f"{wr*100:.1f}% [{w_lo*100:.1f}, {w_hi*100:.1f}]",
            "Draw Rate (95% CI)": f"{dr*100:.1f}% [{d_lo*100:.1f}, {d_hi*100:.1f}]",
            "Loss Rate (95% CI)": f"{lr*100:.1f}% [{l_lo*100:.1f}, {l_hi*100:.1f}]",
            "Mean Cost (95% CI)": f"{cm:.1f} [{c_lo:.1f}, {c_hi:.1f}]",
            "Mean Replans": f"{sub['total_replans'].mean():.1f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    tables = save_table(
        df_summary,
        name="partC_games",
        caption=f"Final {len(suite)}-Game Evaluation Against Assigned Chaser Rival (Eval Seeds k=0..{len(suite)-1})",
    )
    save_table(df_raw, name="partC_games_raw", caption="Raw 50-Game Evaluation Log")

    # Render replay strips
    if win_showcase is not None:
        _render_replay_strip(
            win_showcase[0], win_showcase[1],
            name="partC_replay_win",
            title_prefix="Actual Win Replay Strip",
        )
    if loss_showcase is not None:
        _render_replay_strip(
            loss_showcase[0], loss_showcase[1],
            name="partC_replay_loss",
            title_prefix="Actual Loss (Capture) Replay Strip",
        )
    elif alt_showcase is not None:
        _render_replay_strip(
            alt_showcase[0], alt_showcase[1],
            name="partC_replay_draw_alt",
            title_prefix="Representative Non-Win (Step-Cap Draw) Replay Strip",
        )

    print(f"[FINAL GAMES COMPLETE] Saved summary table to {tables['md']}")
    return {
        "summary_df": df_summary,
        "raw_df": df_raw,
    }


if __name__ == "__main__":
    run_final_games(n_games=50)
