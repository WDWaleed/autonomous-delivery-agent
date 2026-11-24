"""Evaluation Weight Tuning on Disjoint Seeds (§7.5, M7.5).

Uses a fixed-budget predeclared + seeded candidate set over EvalWeights
evaluated strictly on disjoint tuning scenarios (k = 100..107), never
inspecting frozen evaluation seeds (k = 0..49).
Caches results by configuration + seed + source hash and writes config/tuned_weights.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.agents.hybrid import HybridAgent
from delivery.experiments.scenarios import scenario
from delivery.game.evaluation import EvalWeights, MultiFeatureEvaluator
from delivery.game.window import WindowGame, compute_canonical_baseline
from delivery.sim.env import DeliveryEnv
from delivery.sim.runner import run
from delivery.stamp import STAMP, banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.world.grid import GridWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
TUNED_WEIGHTS_PATH = CONFIG_DIR / "tuned_weights.json"
TUNING_CACHE_PATH = PROJECT_ROOT / "results" / "cache" / "tuning_cache.json"

# Disjoint tuning scenario indices (strictly disjoint from evaluation indices 0..49)
TUNING_SCENARIO_INDICES = list(range(100, 108))


def generate_candidate_weights(seed: int = 7500, n_candidates: int = 8) -> List[Tuple[str, EvalWeights]]:
    """Generates a predeclared default + seeded candidate weight configurations."""
    candidates: List[Tuple[str, EvalWeights]] = [
        ("default", EvalWeights(350.0, -300.0, 180.0, 70.0, 50.0)),
        ("evasion_heavy", EvalWeights(300.0, -250.0, 260.0, 100.0, 40.0)),
        ("route_heavy", EvalWeights(350.0, -380.0, 130.0, 50.0, 40.0)),
        ("mobility_focus", EvalWeights(320.0, -280.0, 190.0, 120.0, 40.0)),
    ]
    rng = np.random.default_rng(seed)
    while len(candidates) < n_candidates:
        idx = len(candidates)
        w = EvalWeights(
            w_delivered=float(rng.choice([300.0, 350.0, 400.0])),
            w_remaining_route=float(-rng.choice([250.0, 300.0, 360.0])),
            w_rival_dist=float(rng.choice([140.0, 180.0, 240.0])),
            w_mobility=float(rng.choice([50.0, 70.0, 110.0])),
            w_cost_margin=float(rng.choice([30.0, 50.0, 70.0])),
        )
        candidates.append((f"seeded_{idx}", w))
    return candidates[:n_candidates]


def _load_cache() -> Dict[str, Any]:
    if TUNING_CACHE_PATH.exists():
        try:
            with open(TUNING_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    TUNING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TUNING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def run_weight_tuning(
    quick: bool = False,
    seed: int = 7500,
) -> Dict[str, Any]:
    """Tunes EvalWeights on disjoint tuning scenarios and saves config/tuned_weights.json."""
    print(banner("PART C: EVALUATION WEIGHT TUNING ON DISJOINT SEEDS (M7.5)"))

    indices = TUNING_SCENARIO_INDICES[:3] if quick else TUNING_SCENARIO_INDICES
    candidates = generate_candidate_weights(seed=seed, n_candidates=4 if quick else 8)

    # Prebuild tuning scenarios & baselines
    tuning_envs = []
    for k in indices:
        sc = scenario(k)
        if not sc.feasible:
            continue
        world = GridWorld(sc.instance)
        twists = build_twist_stack(sc.instance, world)
        c_star, l_star, _ = compute_canonical_baseline(sc.instance, world=world, twists=twists)
        tuning_envs.append((sc, world, twists, c_star or 200.0, l_star or 60))

    cache = _load_cache()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

    rows: List[Dict[str, Any]] = []
    best_name = "default"
    best_weights = candidates[0][1]
    best_score = (-1.0, float("inf"))  # (win_rate, -mean_cost) -> maximize win_rate, minimize cost

    for name, weights in candidates:
        w_dict = asdict(weights)
        w_hash = hashlib.sha256(json.dumps(w_dict, sort_keys=True).encode()).hexdigest()[:12]

        wins = 0
        total = 0
        costs: List[float] = []

        for sc, world, twists, c_star, l_star in tuning_envs:
            cache_key = f"{w_hash}_{sc.key}_{source_hash}"
            if cache_key in cache:
                entry = cache[cache_key]
                win = bool(entry["win"])
                cost = float(entry["cost"])
            else:
                max_steps = min(180, 4 * l_star + 20)
                env = DeliveryEnv(
                    world=world,
                    twists=twists,
                    slip_p=sc.instance.slip_probability,
                    seed=sc.instance.seed + 7500,
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
                evaluator = MultiFeatureEvaluator(
                    problem=None,
                    own_heuristic=None,
                    weights=weights,
                )
                # Wire problem & own heuristic into evaluator
                from delivery.game.evaluation import make_default_evaluator
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
                win = bool(res.success)
                cost = float(res.total_cost)
                cache[cache_key] = {"win": win, "cost": cost}

            total += 1
            if win:
                wins += 1
            costs.append(cost)

        win_rate = wins / max(1, total)
        mean_cost = float(np.mean(costs)) if costs else float("inf")
        rows.append({
            "Candidate": name,
            "w_delivered": weights.w_delivered,
            "w_remaining": weights.w_remaining_route,
            "w_rival": weights.w_rival_dist,
            "w_mobility": weights.w_mobility,
            "w_margin": weights.w_cost_margin,
            "Win Rate": f"{win_rate*100:.1f}%",
            "win_rate_val": win_rate,
            "Mean Cost": round(mean_cost, 2),
        })

        # Primary: higher win_rate; tie-break: lower mean_cost
        if (win_rate > best_score[0] + 1e-9) or (
            abs(win_rate - best_score[0]) <= 1e-9 and mean_cost < best_score[1]
        ):
            best_score = (win_rate, mean_cost)
            best_name = name
            best_weights = weights

    _save_cache(cache)

    # Save chosen weights to config/tuned_weights.json
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tuned_payload = {
        "stamp": STAMP,
        "selected_candidate": best_name,
        "tuning_scenario_indices": indices,
        "disjoint_from_eval_indices": "0..49",
        "win_rate": best_score[0],
        "mean_cost": best_score[1],
        "weights": asdict(best_weights),
    }
    with open(TUNED_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(tuned_payload, f, indent=2)

    df_summary = pd.DataFrame(rows)
    tables = save_table(
        df_summary.drop(columns=["win_rate_val"]),
        name="partC_tuning",
        caption=f"Evaluation Weight Tuning on Disjoint Seeds (k={indices[0]}..{indices[-1]})",
    )

    # Render bar/heatmap comparison of candidates
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df_summary))
    colors = ["#2e7d32" if n == best_name else "#5c6bc0" for n in df_summary["Candidate"]]
    ax.bar(x, df_summary["Mean Cost"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["Candidate"], rotation=20, ha="right")
    ax.set_ylabel("Mean Realised Cost on Tuning Seeds")
    ax.set_title(f"Weight Tuning Candidates (Selected: {best_name} | Win Rate: {best_score[0]*100:.1f}%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    png_path, _ = save_figure(fig, "partC_tuning_comparison")
    plt.close(fig)

    print(f"[TUNING COMPLETE] Selected '{best_name}' -> {TUNED_WEIGHTS_PATH}")
    return {
        "best_name": best_name,
        "best_weights": best_weights,
        "summary_df": df_summary,
        "figure_png": png_path,
    }


if __name__ == "__main__":
    run_weight_tuning()
