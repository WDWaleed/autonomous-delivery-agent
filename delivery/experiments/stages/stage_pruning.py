"""Part C Move Ordering and Alpha-Beta Pruning Experiment (§6.1, §7.5, M6.3).

Samples >= 200 game positions from simulated trajectories over the 30 certified scenarios,
compares Minimax vs Alpha-Beta under 'natural', 'random(seeded)', and 'own' move orderings
across depths 2..adversary_search_depth (6), and reports:
- Value invariance (Alpha-Beta == Minimax)
- Mean node expansions and evaluations with 95% CI
- Percentage nodes saved by Alpha-Beta and by 'own' ordering vs 'natural'/'random'
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
from delivery.game.alphabeta import alphabeta_search
from delivery.game.minimax import minimax_search
from delivery.game.ordering import make_own_ordering, make_random_ordering, natural_ordering
from delivery.game.window import AGENT, RIVAL, GameState, WindowGame
from delivery.stamp import banner, save_figure, save_table
from delivery.twists.registry import build_twist_stack
from delivery.world.distance import distance_field
from delivery.world.grid import GridWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RANDOM_ORDER_SEED = 20260603


def _sample_positions_from_scenarios(
    n_scenarios: int = 30,
    target_positions: int = 210,
    seed: int = 6300,
) -> List[Tuple[WindowGame, GameState, List[np.ndarray]]]:
    """Samples diverse mid-game positions across the 30 certified scenarios."""
    rng = np.random.default_rng(seed)
    suite = generate_scenario_suite(n_scenarios=n_scenarios)
    samples: List[Tuple[WindowGame, GameState, List[np.ndarray]]] = []

    per_scenario = max(1, (target_positions + len(suite) - 1) // len(suite))

    for sc in suite:
        if not sc.feasible:
            continue
        inst = sc.instance
        world = GridWorld(inst)
        twists = build_twist_stack(inst, world)
        relaxed = twists.relaxed_graph(world)
        stop_fields = [
            distance_field(world, s, reverse=True, edge_model=relaxed)
            for s in inst.delivery_stops
        ]

        passable = [(r, c) for r in range(inst.n) for c in range(inst.n) if world.passable[r, c]]
        if len(passable) < 2:
            continue

        for _ in range(per_scenario):
            a_pos = passable[int(rng.integers(0, len(passable)))]
            # Place rival within Chebyshev distance 2..4 so tactical interaction occurs
            nearby = [
                p for p in passable
                if p != a_pos and 1 <= max(abs(p[0] - a_pos[0]), abs(p[1] - a_pos[1])) <= 4
            ]
            r_pos = nearby[int(rng.integers(0, len(nearby)))] if nearby else passable[int(rng.integers(0, len(passable)))]

            game = WindowGame(
                inst,
                world=world,
                twists=twists,
                window_radius=4,
                window_center=a_pos,
                allow_wait=True,
            )
            mask = int(rng.integers(0, max(1, (1 << len(inst.delivery_stops)) - 1)))
            turn = AGENT if rng.random() < 0.5 else RIVAL
            state = game.initial_state(
                agent_pos=a_pos,
                rival_pos=r_pos,
                delivered_mask=mask,
                turn=turn,
            )
            if not game.is_terminal(state):
                samples.append((game, state, stop_fields))
            if len(samples) >= target_positions:
                return samples

    return samples


def run_pruning_experiment(
    n_scenarios: int = 30,
    target_positions: int = 210,
    quick: bool = False,
) -> Dict[str, Any]:
    """Executes the M6.3 move ordering & Alpha-Beta pruning experiment."""
    print(banner("PART C: MOVE ORDERING & ALPHA-BETA PRUNING EXPERIMENT (M6.3)"))

    n_pos = 30 if quick else target_positions
    n_scn = 5 if quick else n_scenarios
    samples = _sample_positions_from_scenarios(n_scenarios=n_scn, target_positions=n_pos)
    print(f"Sampled {len(samples)} non-terminal game positions across {n_scn} scenarios.")

    max_depth = 6
    depths = list(range(2, max_depth + 1))
    rand_order = make_random_ordering(RANDOM_ORDER_SEED)

    records: List[Dict[str, Any]] = []
    value_mismatches = 0

    for pos_idx, (game, state, stop_fields) in enumerate(samples):
        own_order = make_own_ordering(stop_fields=stop_fields)
        pv_hint: Optional[List[int]] = None

        for d in depths:
            # Minimax baseline (value & full unpruned tree size; run on all positions for d<=4, and first 30 for d>=5)
            mm_evals: Optional[int] = None
            mm_exp: Optional[int] = None
            mm_val: Optional[float] = None
            if d <= 4 or pos_idx < 30:
                res_mm = minimax_search(game, state, depth_limit=d, order_fn=natural_ordering)
                mm_evals = res_mm.stats.evaluations
                mm_exp = res_mm.stats.nodes_expanded
                mm_val = res_mm.value

            # Alpha-Beta under 3 orderings
            res_nat = alphabeta_search(game, state, depth_limit=d, order_fn=natural_ordering)
            res_rnd = alphabeta_search(game, state, depth_limit=d, order_fn=rand_order)
            res_own = alphabeta_search(game, state, depth_limit=d, order_fn=own_order, pv_hint=pv_hint)
            pv_hint = res_own.principal_variation

            # Verify value agreement across all orderings and Minimax
            if abs(res_nat.value - res_rnd.value) > 1e-6 or abs(res_nat.value - res_own.value) > 1e-6:
                value_mismatches += 1
            if mm_val is not None and abs(res_nat.value - mm_val) > 1e-6:
                value_mismatches += 1

            for ord_name, res in [("natural", res_nat), ("random", res_rnd), ("own", res_own)]:
                saved_vs_mm = (
                    ((mm_evals - res.stats.evaluations) / mm_evals) * 100.0
                    if (mm_evals is not None and mm_evals > 0)
                    else np.nan
                )
                saved_vs_nat = (
                    ((res_nat.stats.evaluations - res.stats.evaluations) / res_nat.stats.evaluations) * 100.0
                    if res_nat.stats.evaluations > 0
                    else 0.0
                )
                records.append({
                    "pos_idx": pos_idx,
                    "depth": d,
                    "ordering": ord_name,
                    "ab_evaluations": res.stats.evaluations,
                    "ab_expanded": res.stats.nodes_expanded,
                    "ab_cutoffs": res.stats.cutoffs,
                    "mm_evaluations": mm_evals if mm_evals is not None else np.nan,
                    "mm_expanded": mm_exp if mm_exp is not None else np.nan,
                    "saved_vs_mm_pct": saved_vs_mm,
                    "saved_vs_nat_pct": saved_vs_nat,
                })

    assert value_mismatches == 0, f"Value mismatch detected across {value_mismatches} comparisons!"

    df_raw = pd.DataFrame(records)

    summary_rows: List[Dict[str, Any]] = []
    for d in depths:
        df_d = df_raw[df_raw["depth"] == d]
        mm_sub = df_d[df_d["ordering"] == "natural"]["mm_evaluations"].dropna()
        mm_mean = mm_sub.mean() if len(mm_sub) > 0 else np.nan

        for ord_name in ["natural", "random", "own"]:
            sub = df_d[df_d["ordering"] == ord_name]
            ev_mean, ev_lo, ev_hi = mean_ci(sub["ab_evaluations"])
            sav_mm = sub["saved_vs_mm_pct"].dropna()
            sav_mm_mean, sav_mm_lo, sav_mm_hi = mean_ci(sav_mm) if len(sav_mm) > 1 else (sav_mm.mean(), 0.0, 0.0)
            sav_nat_mean, sav_nat_lo, sav_nat_hi = mean_ci(sub["saved_vs_nat_pct"])

            summary_rows.append({
                "Depth (Plies)": d,
                "Ordering": ord_name,
                "Minimax Evals (Mean)": f"{mm_mean:.1f}" if not np.isnan(mm_mean) else "N/A",
                "AB Evals (Mean ± 95% CI)": f"{ev_mean:.1f} [{ev_lo:.1f}, {ev_hi:.1f}]",
                "Saved vs Minimax (%)": f"{sav_mm_mean:.1f}% [{sav_mm_lo:.1f}, {sav_mm_hi:.1f}]" if not np.isnan(sav_mm_mean) else "N/A",
                "Saved vs Natural AB (%)": f"{sav_nat_mean:.1f}% [{sav_nat_lo:.1f}, {sav_nat_hi:.1f}]",
            })

    df_summary = pd.DataFrame(summary_rows)
    tables = save_table(
        df_summary,
        name="partC_pruning",
        caption=f"Minimax vs Alpha-Beta Pruning Across Depths 2..{max_depth} ({len(samples)} Positions, Random Seed={RANDOM_ORDER_SEED})",
    )
    save_table(df_raw, name="partC_pruning_raw", caption="Raw Alpha-Beta Pruning Experiment Log")

    # Plot evaluations vs depth across orderings
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"natural": "#1976d2", "random": "#fb8c00", "own": "#2e7d32"}
    markers = {"natural": "s--", "random": "^:", "own": "o-"}

    for ord_name in ["natural", "random", "own"]:
        means = [df_raw[(df_raw["depth"] == d) & (df_raw["ordering"] == ord_name)]["ab_evaluations"].mean() for d in depths]
        ax.plot(depths, means, markers[ord_name], color=colors[ord_name], lw=2.2, label=f"Alpha-Beta ({ord_name})")

    mm_depths = [d for d in depths if df_raw[(df_raw["depth"] == d) & (df_raw["ordering"] == "natural")]["mm_evaluations"].notna().any()]
    mm_means = [df_raw[(df_raw["depth"] == d) & (df_raw["ordering"] == "natural")]["mm_evaluations"].mean() for d in mm_depths]
    ax.plot(mm_depths, mm_means, "x-", color="#c62828", lw=2.0, label="Minimax (unpruned)")

    ax.set_yscale("log")
    ax.set_xlabel("Search Depth (Plies)")
    ax.set_ylabel("Mean Leaf Evaluations (Log Scale)")
    ax.set_title(f"Move Ordering & Alpha-Beta Pruning ({len(samples)} Sampled Positions)")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path, pdf_path = save_figure(fig, "partC_pruning_curves")
    plt.close(fig)

    print(f"\n[PRUNING COMPLETE] Value mismatches: {value_mismatches}")
    print(f"Saved summary table to {tables['md']}")
    print(f"Saved pruning figure to {png_path}")

    return {
        "n_positions": len(samples),
        "value_mismatches": value_mismatches,
        "summary_df": df_summary,
        "figure_png": png_path,
    }


if __name__ == "__main__":
    run_pruning_experiment(n_scenarios=30, target_positions=210)
