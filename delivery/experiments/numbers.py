"""Experiment Numbers Export and LaTeX Macro Generator (§6.1, M9.3).

Reads verified full-mode experiment tables under results/tables/, builds
results/numbers.json with completeness metadata, and generates
report/generated_numbers.tex (\\newcommand macros) without manual editing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from delivery.stamp import STAMP, banner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"{name}.csv", comment="#")


def _tex_safe_macro_name(key: str) -> str:
    """Converts a snake_case key to CamelCase letters-only TeX macro name."""
    parts = re.split(r"[^A-Za-z0-9]+", key)
    out = "".join(p.capitalize() for p in parts if p)
    # TeX macro names cannot contain digits
    digit_map = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    for d, word in digit_map.items():
        out = out.replace(d, word)
    return f"num{out}"


def export_numbers() -> Dict[str, Any]:
    """Aggregates verified full-mode numbers into results/numbers.json and report/generated_numbers.tex."""
    print(banner("M9.3: NUMBERS EXPORT & LATEX MACROS"))

    full_numbers: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    completeness: Dict[str, bool] = {}

    # 1. Part A Reflex
    if (TABLES_DIR / "partA_reflex_scenarios.csv").exists():
        df_a = _read_csv("partA_reflex_scenarios")
        n_a = len(df_a)
        succ_a = int((df_a["Success"] == True).sum())  # noqa: E712
        full_numbers["partA_reflex_scenarios"] = n_a
        full_numbers["partA_reflex_successes"] = succ_a
        full_numbers["partA_reflex_success_pct"] = round(100.0 * succ_a / max(1, n_a), 1)
        provenance["partA_reflex"] = "results/tables/partA_reflex_scenarios.csv"
        completeness["partA_reflex"] = n_a >= 30
    else:
        completeness["partA_reflex"] = False

    # 2. Part B Heuristics & Six Algorithms
    if (TABLES_DIR / "partB_heuristics_raw.csv").exists():
        df_bh = _read_csv("partB_heuristics_raw")
        ucs_sub = df_bh[df_bh["algo"] == "ucs"]
        own_sub = df_bh[(df_bh["algo"] == "astar") & (df_bh["heuristic"] == "own")]
        b2_sub = df_bh[(df_bh["algo"] == "astar") & (df_bh["heuristic"] == "b2")]

        full_numbers["partB_scenarios"] = int(len(own_sub))
        full_numbers["partB_ucs_mean_exp"] = round(float(ucs_sub["expanded_nodes"].mean()), 1)
        full_numbers["partB_b2_mean_exp"] = round(float(b2_sub["expanded_nodes"].mean()), 1)
        full_numbers["partB_own_mean_exp"] = round(float(own_sub["expanded_nodes"].mean()), 1)
        full_numbers["partB_own_mean_cost"] = round(float(own_sub["cost"].mean()), 2)
        provenance["partB_heuristics"] = "results/tables/partB_heuristics_raw.csv"
        completeness["partB_heuristics"] = len(own_sub) >= 30
    else:
        completeness["partB_heuristics"] = False

    # 3. Part C Pruning & Final Games
    if (TABLES_DIR / "partC_pruning_raw.csv").exists():
        df_cp = _read_csv("partC_pruning_raw")
        n_pos = int(df_cp["pos_idx"].nunique())
        d6_own = df_cp[(df_cp["depth"] == 6) & (df_cp["ordering"] == "own")]
        full_numbers["partC_pruning_positions"] = n_pos
        full_numbers["partC_pruning_d6_own_evals"] = round(float(d6_own["ab_evaluations"].mean()), 1)
        full_numbers["partC_pruning_d6_saved_mm_pct"] = round(float(d6_own["saved_vs_mm_pct"].dropna().mean()), 1)
        provenance["partC_pruning"] = "results/tables/partC_pruning_raw.csv"
        completeness["partC_pruning"] = n_pos >= 200
    else:
        completeness["partC_pruning"] = False

    if (TABLES_DIR / "partC_games_raw.csv").exists():
        df_cg = _read_csv("partC_games_raw")
        tuned = df_cg[df_cg["config"] == "Tuned Weights"]
        default = df_cg[df_cg["config"] == "Default Weights"]
        full_numbers["partC_games_count"] = int(len(tuned))
        full_numbers["partC_tuned_win_pct"] = round(float(tuned["win"].mean() * 100.0), 1)
        full_numbers["partC_tuned_mean_cost"] = round(float(tuned["cost"].mean()), 1)
        full_numbers["partC_default_mean_cost"] = round(float(default["cost"].mean()), 1)
        provenance["partC_games"] = "results/tables/partC_games_raw.csv"
        completeness["partC_games"] = len(tuned) >= 30
    else:
        completeness["partC_games"] = False

    # 4. Part D LLM Dispatcher
    if (TABLES_DIR / "partD_llm_raw.csv").exists():
        df_d = _read_csv("partD_llm_raw")
        noval = df_d[df_d["mode"] == "ReAct_NoValidation"]
        withval = df_d[df_d["mode"] == "ReAct_WithValidation"]
        full_numbers["partD_requests"] = int(withval["request_id"].nunique())
        full_numbers["partD_noval_success_pct"] = round(float(noval["passed"].mean() * 100.0), 1)
        full_numbers["partD_withval_success_pct"] = round(float(withval["passed"].mean() * 100.0), 1)
        provenance["partD_llm"] = "results/tables/partD_llm_raw.csv"
        completeness["partD_llm"] = withval["request_id"].nunique() == 20
    else:
        completeness["partD_llm"] = False

    # 5. M9.1 Ablation
    if (TABLES_DIR / "ablation_raw.csv").exists():
        df_ab = _read_csv("ablation_raw")
        mst_only = df_ab[df_ab["component"].str.contains("MST-Only")]
        full_own = df_ab[df_ab["component"].str.contains("Full Own")]
        full_numbers["m9_ablation_mst_only_exp"] = round(float(mst_only["expanded"].mean()), 1)
        full_numbers["m9_ablation_full_own_exp"] = round(float(full_own["expanded"].mean()), 1)
        provenance["ablation"] = "results/tables/ablation_raw.csv"
        completeness["ablation"] = len(full_own) >= 30
    else:
        completeness["ablation"] = False

    payload = {
        "stamp": STAMP,
        "mode": "full",
        "completeness": completeness,
        "provenance": provenance,
        "full": full_numbers,
        "quick": {},
        "timing": {},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    numbers_path = RESULTS_DIR / "numbers.json"
    with open(numbers_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = REPORT_DIR / "generated_numbers.tex"
    tex_lines = [
        f"% Auto-generated report numbers | {STAMP}",
        "% Do not edit manually; regenerate via python -m delivery.experiments.numbers",
    ]
    for k, v in sorted(full_numbers.items()):
        macro = _tex_safe_macro_name(k)
        tex_lines.append(f"\\newcommand{{\\{macro}}}{{{v}}}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    print(f"[NUMBERS EXPORT COMPLETE] Wrote {numbers_path} and {tex_path}")
    return payload


if __name__ == "__main__":
    export_numbers()
