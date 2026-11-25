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

from delivery.stamp import STAMP, banner, get_results_dir, get_tables_dir, is_quick_mode

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "report"


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(get_tables_dir() / f"{name}.csv", comment="#")


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
    """Aggregates verified experiment numbers into numbers.json (and generated_numbers.tex in full mode)."""
    quick = is_quick_mode()
    tables_dir = get_tables_dir()
    results_dir = get_results_dir()
    mode_label = "QUICK" if quick else "FULL"
    print(banner(f"M9.3: NUMBERS EXPORT & LATEX MACROS ({mode_label})"))

    extracted: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    completeness: Dict[str, bool] = {}

    # 1. Part A Reflex
    if (tables_dir / "partA_reflex_scenarios.csv").exists():
        df_a = _read_csv("partA_reflex_scenarios")
        n_a = len(df_a)
        succ_a = int((df_a["Success"] == True).sum())  # noqa: E712
        extracted["partA_reflex_scenarios"] = n_a
        extracted["partA_reflex_successes"] = succ_a
        extracted["partA_reflex_success_pct"] = round(100.0 * succ_a / max(1, n_a), 1)
        provenance["partA_reflex"] = str((tables_dir / "partA_reflex_scenarios.csv").relative_to(PROJECT_ROOT)).replace("\\", "/")
        completeness["partA_reflex"] = n_a >= 30
    else:
        completeness["partA_reflex"] = False

    # 2. Part B Heuristics & Six Algorithms
    if (tables_dir / "partB_heuristics_raw.csv").exists():
        df_bh = _read_csv("partB_heuristics_raw")
        ucs_sub = df_bh[df_bh["algo"] == "ucs"]
        own_sub = df_bh[(df_bh["algo"] == "astar") & (df_bh["heuristic"] == "own")]
        b2_sub = df_bh[(df_bh["algo"] == "astar") & (df_bh["heuristic"] == "b2")]

        extracted["partB_scenarios"] = int(len(own_sub))
        extracted["partB_ucs_mean_exp"] = round(float(ucs_sub["expanded_nodes"].mean()), 1)
        extracted["partB_b2_mean_exp"] = round(float(b2_sub["expanded_nodes"].mean()), 1)
        extracted["partB_own_mean_exp"] = round(float(own_sub["expanded_nodes"].mean()), 1)
        extracted["partB_own_mean_cost"] = round(float(own_sub["cost"].mean()), 2)
        provenance["partB_heuristics"] = str((tables_dir / "partB_heuristics_raw.csv").relative_to(PROJECT_ROOT)).replace("\\", "/")
        completeness["partB_heuristics"] = len(own_sub) >= 30
    else:
        completeness["partB_heuristics"] = False

    # 3. Part C Pruning & Final Games
    if (tables_dir / "partC_pruning_raw.csv").exists():
        df_cp = _read_csv("partC_pruning_raw")
        n_pos = int(df_cp["pos_idx"].nunique())
        d6_own = df_cp[(df_cp["depth"] == 6) & (df_cp["ordering"] == "own")]
        extracted["partC_pruning_positions"] = n_pos
        extracted["partC_pruning_d6_own_evals"] = round(float(d6_own["ab_evaluations"].mean()), 1) if len(d6_own) > 0 else 0.0
        extracted["partC_pruning_d6_saved_mm_pct"] = round(float(d6_own["saved_vs_mm_pct"].dropna().mean()), 1) if len(d6_own) > 0 else 0.0
        provenance["partC_pruning"] = str((tables_dir / "partC_pruning_raw.csv").relative_to(PROJECT_ROOT)).replace("\\", "/")
        completeness["partC_pruning"] = n_pos >= 200
    else:
        completeness["partC_pruning"] = False

    if (tables_dir / "partC_games_raw.csv").exists():
        df_cg = _read_csv("partC_games_raw")
        tuned = df_cg[df_cg["config"] == "Tuned Weights"]
        default = df_cg[df_cg["config"] == "Default Weights"]
        extracted["partC_games_count"] = int(len(tuned))
        extracted["partC_tuned_win_pct"] = round(float(tuned["win"].mean() * 100.0), 1)
        extracted["partC_tuned_mean_cost"] = round(float(tuned["cost"].mean()), 1)
        extracted["partC_default_mean_cost"] = round(float(default["cost"].mean()), 1)
        provenance["partC_games"] = str((tables_dir / "partC_games_raw.csv").relative_to(PROJECT_ROOT)).replace("\\", "/")
        completeness["partC_games"] = len(tuned) >= 30
    else:
        completeness["partC_games"] = False

    # 4. Part D LLM Dispatcher
    if (tables_dir / "partD_llm_raw.csv").exists():
        df_d = _read_csv("partD_llm_raw")
        noval = df_d[df_d["mode"] == "ReAct_NoValidation"]
        withval = df_d[df_d["mode"] == "ReAct_WithValidation"]
        extracted["partD_requests"] = int(withval["request_id"].nunique())
        extracted["partD_noval_success_pct"] = round(float(noval["passed"].mean() * 100.0), 1)
        extracted["partD_withval_success_pct"] = round(float(withval["passed"].mean() * 100.0), 1)
        provenance["partD_llm"] = str((tables_dir / "partD_llm_raw.csv").relative_to(PROJECT_ROOT)).replace("\\", "/")
        completeness["partD_llm"] = withval["request_id"].nunique() == 20
    else:
        completeness["partD_llm"] = False

    # 5. M9.1 Ablation
    if (tables_dir / "ablation_raw.csv").exists():
        df_ab = _read_csv("ablation_raw")
        mst_only = df_ab[df_ab["component"].str.contains("MST-Only")]
        full_own = df_ab[df_ab["component"].str.contains("Full Own")]
        extracted["m9_ablation_mst_only_exp"] = round(float(mst_only["expanded"].mean()), 1)
        extracted["m9_ablation_full_own_exp"] = round(float(full_own["expanded"].mean()), 1)
        provenance["ablation"] = str((tables_dir / "ablation_raw.csv").relative_to(PROJECT_ROOT)).replace("\\", "/")
        completeness["ablation"] = len(full_own) >= 30
    else:
        completeness["ablation"] = False

    payload = {
        "stamp": STAMP,
        "mode": "quick" if quick else "full",
        "completeness": completeness,
        "provenance": provenance,
        "full": {} if quick else extracted,
        "quick": extracted if quick else {},
        "timing": {},
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    numbers_path = results_dir / "numbers.json"
    with open(numbers_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not quick:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        tex_path = REPORT_DIR / "generated_numbers.tex"
        tex_lines = [
            f"% Auto-generated report numbers | {STAMP}",
            "% Do not edit manually; regenerate via python -m delivery.experiments.numbers",
        ]
        for k, v in sorted(extracted.items()):
            macro = _tex_safe_macro_name(k)
            tex_lines.append(f"\\newcommand{{\\{macro}}}{{{v}}}")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tex_lines) + "\n")
        print(f"[NUMBERS EXPORT COMPLETE] Wrote {numbers_path} and {tex_path}")
    else:
        print(f"[NUMBERS EXPORT COMPLETE (QUICK)] Wrote {numbers_path} (report macros untouched)")

    return payload


if __name__ == "__main__":
    export_numbers()
