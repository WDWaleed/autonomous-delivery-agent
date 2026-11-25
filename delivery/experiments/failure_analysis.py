"""Cross-Stage Failure Analysis Aggregator (§6.1, M9.2).

Aggregates actual failures traced directly to raw experiment CSVs across all stages:
- Part A: Reflex agent failures (local minima / step cap)
- Part B: Planner DNF / CUTOFF / NODELIMIT (e.g. DLS, IDS on full 5-stop instances)
- Part C: Game losses (captures) and draws (step-cap / excess cost)
- Part D: LLM dispatcher failures (rate limits, tool schema/JSON errors, premature stop)
- UI / Runtime: Explicitly labelled as PENDING_M10 (not assumed zero before M10 manual test)

Exports stamped results/tables/failure_analysis.csv/.md/.tex.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from delivery.stamp import banner, save_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def _read_csv_skip_stamp(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def run_failure_analysis() -> pd.DataFrame:
    """Aggregates cross-stage failures from raw experiment tables."""
    print(banner("M9.2: CROSS-STAGE FAILURE ANALYSIS"))

    rows: List[Dict[str, Any]] = []

    # 1. Part A Reflex Failures
    reflex_raw = TABLES_DIR / "partA_reflex_scenarios.csv"
    if reflex_raw.exists():
        df_a = _read_csv_skip_stamp(reflex_raw)
        fails_a = df_a[df_a["Success"] == False]  # noqa: E712
        ex_seed = str(fails_a.iloc[0]["Key"]) if len(fails_a) > 0 else "N/A"
        rows.append({
            "Category": "Part A: Reflex Agent Failures",
            "Count": int(len(fails_a)),
            "Total_Evaluated": int(len(df_a)),
            "Example_Seed_Or_Request": ex_seed,
            "Root_Cause": "Stateless greedy/reflex agent trapped in local minima/wall oscillations without memory.",
            "Mitigation": "Replace with stateful A* search planner (Part B).",
        })
    else:
        rows.append({
            "Category": "Part A: Reflex Agent Failures",
            "Count": "MISSING_LOG",
            "Total_Evaluated": "N/A",
            "Example_Seed_Or_Request": "N/A",
            "Root_Cause": "partA_reflex_scenarios.csv not found.",
            "Mitigation": "Run stage_reflex.",
        })

    # 2. Part B Planner DNF / Limits
    six_raw = TABLES_DIR / "partB_six_algorithms_raw.csv"
    if six_raw.exists():
        df_b = _read_csv_skip_stamp(six_raw)
        fails_b = df_b[df_b["success"] == False]  # noqa: E712
        ex_b = f"{fails_b.iloc[0]['algo']} ({fails_b.iloc[0]['scenario_key']})" if len(fails_b) > 0 else "N/A"
        rows.append({
            "Category": "Part B: Uninformed Bounded Search DNF (DLS/IDS)",
            "Count": int(len(fails_b)),
            "Total_Evaluated": int(len(df_b)),
            "Example_Seed_Or_Request": ex_b,
            "Root_Cause": "DLS(d=10) hits CUTOFF below optimal depth; DLS(d=25)/IDS hit NODELIMIT on 5-stop branching.",
            "Mitigation": "Use A* with admissible CombinedMSTHeuristic (30/30 solved).",
        })
    else:
        rows.append({
            "Category": "Part B: Uninformed Bounded Search DNF (DLS/IDS)",
            "Count": "MISSING_LOG",
            "Total_Evaluated": "N/A",
            "Example_Seed_Or_Request": "N/A",
            "Root_Cause": "partB_six_algorithms_raw.csv not found.",
            "Mitigation": "Run stage_six_algorithms.",
        })

    # 3. Part C Game Losses (Captures) & Draws
    games_raw = TABLES_DIR / "partC_games_raw.csv"
    if games_raw.exists():
        df_c = _read_csv_skip_stamp(games_raw)
        losses_c = df_c[df_c["loss"] == True]  # noqa: E712
        draws_c = df_c[df_c["draw"] == True]  # noqa: E712
        ex_loss = f"{losses_c.iloc[0]['config']} k={losses_c.iloc[0]['scenario_k']}" if len(losses_c) > 0 else "N/A"
        ex_draw = f"{draws_c.iloc[0]['config']} k={draws_c.iloc[0]['scenario_k']}" if len(draws_c) > 0 else "N/A"
        rows.append({
            "Category": "Part C: Rival Capture Losses",
            "Count": int(len(losses_c)),
            "Total_Evaluated": int(len(df_c)),
            "Example_Seed_Or_Request": ex_loss,
            "Root_Cause": "Stochastic slip (p=0.23) pushed agent toward Chaser or corridor trap.",
            "Mitigation": "Tactical Alpha-Beta window evasion + slip-triggered A* replanning.",
        })
        rows.append({
            "Category": "Part C: Step-Cap / Excess-Cost Draws",
            "Count": int(len(draws_c)),
            "Total_Evaluated": int(len(df_c)),
            "Example_Seed_Or_Request": ex_draw,
            "Root_Cause": "Repeated rival evasion + slip detours exhausted T_max or kappa*C* budget.",
            "Mitigation": "Route-heavy evaluation weights (tuned) reduced draws and mean cost.",
        })

    # 4. Part D LLM Dispatcher Failures
    llm_fail = TABLES_DIR / "partD_failures.csv"
    llm_raw = TABLES_DIR / "partD_llm_raw.csv"
    if llm_fail.exists() and llm_raw.exists():
        df_dfail = _read_csv_skip_stamp(llm_fail)
        df_draw = _read_csv_skip_stamp(llm_raw)
        real_fails = df_dfail[df_dfail["request_id"] != "NONE"]
        ex_d = f"{real_fails.iloc[0]['request_id']} ({real_fails.iloc[0]['mode']})" if len(real_fails) > 0 else "N/A"
        rows.append({
            "Category": "Part D: LLM Dispatcher Failures (Live Groq)",
            "Count": int(len(real_fails)),
            "Total_Evaluated": int(len(df_draw[df_draw["mode"] != "Direct_Planner"])),
            "Example_Seed_Or_Request": ex_d,
            "Root_Cause": "Groq free-tier HTTP 429 rate limits, tool JSON schema errors, and premature stop (REQ-09).",
            "Mitigation": "ArgumentValidator improved accuracy (+5% WithVal vs NoVal) and blocked contradictions.",
        })

    # 5. UI / Runtime Failures (explicitly labelled pending M10)
    rows.append({
        "Category": "UI / Runtime Failures",
        "Count": "PENDING_M10",
        "Total_Evaluated": "0",
        "Example_Seed_Or_Request": "N/A",
        "Root_Cause": "Streamlit UI (M10) manual verification not yet executed.",
        "Mitigation": "Complete M10.4 manual checklist before final v0.5.0 release.",
    })

    df_out = pd.DataFrame(rows)
    tables = save_table(
        df_out,
        name="failure_analysis",
        caption="Cross-Stage Failure Analysis Summary Traced to Raw Logs",
    )
    print(f"[FAILURE ANALYSIS COMPLETE] Saved to {tables['md']}")
    return df_out


if __name__ == "__main__":
    run_failure_analysis()
