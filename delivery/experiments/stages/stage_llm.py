"""Part D LLM Dispatcher Evaluation Stage (§6.1, §7.6, M8.6).

Evaluates the 20 dispatcher requests (data/requests.yaml) comparing:
1. Direct-Planner Baseline
2. ReAct Dispatcher WITHOUT Validation
3. ReAct Dispatcher WITH Argument Validation & Self-Repair

Records:
- Success, intent/tool accuracy, constraint satisfaction, plan validity, cost ratio
- Tool calls, validation interventions, latency
- Failure log (wrong_tool, invented_argument, ignored_constraint, invalid_plan, unrecognised_infeasibility)
- Stamped traces under results/traces/
- Comparison figure results/figures/partD_llm_comparison.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delivery.experiments.stats import mean_ci, wilson_ci
from delivery.llm.checker import check_dispatcher_result, load_requests, run_direct_planner
from delivery.llm.client import LLMClient, OpenAICompatibleClient, ReplayClient
from delivery.llm.react import PROMPT_VERSION, ReActResult, run_react_dispatcher
from delivery.llm.tools import DispatcherToolbox
from delivery.llm.validation import ArgumentValidator
from delivery.stamp import STAMP, banner, save_figure, save_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRACES_DIR = PROJECT_ROOT / "results" / "traces"


def _run_with_backoff(fn, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg and attempt < max_retries:
                wait_s = 5.0 * (attempt + 1)
                print(f"    [429 rate limit] backing off {wait_s:.0f}s...", flush=True)
                time.sleep(wait_s)
                continue
            raise


def run_partD_evaluation(
    llm_mode: str = "live",
    reps: int = 1,
    quick: bool = False,
    client_override: Optional[LLMClient] = None,
    trace_dir: Optional[Path] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Runs the Part D 20-request evaluation with and without validation."""
    print(banner(f"PART D: LLM DISPATCHER EVALUATION (mode={llm_mode}, R={reps})"))

    reqs = load_requests()
    if quick:
        reqs = reqs[:5]

    if client_override is not None:
        client = client_override
    elif llm_mode == "replay":
        client = ReplayClient()
    else:
        client = OpenAICompatibleClient()

    print(f"Backend: provider={client.provider}, model={client.model}, prompt={PROMPT_VERSION}")

    toolbox = DispatcherToolbox()
    validator = ArgumentValidator(toolbox.instance)

    raw_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    saved_traces: Dict[str, ReActResult] = {}

    for req in reqs:
        direct_res = run_direct_planner(req, toolbox)
        direct_ok = (
            (req.expected_outcome == "PLAN" and direct_res.get("status") == "SUCCESS")
            or (req.expected_outcome == "REFUSE_INFEASIBLE" and direct_res.get("status") != "SUCCESS")
            or (req.expected_outcome == "CLARIFY" and direct_res.get("status") == "CLARIFY_REQUIRED")
        )
        raw_rows.append({
            "request_id": req.id,
            "level": req.level,
            "mode": "Direct_Planner",
            "rep": 1,
            "passed": direct_ok,
            "intent_accurate": direct_ok,
            "tool_accurate": True,
            "constraints_satisfied": direct_ok,
            "plan_valid": direct_ok,
            "cost_ratio": 1.0 if req.expected_outcome == "PLAN" else np.nan,
            "tool_calls": 1 if req.expected_outcome != "CLARIFY" else 0,
            "interventions": 0,
            "latency_s": 0.01,
            "failure_category": None,
            "cached": True,
        })

        for r_idx in range(1, reps + 1):
            bypass = (llm_mode == "live")

            # 1. Without Validation
            try:
                res_noval = _run_with_backoff(
                    lambda: run_react_dispatcher(
                        request_text=req.text,
                        client=client,
                        toolbox=toolbox,
                        request_id=f"{req.id}_noval_r{r_idx}",
                        max_steps=6,
                        validator=None,
                        bypass_cache=bypass,
                        save_trace=True,
                        trace_dir=trace_dir or TRACES_DIR,
                    )
                )
            except Exception as e:
                res_noval = ReActResult(
                    request_id=f"{req.id}_noval_r{r_idx}",
                    status="ERROR",
                    final_answer=str(e),
                    provider=client.provider,
                    model=client.model,
                )

            chk_noval = check_dispatcher_result(req, res_noval, direct_res, toolbox)
            raw_rows.append({
                "request_id": req.id,
                "level": req.level,
                "mode": "ReAct_NoValidation",
                "rep": r_idx,
                "passed": chk_noval.passed,
                "intent_accurate": chk_noval.intent_accurate,
                "tool_accurate": chk_noval.tool_accurate,
                "constraints_satisfied": chk_noval.constraints_satisfied,
                "plan_valid": chk_noval.plan_valid,
                "cost_ratio": chk_noval.cost_ratio if chk_noval.cost_ratio is not None else np.nan,
                "tool_calls": res_noval.tool_calls_count,
                "interventions": 0,
                "latency_s": res_noval.total_latency_seconds,
                "failure_category": chk_noval.failure_category,
                "cached": any(st.cached for st in res_noval.steps),
            })
            if not chk_noval.passed:
                failure_rows.append({
                    "request_id": req.id,
                    "mode": "ReAct_NoValidation",
                    "rep": r_idx,
                    "category": chk_noval.failure_category or "unknown",
                    "tool_calls": ", ".join(st.tool_name for st in res_noval.steps if st.tool_name) or "none",
                    "evidence": chk_noval.reason,
                    "root_cause": res_noval.final_answer[:140],
                })

            # 2. With Validation
            validator.reset()
            try:
                res_val = _run_with_backoff(
                    lambda: run_react_dispatcher(
                        request_text=req.text,
                        client=client,
                        toolbox=toolbox,
                        request_id=f"{req.id}_val_r{r_idx}",
                        max_steps=6,
                        validator=validator,
                        max_repair_attempts=2,
                        bypass_cache=bypass,
                        save_trace=True,
                        trace_dir=trace_dir or TRACES_DIR,
                    )
                )
            except Exception as e:
                res_val = ReActResult(
                    request_id=f"{req.id}_val_r{r_idx}",
                    status="ERROR",
                    final_answer=str(e),
                    provider=client.provider,
                    model=client.model,
                )

            saved_traces[req.id] = res_val
            chk_val = check_dispatcher_result(req, res_val, direct_res, toolbox)
            raw_rows.append({
                "request_id": req.id,
                "level": req.level,
                "mode": "ReAct_WithValidation",
                "rep": r_idx,
                "passed": chk_val.passed,
                "intent_accurate": chk_val.intent_accurate,
                "tool_accurate": chk_val.tool_accurate,
                "constraints_satisfied": chk_val.constraints_satisfied,
                "plan_valid": chk_val.plan_valid,
                "cost_ratio": chk_val.cost_ratio if chk_val.cost_ratio is not None else np.nan,
                "tool_calls": res_val.tool_calls_count,
                "interventions": res_val.validation_interventions,
                "latency_s": res_val.total_latency_seconds,
                "failure_category": chk_val.failure_category,
                "cached": any(st.cached for st in res_val.steps),
            })
            if not chk_val.passed:
                failure_rows.append({
                    "request_id": req.id,
                    "mode": "ReAct_WithValidation",
                    "rep": r_idx,
                    "category": chk_val.failure_category or "unknown",
                    "tool_calls": ", ".join(st.tool_name for st in res_val.steps if st.tool_name) or "none",
                    "evidence": chk_val.reason,
                    "root_cause": res_val.final_answer[:140],
                })

            print(
                f"  [{req.id}] ({req.level}) | NoVal={'PASS' if chk_noval.passed else 'FAIL'} | "
                f"WithVal={'PASS' if chk_val.passed else 'FAIL'} (interventions={res_val.validation_interventions})",
                flush=True,
            )

    df_raw = pd.DataFrame(raw_rows)

    # Cluster-aware aggregation by request_id first, then across requests
    summary_rows: List[Dict[str, Any]] = []
    for mode in ["Direct_Planner", "ReAct_NoValidation", "ReAct_WithValidation"]:
        sub = df_raw[df_raw["mode"] == mode]
        # Aggregate per request_id (cluster-aware)
        req_agg = sub.groupby("request_id").mean(numeric_only=True)
        n_reqs = len(req_agg)
        n_pass = int((req_agg["passed"] >= 0.5).sum())
        sr, sr_lo, sr_hi = wilson_ci(n_pass, n_reqs)

        cr_vals = req_agg["cost_ratio"].dropna()
        cr_mean = cr_vals.mean() if len(cr_vals) > 0 else np.nan
        lat_mean = req_agg["latency_s"].mean()

        summary_rows.append({
            "Mode": mode,
            "Requests": n_reqs,
            "Success Rate (95% Wilson CI)": f"{sr*100:.1f}% [{sr_lo*100:.1f}, {sr_hi*100:.1f}]",
            "Tool Accuracy": f"{req_agg['tool_accurate'].mean()*100:.1f}%",
            "Constraint Sat.": f"{req_agg['constraints_satisfied'].mean()*100:.1f}%",
            "Mean Cost Ratio": f"{cr_mean:.3f}" if not np.isnan(cr_mean) else "N/A",
            "Mean Tool Calls": f"{req_agg['tool_calls'].mean():.2f}",
            "Interventions": f"{req_agg['interventions'].mean():.2f}",
            "Mean Latency (s)": f"{lat_mean:.2f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_failures = pd.DataFrame(failure_rows) if failure_rows else pd.DataFrame(
        [{"request_id": "NONE", "mode": "-", "rep": 0, "category": "none", "tool_calls": "-", "evidence": "Zero failures observed", "root_cause": "-"}]
    )

    if save_outputs:
        tables = save_table(
            df_summary,
            name="partD_llm_summary",
            caption=f"Part D LLM Dispatcher vs Direct Planner ({client.provider}/{client.model}, mode={llm_mode}, R={reps})",
        )
        save_table(df_raw, name="partD_llm_raw", caption="Raw Per-Request Part D Evaluation Log")
        save_table(df_failures, name="partD_failures", caption="Part D Dispatcher Failure Log")

        # Comparison figure
        fig, ax = plt.subplots(figsize=(9, 4.5))
        modes = ["Direct_Planner", "ReAct_NoValidation", "ReAct_WithValidation"]
        rates = [
            df_raw[df_raw["mode"] == m].groupby("request_id")["passed"].mean().mean() * 100.0
            for m in modes
        ]
        colors = ["#546e7a", "#ef6c00", "#2e7d32"]
        ax.bar(modes, rates, color=colors)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Request Success Rate (%)")
        ax.set_title(f"LLM Dispatcher Accuracy: Direct vs ReAct (With & Without Validation) [{client.model}]")
        for idx, val in enumerate(rates):
            ax.text(idx, val + 2.5, f"{val:.1f}%", ha="center", weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        png_path, _ = save_figure(fig, "partD_llm_comparison")
        plt.close(fig)

        print(f"\n[PART D COMPLETE] Summary saved to {tables['md']}")
        print(f"Comparison figure saved to {png_path}")

    return {
        "summary_df": df_summary,
        "raw_df": df_raw,
        "failures_df": df_failures,
        "traces": saved_traces,
        "provider": client.provider,
        "model": client.model,
    }


if __name__ == "__main__":
    run_partD_evaluation(llm_mode="live", reps=1)
