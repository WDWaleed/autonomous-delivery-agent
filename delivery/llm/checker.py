"""Dispatcher Request Loader, Direct-Planner Baseline, and Automatic Checker (§7.6, M8.4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from delivery.llm.react import ReActResult
from delivery.llm.tools import DispatcherToolbox

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REQUESTS_PATH = PROJECT_ROOT / "data" / "requests.yaml"


@dataclass
class DispatcherRequest:
    """Single natural-language dispatcher benchmark request."""

    id: str
    level: str
    text: str
    expected_outcome: str  # PLAN, CLARIFY, REFUSE_INFEASIBLE
    expected_tool: str
    ground_truth_spec: Dict[str, Any]


def load_requests(path: Optional[Path] = None) -> List[DispatcherRequest]:
    target = path or DEFAULT_REQUESTS_PATH
    with open(target, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [
        DispatcherRequest(
            id=item["id"],
            level=item["level"],
            text=item["text"],
            expected_outcome=item["expected_outcome"],
            expected_tool=item["expected_tool"],
            ground_truth_spec=dict(item.get("ground_truth_spec", {})),
        )
        for item in raw
    ]


def run_direct_planner(req: DispatcherRequest, toolbox: DispatcherToolbox) -> Dict[str, Any]:
    """Direct-planner baseline: converts ground-truth spec directly into one plan_route call."""
    if req.expected_outcome == "CLARIFY":
        return {"status": "CLARIFY_REQUIRED", "cost": None, "plan": None}

    spec = req.ground_truth_spec
    return toolbox.plan_route(
        algorithm=spec.get("algorithm", "astar"),
        heuristic=spec.get("heuristic", "own"),
        constraints=spec.get("constraints", {}),
    )


@dataclass
class CheckResult:
    """Automated evaluation outcome for a dispatcher response against ground truth."""

    request_id: str
    level: str
    expected_outcome: str
    passed: bool
    intent_accurate: bool
    tool_accurate: bool
    constraints_satisfied: bool
    plan_valid: bool
    cost_ratio: Optional[float]
    failure_category: Optional[str] = None
    reason: str = ""


def check_dispatcher_result(
    req: DispatcherRequest,
    react_res: ReActResult,
    direct_res: Dict[str, Any],
    toolbox: DispatcherToolbox,
) -> CheckResult:
    """Checks ReAct dispatcher result against ground-truth spec and simulator."""
    payload = react_res.final_payload or {}
    ans_lower = (react_res.final_answer or "").lower()
    used_tools = [st.tool_name for st in react_res.steps if st.tool_name]

    # 1. Ambiguous / CLARIFY requests
    if req.expected_outcome == "CLARIFY":
        clarified = (
            payload.get("status") in ("CLARIFY", "AMBIGUOUS")
            or any(w in ans_lower for w in ("clarif", "ambigu", "specify", "which stop", "unclear", "undefined"))
        )
        return CheckResult(
            request_id=req.id,
            level=req.level,
            expected_outcome=req.expected_outcome,
            passed=clarified,
            intent_accurate=clarified,
            tool_accurate=True,
            constraints_satisfied=True,
            plan_valid=True,
            cost_ratio=None,
            failure_category=None if clarified else "ignored_constraint",
            reason="Correctly requested clarification" if clarified else "Failed to identify ambiguity",
        )

    # 2. Infeasible / Conflicting requests
    if req.expected_outcome == "REFUSE_INFEASIBLE":
        identified_infeasible = (
            payload.get("status") in ("INFEASIBLE", "FAILED", "REFUSED", "COSTLIMIT")
            or any(w in ans_lower for w in ("infeas", "impossib", "cannot", "conflict", "no valid", "exceed"))
        )
        tool_ok = ("plan_route" in used_tools) or identified_infeasible
        return CheckResult(
            request_id=req.id,
            level=req.level,
            expected_outcome=req.expected_outcome,
            passed=identified_infeasible,
            intent_accurate=identified_infeasible,
            tool_accurate=tool_ok,
            constraints_satisfied=identified_infeasible,
            plan_valid=True,
            cost_ratio=None,
            failure_category=None if identified_infeasible else "unrecognised_infeasibility",
            reason="Correctly identified infeasibility" if identified_infeasible else "Failed to identify infeasible request",
        )

    # 3. Feasible PLAN requests
    tool_ok = (req.expected_tool in used_tools) if req.expected_tool != "none" else True

    # Extract plan from payload or last successful plan_route observation
    plan = payload.get("plan")
    reported_cost = payload.get("cost")
    if plan is None:
        for st in reversed(react_res.steps):
            if st.tool_name == "plan_route" and st.observation and st.observation.get("status") == "SUCCESS":
                plan = st.observation.get("plan")
                reported_cost = st.observation.get("cost")
                break

    if not plan:
        return CheckResult(
            request_id=req.id,
            level=req.level,
            expected_outcome=req.expected_outcome,
            passed=False,
            intent_accurate=False,
            tool_accurate=tool_ok,
            constraints_satisfied=False,
            plan_valid=False,
            cost_ratio=None,
            failure_category="wrong_tool" if not tool_ok else "invalid_plan",
            reason="No valid plan produced",
        )

    # Verify plan in simulator on constrained world
    spec = req.ground_truth_spec
    constraints = spec.get("constraints", {})
    _, c_world, c_twists, _ = toolbox._build_constrained_instance(constraints)
    c_tb = DispatcherToolbox(
        instance=toolbox._build_constrained_instance(constraints)[0],
        world=c_world,
        twists=c_twists,
    )
    sim_res = c_tb.simulate(plan=plan, slip_p=0.0)
    plan_valid = sim_res["status"] == "SUCCESS" and sim_res["route_summary"]["illegal_steps"] == 0

    max_cost = constraints.get("max_cost")
    cost_ok = True if max_cost is None else (sim_res["cost"] <= float(max_cost) + 1e-9)

    direct_cost = direct_res.get("cost")
    cost_ratio = (sim_res["cost"] / direct_cost) if (direct_cost and direct_cost > 0) else 1.0

    passed = plan_valid and cost_ok and tool_ok
    fail_cat = None
    if not passed:
        if not tool_ok:
            fail_cat = "wrong_tool"
        elif not cost_ok:
            fail_cat = "ignored_constraint"
        else:
            fail_cat = "invalid_plan"

    return CheckResult(
        request_id=req.id,
        level=req.level,
        expected_outcome=req.expected_outcome,
        passed=passed,
        intent_accurate=True,
        tool_accurate=tool_ok,
        constraints_satisfied=cost_ok and plan_valid,
        plan_valid=plan_valid,
        cost_ratio=cost_ratio,
        failure_category=fail_cat,
        reason="Verified against simulator and ground truth" if passed else f"Failed check ({fail_cat})",
    )
