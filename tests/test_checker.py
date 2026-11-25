"""Tests for 20 Dispatcher Requests and Automatic Checker (§7.6, M8.4)."""

from delivery.llm.checker import check_dispatcher_result, load_requests, run_direct_planner
from delivery.llm.client import MockClient
from delivery.llm.react import run_react_dispatcher
from delivery.llm.tools import DispatcherToolbox


def test_load_all_20_requests_and_direct_planner_baseline():
    reqs = load_requests()
    assert len(reqs) == 20
    levels = {r.level for r in reqs}
    assert levels == {
        "single_constraint",
        "multiple_constraints",
        "twist_specific",
        "ambiguous",
        "infeasible_conflicting",
    }

    tb = DispatcherToolbox()
    for r in reqs:
        direct = run_direct_planner(r, tb)
        if r.expected_outcome == "PLAN":
            assert direct["status"] == "SUCCESS", f"{r.id} direct planner failed: {direct}"
        elif r.expected_outcome == "REFUSE_INFEASIBLE":
            assert direct["status"] != "SUCCESS", f"{r.id} should be infeasible, got {direct}"
        else:
            assert direct["status"] == "CLARIFY_REQUIRED"


def test_checker_evaluates_plan_clarify_and_infeasible():
    reqs = {r.id: r for r in load_requests()}
    tb = DispatcherToolbox()

    # 1. PLAN check
    r_plan = reqs["REQ-01"]
    direct1 = run_direct_planner(r_plan, tb)
    c1 = MockClient(
        responses=[
            {
                "content": "Reason: Run A* with own heuristic.",
                "tool_calls": [{"id": "1", "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "heuristic": "own"}'}}],
            },
            'Reason: Done.\nFINAL_ANSWER: {"status": "SUCCESS"}',
        ]
    )
    res1 = run_react_dispatcher(r_plan.text, c1, tb, request_id=r_plan.id)
    chk1 = check_dispatcher_result(r_plan, res1, direct1, tb)
    assert chk1.passed is True

    # 2. CLARIFY check
    r_clar = reqs["REQ-13"]
    c2 = MockClient(responses=['Reason: Stop is unspecified.\nFINAL_ANSWER: {"status": "CLARIFY", "message": "Please specify which stop."}'])
    res2 = run_react_dispatcher(r_clar.text, c2, tb, request_id=r_clar.id)
    chk2 = check_dispatcher_result(r_clar, res2, run_direct_planner(r_clar, tb), tb)
    assert chk2.passed is True

    # 3. INFEASIBLE check
    r_inf = reqs["REQ-17"]
    c3 = MockClient(
        responses=[
            {
                "content": "Reason: Check feasibility with max_cost=15.",
                "tool_calls": [{"id": "2", "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "heuristic": "own", "constraints": {"max_cost": 15.0}}'}}],
            },
            'Reason: Cost 15.0 is impossible.\nFINAL_ANSWER: {"status": "INFEASIBLE"}',
        ]
    )
    res3 = run_react_dispatcher(r_inf.text, c3, tb, request_id=r_inf.id)
    chk3 = check_dispatcher_result(r_inf, res3, run_direct_planner(r_inf, tb), tb)
    assert chk3.passed is True
