"""Tests for Dispatcher Tools (§7.6, M8.2)."""

import pytest

from delivery.llm.client import MockClient
from delivery.llm.tools import TOOL_SCHEMAS, DispatcherToolbox


def test_tool_schemas_well_formed():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {"plan_route", "simulate", "inspect_twists"}


def test_plan_route_and_simulate_deterministic():
    tb = DispatcherToolbox()
    r1 = tb.execute("plan_route", {"algorithm": "astar", "heuristic": "own"})
    r2 = tb.execute("plan_route", {"algorithm": "astar", "heuristic": "own"})

    assert r1["status"] == "SUCCESS"
    assert r1["cost"] == pytest.approx(r2["cost"])
    assert r1["plan"] == r2["plan"]

    sim1 = tb.execute("simulate", {"plan": r1["plan"], "slip_p": 0.0})
    sim2 = tb.execute("simulate", {"plan": r1["plan"], "slip_p": 0.0})
    assert sim1["status"] == "SUCCESS"
    assert sim1["cost"] == pytest.approx(r1["cost"])
    assert sim1["cost"] == pytest.approx(sim2["cost"])


def test_inspect_twists_and_invalid_args():
    tb = DispatcherToolbox()
    res = tb.execute("inspect_twists", {"pos": [4, 20], "fragile_delivered": False})
    assert res["status"] == "SUCCESS"
    assert "active_twists" in res["route_summary"]

    bad_algo = tb.execute("plan_route", {"algorithm": "nonexistent"})
    assert bad_algo["status"] == "INVALID_ARG"

    bad_pos = tb.execute("inspect_twists", {"pos": [999, 999]})
    assert bad_pos["status"] == "INVALID_ARG"


def test_mock_client_scripted_tool_flow():
    client = MockClient(
        responses=[
            {
                "content": "Planning route using A* with own heuristic.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "heuristic": "own"}'},
                    }
                ],
            }
        ]
    )
    resp = client.generate([{"role": "user", "content": "Find optimal route"}], tools=TOOL_SCHEMAS)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "plan_route"
