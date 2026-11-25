"""Tests for ReAct Dispatcher Loop and Safe Trace Logging (§7.6, M8.3)."""

from pathlib import Path

from delivery.llm.client import MockClient
from delivery.llm.react import run_react_dispatcher
from delivery.llm.tools import DispatcherToolbox
from delivery.stamp import STAMP


def test_react_loop_happy_path_and_trace_logging(tmp_path: Path):
    tb = DispatcherToolbox()
    client = MockClient(
        responses=[
            {
                "content": "Reason: Computing optimal route with A* and own heuristic.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "heuristic": "own"}'},
                    }
                ],
            },
            'Reason: Route planned successfully.\nFINAL_ANSWER: {"status": "SUCCESS", "cost": 215.0}',
        ]
    )

    res = run_react_dispatcher(
        request_text="Find the optimal delivery route.",
        client=client,
        toolbox=tb,
        request_id="TEST-REACT-1",
        max_steps=5,
        save_trace=True,
        trace_dir=tmp_path,
    )

    assert res.status == "COMPLETED"
    assert res.tool_calls_count == 1
    assert res.final_payload.get("status") == "SUCCESS"

    jsonl_file = tmp_path / "TEST-REACT-1.jsonl"
    txt_file = tmp_path / "TEST-REACT-1.txt"
    assert jsonl_file.exists()
    assert txt_file.exists()
    txt_content = txt_file.read_text(encoding="utf-8")
    assert STAMP in txt_content


def test_react_loop_handles_malformed_calls_and_step_budget():
    tb = DispatcherToolbox()
    client = MockClient(
        responses=[
            "Action: plan_route\nAction Input: {bad json",
            "Incomplete text with no action or final answer",
        ]
    )
    res = run_react_dispatcher(
        request_text="Plan route",
        client=client,
        toolbox=tb,
        request_id="TEST-BAD",
        max_steps=2,
    )
    assert res.status == "STEP_LIMIT"
    assert res.malformed_count == 2
    assert res.tool_calls_count == 0
