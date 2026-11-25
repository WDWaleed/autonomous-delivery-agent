"""Tests for Part D LLM Evaluation Stage using MockClient (§7.6, M8.6)."""

from delivery.experiments.stages.stage_llm import run_partD_evaluation
from delivery.llm.client import MockClient


def test_partD_stage_runs_cleanly_with_mock_client(tmp_path):
    client = MockClient(
        responses=[
            {
                "content": "Reason: Plan route with A*.",
                "tool_calls": [{"id": "1", "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "heuristic": "own"}'}}],
            },
            'Reason: Done.\nFINAL_ANSWER: {"status": "SUCCESS"}',
        ]
    )
    res = run_partD_evaluation(
        llm_mode="mock",
        reps=1,
        quick=True,
        client_override=client,
        trace_dir=tmp_path,
        save_outputs=False,
    )
    assert len(res["summary_df"]) == 3
    assert set(res["summary_df"]["Mode"]) == {"Direct_Planner", "ReAct_NoValidation", "ReAct_WithValidation"}
