"""Tests for UI Rival Game Panel and LLM Dispatcher Tab (§7.7, M10.3)."""

from delivery.llm.client import MockClient
from delivery.ui.backend import (
    build_configured_instance,
    get_provider_status,
    run_ui_dispatcher,
    run_ui_rival_game,
)


def test_ui_rival_game_runs_and_reports_outcome_and_replans():
    inst = build_configured_instance(
        slip_probability=0.0,
        start=(4, 20),
        stops=[(5, 15)],
        twists=["one_way_streets"],
        rival_depth=2,
        seed=42,
    )
    res = run_ui_rival_game(inst, window_radius=4, use_tuned_weights=True, max_steps=60)
    assert res["outcome"] in ("WIN", "LOSS (CAPTURED)") or res["outcome"].startswith("DRAW")
    assert res["total_steps"] > 0
    assert "total_replans" in res["replans"]


def test_ui_dispatcher_and_provider_status_hides_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecret1234567890")
    status = get_provider_status()
    assert status["key_configured"] is True
    assert "gsk_supersecret" not in str(status)

    inst = build_configured_instance(
        slip_probability=0.23,
        start=(4, 20),
        stops=[(5, 15)],
        twists=[],
        rival_depth=4,
        seed=42,
    )
    res = run_ui_dispatcher(
        inst,
        request_text="Plan route to Stop 0.",
        llm_mode="mock",
        use_validation=True,
    )
    assert res["status"] == "COMPLETED"
    assert res["tool_calls_count"] == 1
