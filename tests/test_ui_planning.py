"""Tests for UI Planning Parity and Step-by-Step Execution (§7.7, M10.2)."""

import pytest

from delivery.search.astar import astar_search
from delivery.search.problem import DeliveryProblem
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.twists.registry import build_twist_stack
from delivery.ui.backend import UISimSession, build_configured_instance, run_ui_plan
from delivery.world.grid import GridWorld


def test_ui_plan_matches_cli_backend_parity():
    inst = build_configured_instance(
        slip_probability=0.0,
        start=(4, 20),
        stops=[(5, 15), (6, 10)],
        twists=["one_way_streets"],
        rival_depth=4,
        seed=42,
    )
    ui_res = run_ui_plan(inst, algo="astar", heuristic="own")

    world = GridWorld(inst)
    twists = build_twist_stack(inst, world)
    prob = DeliveryProblem(inst, world=world, twist_stack=twists)
    cli_res = astar_search(prob, CombinedMSTHeuristic(prob))

    assert ui_res["success"] is True
    assert ui_res["cost"] == pytest.approx(cli_res.cost)
    assert ui_res["plan"] == cli_res.plan
    assert ui_res["expanded"] == cli_res.metrics.expanded_nodes


def test_ui_plan_reports_honest_dnf_on_tight_dls():
    inst = build_configured_instance(
        slip_probability=0.0,
        start=(4, 20),
        stops=[(5, 15), (16, 1), (15, 12), (9, 7), (6, 10)],
        twists=["one_way_streets"],
        rival_depth=4,
        seed=42,
    )
    res = run_ui_plan(inst, algo="dls", depth_limit=5)
    assert res["success"] is False
    assert res["status"] in ("CUTOFF", "NODELIMIT", "FAILED")


def test_ui_sim_session_step_play_reset():
    inst = build_configured_instance(
        slip_probability=0.0,
        start=(4, 20),
        stops=[(5, 15)],
        twists=[],
        rival_depth=2,
        seed=42,
    )
    sess = UISimSession(inst, enable_rival=False)
    ev1 = sess.step_once()
    assert ev1["step"] == 1
    assert len(sess.trajectory) == 2

    sess.play_all(max_steps=50)
    assert sess.env.is_done() is True
    assert sess.env.total_cost > 0
