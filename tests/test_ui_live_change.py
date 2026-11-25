"""Tests for M10.4 Live Parameter Changes, Cache Invalidation, and CLI Rerun Parity."""

import time

import pytest

from delivery.cli import cmd_rerun
from delivery.ui.backend import build_configured_instance, compute_config_hash, run_ui_plan


def test_live_change_cache_invalidation_and_fast_rerun():
    base_cfg = {"algo": "astar", "heuristic": "own", "slip_p": 0.23, "start": [4, 20], "twists": ["one_way_streets"]}
    mod_cfg = {"algo": "astar", "heuristic": "own", "slip_p": 0.0, "start": [4, 20], "twists": ["one_way_streets"]}
    twist_cfg = {"algo": "astar", "heuristic": "own", "slip_p": 0.23, "start": [4, 20], "twists": []}

    h_base = compute_config_hash(base_cfg)
    h_mod = compute_config_hash(mod_cfg)
    h_twist = compute_config_hash(twist_cfg)
    assert h_base != h_mod
    assert h_base != h_twist

    t0 = time.perf_counter()
    inst1 = build_configured_instance(0.23, (4, 20), [(5, 15), (6, 10)], ["one_way_streets"], 4, 42)
    res1 = run_ui_plan(inst1, algo="astar", heuristic="own")
    inst2 = build_configured_instance(0.0, (4, 20), [(5, 15), (6, 10)], [], 4, 42)
    res2 = run_ui_plan(inst2, algo="astar", heuristic="own")
    elapsed = time.perf_counter() - t0

    assert res1["success"] is True
    assert res2["success"] is True
    assert elapsed < 5.0  # Lightweight live rerun completes well within target few seconds


def test_cli_rerun_matches_ui_backend(capsys):
    import argparse
    import json

    args = argparse.Namespace(
        algo="astar",
        heuristic="own",
        depth_limit=80,
        set=["slip_probability=0.0"],
        json=True,
    )
    cmd_rerun(args)
    out = capsys.readouterr().out
    cli_json = json.loads(out)

    from delivery.instance import get_canonical_instance
    base = get_canonical_instance()
    inst = build_configured_instance(
        slip_probability=0.0,
        start=base.start,
        stops=list(base.delivery_stops),
        twists=list(base.twists),
        rival_depth=base.adversary_search_depth,
        seed=base.seed,
    )
    ui_res = run_ui_plan(inst, algo="astar", heuristic="own")
    assert cli_json["cost"] == pytest.approx(ui_res["cost"])
    assert cli_json["length"] == ui_res["length"]
