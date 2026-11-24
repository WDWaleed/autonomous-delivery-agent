"""Smoke tests for basic repository setup and importability."""

import subprocess
import sys
from pathlib import Path


def test_packages_importable():
    import delivery
    import delivery.world
    import delivery.twists
    import delivery.sim
    import delivery.agents
    import delivery.search
    import delivery.heuristics
    import delivery.game
    import delivery.llm
    import delivery.experiments
    import delivery.experiments.stages
    import delivery.viz
    import delivery.ui

    assert delivery.__version__ == "0.1.0"


def test_instructor_script_exists_and_interfaces():
    import oel_instance

    assert hasattr(oel_instance, "make_instance")
    assert hasattr(oel_instance, "stamp")
    assert hasattr(oel_instance, "reachable")


def test_run_all_help():
    res = subprocess.run(
        [sys.executable, "run_all.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "reproduce all oel experiment" in res.stdout.lower()
