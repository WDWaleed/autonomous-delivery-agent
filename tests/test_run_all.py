"""Tests for M11.1 one-command reproduction pipeline (run_all.py)."""

from __future__ import annotations

import json
from pathlib import Path

from delivery.experiments.stages import STAGE_SPECS, STAGES
from run_all import PROJECT_ROOT, run_pipeline


def test_stage_registry_dependency_order():
    """Verifies all required stages are registered in valid topological order."""
    required = {
        "partA_benchmark",
        "partA_reflex",
        "partB_heuristics",
        "partB_inadmissible",
        "partB_ids_memory",
        "partB_six_algorithms",
        "partC_pruning",
        "partC_slip",
        "partC_games",
        "partD",
        "ablation",
        "failure",
        "numbers",
    }
    assert required.issubset(set(STAGES))
    seen = set()
    for name in STAGES:
        for dep in STAGE_SPECS[name].dependencies:
            assert dep in seen, f"Stage {name} lists dependency {dep} before it runs"
        seen.add(name)


def test_failure_injection_skips_dependents_only():
    """A failed independent stage must not kill unrelated stages, and must skip dependents."""
    manifest = run_pipeline(
        quick=True,
        only="partA_benchmark,partB_heuristics,ablation,numbers",
        fresh=True,
        inject_failure="partB_heuristics",
    )
    stages = manifest["stages"]
    assert stages["partA_benchmark"]["status"] == "COMPLETED"
    assert stages["partB_heuristics"]["status"] == "FAILED"
    assert stages["ablation"]["status"] == "SKIPPED"
    assert "partB_heuristics" in stages["ablation"]["reason"]
    assert stages["numbers"]["status"] == "SKIPPED"
    assert manifest["has_failures"] is True


def test_quick_mode_isolation_does_not_overwrite_full_numbers():
    """Quick mode writes to results/quick/ and leaves results/numbers.json untouched (D19)."""
    full_numbers_path = PROJECT_ROOT / "results" / "numbers.json"
    before_bytes = full_numbers_path.read_bytes() if full_numbers_path.exists() else None

    manifest = run_pipeline(quick=True, only="partA_benchmark,numbers", fresh=True)
    assert manifest["run_mode"] == "quick"
    assert (PROJECT_ROOT / "results" / "quick" / "numbers.json").exists()
    assert (PROJECT_ROOT / "results" / "quick" / "MANIFEST.json").exists()
    assert (PROJECT_ROOT / "results" / "MANIFEST.json").exists()

    if before_bytes is not None:
        assert full_numbers_path.read_bytes() == before_bytes
