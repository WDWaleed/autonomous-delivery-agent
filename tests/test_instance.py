"""Tests for instance generation, loading, serialization, and invariants."""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from delivery.instance import (
    PINNED_SCRIPT_SHA256,
    Instance,
    derive_seed_from_ids,
    get_canonical_instance,
    load_instance,
    load_saved_instance,
    save_instance,
    verify_instructor_script_hash,
)
import oel_instance


def test_oel_instance_hash_pinned():
    """Ensures oel_instance.py has not been modified and matches the pinned SHA-256."""
    computed_hash = verify_instructor_script_hash()
    assert computed_hash == PINNED_SCRIPT_SHA256


def test_instance_invariance_case_whitespace_order():
    """Ensures ID order, case, and whitespace do not change the generated instance."""
    set1 = ["513303", "522356", "514549"]
    set2 = [" 522356 ", "514549", "513303"]
    set3 = ["514549", "513303", "522356"]

    inst1 = load_instance(set1)
    inst2 = load_instance(set2)
    inst3 = load_instance(set3)

    assert inst1.id == inst2.id == inst3.id
    assert inst1.start == inst2.start == inst3.start
    assert inst1.delivery_stops == inst2.delivery_stops == inst3.delivery_stops
    assert np.array_equal(inst1.grid, inst2.grid)
    assert np.array_equal(inst1.grid, inst3.grid)


def test_instance_invariants():
    """Validates all mathematical and problem invariants on the generated instance."""
    inst = get_canonical_instance()

    # 1. Grid size n in [20, 40]
    assert 20 <= inst.n <= 40
    assert inst.grid.shape == (inst.n, inst.n)

    # 2. Delivery stops in [3, 6]
    assert 3 <= len(inst.delivery_stops) <= 6

    # 3. Start and stops must be unblocked
    assert inst.grid[inst.start[0], inst.start[1]] >= 0
    for r, c in inst.delivery_stops:
        assert inst.grid[r, c] >= 0

    # 4. 4-connected reachability
    assert oel_instance.reachable(inst.grid.tolist(), inst.start, inst.delivery_stops)

    # 5. Adversary and depth
    assert inst.adversary in ["chaser", "blocker", "interceptor"]
    assert 3 <= inst.adversary_search_depth <= 6

    # 6. Slip probability
    assert 0.10 <= inst.slip_probability <= 0.30

    # 7. Terrain costs strictly positive and sorted
    costs = inst.terrain_cost
    assert set(costs.keys()) == {"road", "grass", "mud", "ford"}
    assert 1 <= costs["road"] < costs["grass"] < costs["mud"] < costs["ford"] <= 9

    # 8. Exactly 2 distinct assigned twists
    assert len(inst.twists) == 2
    assert len(set(inst.twists)) == 2


def test_save_and_load_saved_instance(tmp_path):
    """Ensures save_instance writes all required files and load_saved_instance recovers it identically."""
    inst = get_canonical_instance()
    out_dir = save_instance(inst, target_dir=tmp_path / inst.id)

    assert (out_dir / "instance.json").exists()
    assert (out_dir / "grid.npy").exists()
    assert (out_dir / "twists.json").exists()
    assert (out_dir / "meta.json").exists()

    loaded = load_saved_instance(inst.id, instances_root=tmp_path)
    assert loaded.id == inst.id
    assert loaded.n == inst.n
    assert loaded.start == inst.start
    assert loaded.delivery_stops == inst.delivery_stops
    assert np.array_equal(loaded.grid, inst.grid)


def test_d8_scenario_token_acceptance():
    """Verifies D8 requirement that synthetic scenario tokens are accepted and deterministic."""
    base_ids = ["513303", "522356", "514549"]
    token_ids_1 = base_ids + ["SCN-001"]
    token_ids_2 = base_ids + ["SCN-002"]

    inst1 = load_instance(token_ids_1)
    inst2 = load_instance(token_ids_2)

    assert inst1.id != inst2.id
    assert inst1.id != get_canonical_instance().id
    # Re-running same token yields identical instance
    inst1_repeat = load_instance(token_ids_1)
    assert inst1.id == inst1_repeat.id
    assert np.array_equal(inst1.grid, inst1_repeat.grid)


def test_cli_instance_command():
    """Validates CLI command execution."""
    res = subprocess.run(
        [sys.executable, "-m", "delivery.cli", "instance"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "CANONICAL PROBLEM INSTANCE" in res.stdout
    assert "432382d9" in res.stdout
    assert "chaser" in res.stdout
