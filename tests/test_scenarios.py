"""Unit and suite verification tests for scenario generation (D8, M2.6)."""

import numpy as np
import pytest

from delivery.experiments.scenarios import (
    Scenario,
    check_base_reachability,
    generate_scenario_suite,
    scenario,
)


def test_single_scenario_generation():
    sc0 = scenario(0)
    assert sc0.k == 0
    assert sc0.key == "SCN-000"
    assert sc0.is_direct is True
    assert sc0.feasible is True
    assert 20 <= sc0.instance.n <= 30

    # Determinism: repeating yields identical scenario
    sc0_repeat = scenario(0)
    assert sc0.instance.seed == sc0_repeat.instance.seed
    assert np.array_equal(sc0.instance.grid, sc0_repeat.instance.grid)
    assert sc0.instance.start == sc0_repeat.instance.start
    assert sc0.instance.delivery_stops == sc0_repeat.instance.delivery_stops


def test_scenario_suite_30_feasible_scenarios():
    """Verifies that 30 independently seeded scenarios are feasible and unique (§6.1, M2.6)."""
    suite = generate_scenario_suite(n_scenarios=30)
    assert len(suite) == 30

    seeds = set()
    for sc in suite:
        assert sc.feasible is True, f"Scenario {sc.key} failed feasibility certification!"
        assert sc.is_direct is True
        assert sc.instance.seed not in seeds, f"Duplicate seed in suite: {sc.instance.seed}"
        seeds.add(sc.instance.seed)

        # Base reachability check
        assert check_base_reachability(sc.instance.grid, sc.instance.start, sc.instance.delivery_stops) is True


def test_scenario_with_overrides():
    sc = scenario(1, overrides={"slip_probability": 0.05, "adversary": "random"})
    assert sc.instance.slip_probability == 0.05
    assert sc.instance.adversary == "random"
