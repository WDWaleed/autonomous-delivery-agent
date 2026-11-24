"""Unit tests for statistical calculators and confidence intervals (D9, M2.6)."""

import math
import numpy as np
import pytest
from scipy import stats

from delivery.experiments.stats import bootstrap_ci, mean_ci, paired_diff_ci, wilson_ci


def test_mean_ci_known_t_critical():
    # n = 30 data points
    n = 30
    data = np.arange(1, n + 1, dtype=float)  # 1..30
    mean_val, low, high = mean_ci(data, confidence=0.95)

    assert mean_val == pytest.approx(15.5)

    # Hand-computed with exact t critical value
    t_crit = stats.t.ppf(0.975, df=29)  # ~2.04523
    assert t_crit == pytest.approx(2.04523, rel=1e-4)

    expected_margin = t_crit * (np.std(data, ddof=1) / math.sqrt(n))
    assert (high - mean_val) == pytest.approx(expected_margin)
    assert (mean_val - low) == pytest.approx(expected_margin)


def test_mean_ci_edge_cases():
    # Empty
    assert mean_ci([]) == (0.0, 0.0, 0.0)
    # Single element
    assert mean_ci([42.0]) == (42.0, 42.0, 42.0)
    # Identical elements (zero variance)
    assert mean_ci([5.0, 5.0, 5.0]) == (5.0, 5.0, 5.0)


def test_wilson_ci_boundary_cases():
    # 0 successes out of 30
    p, low, high = wilson_ci(0, 30, confidence=0.95)
    assert p == 0.0
    assert low == 0.0
    assert high > 0.0
    assert high < 0.15  # Wilson upper bound for 0/30 is ~0.1135

    # 30 successes out of 30
    p, low, high = wilson_ci(30, 30, confidence=0.95)
    assert p == 1.0
    assert high == pytest.approx(1.0)
    assert low > 0.85

    # 15 out of 30: exactly centered at 0.5
    p, low, high = wilson_ci(15, 30, confidence=0.95)
    assert p == 0.5
    assert (0.5 - low) == pytest.approx(high - 0.5)


def test_paired_diff_ci():
    a = [10.0, 12.0, 14.0, 16.0]
    b = [8.0, 9.0, 11.0, 12.0]
    # differences: [2, 3, 3, 4] -> mean = 3.0
    diff_mean, low, high = paired_diff_ci(a, b, confidence=0.95)
    assert diff_mean == pytest.approx(3.0)
    assert low < 3.0 < high


def test_bootstrap_ci():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=100.0, scale=10.0, size=50)

    mean_est, low, high = bootstrap_ci(data, stat_fn=np.mean, n_resamples=1000, rng=rng)
    assert mean_est == pytest.approx(np.mean(data))
    assert low < mean_est < high
    # Reasonable 95% CI around 100 for n=50, scale=10 is approx [97, 103]
    assert 95.0 < low < 100.0
    assert 100.0 < high < 105.0
