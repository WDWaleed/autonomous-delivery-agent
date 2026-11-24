"""Statistical analysis and confidence interval calculators (§6.1, D9, M2.6).

Implements t-based mean CIs, Wilson score CIs for proportions, paired difference CIs,
and non-parametric percentile bootstrap CIs with cluster support.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats


def mean_ci(
    data: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Computes sample mean and Student's t-based confidence interval.

    Returns:
    - (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)

    mean_val = float(np.mean(arr))
    if n == 1:
        return (mean_val, mean_val, mean_val)

    std_err = float(stats.sem(arr))
    if std_err == 0.0 or np.isnan(std_err):
        return (mean_val, mean_val, mean_val)

    df = n - 1
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=df))
    margin = t_crit * std_err

    return (mean_val, mean_val - margin, mean_val + margin)


def wilson_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Computes Wilson score interval for binomial proportions (exact at 0/n and n/n bounds).

    Returns:
    - (proportion, ci_lower, ci_upper)
    """
    if total <= 0:
        return (0.0, 0.0, 0.0)

    p_hat = successes / total
    z = float(stats.norm.ppf((1.0 + confidence) / 2.0))
    z2 = z * z

    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom

    radicand = (p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total))
    half_width = (z * math.sqrt(max(0.0, radicand))) / denom

    ci_lower = max(0.0, center - half_width)
    ci_upper = min(1.0, center + half_width)

    if successes == 0:
        ci_lower = 0.0
    if successes == total:
        ci_upper = 1.0

    return (p_hat, ci_lower, ci_upper)


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Computes mean difference (a - b) and Student's t-based paired difference CI.

    Returns:
    - (mean_diff, ci_lower, ci_upper)
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"Length mismatch: len(a)={len(arr_a)} != len(b)={len(arr_b)}")

    diff = arr_a - arr_b
    return mean_ci(diff, confidence=confidence)


def bootstrap_ci(
    data: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    cluster_ids: Optional[Sequence[Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Computes non-parametric percentile bootstrap CI with optional cluster-level resampling.

    Returns:
    - (point_estimate, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)

    point_est = float(stat_fn(arr))
    if n == 1:
        return (point_est, point_est, point_est)

    generator = rng or np.random.default_rng(42)

    if cluster_ids is None:
        # Standard observation-level bootstrap
        indices = generator.integers(0, n, size=(n_resamples, n))
        bootstrap_samples = arr[indices]
        # Evaluate statistic along resample axis
        bootstrap_stats = np.apply_along_axis(stat_fn, 1, bootstrap_samples)
    else:
        # Clustered bootstrap resampling
        clusters: Dict[Any, List[int]] = {}
        for idx, cid in enumerate(cluster_ids):
            clusters.setdefault(cid, []).append(idx)
        unique_clusters = list(clusters.keys())
        m = len(unique_clusters)

        bootstrap_stats = np.empty(n_resamples, dtype=np.float64)
        for b in range(n_resamples):
            sampled_c = generator.choice(unique_clusters, size=m, replace=True)
            resampled_indices: List[int] = []
            for cid in sampled_c:
                resampled_indices.extend(clusters[cid])
            resampled_data = arr[resampled_indices]
            bootstrap_stats[b] = stat_fn(resampled_data)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_stats, 100.0 * (1.0 - alpha / 2.0)))

    return (point_est, ci_lower, ci_upper)
