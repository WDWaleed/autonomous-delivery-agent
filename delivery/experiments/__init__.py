"""Experiment runners, scenario generation, statistics and caching (§6.1, M2.6)."""

from delivery.experiments.cache import ResultCache, compute_cache_key
from delivery.experiments.parallel import map_tasks
from delivery.experiments.scenarios import Scenario, generate_scenario_suite, scenario
from delivery.experiments.stats import bootstrap_ci, mean_ci, paired_diff_ci, wilson_ci

__all__ = [
    "scenario",
    "generate_scenario_suite",
    "Scenario",
    "mean_ci",
    "wilson_ci",
    "paired_diff_ci",
    "bootstrap_ci",
    "ResultCache",
    "compute_cache_key",
    "map_tasks",
]
