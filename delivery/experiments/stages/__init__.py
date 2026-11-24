"""Experiment stages registry."""
from delivery.experiments.stages.stage_heuristics import run_heuristics_benchmark
from delivery.experiments.stages.stage_reflex import run_reflex_benchmark

STAGES = [
    "stage1_world_benchmark",
    "stage_reflex",
    "stage_heuristics",
    "stage2_search_comparison",
    "stage3_heuristic_proof",
    "stage4_pruning_efficiency",
    "stage5_slip_replanning",
    "stage6_llm_benchmark",
    "stage7_ablation",
    "stage8_consolidation",
]

__all__ = [
    "STAGES",
    "run_reflex_benchmark",
    "run_heuristics_benchmark",
]
