"""Experiment stages registry in dependency order (M11.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from delivery.experiments.failure_analysis import run_failure_analysis
from delivery.experiments.numbers import export_numbers
from delivery.experiments.stages.stage1_world_benchmark import run_stage1
from delivery.experiments.stages.stage_ablation import run_ablation_study
from delivery.experiments.stages.stage_games import run_final_games
from delivery.experiments.stages.stage_heuristics import run_heuristics_benchmark
from delivery.experiments.stages.stage_ids_memory import run_ids_memory_budget_experiment
from delivery.experiments.stages.stage_inadmissible import run_inadmissible_benchmark
from delivery.experiments.stages.stage_llm import run_partD_evaluation
from delivery.experiments.stages.stage_pruning import run_pruning_experiment
from delivery.experiments.stages.stage_reflex import run_reflex_benchmark
from delivery.experiments.stages.stage_six_algorithms import run_full_algorithms_benchmark
from delivery.experiments.stages.stage_slip import run_slip_experiment


@dataclass(frozen=True)
class StageSpec:
    name: str
    runner: Callable[..., Any]
    dependencies: List[str]
    description: str


STAGE_SPECS: Dict[str, StageSpec] = {
    "partA_benchmark": StageSpec(
        name="partA_benchmark",
        runner=lambda quick=False, **_: run_stage1(quick=quick),
        dependencies=[],
        description="Part A: NumPy vectorized vs pure-Python world cost-map benchmark",
    ),
    "partA_reflex": StageSpec(
        name="partA_reflex",
        runner=lambda quick=False, **_: run_reflex_benchmark(quick=quick),
        dependencies=[],
        description="Part A: Simple reflex agent failure benchmark",
    ),
    "partB_heuristics": StageSpec(
        name="partB_heuristics",
        runner=lambda quick=False, **_: run_heuristics_benchmark(quick=quick),
        dependencies=[],
        description="Part B: Own heuristic vs baselines paired comparison",
    ),
    "partB_inadmissible": StageSpec(
        name="partB_inadmissible",
        runner=lambda quick=False, **_: run_inadmissible_benchmark(quick=quick),
        dependencies=[],
        description="Part B: Weighted A* inadmissible suboptimality gap",
    ),
    "partB_ids_memory": StageSpec(
        name="partB_ids_memory",
        runner=lambda quick=False, **_: run_ids_memory_budget_experiment(quick=quick),
        dependencies=[],
        description="Part B: IDS vs A* under stored-record budget",
    ),
    "partB_six_algorithms": StageSpec(
        name="partB_six_algorithms",
        runner=lambda quick=False, **_: run_full_algorithms_benchmark(quick=quick),
        dependencies=[],
        description="Part B: Six-algorithm comparison (BFS, UCS, DLS, IDS, Greedy, A*)",
    ),
    "partC_pruning": StageSpec(
        name="partC_pruning",
        runner=lambda quick=False, **_: run_pruning_experiment(quick=quick),
        dependencies=[],
        description="Part C: Minimax vs Alpha-Beta move-ordering pruning study",
    ),
    "partC_slip": StageSpec(
        name="partC_slip",
        runner=lambda quick=False, **_: run_slip_experiment(quick=quick),
        dependencies=[],
        description="Part C: Stochastic slip & replanning mismatch analysis",
    ),
    "partC_games": StageSpec(
        name="partC_games",
        runner=lambda quick=False, **_: run_final_games(quick=quick),
        dependencies=[],
        description="Part C: Final games vs Chaser (Default vs Tuned weights)",
    ),
    "partD": StageSpec(
        name="partD",
        runner=lambda quick=False, llm_mode="replay", **_: run_partD_evaluation(
            llm_mode=llm_mode, quick=quick
        ),
        dependencies=[],
        description="Part D: LLM dispatcher vs direct planner evaluation",
    ),
    "ablation": StageSpec(
        name="ablation",
        runner=lambda quick=False, **_: run_ablation_study(quick=quick),
        dependencies=["partB_heuristics"],
        description="M9.1: Heuristic component ablation study",
    ),
    "failure": StageSpec(
        name="failure",
        runner=lambda **_: {"status": "COMPLETED", "rows": len(run_failure_analysis())},
        dependencies=["partA_reflex", "partB_six_algorithms", "partC_games", "partD"],
        description="M9.2: Cross-stage failure analysis aggregation",
    ),
    "numbers": StageSpec(
        name="numbers",
        runner=lambda **_: export_numbers(),
        dependencies=[
            "partA_reflex",
            "partB_heuristics",
            "partC_pruning",
            "partC_games",
            "partD",
            "ablation",
            "failure",
        ],
        description="M9.3: Numbers JSON and LaTeX macro export",
    ),
}

STAGES: List[str] = list(STAGE_SPECS.keys())

STAGE_ALIASES: Dict[str, str] = {
    "stage1_world_benchmark": "partA_benchmark",
    "stage_reflex": "partA_reflex",
    "stage_heuristics": "partB_heuristics",
    "stage2_search_comparison": "partB_six_algorithms",
    "stage3_heuristic_proof": "partB_heuristics",
    "stage4_pruning_efficiency": "partC_pruning",
    "stage5_slip_replanning": "partC_slip",
    "stage6_llm_benchmark": "partD",
    "stage7_ablation": "ablation",
    "stage8_consolidation": "numbers",
}

__all__ = [
    "STAGES",
    "STAGE_SPECS",
    "STAGE_ALIASES",
    "StageSpec",
    "run_reflex_benchmark",
    "run_heuristics_benchmark",
]
