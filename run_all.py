#!/usr/bin/env python3
"""Single reproduction command for the CS-370 AI OEL delivery agent.

Usage:
    python run_all.py [--quick] [--only STAGES] [--fresh] [--workers N]
                      [--llm-mode {replay,live}] [--set KEY=VALUE]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from delivery.experiments.stages import STAGES

DEFAULT_BANNER = "CS-370 AI OEL Autonomous Delivery Agent | Reproduction Pipeline"


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted"


def get_numpy_version() -> str:
    try:
        import numpy as np
        return np.__version__
    except ImportError:
        return "not-installed"


def print_banner(stamp_line: str = DEFAULT_BANNER):
    border = "=" * 78
    print(border)
    print(stamp_line)
    print(border)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce all OEL experiment figures, tables, and metrics."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in fast quick-mode with reduced iterations/scenarios for sanity checking.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated list of stages to run (e.g. stage1_world_benchmark,stage2_search_comparison).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore cached results and recompute all outputs from scratch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for embarrassingly parallel stages (default 1).",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["replay", "live"],
        default="replay",
        help="LLM execution mode: 'replay' uses committed cache without API calls; 'live' calls API.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        help="Override configuration parameters (format: KEY=VALUE). Can be specified multiple times.",
    )
    return parser.parse_args()


STAGE_ALIASES = {
    "partA_benchmark": "stage1_world_benchmark",
    "partA_reflex": "stage_reflex",
    "partB_heuristics": "stage_heuristics",
    "partD": "stage6_llm_benchmark",
}


def main():
    args = parse_args()
    print_banner()

    raw_stages = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else STAGES
    target_stages = [STAGE_ALIASES.get(s, s) for s in raw_stages]
    invalid_stages = [s for s in target_stages if s not in STAGES]
    if invalid_stages:
        print(f"Error: Unknown stages requested: {invalid_stages}", file=sys.stderr)
        print(f"Available stages: {STAGES} (aliases: {list(STAGE_ALIASES.keys())})", file=sys.stderr)
        sys.exit(1)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    manifest_path = results_dir / "MANIFEST.json"

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "numpy_version": get_numpy_version(),
        "quick_mode": args.quick,
        "llm_mode": args.llm_mode,
        "workers": args.workers,
        "overrides": args.overrides,
        "stages": {},
    }

    print(f"Reproduction mode: {'QUICK (validation only)' if args.quick else 'FULL (official benchmark)'}")
    print(f"Stages scheduled ({len(target_stages)}): {', '.join(target_stages)}")
    print(f"Git commit: {manifest['git_commit']} | Python: {sys.version.split()[0]} | NumPy: {manifest['numpy_version']}")
    print("-" * 78)

    start_all = time.perf_counter()
    for stage_name in target_stages:
        stage_start = time.perf_counter()
        print(f"[{stage_name}] Starting...")
        meta = {}
        if stage_name == "stage1_world_benchmark":
            from delivery.experiments.stages.stage1_world_benchmark import run_stage1
            res = run_stage1(quick=args.quick)
            status = res.get("status", "COMPLETED")
            meta = {"max_speedup": res.get("max_speedup")}
        elif stage_name == "stage_reflex":
            from delivery.experiments.stages.stage_reflex import run_reflex_benchmark
            res = run_reflex_benchmark(quick=args.quick)
            status = res.get("status", "COMPLETED")
            meta = {
                "n_scenarios": res.get("n_scenarios"),
                "n_success": res.get("n_success"),
                "success_rate": res.get("success_rate"),
            }
        elif stage_name in ("stage_heuristics", "stage2_search_comparison"):
            from delivery.experiments.stages.stage_heuristics import run_heuristics_benchmark
            res = run_heuristics_benchmark(quick=args.quick)
            status = "COMPLETED"
            meta = {"n_scenarios": 5 if args.quick else 30}
        elif stage_name == "stage6_llm_benchmark":
            from delivery.experiments.stages.stage_llm import run_partD_evaluation
            res = run_partD_evaluation(llm_mode=args.llm_mode, quick=args.quick)
            status = "COMPLETED"
            meta = {"provider": res.get("provider"), "model": res.get("model")}
        else:
            # Stubs for subsequent milestones
            status = "COMPLETED_STUB"

        elapsed = time.perf_counter() - stage_start
        manifest["stages"][stage_name] = {
            "status": status,
            "duration_seconds": round(elapsed, 4),
            **meta,
        }
        print(f"[{stage_name}] {status} in {elapsed:.4f}s")

    manifest["total_duration_seconds"] = round(time.perf_counter() - start_all, 4)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 78)
    print(f"Reproduction complete in {manifest['total_duration_seconds']}s.")
    print(f"Manifest written to: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
