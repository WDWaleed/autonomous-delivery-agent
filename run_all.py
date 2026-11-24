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


def main():
    args = parse_args()
    print_banner()

    target_stages = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else STAGES
    invalid_stages = [s for s in target_stages if s not in STAGES]
    if invalid_stages:
        print(f"Error: Unknown stages requested: {invalid_stages}", file=sys.stderr)
        print(f"Available stages: {STAGES}", file=sys.stderr)
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
        # Stubs for subsequent milestones
        status = "COMPLETED_STUB"
        elapsed = time.perf_counter() - stage_start
        manifest["stages"][stage_name] = {
            "status": status,
            "duration_seconds": round(elapsed, 4),
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
