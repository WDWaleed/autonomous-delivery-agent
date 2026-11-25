#!/usr/bin/env python3
"""Single reproduction command for the CS-370 AI OEL delivery agent (M11.1).

Usage:
    python run_all.py [--quick] [--only STAGES] [--fresh] [--workers N]
                      [--llm-mode {replay,live}] [--set KEY=VALUE]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from delivery.experiments.cache import ResultCache, compute_cache_key
from delivery.experiments.stages import STAGE_ALIASES, STAGE_SPECS, STAGES
from delivery.instance import get_canonical_instance
from delivery.stamp import STAMP, get_results_dir, set_quick_mode

DEFAULT_BANNER = f"CS-370 AI OEL Autonomous Delivery Agent | {STAMP}"


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


def get_package_versions() -> Dict[str, str]:
    pkgs = ["numpy", "scipy", "pandas", "matplotlib", "streamlit", "pytest", "pyyaml", "python-dotenv"]
    versions: Dict[str, str] = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_artifact_hashes(results_dir: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for sub in ("tables", "figures"):
        sub_dir = results_dir / sub
        if sub_dir.exists():
            for p in sorted(sub_dir.iterdir()):
                if p.is_file():
                    rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    hashes[rel] = _sha256_file(p)
    num_file = results_dir / "numbers.json"
    if num_file.is_file():
        rel = str(num_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
        hashes[rel] = _sha256_file(num_file)
    return hashes


def print_banner(stamp_line: str = DEFAULT_BANNER) -> None:
    border = "=" * 78
    print(border)
    print(stamp_line)
    print(border)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce all OEL experiment figures, tables, and metrics."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in fast quick-mode (3 scenarios, 4 games, isolated outputs under results/quick/).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated list of stages to run (e.g. partA_benchmark,partB_heuristics).",
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
        help="Number of parallel worker processes (serial default 1).",
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
    parser.add_argument(
        "--inject-failure",
        type=str,
        default="",
        help="Testing hook: force a named stage to fail so dependency skipping can be verified.",
    )
    return parser.parse_args(argv)


def _extract_stage_meta(res: Any) -> Dict[str, Any]:
    if not isinstance(res, dict):
        return {}
    meta: Dict[str, Any] = {}
    for k in ("n_scenarios", "n_success", "success_rate", "max_speedup", "provider", "model", "rows"):
        if k in res and res[k] is not None:
            meta[k] = res[k]
    return meta


def run_pipeline(
    quick: bool = False,
    only: str = "",
    fresh: bool = False,
    workers: int = 1,
    llm_mode: str = "replay",
    overrides: Optional[List[str]] = None,
    inject_failure: str = "",
) -> Dict[str, Any]:
    """Executes the dependency-ordered reproduction pipeline and writes MANIFEST.json."""
    overrides = overrides or []
    set_quick_mode(quick)
    try:
        print_banner()

        raw_stages = [s.strip() for s in only.split(",") if s.strip()] if only else list(STAGES)
        target_stages = [STAGE_ALIASES.get(s, s) for s in raw_stages]
        invalid_stages = [s for s in target_stages if s not in STAGE_SPECS]
        if invalid_stages:
            raise ValueError(f"Unknown stages requested: {invalid_stages}. Available: {STAGES}")

        canonical_inst = get_canonical_instance()
        pkg_versions = get_package_versions()
        git_commit = get_git_commit()
        results_dir = get_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        cache = ResultCache(cache_dir=results_dir / "cache", fresh=fresh)

        manifest: Dict[str, Any] = {
            "stamp": STAMP,
            "timestamp": datetime.now().isoformat(),
            "run_mode": "quick" if quick else "full",
            "quick_mode": quick,
            "git_commit": git_commit,
            "python_version": sys.version,
            "numpy_version": pkg_versions.get("numpy", "unknown"),
            "package_versions": pkg_versions,
            "configuration": {
                "llm_mode": llm_mode,
                "workers": workers,
                "fresh": fresh,
                "overrides": overrides,
            },
            "seeds": {
                "canonical_cms_ids": canonical_inst.cms_ids,
                "canonical_instance_id": canonical_inst.id,
                "scenario_token_scheme": "cms_ids + [f'SCN-{k:03d}']",
            },
            "stages": {},
            "artifact_hashes": {},
        }

        print(f"Reproduction mode: {'QUICK (isolated in results/quick/)' if quick else 'FULL (official benchmark)'}")
        print(f"Stages scheduled ({len(target_stages)}): {', '.join(target_stages)}")
        print(f"Git commit: {git_commit} | Python: {sys.version.split()[0]} | NumPy: {manifest['numpy_version']}")
        print("-" * 78)

        start_all = time.perf_counter()
        failed_or_skipped: Dict[str, str] = {}

        for stage_name in target_stages:
            spec = STAGE_SPECS[stage_name]
            blocked_deps = [d for d in spec.dependencies if d in failed_or_skipped]
            if blocked_deps:
                reason = f"Skipped due to failed/skipped dependency: {', '.join(blocked_deps)}"
                failed_or_skipped[stage_name] = reason
                manifest["stages"][stage_name] = {
                    "status": "SKIPPED",
                    "dependencies": spec.dependencies,
                    "duration_seconds": 0.0,
                    "reason": reason,
                }
                print(f"[{stage_name}] SKIPPED ({reason})")
                continue

            stage_start = time.perf_counter()
            print(f"[{stage_name}] Starting...")

            cache_key = compute_cache_key(
                config={"stage": stage_name, "llm_mode": llm_mode, "overrides": overrides},
                seed=0,
                source_hashes={"git_commit": git_commit},
                run_mode="quick" if quick else "full",
            )

            try:
                if inject_failure and stage_name == inject_failure:
                    raise RuntimeError(f"Injected failure in stage '{stage_name}'")

                res = spec.runner(quick=quick, llm_mode=llm_mode, workers=workers, overrides=overrides)
                status = res.get("status", "COMPLETED") if isinstance(res, dict) else "COMPLETED"
                if status == "FAILED":
                    raise RuntimeError(res.get("error", f"Stage {stage_name} returned FAILED"))

                meta = _extract_stage_meta(res)
                elapsed = round(time.perf_counter() - stage_start, 4)
                cache.set(cache_key, {"status": status, "meta": meta, "duration_seconds": elapsed})

                manifest["stages"][stage_name] = {
                    "status": status,
                    "dependencies": spec.dependencies,
                    "duration_seconds": elapsed,
                    "cache_key": cache_key,
                    **meta,
                }
                print(f"[{stage_name}] {status} in {elapsed:.4f}s")
            except Exception as exc:
                elapsed = round(time.perf_counter() - stage_start, 4)
                err_msg = str(exc)
                failed_or_skipped[stage_name] = err_msg
                manifest["stages"][stage_name] = {
                    "status": "FAILED",
                    "dependencies": spec.dependencies,
                    "duration_seconds": elapsed,
                    "error": err_msg,
                }
                print(f"[{stage_name}] FAILED in {elapsed:.4f}s: {err_msg}", file=sys.stderr)

        manifest["total_duration_seconds"] = round(time.perf_counter() - start_all, 4)
        manifest["artifact_hashes"] = _collect_artifact_hashes(results_dir)
        manifest["has_failures"] = len(failed_or_skipped) > 0

        # Always write results/MANIFEST.json (and results/quick/MANIFEST.json in quick mode)
        official_manifest_path = PROJECT_ROOT / "results" / "MANIFEST.json"
        official_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(official_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if quick:
            quick_manifest_path = results_dir / "MANIFEST.json"
            with open(quick_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

        print("-" * 78)
        print(f"Reproduction complete in {manifest['total_duration_seconds']}s.")
        print(f"Manifest written to: {official_manifest_path.relative_to(PROJECT_ROOT)}")
        return manifest
    finally:
        set_quick_mode(False)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        manifest = run_pipeline(
            quick=args.quick,
            only=args.only,
            fresh=args.fresh,
            workers=args.workers,
            llm_mode=args.llm_mode,
            overrides=args.overrides,
            inject_failure=args.inject_failure,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if manifest.get("has_failures"):
        print("Pipeline finished with FAILED/SKIPPED stages.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
