"""Clean-clone and determinism reproducibility test (M11.2).

Marked @pytest.mark.slow. Copies the repository to a temporary directory without
.env, creates a fresh virtual environment, installs pinned dependencies, runs
`python run_all.py --quick` twice, and verifies canonical non-timing outputs match.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IGNORE_DIRS = {
    ".venv",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "results",
}


def _copy_clean_repo(dst: Path) -> None:
    def _ignore(directory: str, names: list[str]) -> set[str]:
        rel = Path(directory).relative_to(PROJECT_ROOT) if Path(directory) != PROJECT_ROOT else Path(".")
        ignored = set()
        for name in names:
            if name in {".venv", ".git", ".pytest_cache", "__pycache__", ".env"}:
                ignored.add(name)
            elif rel == Path("results") and name in {"quick", "cache", "raw", "logs"}:
                ignored.add(name)
        return ignored

    shutil.copytree(PROJECT_ROOT, dst, ignore=_ignore)
    assert not (dst / ".env").exists(), ".env must not be copied to clean-clone test"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _strip_manifest_timing(manifest: dict) -> dict:
    cleaned = dict(manifest)
    cleaned.pop("timestamp", None)
    cleaned.pop("total_duration_seconds", None)
    cleaned.pop("python_version", None)
    cleaned_stages = {}
    for k, v in cleaned.get("stages", {}).items():
        st = dict(v)
        st.pop("duration_seconds", None)
        st.pop("max_speedup", None)  # wall-clock speedup ratio
        cleaned_stages[k] = st
    cleaned["stages"] = cleaned_stages
    return cleaned


TIMING_ONLY_CSVS = {"stage1_world_speedup.csv", "partA_speedup.csv"}
TIMING_COL_SUBSTRINGS = ("runtime", "duration", "latency", "time_s", "time (", "speedup", "loop_ms", "vec_ms")


def _canonical_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    rows = []
    for row in reader:
        filtered = {
            k: v
            for k, v in row.items()
            if not any(tok in k.lower() for tok in TIMING_COL_SUBSTRINGS)
        }
        rows.append(filtered)
    return rows


@pytest.mark.slow
def test_clean_clone_quick_reproducibility(tmp_path: Path):
    """Two clean quick runs without .env reproduce identical canonical non-timing outputs."""
    clone_dir = tmp_path / "clean_repo"
    _copy_clean_repo(clone_dir)

    venv_dir = tmp_path / "clean_venv"
    uv_bin = shutil.which("uv")
    if uv_bin:
        subprocess.run([uv_bin, "venv", str(venv_dir), "--python", sys.executable], check=True)
        py = _venv_python(venv_dir)
        subprocess.run(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                str(py),
                "-r",
                "requirements.txt",
                "-c",
                "constraints.txt",
            ],
            cwd=str(clone_dir),
            check=True,
        )
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        py = _venv_python(venv_dir)
        subprocess.run(
            [str(py), "-m", "pip", "install", "-r", "requirements.txt", "-c", "constraints.txt"],
            cwd=str(clone_dir),
            check=True,
        )

    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("GROQ_API_KEY", None)
    env["PYTHONHASHSEED"] = "0"

    # Run 1
    subprocess.run(
        [str(py), "run_all.py", "--quick", "--fresh", "--llm-mode", "replay"],
        cwd=str(clone_dir),
        env=env,
        check=True,
    )
    run1_numbers = json.loads((clone_dir / "results" / "quick" / "numbers.json").read_text(encoding="utf-8"))
    run1_manifest = _strip_manifest_timing(
        json.loads((clone_dir / "results" / "quick" / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    run1_tables = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted((clone_dir / "results" / "quick" / "tables").glob("*.csv"))
    }

    # Run 2
    subprocess.run(
        [str(py), "run_all.py", "--quick", "--fresh", "--llm-mode", "replay"],
        cwd=str(clone_dir),
        env=env,
        check=True,
    )
    run2_numbers = json.loads((clone_dir / "results" / "quick" / "numbers.json").read_text(encoding="utf-8"))
    run2_manifest = _strip_manifest_timing(
        json.loads((clone_dir / "results" / "quick" / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    run2_tables = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted((clone_dir / "results" / "quick" / "tables").glob("*.csv"))
    }

    assert run1_numbers == run2_numbers
    assert run1_manifest["stages"] == run2_manifest["stages"]
    assert run1_tables.keys() == run2_tables.keys()

    # Compare non-timing CSV content directly between run1 and run2
    import csv
    for name in run1_tables:
        if name in TIMING_ONLY_CSVS:
            continue
        lines1 = [ln for ln in run1_tables[name].splitlines() if not ln.startswith("#")]
        lines2 = [ln for ln in run2_tables[name].splitlines() if not ln.startswith("#")]
        rows1 = [
            {k: v for k, v in r.items() if not any(tok in k.lower() for tok in TIMING_COL_SUBSTRINGS)}
            for r in csv.DictReader(lines1)
        ]
        rows2 = [
            {k: v for k, v in r.items() if not any(tok in k.lower() for tok in TIMING_COL_SUBSTRINGS)}
            for r in csv.DictReader(lines2)
        ]
        assert rows1 == rows2, f"Non-deterministic non-timing CSV output in {name}"
