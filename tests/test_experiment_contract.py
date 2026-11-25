"""Experiment Contract Verification Tests (§6.1, M9.3).

Verifies:
- Full-mode sample sizes N >= 30 where required (Part A, Part B, Part C games, Ablation)
- Pruning sample >= 200 positions, Part D == 20 requests
- Summary tables contain CI columns and group stamp
- results/numbers.json and report/generated_numbers.tex are present and consistent
- Provenance links point to existing files
"""

import json
from pathlib import Path

import pytest

from delivery.experiments.numbers import PROJECT_ROOT, export_numbers
from delivery.stamp import STAMP


def test_experiment_contract_and_numbers_export():
    payload = export_numbers()

    # 1. Completeness checks for full mode
    comp = payload["completeness"]
    assert comp["partA_reflex"] is True
    assert comp["partB_heuristics"] is True
    assert comp["partC_pruning"] is True
    assert comp["partC_games"] is True
    assert comp["partD_llm"] is True
    assert comp["ablation"] is True

    # 2. Sample size contracts
    full = payload["full"]
    assert full["partA_reflex_scenarios"] >= 30
    assert full["partB_scenarios"] >= 30
    assert full["partC_pruning_positions"] >= 200
    assert full["partC_games_count"] >= 30
    assert full["partD_requests"] == 20

    # 3. Provenance files exist
    for key, rel_path in payload["provenance"].items():
        assert (PROJECT_ROOT / rel_path).exists(), f"Missing provenance file for {key}: {rel_path}"

    # 4. Summary tables contain group stamp & CI columns
    tables_dir = PROJECT_ROOT / "results" / "tables"
    for tbl_name in [
        "partA_reflex.md",
        "partB_heuristics.md",
        "partB_six_algorithms.md",
        "partC_pruning.md",
        "partC_games.md",
        "partD_llm_summary.md",
        "ablation.md",
        "failure_analysis.md",
    ]:
        p = tables_dir / tbl_name
        assert p.exists(), f"Missing table {tbl_name}"
        txt = p.read_text(encoding="utf-8")
        assert STAMP in txt, f"Missing group stamp in {tbl_name}"
        if tbl_name != "failure_analysis.md":
            assert ("CI" in txt) or ("Confidence Interval" in txt), f"Missing CI column in {tbl_name}"

    # 5. generated_numbers.tex exists and has macros
    tex_path = PROJECT_ROOT / "report" / "generated_numbers.tex"
    assert tex_path.exists()
    tex_txt = tex_path.read_text(encoding="utf-8")
    assert "\\newcommand{" in tex_txt
    assert STAMP in tex_txt
