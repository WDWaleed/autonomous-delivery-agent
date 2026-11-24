"""Tests for stamp infrastructure and output helpers."""

import re
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from delivery.stamp import (
    MEMBERS,
    STAMP,
    banner,
    generate_report_stamp_tex,
    load_members,
    save_figure,
    save_table,
)
import oel_instance


def test_stamp_string_matches_oel_instance():
    members = load_members()
    assert len(members) >= 3
    expected_stamp = oel_instance.stamp(members)
    assert STAMP == expected_stamp
    # Ensure specific group members appear in stamp
    assert "Muhammad Waleed Ahmed" in STAMP
    assert "513303" in STAMP
    assert "Ahmad Raza" in STAMP
    assert "522356" in STAMP
    assert "Ali Hamza" in STAMP
    assert "514549" in STAMP


def test_banner_contains_stamp():
    b = banner("Test Title")
    assert STAMP in b
    assert "Test Title" in b


def test_save_figure_with_png_metadata_and_pdf(tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Test Plot")

    png_path, pdf_path = save_figure(fig, "test_fig", output_dir=tmp_path, report_copy=False)
    plt.close(fig)

    assert png_path.exists()
    assert pdf_path.exists()

    # Verify PNG metadata
    with Image.open(png_path) as im:
        assert im.info.get("Author") == STAMP
        assert im.info.get("Comment") == STAMP


def test_save_table_formats(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    out_files = save_table(df, "test_table", output_dir=tmp_path, caption="Custom Caption")

    # 1. Verify CSV
    csv_file = out_files["csv"]
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        assert first_line == f"# {STAMP}"

    # 2. Verify Markdown
    md_file = out_files["md"]
    assert md_file.exists()
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert STAMP in content
        assert "Custom Caption" in content

    # 3. Verify LaTeX
    tex_file = out_files["tex"]
    assert tex_file.exists()
    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "\\stampline" in content
        assert "Custom Caption" in content


def test_generate_report_stamp_tex(tmp_path):
    tex_path = generate_report_stamp_tex(tmp_path / "stamp.tex")
    assert tex_path.exists()
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "\\newcommand{\\stampline}" in content
    # TeX safe check
    assert "513303" in content
    assert "522356" in content
    assert "514549" in content
