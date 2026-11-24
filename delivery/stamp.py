"""Stamp infrastructure for CS-370 AI OEL.

Loads group members from config/members.json and provides stamping helpers
for console logs, figures, tables, UI, and LaTeX reports.
Every plot, table, console log, UI screen and report figure must carry the stamp.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, PngImagePlugin

import oel_instance

# Resolve repository paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MEMBERS_FILE = CONFIG_DIR / "members.json"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"


def load_members(path: Optional[Path] = None) -> List[Tuple[str, str]]:
    """Loads group members from config/members.json as a list of (Name, CMS_ID)."""
    target = path or MEMBERS_FILE
    if not target.exists():
        raise FileNotFoundError(f"Members configuration not found at {target}")
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(m["name"], str(m["cms_id"])) for m in data]


# Authoritative stamp string derived via oel_instance.stamp()
try:
    MEMBERS = load_members()
    STAMP = oel_instance.stamp(MEMBERS)
except Exception:
    MEMBERS = []
    STAMP = "CS-370 AI OEL Stamp | Unconfigured"


def banner(title: str = "CS-370 AI OEL Autonomous Delivery Agent") -> str:
    """Formats a stamped terminal banner with border."""
    border = "=" * 80
    lines = [
        border,
        title.center(80),
        STAMP.center(80),
        border,
    ]
    return "\n".join(lines)


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger with the group stamp in the log record format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"[%(asctime)s] [{name}] [{STAMP}]\n%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    report_copy: bool = True,
) -> Tuple[Path, Path]:
    """Saves a matplotlib figure with the stamp footer and embedded PNG metadata.

    Writes both .png and .pdf formats to results/figures/ (and report/figures/ if enabled).
    """
    target_dir = output_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Add stamp to figure footer if not already added
    has_stamp = any(STAMP in (text.get_text() or "") for text in fig.texts)
    if not has_stamp:
        fig.text(
            0.5,
            0.01,
            STAMP,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
            family="sans-serif",
        )
        # Ensure bottom margin accommodates footer
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.08))

    png_path = target_dir / f"{name}.png"
    pdf_path = target_dir / f"{name}.pdf"

    # Save initial PNG and PDF
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    # Embed stamp in PNG metadata using Pillow
    im = Image.open(png_path)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Author", STAMP)
    meta.add_text("Comment", STAMP)
    meta.add_text("Description", f"OEL delivery agent figure: {name} | {STAMP}")
    im.save(png_path, "PNG", pnginfo=meta)

    # Also copy to report/figures if requested
    if report_copy:
        rep_fig_dir = REPORT_DIR / "figures"
        rep_fig_dir.mkdir(parents=True, exist_ok=True)
        im.save(rep_fig_dir / f"{name}.png", "PNG", pnginfo=meta)
        fig.savefig(rep_fig_dir / f"{name}.pdf", bbox_inches="tight")

    return png_path, pdf_path


def df_to_markdown(df: pd.DataFrame) -> str:
    """Renders a pandas DataFrame as a GitHub-flavored Markdown table without tabulate."""
    cols = [str(c) for c in df.columns]
    rows = [[str(val) for val in row] for row in df.itertuples(index=False)]
    widths = [
        max(len(c), max((len(r[i]) for r in rows), default=0))
        for i, c in enumerate(cols)
    ]
    header = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        for r in rows
    ]
    return "\n".join([header, sep] + body)


def save_table(
    df: pd.DataFrame,
    name: str,
    output_dir: Optional[Path] = None,
    caption: Optional[str] = None,
) -> Dict[str, Path]:
    r"""Saves a pandas DataFrame with the stamp in CSV, Markdown, and LaTeX formats.

    - CSV: first line is `# <STAMP>`
    - Markdown: includes caption with STAMP
    - LaTeX: table environment with `\caption{... | \stampline}`
    """
    target_dir = output_dir or TABLES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / f"{name}.csv"
    md_path = target_dir / f"{name}.md"
    tex_path = target_dir / f"{name}.tex"

    # 1. CSV with first-line stamp
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(f"# {STAMP}\n")
        df.to_csv(f, index=False)

    # 2. Markdown with caption
    table_caption = caption or f"Table: {name}"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"**{table_caption}**\n\n")
        f.write(f"*Group Stamp: {STAMP}*\n\n")
        f.write(df_to_markdown(df))
        f.write("\n")

    # 3. LaTeX table with \stampline in caption
    latex_body = df.to_latex(index=False, escape=True)
    tex_content = (
        f"% {STAMP}\n"
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{table_caption} | \\stampline}}\n"
        f"\\label{{tab:{name}}}\n"
        f"{latex_body}\n"
        f"\\end{{table}}\n"
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def stamp_text_file(path: Path, content: str, position: str = "both"):
    """Writes content to a text file with header and/or footer stamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"=== {STAMP} ===\n\n" if position in ("header", "both") else ""
    footer = f"\n\n=== {STAMP} ===" if position in ("footer", "both") else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{header}{content}{footer}\n")


def generate_report_stamp_tex(output_path: Optional[Path] = None) -> Path:
    """Generates report/stamp.tex defining \\stampline for the IEEE report."""
    target = output_path or (REPORT_DIR / "stamp.tex")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Escape special TeX characters if any
    safe_stamp = STAMP.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    content = (
        f"% Auto-generated stamp definitions for CS-370 AI OEL\n"
        f"\\newcommand{{\\stampline}}{{{safe_stamp}}}\n"
    )
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return target


def ui_header(st_module: Any = None):
    """UI Header stub for Streamlit displaying the persistent group stamp."""
    if st_module is not None:
        st_module.markdown(
            f"<div style='text-align: center; color: #555555; font-size: 0.85em; "
            f"border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-bottom: 15px;'>"
            f"<strong>{STAMP}</strong></div>",
            unsafe_allow_html=True,
        )


def ui_footer(st_module: Any = None):
    """UI Footer stub for Streamlit displaying the persistent group stamp."""
    if st_module is not None:
        st_module.markdown(
            f"<div style='text-align: center; color: #555555; font-size: 0.8em; "
            f"border-top: 1px solid #ddd; padding-top: 8px; margin-top: 30px;'>"
            f"{STAMP}</div>",
            unsafe_allow_html=True,
        )


def demo():
    """Generates a demo figure, table, and banner showcasing the stamp."""
    print(banner("STAMP INFRASTRUCTURE DEMO"))

    # Generate demo figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9], "b-o", label="Demo Curve")
    ax.set_title("Stamp Verification Demo Plot")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Metric value")
    ax.legend()
    png_path, pdf_path = save_figure(fig, "stamp_demo")
    plt.close(fig)
    print(f"Generated demo figure: {png_path.relative_to(PROJECT_ROOT)}")

    # Generate demo table
    df = pd.DataFrame({
        "Algorithm": ["A*", "UCS", "IDS"],
        "Nodes Expanded": [142, 580, 2140],
        "Solution Cost": [28.0, 28.0, 28.0],
    })
    table_files = save_table(df, "stamp_demo_table", caption="Sample Performance Table")
    print(f"Generated demo table: {table_files['csv'].relative_to(PROJECT_ROOT)}")

    # Generate report/stamp.tex
    tex_path = generate_report_stamp_tex()
    print(f"Generated LaTeX stamp definition: {tex_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        print(banner())
