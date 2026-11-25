"""Build Word (.docx) report from report/main.tex via Pandoc (M12.1).

Expands macros from stamp.tex and generated_numbers.tex before invoking pandoc
so both LaTeX (Overleaf) and Word (.docx) stay synchronized from one source.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent


def _load_macros(*paths: Path) -> dict[str, str]:
    macros: dict[str, str] = {}
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
    for path in paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                m = pattern.search(line)
                if m:
                    macros[m.group(1)] = m.group(2)
    return macros


def build_docx() -> Path:
    pandoc = shutil.which("pandoc") or r"C:\Program Files\Pandoc\pandoc.exe"
    if not Path(pandoc).exists() and shutil.which("pandoc") is None:
        raise FileNotFoundError("pandoc not found on PATH or C:\\Program Files\\Pandoc\\pandoc.exe")

    tex_src = (REPORT_DIR / "main.tex").read_text(encoding="utf-8")
    macros = _load_macros(REPORT_DIR / "stamp.tex", REPORT_DIR / "generated_numbers.tex")

    # Remove \input{stamp.tex} / \input{generated_numbers.tex} and inline macro values
    expanded = re.sub(r"\\input\{(?:stamp|generated_numbers)\.tex\}\s*", "", tex_src)
    for name, val in macros.items():
        expanded = re.sub(rf"\\{name}(?![A-Za-z])", lambda _m, v=val: v, expanded)

    tmp_tex = REPORT_DIR / "_main_expanded.tex"
    out_docx = REPORT_DIR / "main.docx"
    tmp_tex.write_text(expanded, encoding="utf-8")
    try:
        subprocess.run(
            [
                pandoc,
                str(tmp_tex),
                "--from=latex",
                "--to=docx",
                f"--resource-path={REPORT_DIR};{REPORT_DIR / 'figures'}",
                f"--bibliography={REPORT_DIR / 'references.bib'}",
                "--citeproc",
                "-o",
                str(out_docx),
            ],
            cwd=str(REPORT_DIR),
            check=True,
        )
    finally:
        if tmp_tex.exists():
            tmp_tex.unlink()

    print(f"[WORD BUILD COMPLETE] Wrote {out_docx}")
    return out_docx


if __name__ == "__main__":
    build_docx()
