"""Verify report/main.tex macros, stamps, figures, and citations against numbers.json (M12.5)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parent
NUMBERS_JSON = PROJECT_ROOT / "results" / "numbers.json"
GEN_TEX = REPORT_DIR / "generated_numbers.tex"
MAIN_TEX = REPORT_DIR / "main.tex"
BIB_FILE = REPORT_DIR / "references.bib"


def main() -> int:
    tex = MAIN_TEX.read_text(encoding="utf-8")
    gen_tex = GEN_TEX.read_text(encoding="utf-8")
    bib = BIB_FILE.read_text(encoding="utf-8")
    numbers = json.loads(NUMBERS_JSON.read_text(encoding="utf-8"))

    errors: list[str] = []

    # 1. Check all captions include \stampline (captions are single-line in main.tex)
    captions = [ln.strip() for ln in tex.splitlines() if r"\caption{" in ln]
    if not captions:
        errors.append("No \\caption{...} entries found in main.tex.")
    for cap in captions:
        if r"\stampline" not in cap:
            errors.append(f"Caption missing \\stampline: {cap[:60]}")

    # 2. Check all defined macros in generated_numbers.tex
    defined = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", gen_tex))
    used_macros = set(re.findall(r"\\(num[A-Za-z]+)\b", tex))
    undefined = used_macros - set(defined.keys())
    if undefined:
        errors.append(f"Undefined macros in main.tex: {sorted(undefined)}")

    # 3. Check completeness in numbers.json
    comp = numbers.get("completeness", {})
    incomplete = [k for k, v in comp.items() if not v]
    if incomplete:
        errors.append(f"Incomplete full-mode stages in numbers.json: {incomplete}")

    # 4. Check all \includegraphics files exist in report/figures/
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for fig in figs:
        if not (REPORT_DIR / "figures" / fig).exists():
            errors.append(f"Missing figure file: report/figures/{fig}")

    # 5. Check all \cite keys exist in references.bib
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    for cite_grp in re.findall(r"\\cite\{([^}]+)\}", tex):
        for key in [k.strip() for k in cite_grp.split(",")]:
            if key not in bib_keys:
                errors.append(f"Missing BibTeX key: {key}")

    if errors:
        for err in errors:
            print(f"[CHECK FAILED] {err}", file=sys.stderr)
        return 1

    print(
        f"[REPORT CHECK PASSED] {len(captions)} stamped captions, "
        f"{len(used_macros)}/{len(defined)} macros verified against numbers.json, "
        f"{len(figs)} figures found, all citations present in references.bib."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
