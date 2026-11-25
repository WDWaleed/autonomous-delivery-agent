"""Build IEEE two-column Word (.docx) report from report/main.tex via Pandoc.

1. Expands macros from stamp.tex and generated_numbers.tex.
2. Normalizes IEEEtran-specific author/keywords environments for Pandoc.
3. Invokes Pandoc to generate main.docx.
4. Post-processes OOXML (document.xml & styles.xml) to enforce IEEEtran layout:
   - US Letter page size, IEEE margins (0.75" top, 1.0" bottom, 0.625" L/R)
   - Single-column Title / Authors / Abstract / Keywords header
   - Continuous section break into 2-column body (0.25" column gutter)
   - Times New Roman typography (24pt title, 10pt body, 8pt captions/tables)
   - Roman-numeral centered Heading 1 sections (I., II., ...)
   - Column-fitted images (<= 3.30" wide) and compact tables
"""

from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def _q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


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


def _prepare_tex_for_pandoc(tex_src: str, macros: dict[str, str]) -> str:
    expanded = re.sub(r"\\input\{(?:stamp|generated_numbers)\.tex\}\s*", "", tex_src)
    for name, val in macros.items():
        expanded = re.sub(rf"\\{name}(?![A-Za-z])", lambda _m, v=val: v, expanded)

    # Replace IEEEtran author block with Pandoc-friendly \author{...}
    expanded = re.sub(
        r"\\author\{.*?\\IEEEauthorblockN\{([^}]*)\}.*?\\IEEEauthorblockA\{([^}]*)\}\s*\}",
        lambda m: f"\\author{{{m.group(1)} \\\\ {m.group(2)}}}",
        expanded,
        flags=re.DOTALL,
    )
    # Prefix IEEEkeywords cleanly
    expanded = expanded.replace(
        r"\begin{IEEEkeywords}",
        "\n\\noindent\\textbf{\\textit{Index Terms---}}",
    ).replace(r"\end{IEEEkeywords}", "")
    return expanded


def _make_sect_pr(cols: int, continuous: bool = False) -> ET.Element:
    sect = ET.Element(_q(W_NS, "sectPr"))
    if continuous:
        ET.SubElement(sect, _q(W_NS, "type"), {_q(W_NS, "val"): "continuous"})
    ET.SubElement(
        sect,
        _q(W_NS, "pgSz"),
        {_q(W_NS, "w"): "12240", _q(W_NS, "h"): "15840"},
    )
    ET.SubElement(
        sect,
        _q(W_NS, "pgMar"),
        {
            _q(W_NS, "top"): "1080",
            _q(W_NS, "right"): "900",
            _q(W_NS, "bottom"): "1440",
            _q(W_NS, "left"): "900",
            _q(W_NS, "header"): "540",
            _q(W_NS, "footer"): "720",
            _q(W_NS, "gutter"): "0",
        },
    )
    if cols > 1:
        ET.SubElement(
            sect,
            _q(W_NS, "cols"),
            {_q(W_NS, "num"): str(cols), _q(W_NS, "space"): "360"},
        )
    else:
        ET.SubElement(sect, _q(W_NS, "cols"), {_q(W_NS, "space"): "360"})
    return sect


def _apply_ieee_styles(styles_xml: bytes) -> bytes:
    ET.register_namespace("w", W_NS)
    root = ET.fromstring(styles_xml)

    # Set default font to Times New Roman 10pt
    doc_defaults = root.find(_q(W_NS, "docDefaults"))
    if doc_defaults is not None:
        rpr_def = doc_defaults.find(_q(W_NS, "rPrDefault"))
        if rpr_def is not None:
            rpr = rpr_def.find(_q(W_NS, "rPr"))
            if rpr is None:
                rpr = ET.SubElement(rpr_def, _q(W_NS, "rPr"))
            fonts = rpr.find(_q(W_NS, "rFonts"))
            if fonts is None:
                fonts = ET.SubElement(rpr, _q(W_NS, "rFonts"))
            for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
                fonts.set(_q(W_NS, attr), "Times New Roman")
            for sz_tag in ("sz", "szCs"):
                sz = rpr.find(_q(W_NS, sz_tag))
                if sz is None:
                    sz = ET.SubElement(rpr, _q(W_NS, sz_tag))
                sz.set(_q(W_NS, "val"), "20")  # 10pt

    style_specs = {
        "Normal": {"sz": "20", "jc": "both"},
        "Compact": {"sz": "20", "jc": "both"},
        "Title": {"sz": "48", "bold": True, "jc": "center"},
        "Author": {"sz": "22", "bold": False, "jc": "center"},
        "AbstractTitle": {"sz": "20", "bold": True, "jc": "left"},
        "Abstract": {"sz": "20", "bold": False, "jc": "both"},
        "Heading1": {"sz": "20", "bold": True, "jc": "center"},
        "Heading2": {"sz": "20", "bold": False, "italic": True, "jc": "left"},
        "BodyText": {"sz": "20", "jc": "both"},
        "FirstParagraph": {"sz": "20", "jc": "both"},
        "Definition": {"sz": "20", "jc": "both"},
        "Bibliography": {"sz": "18", "jc": "both"},
        "ImageCaption": {"sz": "16", "jc": "left"},
        "TableCaption": {"sz": "16", "jc": "center"},
        "Caption": {"sz": "16", "jc": "left"},
        "VerbatimChar": {"sz": "18", "font": "Consolas"},
        "SourceCode": {"sz": "18", "font": "Consolas"},
    }

    for st in root.findall(_q(W_NS, "style")):
        sid = st.attrib.get(_q(W_NS, "styleId"), "")
        spec = style_specs.get(sid, {})
        font_name = spec.get("font", "Times New Roman")
        ppr = st.find(_q(W_NS, "pPr"))
        if "jc" in spec:
            if ppr is None:
                ppr = ET.SubElement(st, _q(W_NS, "pPr"))
            jc = ppr.find(_q(W_NS, "jc"))
            if jc is None:
                jc = ET.SubElement(ppr, _q(W_NS, "jc"))
            jc.set(_q(W_NS, "val"), spec["jc"])

        rpr = st.find(_q(W_NS, "rPr"))
        if rpr is None and (sid in style_specs or st.attrib.get(_q(W_NS, "type")) in ("paragraph", "character")):
            rpr = ET.SubElement(st, _q(W_NS, "rPr"))
        if rpr is not None:
            fonts = rpr.find(_q(W_NS, "rFonts"))
            if fonts is None:
                fonts = ET.SubElement(rpr, _q(W_NS, "rFonts"))
            for attr in ("ascii", "hAnsi", "cs"):
                fonts.set(_q(W_NS, attr), font_name)
            if "sz" in spec:
                for sz_tag in ("sz", "szCs"):
                    sz = rpr.find(_q(W_NS, sz_tag))
                    if sz is None:
                        sz = ET.SubElement(rpr, _q(W_NS, sz_tag))
                    sz.set(_q(W_NS, "val"), spec["sz"])
            if spec.get("bold") and rpr.find(_q(W_NS, "b")) is None:
                ET.SubElement(rpr, _q(W_NS, "b"))
            if spec.get("italic") and rpr.find(_q(W_NS, "i")) is None:
                ET.SubElement(rpr, _q(W_NS, "i"))

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _apply_ieee_document(doc_xml: bytes, stamp_line: str) -> bytes:
    ET.register_namespace("w", W_NS)
    ET.register_namespace("wp", WP_NS)
    ET.register_namespace("a", A_NS)
    root = ET.fromstring(doc_xml)
    body = root.find(_q(W_NS, "body"))
    assert body is not None

    # 1. Insert stamp line right after Author paragraph, and continuous 1-col sectPr before first Heading1
    children = list(body)
    first_h1_idx = None
    h1_count = 0
    for idx, el in enumerate(children):
        if el.tag == _q(W_NS, "p"):
            pstyle = el.find(f".//{_q(W_NS, 'pStyle')}")
            sval = pstyle.attrib.get(_q(W_NS, "val"), "") if pstyle is not None else ""
            if sval == "Heading1":
                if first_h1_idx is None:
                    first_h1_idx = idx
                # Add Roman numeral prefix and uppercase text
                t_nodes = el.findall(f".//{_q(W_NS, 't')}")
                if t_nodes and h1_count < len(ROMAN):
                    full_txt = "".join(t.text or "" for t in t_nodes).strip()
                    t_nodes[0].text = f"{ROMAN[h1_count]}. {full_txt.upper()}"
                    for extra_t in t_nodes[1:]:
                        extra_t.text = ""
                    h1_count += 1

    # Add stamp banner under Title/Author and continuous 1-col break before Section I
    if first_h1_idx is not None:
        break_p = ET.Element(_q(W_NS, "p"))
        break_ppr = ET.SubElement(break_p, _q(W_NS, "pPr"))
        break_ppr.append(_make_sect_pr(cols=1, continuous=True))
        body.insert(first_h1_idx, break_p)

    # Insert visible stamp line right at top of document
    stamp_p = ET.Element(_q(W_NS, "p"))
    sp_ppr = ET.SubElement(stamp_p, _q(W_NS, "pPr"))
    ET.SubElement(sp_ppr, _q(W_NS, "jc"), {_q(W_NS, "val"): "center"})
    sp_r = ET.SubElement(stamp_p, _q(W_NS, "r"))
    sp_rpr = ET.SubElement(sp_r, _q(W_NS, "rPr"))
    ET.SubElement(sp_rpr, _q(W_NS, "sz"), {_q(W_NS, "val"): "16"})
    ET.SubElement(sp_rpr, _q(W_NS, "szCs"), {_q(W_NS, "val"): "16"})
    ET.SubElement(sp_rpr, _q(W_NS, "color"), {_q(W_NS, "val"): "444444"})
    sp_t = ET.SubElement(sp_r, _q(W_NS, "t"))
    sp_t.text = stamp_line
    body.insert(0, stamp_p)

    # 2. Scale all images to max width 3.25 in (2971800 EMUs) to fit IEEE single column cleanly
    max_cx = 2971800
    for extent in root.findall(f".//{_q(WP_NS, 'extent')}"):
        cx = int(extent.attrib.get("cx", "0"))
        cy = int(extent.attrib.get("cy", "0"))
        if cx > max_cx and cx > 0:
            scale = max_cx / cx
            new_cx = max_cx
            new_cy = int(cy * scale)
            extent.set("cx", str(new_cx))
            extent.set("cy", str(new_cy))
    for ext in root.findall(f".//{_q(A_NS, 'ext')}"):
        cx = int(ext.attrib.get("cx", "0"))
        cy = int(ext.attrib.get("cy", "0"))
        if cx > max_cx and cx > 0:
            scale = max_cx / cx
            ext.set("cx", str(max_cx))
            ext.set("cy", str(int(cy * scale)))

    # 3. Strip stray Pandoc direct font/size overrides on body runs, then enforce uniform 8pt in tables
    for r in body.findall(f".//{_q(W_NS, 'r')}"):
        rpr = r.find(_q(W_NS, "rPr"))
        if rpr is not None:
            for tag in ("sz", "szCs", "rFonts"):
                node = rpr.find(_q(W_NS, tag))
                if node is not None:
                    rpr.remove(node)

    for tbl in root.findall(f".//{_q(W_NS, 'tbl')}"):
        for r in tbl.findall(f".//{_q(W_NS, 'r')}"):
            rpr = r.find(_q(W_NS, "rPr"))
            if rpr is None:
                rpr = ET.Element(_q(W_NS, "rPr"))
                r.insert(0, rpr)
            for sz_tag in ("sz", "szCs"):
                sz = rpr.find(_q(W_NS, sz_tag))
                if sz is None:
                    sz = ET.SubElement(rpr, _q(W_NS, sz_tag))
                sz.set(_q(W_NS, "val"), "16")

    # 4. Replace trailing sectPr with 2-column IEEE body sectPr
    old_sect = body.find(_q(W_NS, "sectPr"))
    if old_sect is not None:
        body.remove(old_sect)
    body.append(_make_sect_pr(cols=2, continuous=True))

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_docx() -> Path:
    pandoc = shutil.which("pandoc") or r"C:\Program Files\Pandoc\pandoc.exe"
    if not Path(pandoc).exists() and shutil.which("pandoc") is None:
        raise FileNotFoundError("pandoc not found on PATH or C:\\Program Files\\Pandoc\\pandoc.exe")

    tex_src = (REPORT_DIR / "main.tex").read_text(encoding="utf-8")
    macros = _load_macros(REPORT_DIR / "stamp.tex", REPORT_DIR / "generated_numbers.tex")
    stamp_line = macros.get("stampline", "CS-370 AI OEL")
    expanded = _prepare_tex_for_pandoc(tex_src, macros)

    tmp_tex = REPORT_DIR / "_main_expanded.tex"
    out_docx = REPORT_DIR / "main.docx"
    tmp_docx = REPORT_DIR / "_main_raw.docx"
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
                str(tmp_docx),
            ],
            cwd=str(REPORT_DIR),
            check=True,
        )

        # Post-process OOXML to apply IEEE two-column formatting
        with zipfile.ZipFile(tmp_docx, "r") as zin, zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = _apply_ieee_document(data, stamp_line)
                elif item.filename == "word/styles.xml":
                    data = _apply_ieee_styles(data)
                zout.writestr(item, data)
    finally:
        for p in (tmp_tex, tmp_docx):
            if p.exists():
                p.unlink()

    print(f"[WORD BUILD COMPLETE (IEEE 2-Col)] Wrote {out_docx}")
    return out_docx


if __name__ == "__main__":
    build_docx()
