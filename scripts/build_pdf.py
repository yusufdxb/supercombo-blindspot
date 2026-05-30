"""Assemble the arXiv-preprint markdown into a single build file for pandoc.

Inserts the figures inline at their experiment sections, drops the internal
figure manifest, generates a References section from references.bib, and adds
title/author metadata. Output: drafts/paper_assembled.md, then build with:

    pandoc drafts/paper_assembled.md -o drafts/paper.pdf \
        --pdf-engine=weasyprint --css=docs/paper.css --toc

Reads drafts/rewritten_draft.md + drafts/references.bib. Changes no content;
only reorders/inserts figures and references for the PDF build.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "drafts" / "rewritten_draft.md"
BIB = ROOT / "drafts" / "references.bib"
FIGDIR = ROOT / "report" / "figures"
OUT = ROOT / "drafts" / "paper_assembled.md"

# heading-substring -> [(figure file, caption), ...] inserted after that heading
FIGURES = {
    "## Abstract": [("hero.png", "Four findings at a glance: output collapse (E1), uncertainty silence (E3), the Subaru cliff (E4), and recurrent-state monitor detection (E6).")],
    "### 5.1 E1": [("e1_head_collapse.png", "E1: per-head CARLA-to-real temporal-activity ratio. Eight of ten output heads collapse below 1 percent of real activity; pose and meta survive.")],
    "### 5.2 E2": [("e2_feature_ood.png", "E2: projected 512-D recurrent feature space. Real-driving states spread out; CARLA states freeze to a single point.")],
    "### 5.3 E3": [("e3_confidence.png", "E3: predictive-uncertainty distributions, real versus CARLA, against the real-driving 95th percentile. Uncertainty barely moves while outputs collapse.")],
    "### 5.4 E4": [("e4_interpolation.png", "E4 (Subaru): a hard cliff. Output activity falls from 0.9 to 0.1 of real over a transition width of 0.015."),
                   ("e4_ram_interpolation.png", "E4 (RAM): a gradient (transition width 0.274), not a cliff. The cliff shape is segment-dependent.")],
    "### 5.5 E5": [("e5_layer_localization.png", "E5a: per-stage vision-encoder activity ratio. Every encoder stage stays at or above real activity; the collapse is not in perception."),
                   ("e5_submodule_localization.png", "E5b: per-submodule cliff-alpha. The collapse enters at the recurrent summarizer (cliff 0.900) and the action-block feedback path (cliff 0.500).")],
    "### 5.6 E6": [("e6_detector.png", "E6: monitor fired fraction versus alpha. The monitor fires at alpha 0.550, before the output cliff at 0.784."),
                   ("auroc_vs_alpha.png", "E6: AUROC versus alpha for all five detectors.")],
    "### 5.7 E7": [("e7_auroc_heatmap.png", "E7a: monitor AUROC across the 15 by 5 corruption-severity grid."),
                   ("e7_severity_sweep.png", "E7b: monitor fire rate versus severity per corruption family."),
                   ("e7_overlay.png", "E7c: cell-for-cell output-collapse count versus monitor AUROC. No corruption cell reaches the collapse cutoff.")],
}


def insert_figures(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        out.append(line)
        for key, figs in FIGURES.items():
            if line.strip().startswith(key):
                out.append("")
                for fname, cap in figs:
                    p = FIGDIR / fname
                    out.append(f"![{cap}]({p})")
                    out.append("")
                break
    return "\n".join(out)


def strip_manifest(text: str) -> str:
    # remove everything from the "## Figure and Table Manifest" heading to EOF
    idx = text.find("## Figure and Table Manifest")
    if idx == -1:
        return text
    # also drop a preceding horizontal rule if present
    head = text[:idx].rstrip()
    head = re.sub(r"\n-{3,}\s*$", "", head)
    return head + "\n"


def parse_bib(bib_text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bib_text, re.DOTALL):
        body = m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.DOTALL):
            fields[fm.group(1).strip().lower()] = " ".join(fm.group(2).split())
        fields["_key"] = m.group(2).strip()
        fields["_type"] = m.group(1).strip().lower()
        entries.append(fields)
    return entries


def format_reference(e: dict) -> str:
    author = e.get("author", "").replace(" and ", ", ")
    year = e.get("year", "n.d.")
    title = e.get("title", "").replace("{", "").replace("}", "")
    venue = e.get("booktitle") or e.get("journal") or e.get("howpublished") or ""
    eprint = e.get("eprint", "")
    doi = e.get("doi", "")
    parts = [p for p in [author, f"({year})", title + ".", venue + ("." if venue else "")] if p]
    s = " ".join(parts)
    if eprint:
        s += f" arXiv:{eprint}."
    if doi:
        s += f" DOI: {doi}."
    return s.strip()


def references_section(bib_text: str) -> str:
    entries = parse_bib(bib_text)
    entries.sort(key=lambda e: (e.get("author", "zzz").lower(), e.get("year", "")))
    lines = ["", "---", "", "## References", ""]
    for e in entries:
        lines.append(f"- {format_reference(e)}")
    lines.append("")
    return "\n".join(lines), len(entries)


def main() -> int:
    text = DRAFT.read_text()
    text = strip_manifest(text)
    text = insert_figures(text)
    refs, n = references_section(BIB.read_text())
    meta = (
        "---\n"
        "title: 'Silent Collapse: A Distribution-Shift Teardown of a Production Driving Model and a Zero-Retraining Recurrent-State Monitor'\n"
        "author: 'Yusuf Guenena (Wayne State University)'\n"
        "---\n\n"
    )
    OUT.write_text(meta + text + refs)
    print(f"wrote {OUT} ({len(OUT.read_text().split())} words, {n} references, figures inserted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
