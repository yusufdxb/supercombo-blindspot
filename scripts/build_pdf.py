"""Build author and anonymous paper PDFs with Pandoc and WeasyPrint."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "manuscript.md"
BIB = ROOT / "paper" / "references.bib"
FIGURES = ROOT / "report" / "figures"
BUILD = ROOT / "paper" / "build"
CSS = ROOT / "docs" / "paper.css"

FIGURE_INSERTS = {
    "## Abstract": [
        ("hero.png", "Four findings: output collapse, uncertainty silence, source-dependent transition shape, and recurrent-state monitoring."),
    ],
    "### 5.1 E1": [("e1_head_collapse.png", "E1: per-head CARLA-to-real temporal-activity ratio.")],
    "### 5.2 E2": [("e2_feature_ood.png", "E2: projected recurrent feature states for real and CARLA inputs.")],
    "### 5.3 E3": [("e3_confidence.png", "E3: monitored uncertainty distributions against real-driving thresholds.")],
    "### 5.4 E4": [
        ("e4_interpolation.png", "E4 Subaru overlay: a 0.015-wide output transition."),
        ("e4_ram_interpolation.png", "E4 RAM overlay: a 0.274-wide output transition."),
    ],
    "### 5.5 E5": [
        ("e5_layer_localization.png", "E5a: vision-encoder activity remains at or above the real baseline."),
        ("e5_submodule_localization.png", "E5b: selected downstream submodule activity across the overlay sweep."),
    ],
    "### 5.6 E6": [
        ("e6_detector.png", "E6: monitor fire rate across the Subaru overlay sweep."),
        ("auroc_vs_alpha.png", "E6 and baseline AUROC across the overlay sweep."),
    ],
    "### 5.7 E7": [
        ("e7_auroc_heatmap.png", "E7a: monitor AUROC across corruption-severity cells."),
        ("e7_severity_sweep.png", "E7b: monitor fire rate across corruption severity."),
        ("e7_overlay.png", "E7c: output-collapse count against monitor AUROC."),
    ],
    "### 5.9 Confound controls": [
        ("e9_pixelstat.png", "E9: readouts below 1% of real activity and recurrent-feature spread under "
                             "raw, moment-matched, histogram-matched, and Fourier-matched CARLA input."),
        ("e9b_geomwarp.png", "E9b: readouts below 1% and recurrent-feature spread for real footage under "
                             "the zero-calibration warp against CARLA under the identical warp."),
    ],
}

CITED_KEYS = {
    "commaai20704",
    "vonstein2022",
    "chen2022deepdive",
    "adversarial2025",
    "hendrycks2017msp",
    "liu2020energy",
    "lee2018",
    "ren2021",
    "sun2022",
    "vim2022",
    "muellerplus2025",
    "yang2022openood",
    "keser2025",
    "guosu2026",
    "filos2020",
    "stocco2020",
    "grewal2024",
    "hodge2025",
    "yuhas2023",
    "saemann2021",
    "cheng2018",
    "sastry2020",
    "eigentrack2025",
    "neco2024",
    "hendrycks2019imgnetc",
    "michaelis2019",
    "dosovitskiy2017carla",
    "deeptest2018",
    "deeproad2018",
    "marmot2024",
}


def strip_source_header(text: str) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("manuscript must start with a level-one title")
    body = "\n".join(lines[6:])
    marker = "## Figure and Table Manifest"
    if marker in body:
        body = body.split(marker, 1)[0].rstrip()
    body = re.sub(r"\n-{3,}\s*$", "", body).rstrip()
    return body + "\n"


def insert_figures(text: str) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        out.append(line)
        for heading, figures in FIGURE_INSERTS.items():
            if line.strip().startswith(heading):
                seen.add(heading)
                out.append("")
                for filename, caption in figures:
                    path = FIGURES / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    out.extend((f"![{caption}](report/figures/{filename})", ""))
                break
    missing = set(FIGURE_INSERTS) - seen
    if missing:
        raise ValueError(f"figure insertion headings missing: {sorted(missing)}")
    return "\n".join(out)


def parse_bibliography(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.DOTALL):
        fields: dict[str, str] = {"key": match.group(2).strip()}
        for field in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", match.group(3), re.DOTALL):
            fields[field.group(1).lower()] = " ".join(field.group(2).split())
        entries.append(fields)
    if len(entries) < 30:
        raise ValueError(f"bibliography parse produced only {len(entries)} entries")
    return entries


def clean_latex(text: str) -> str:
    substitutions = {
        r"\ss": "ss",
        r"\o": "o",
        r"\O": "O",
        r"\ae": "ae",
        r"\AE": "AE",
        r"\&": "&",
    }
    for source, replacement in substitutions.items():
        text = text.replace(source, replacement)
    accents = {
        '"': "\u0308",
        "'": "\u0301",
        "`": "\u0300",
        "^": "\u0302",
        "~": "\u0303",
        "=": "\u0304",
        ".": "\u0307",
        "c": "\u0327",
        "v": "\u030c",
    }

    def replace_accent(match: re.Match[str]) -> str:
        return unicodedata.normalize("NFC", match.group(2) + accents[match.group(1)])

    text = re.sub(r"\\([\"'`\^~=\.cv])\{?([A-Za-z])\}?", replace_accent, text)
    return text.replace("{", "").replace("}", "").replace("--", "-")


def render_references(entries: list[dict[str, str]]) -> str:
    by_key = {entry["key"]: entry for entry in entries}
    missing = CITED_KEYS - set(by_key)
    if missing:
        raise ValueError(f"cited bibliography keys missing: {sorted(missing)}")
    entries = [by_key[key] for key in CITED_KEYS]
    entries.sort(key=lambda item: (item.get("author", "").lower(), item.get("year", "")))
    lines = ["## References", ""]
    for entry in entries:
        authors = clean_latex(entry.get("author", "").replace(" and ", ", "))
        title = clean_latex(entry.get("title", ""))
        venue = clean_latex(entry.get("booktitle") or entry.get("journal") or entry.get("howpublished", ""))
        reference = f"{authors} ({entry.get('year', 'n.d.')}). {title}. {venue}."
        if entry.get("doi"):
            reference += f" DOI: {entry['doi']}."
        elif entry.get("url"):
            reference += f" {entry['url']}"
        lines.append(f"- {reference}")
    return "\n".join(lines) + "\n"


def metadata(anonymous: bool) -> str:
    author = "Anonymous submission" if anonymous else "Yusuf Guenena, Wayne State University"
    return (
        "---\n"
        "title: 'Silent Collapse: A Distribution-Shift Teardown of a Production Driving Model and a Zero-Retraining Recurrent-State Monitor'\n"
        f"author: '{author}'\n"
        "lang: en-US\n"
        "---\n\n"
    )


def validate_source(text: str, anonymous: bool) -> None:
    banned = ["[AUTHOR TODO]", "[DRAFT", "parity-exact", "first second-order"]
    if anonymous:
        banned.extend(["Yusuf", "yusufdxb", "Wayne State"])
    found = [token for token in banned if token.lower() in text.lower()]
    if found:
        raise ValueError(f"build blocked by manuscript tokens: {found}")


def build_one(body: str, references: str, anonymous: bool) -> tuple[Path, Path]:
    stem = "manuscript_anonymous" if anonymous else "manuscript"
    markdown = BUILD / f"{stem}.md"
    pdf = BUILD / f"{stem}.pdf"
    document = metadata(anonymous) + body.rstrip() + "\n\n" + references
    validate_source(document, anonymous)
    markdown.write_text(document, encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            str(markdown),
            "--from=markdown",
            "--standalone",
            "--pdf-engine=weasyprint",
            f"--css={CSS}",
            f"--resource-path={ROOT}",
            "--output",
            str(pdf),
        ],
        check=True,
        cwd=ROOT,
    )
    return markdown, pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate sources without building PDFs")
    args = parser.parse_args()

    if not shutil.which("pandoc") and not args.check:
        raise SystemExit("pandoc is required")
    if not shutil.which("weasyprint") and not args.check:
        raise SystemExit("weasyprint is required")

    BUILD.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    body = insert_figures(strip_source_header(source))
    references = render_references(parse_bibliography(BIB.read_text(encoding="utf-8")))
    validate_source(metadata(False) + body + references, False)
    validate_source(metadata(True) + body + references, True)
    if args.check:
        print("paper source, figures, bibliography, and anonymous scrub: PASS")
        return 0

    for anonymous in (False, True):
        markdown, pdf = build_one(body, references, anonymous)
        print(f"wrote {markdown.relative_to(ROOT)} and {pdf.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
