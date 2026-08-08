"""Build the paper as a submission-format IEEE journal PDF (IEEEtran, two column).

Pipeline: paper/manuscript.md
  -> strip title block / abstract / manifest (rendered by the LaTeX preamble instead)
  -> author-year inline citations rewritten to \\cite{key} (numeric IEEE refs)
  -> markdown tables replaced by hand-set LaTeX floats
  -> pandoc markdown -> latex body
  -> figures inserted as figure / figure* floats at their section anchors
  -> IEEEtran preamble + bibtex (IEEEtran.bst) + 3 latex passes

Output: paper/paper_ieee.pdf

Content is not edited. Only citation syntax, float placement, and layout change.

    python3 scripts/build_pdf_ieee.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "manuscript.md"
BIB = ROOT / "paper" / "references.bib"
FIGDIR = ROOT / "report" / "figures_print"
BUILD = ROOT / "paper" / "_ieee_build"
OUT = ROOT / "paper" / "paper_ieee.pdf"

TITLE = ("Silent Collapse: A Distribution-Shift Teardown of a Production "
         "Driving Model and a Zero-Retraining Recurrent-State Monitor")
AUTHOR = "Yusuf Guenena"
AFFIL = ("Y. Guenena is with the Department of Electrical and Computer Engineering, "
         "Wayne State University, Detroit, MI 48202, USA (e-mail: "
         "yusuf.a.guenena@gmail.com). Code and data: "
         "\\url{https://github.com/yusufdxb/supercombo-blindspot}")
KEYWORDS = ("Out-of-distribution detection, autonomous driving, distribution shift, "
            "runtime monitoring, recurrent neural networks, simulation-to-reality gap, "
            "safety-critical machine learning.")

# Inline author-year citation -> bibtex key. Whitespace in the pattern matches a
# newline in the wrapped draft. Order matters: longer patterns first.
CITES: list[tuple[str, str]] = [
    (r"\(Grewal,\s+Tonella,\s+and\s+Stocco\s+2024\)", "grewal2024"),
    (r"\(Hendrycks\s+and\s+Dietterich\s+2019\)", "hendrycks2019imgnetc"),
    (r"\(Hendrycks\s+and\s+Gimpel\s+2017\)", "hendrycks2017msp"),
    (r"\(Mueller\s+and\s+Hein\s+2025\)", "muellerplus2025"),
    (r"\(Sastry\s+and\s+Oore\s+2020\)", "sastry2020"),
    (r"\(Dosovitskiy\s+et\s+al\.\s+2017\)", "dosovitskiy2017carla"),
    (r"\(Michaelis\s+et\s+al\.\s+2019\)", "michaelis2019"),
    (r"\(Stocco\s+et\s+al\.\s+2020\)", "stocco2020"),
    (r"\(Ayerdi\s+et\s+al\.\s+2024\)", "marmot2024"),
    (r"\(Hodge\s+et\s+al\.\s+2025\)", "hodge2025"),
    (r"\(Cheng\s+et\s+al\.\s+2018\)", "cheng2018"),
    (r"\(Chen\s+et\s+al\.\s+2022\)", "chen2022deepdive"),
    (r"\(Zhang\s+et\s+al\.\s+2018\)", "deeproad2018"),
    (r"\(Tian\s+et\s+al\.\s+2018\)", "deeptest2018"),
    (r"\(Wang\s+et\s+al\.\s+2022\)", "vim2022"),
    (r"\(Yang\s+et\s+al\.\s+2022\)", "yang2022openood"),
    (r"\(Liu\s+et\s+al\.\s+2020\)", "liu2020energy"),
    (r"\(Ren\s+et\s+al\.\s+2021\)", "ren2021"),
    (r"\(Sun\s+et\s+al\.\s+2022\)", "sun2022"),
    (r"\(Lee\s+et\s+al\.\s+2018\)", "lee2018"),
    (r"\(Guo\s+and\s+Su\s+2026\)", "guosu2026"),
    (r"\(arXiv:2509\.15735\)", "eigentrack2025"),
    (r"\(arXiv:2505\.11532\)", "adversarial2025"),
    (r"\(ASE\s+2022\)", "vonstein2022"),
    # bare author-year mentions used as running text
    (r"Keser\s+et\s+al\.\s+2025", "keser2025|Keser @@ETAL@@"),
    (r"Filos\s+et\s+al\.\s+2020", "filos2020|Filos @@ETAL@@"),
    (r"Guo\s+and\s+Su\s+2026", "guosu2026|Guo and Su"),
    (r"\(Ben\s+Ammar\s*\n?\s*et\s+al\.\s+2024\)", "neco2024"),
]

# heading-substring -> [(figure file, caption, wide?), ...]
FIGURES: dict[str, list[tuple[str, str, bool]]] = {
    "E1: Output collapse": [("e1_head_collapse.png", "E1: per-head CARLA-to-real temporal-activity ratio. Eight of ten output heads collapse below 1 percent of real activity; pose and meta survive.", False)],
    "E2: Recurrent-feature freeze": [("e2_feature_ood.png", "E2: projected 512-D recurrent feature space. Real-driving states spread out; CARLA states freeze to a single point.", False)],
    "E3: Uncertainty silence": [("e3_confidence.png", "E3: predictive-uncertainty distributions, real versus CARLA, against the real-driving 95th percentile. Uncertainty barely moves while outputs collapse.", False)],
    "E4: Cliff characterization": [("e4_interpolation.png", "E4 (Subaru): a hard cliff. Output activity falls from 0.9 to 0.1 of real over a transition width of 0.015.", False),
                                   ("e4_ram_interpolation.png", "E4 (RAM): a gradient (transition width 0.274), not a cliff. The cliff shape is segment-dependent.", False)],
    "E5: Localization": [("e5_layer_localization.png", "E5a: per-stage vision-encoder activity ratio. Every encoder stage stays at or above real activity; the collapse is not in perception.", False),
                         ("e5_submodule_localization.png", "E5b: per-submodule cliff-alpha. The collapse enters at the recurrent summarizer (cliff 0.900) and the action-block feedback path (cliff 0.500).", False)],
    "E6: Monitor detection": [("e6_detector.png", "E6: monitor fired fraction versus alpha. The monitor fires at alpha 0.550, before the output cliff at 0.784.", False),
                              ("auroc_vs_alpha.png", "E6: AUROC versus alpha for all five detectors.", False)],
    "E7: Corruption sweep": [("e7_auroc_heatmap.png", "E7a: monitor AUROC across the 15 by 5 corruption-severity grid.", False),
                             ("e7_severity_sweep.png", "E7b: monitor fire rate versus severity per corruption family.", True),
                             ("e7_overlay.png", "E7c: cell-for-cell output-collapse count versus monitor AUROC. No corruption cell reaches the collapse cutoff.", False)],
}

HERO = ("hero.png", "Four findings at a glance: output collapse (E1), uncertainty silence (E3), "
        "the Subaru cliff (E4), and recurrent-state monitor detection (E6).")

TABLE_E6OP = r"""
\begin{table}[!t]
\renewcommand{\arraystretch}{1.25}
\caption{Sensitivity-Matched Cross-Corpus Operating Point (LOCO FPR@95\%TPR)}
\label{tab:e6op}
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lccc}
\hline
\textbf{Detector} & \textbf{LOCO mean} & \textbf{95\% CI} & \textbf{LOCO max} \\
 & \textbf{FPR@95\%TPR} & & \textbf{FPR} \\
\hline
E6 (rolling spread) & 0.00\% & [0.00\%, 0.00\%] & 0.00\% \\
KNN-50 & 60.82\% & [35.89\%, 85.74\%] & 100.00\% \\
Mahalanobis & 95.14\% & [91.77\%, 98.51\%] & 100.00\% \\
Relative Mahalanobis & 99.69\% & [99.06\%, 100.00\%] & 100.00\% \\
\hline
\end{tabular}
\end{table}
"""

TABLE_E9 = r"""
\begin{table}[!t]
\renewcommand{\arraystretch}{1.25}
\caption{Per-Readout Activity Ratio vs.\ Real Under Each Pixel-Statistic Intervention}
\label{tab:e9}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\hline
\textbf{Readout} & \textbf{CARLA} & \textbf{+ mean/std} & \textbf{+ histogram} & \textbf{+ Fourier} \\
 & \textbf{(raw)} & \textbf{match} & \textbf{match} & \textbf{(FDA) match} \\
\hline
\texttt{accel\_t0} & 0.0040 & 0.1105 & 0.0478 & 0.0235 \\
\texttt{desired\_curv} & 0.0018 & 0.0030 & 0.0020 & 0.0020 \\
\texttt{lead\_prob} & 0.0058 & 0.0403 & 0.0380 & 0.0111 \\
\texttt{plan} & 0.0057 & 0.0435 & 0.0267 & 0.0132 \\
\texttt{lane\_lines} & 0.0054 & 0.0383 & 0.0177 & 0.0079 \\
\texttt{road\_edges} & 0.0076 & 0.0258 & 0.0143 & 0.0115 \\
\texttt{lead} & 0.0042 & 0.0453 & 0.0302 & 0.0181 \\
\texttt{pose} & 0.1788 & 0.2090 & 0.2015 & 0.1704 \\
\texttt{desire\_state} & 0.0049 & 0.0140 & 0.0060 & 0.0052 \\
\texttt{meta} & 0.7181 & 0.6296 & 0.6324 & 0.6532 \\
\hline
\end{tabular}
\end{table}
"""

TABLE_TAXONOMY = r"""
\begin{table*}[!t]
\renewcommand{\arraystretch}{1.2}
\caption{Claim Taxonomy by Evidential Status}
\label{tab:taxonomy}
\centering
\footnotesize
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.78\textwidth}@{}}
\hline
\textbf{Bucket} & \textbf{Claims} \\
\hline
VERIFIED (v0.9.7, CARLA, Subaru/RAM corpora) & E1: 8/10 tracked output readouts (7 heads plus 3 scalars derived from them) collapse to under 1\% of real activity. E2: recurrent feature is OOD and reaches 87.9\% in-sample centroid-direction classification accuracy against real ($d'=2.19$); held-out evidence is the LOCO analysis, not this figure. E3: exported predictive-uncertainty heads rise only 1.20--1.84$\times$; 0/219 CARLA frames exceed real p95, so the exported uncertainty channel is not a reliable OOD monitor for this collapse. E4: collapse arrives as a hard cliff on the Subaru source (transition width 0.015) and as a gradient on the RAM source (width 0.274). \\
\hline
REPLICATED on v0.9.6 & v0.9.6's exported uncertainty is likewise blind to the shift while its internal feature space remains highly discriminative (100\% in-sample centroid-direction accuracy, $d'=6.8$). \\
\hline
CONTRADICTED / DIFFERS on v0.9.6 & The silent output freeze does not replicate: only 1/10 readouts collapses versus 8/10; the alpha-blend sweep shows chaotic amplification (peaks 14.6$\times$ real) rather than a cliff. The E6 monitor does not transfer (33\% LOCO mean FPR vs 2.4\% on v0.9.7). \\
\hline
MONITOR-ONLY (E6) & The rolling recurrent-spread detector catches the temporal-collapse mode with AUROC 0.996. At a sensitivity-matched cross-corpus operating point (95\% TPR on the collapse set) it flags 0 of 1160 held-out real frames (0\% LOCO FPR@95\%TPR, approximately 94.8\% realised detection) while every location baseline fails to transfer (KNN-50 60.8\%, Mahalanobis 95.1\%, Relative Mahalanobis 99.7\%); under collapse-unaware percentile calibration it holds 2.41\% cross-corpus LOCO FPR ($N=4$; the original $N=2$ subset gave an optimistic 1.03\%). E7 shows it is a collapse detector, not a universal OOD detector: photometric corruptions evade it (mean AUROC 0.52--0.74 across corruption types). \\
\hline
DEPLOYMENT-UNSUPPORTED & Scaling the clean-real calibration set from $N=2$ to $N=4$ raised the LOCO mean FPR from an optimistic 1.03\% to 2.41\% (segment-level bootstrap 95\% CI [0\%, 5.17\%], 6.90\% max). Fleet-scale FPR is still unproven and likely higher; $N=4$ is honest progress, not a production number. \\
\hline
CONFOUNDS EXCLUDED AS SUFFICIENT (E9, E9b) & Matching CARLA's low-level pixel statistics to real (moment, marginal histogram, low-frequency Fourier amplitude) does not lift the recurrent freeze: spread stays 1.26--1.35e-5 of real and in-sample centroid-direction accuracy holds 87.9\%, though output quiescence partly recovers (readouts below 1\% fall 8/10 to 1--3/10). Substituting the zero-calibration warp on real footage does not collapse it (0/10 readouts below either threshold, spread 0.54$\times$, though 89.4\% in-sample centroid-direction accuracy), and CARLA still freezes under the identical warp. Neither low-level statistics nor the calibration warp is a sufficient explanation for the freeze. \\
\hline
HYPOTHESIS / OPEN & A real daytime-dry segment intermittently enters a near-zero recurrent attractor (monitor fires on 60.34\% of analyzed frames) on clean correctly-warped input; the trigger is unexplained and an initial steer/speed hypothesis was falsified. Which property of rendered content actually drives the collapse (semantics, higher-order texture, phase structure) is unidentified: E9 and E9b exclude candidate causes without isolating the operative one. \\
\hline
\end{tabular}
\end{table*}
"""

PREAMBLE = r"""\documentclass[10pt,journal]{IEEEtran}
\usepackage{graphicx}
\usepackage{array}
\usepackage{amsmath,amssymb}
\usepackage{url}
\usepackage{cite}
\usepackage[hidelinks,breaklinks=true]{hyperref}
\usepackage{textcomp}
\interdisplaylinepenalty=2500
\hyphenpenalty=1000
\sloppy
\begin{document}
\title{%(title)s}
\author{%(author)s%%
\thanks{%(affil)s}}
\maketitle
\begin{abstract}
%(abstract)s
\end{abstract}
\begin{IEEEkeywords}
%(keywords)s
\end{IEEEkeywords}
\IEEEpeerreviewmaketitle
"""


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def split_draft(text: str) -> tuple[str, str]:
    """Return (abstract_markdown, body_markdown)."""
    i = text.index("## Abstract")
    j = text.index("## 1. Introduction")
    abstract = text[i + len("## Abstract"):j]
    abstract = abstract.strip().rstrip("-").strip()
    body = text[j:]
    k = body.find("## Figure and Table Manifest")
    if k != -1:
        body = re.sub(r"\n-{3,}\s*$", "", body[:k].rstrip()) + "\n"
    return abstract, body


def apply_cites(text: str) -> tuple[str, list[str]]:
    missing = []
    for pattern, key in CITES:
        if "|" in key:
            bibkey, lead = key.split("|", 1)
            repl = f"{lead}@@CITE:{bibkey}@@"
        else:
            repl = f"@@CITE:{key}@@"
        text, n = re.subn(pattern, lambda _m, r=repl: r, text)
        if n == 0:
            missing.append(pattern)
    # the parenthetical form leaves a space before the token; a citation binds
    # to the preceding word with a non-breaking space instead
    text = re.sub(r"\s+@@CITE:", "@@CITE:", text)
    return text, missing


# purely typographic fixes applied to the generated LaTeX: the draft is written
# in plain ASCII, journals are not. Units, scientific notation, and multipliers
# are set as math; no wording or value changes.
TYPOGRAPHY: list[tuple[str, str]] = [
    # pandoc escapes the draft's literal caret to \^{}, so match the escaped form
    (r"m/s\\\^\{\}2", r"m/s\\textsuperscript{2}"),
    (r"\+/-", r"$\\pm$"),
    (r"(?<![\w.])(\d+(?:\.\d+)?)e-(\d+)\b", r"$\\num{\1}\\times 10^{-\2}$"),
    (r"(?<![\w.])(\d+(?:\.\d+)?)x(?![\w])", r"$\\num{\1}\\times$"),
]


def apply_typography(tex: str) -> str:
    for pat, rep in TYPOGRAPHY:
        tex = re.sub(pat, rep, tex)
    return tex.replace("\\num{", "").replace("}\\times", "\\times")


def detokenize(tex: str) -> str:
    """Resolve the pandoc-safe placeholders into real LaTeX.

    The tokens exist because pandoc escapes a literal tilde to
    \\textasciitilde{} and a backslash to \\textbackslash{}, so \\cite cannot be
    written directly into the markdown that pandoc consumes."""
    tex = re.sub(r"@@CITE:([A-Za-z0-9_]+)@@",
                 lambda m: "~\\cite{" + m.group(1) + "}", tex)
    return tex.replace("@@ETAL@@", "\\emph{et al.}")


def strip_heading_numbers(text: str) -> str:
    def fix(m: re.Match) -> str:
        return m.group(1) + m.group(3)
    return re.sub(r"^(#{1,6} )(\d+(\.\d+)*\.?\s+)(.*)$",
                  lambda m: m.group(1) + m.group(4), text, flags=re.M)


def demote_headings(text: str) -> str:
    """## -> #, ### -> ## so pandoc maps them to section/subsection."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^(#{2,6}) (.*)$", line)
        if m:
            out.append("#" * (len(m.group(1)) - 1) + " " + m.group(2))
        else:
            out.append(line)
    return "\n".join(out)


# Each pipe table in the draft is replaced by a hand-set LaTeX float. A table is
# matched on a string unique to its HEADER row, never by falling through to a
# default: a new table in the draft with no float here must fail the build loudly
# rather than silently render as a duplicate of whichever float was the fallback.
# Matching the header rather than the whole block keeps prose in one table's cells
# (the taxonomy table quotes "LOCO mean FPR") from matching another table's marker.
TABLE_MARKERS: list[tuple[str, str]] = [
    ("Bucket", "@@TABLETAXONOMY@@"),
    ("LOCO mean FPR@95%TPR", "@@TABLEE6OP@@"),
    ("readout", "@@TABLEE9@@"),
]

TABLE_FLOATS: dict[str, str] = {
    "@@TABLETAXONOMY@@": TABLE_TAXONOMY,
    "@@TABLEE6OP@@": TABLE_E6OP,
    "@@TABLEE9@@": TABLE_E9,
}


def drop_markdown_tables(text: str) -> str:
    """Replace each pipe table with its float placeholder.

    Raises ValueError if a table matches no known marker, or if two tables claim
    the same float.
    """
    lines = text.splitlines()
    out, seen, i = [], set(), 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            start = i
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            header = lines[start]
            hits = [tok for key, tok in TABLE_MARKERS if key in header]
            if len(hits) != 1:
                raise ValueError(
                    f"markdown table at draft line {start + 1} matched {len(hits)} "
                    f"table markers, expected exactly 1. Header: "
                    f"{lines[start].strip()[:120]!r}. Add a hand-set float and a "
                    f"unique marker to TABLE_MARKERS."
                )
            token = hits[0]
            if token in seen:
                raise ValueError(
                    f"two markdown tables both matched {token}; markers must be unique"
                )
            seen.add(token)
            out.append(token)
            out.append("")
            continue
        out.append(lines[i])
        i += 1
    unused = set(TABLE_FLOATS) - seen
    if unused:
        raise ValueError(f"no markdown table matched these floats: {sorted(unused)}")
    return "\n".join(out)


def figure_tex(fname: str, caption: str, wide: bool) -> str:
    env = "figure*" if wide else "figure"
    width = r"\textwidth" if wide else r"\columnwidth"
    path = (FIGDIR / fname).as_posix()
    label = "fig:" + fname.rsplit(".", 1)[0].replace("_", "-")
    return (f"\\begin{{{env}}}[!t]\n\\centering\n"
            f"\\includegraphics[width={width}]{{{path}}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\end{{{env}}}\n")


def insert_figures(tex: str) -> tuple[str, list[str]]:
    missing = []
    for key, figs in FIGURES.items():
        block = "\n".join(figure_tex(*f) for f in figs)
        # anchor on the \subsection line whose text contains the key fragment
        pat = re.compile(r"(\\subsection\{[^}]*" + re.escape(key.split(":")[1].strip()[:18]) + r"[^}]*\}\n)")
        tex, n = pat.subn(lambda m: m.group(1) + block, tex, count=1)
        if n == 0:
            missing.append(key)
    return tex, missing


def main() -> int:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    raw = DRAFT.read_text()
    abstract_md, body_md = split_draft(raw)

    abstract_md, miss_a = apply_cites(abstract_md)
    body_md, miss_b = apply_cites(body_md)
    missing_cites = [p for p in miss_a if p in miss_b]

    body_md = strip_heading_numbers(body_md)
    body_md = demote_headings(body_md)
    body_md = drop_markdown_tables(body_md)

    (BUILD / "body.md").write_text(body_md)
    (BUILD / "abstract.md").write_text(abstract_md)

    pandoc = ["pandoc", "--wrap=preserve", "-f",
              "markdown-auto_identifiers", "-t", "latex"]
    r = run(pandoc + [str(BUILD / "body.md"), "-o", str(BUILD / "body.tex")])
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        return 1
    r = run(pandoc + [str(BUILD / "abstract.md"), "-o", str(BUILD / "abstract.tex")])
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        return 1

    body = (BUILD / "body.tex").read_text()
    abstract = (BUILD / "abstract.tex").read_text().strip()

    body = apply_typography(detokenize(body))
    abstract = apply_typography(detokenize(abstract))
    for token, float_tex in TABLE_FLOATS.items():
        if token not in body:
            print(f"ERROR: {token} did not survive pandoc", file=sys.stderr)
            return 1
        body = body.replace(token, float_tex)
    body, missing_figs = insert_figures(body)

    # hero figure directly after the first section opens
    body = re.sub(r"(\\section\{Introduction\}\n)",
                  lambda m: m.group(1) + figure_tex(HERO[0], HERO[1], True),
                  body, count=1)

    doc = (PREAMBLE % dict(title=TITLE, author=AUTHOR, affil=AFFIL,
                           abstract=abstract, keywords=KEYWORDS)
           + body
           + "\n\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n"
           + "\\end{document}\n")
    (BUILD / "paper.tex").write_text(doc)
    shutil.copy(BIB, BUILD / "references.bib")

    for i, cmd in enumerate([["pdflatex", "-interaction=nonstopmode", "paper"],
                             ["bibtex", "paper"],
                             ["pdflatex", "-interaction=nonstopmode", "paper"],
                             ["pdflatex", "-interaction=nonstopmode", "paper"]]):
        r = run(cmd, cwd=BUILD)
        if cmd[0] == "bibtex" and r.returncode:
            print("bibtex:", r.stdout[-2000:], file=sys.stderr)

    pdf = BUILD / "paper.pdf"
    if not pdf.exists():
        log = (BUILD / "paper.log").read_text()
        print("\n".join(l for l in log.splitlines() if l.startswith("!"))[-3000:],
              file=sys.stderr)
        return 1
    shutil.copy(pdf, OUT)

    log = (BUILD / "paper.log").read_text()
    overfull = len([l for l in log.splitlines() if "Overfull \\hbox" in l])
    undefined = sorted(set(re.findall(r"Citation `([^']+)' undefined", log)))
    print(f"wrote {OUT}")
    print(f"  overfull hboxes: {overfull}")
    if undefined:
        print(f"  UNDEFINED CITATIONS: {undefined}")
    if missing_cites:
        print(f"  citation patterns that matched nothing: {missing_cites}")
    if missing_figs:
        print(f"  figure anchors not found: {missing_figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
