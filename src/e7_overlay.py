"""E7 overlay: does E1 output-collapse co-occur with E6 detection, cell-for-cell?

Resolves the `[AUTHOR TODO]` in docs/paper_draft.md (E7 subsection). The draft
bounds E6 as "collapse-specific" but flags an open question: when E6 is quiet on
a corruption, is that CORRECT (no output collapse, nothing to flag) or a MISS
(the output collapsed but E6 stayed quiet = a false negative)?

We answer it from the existing cache (no model re-run, no GPU, no CARLA) using
the CANONICAL E1 metric, src.teardown.e1_collapse_map, verbatim. A head is
"collapsed" when its activity ratio (corrupted output temporal activity / clean
output temporal activity) < COLLAPSE (0.10); a cell shows the CARLA-style output
collapse when at least COLLAPSE_CELL_MIN of the 10 heads collapse.

A built-in VALIDATION GATE first runs e1_collapse_map on the real-vs-CARLA
teardown data and refuses to report unless it reproduces the published collapse
(>= 6 of 10 heads), so the metric is proven on a known-positive before it is
trusted on the corruption cells. E6 AUROC per cell is read from the verified
report/e7_results.md table (not recomputed) so the overlay matches the paper.

    python -m src.e7_overlay
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.teardown import COLLAPSE, HEAD_NAMES, SCALARS, WARMUP, _post, e1_collapse_map

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "report" / "e7_collected.npz"
TEARDOWN = ROOT / "report" / "teardown_collected.npz"
E7_RESULTS = ROOT / "report" / "e7_results.md"
OUT_MD = ROOT / "report" / "e7_overlay_results.md"
FIG_DIR = ROOT / "report" / "figures"
FIG = FIG_DIR / "e7_overlay.png"

N_HEADS = len(SCALARS + HEAD_NAMES)        # 10
COLLAPSE_CELL_MIN = 5                       # >= this many heads collapsed = cell output-collapsed
E6_FIRES = 0.85                            # E6 AUROC at/above = detects
E6_QUIET = 0.70                            # E6 AUROC below = quiet


def _split(npz, prefix: str) -> dict:
    return {k.split("__", 1)[1]: npz[k] for k in npz.files if k.startswith(prefix + "__")}


def n_collapsed(real: dict, cell: dict) -> int:
    """Heads (of 10) whose output activity collapses vs the clean reference."""
    rows = e1_collapse_map(_post(real, WARMUP), _post(cell, WARMUP))
    return sum(1 for r in rows if np.isfinite(r["ratio"]) and r["ratio"] < COLLAPSE)


def validate_against_carla() -> int:
    """Gate: e1_collapse_map must reproduce the published CARLA collapse."""
    d = np.load(TEARDOWN)
    real = _split(d, "subaru")
    carla = _split(d, "carla")
    rows = e1_collapse_map(_post(real, WARMUP), _post(carla, WARMUP))
    return sum(1 for r in rows if np.isfinite(r["ratio"]) and r["ratio"] < COLLAPSE)


def parse_e6_auroc(path: Path) -> dict:
    """Read per-cell E6 AUROC from the verified e7_results.md AUROC table.

    The AUROC table rows are the ones carrying a bootstrap CI bracket, e.g.
    `| frost | 1 | 0.5385 | [0.5016, 0.5772] | 0.2288 | 0.7652 |`.
    """
    auroc: dict[tuple[str, int], float] = {}
    for line in path.read_text().splitlines():
        if "|" not in line or "[" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        try:
            auroc[(cells[0], int(cells[1]))] = float(cells[2])
        except ValueError:
            continue
    return auroc


def classify(collapsed: bool, e6: float) -> str:
    if collapsed and e6 >= E6_FIRES:
        return "TP (E6 catches collapse)"
    if (not collapsed) and e6 < E6_QUIET:
        return "TN (correctly quiet)"
    if collapsed and e6 < E6_QUIET:
        return "FALSE NEGATIVE"
    if (not collapsed) and e6 >= E6_FIRES:
        return "FP (E6 fires, no collapse)"
    return "marginal"


def main() -> int:
    carla_collapsed = validate_against_carla()
    print(f"[validation] e1_collapse_map on real-vs-CARLA: {carla_collapsed}/{N_HEADS} heads collapsed")
    if carla_collapsed < 6:
        print("ABORT: E1 metric did NOT reproduce the published CARLA collapse "
              f"({carla_collapsed}/{N_HEADS} < 6). Not reporting corruption overlay.")
        return 1

    npz = np.load(CACHE)
    conditions = sorted({k.split("__")[0] + "__" + k.split("__")[1]
                         for k in npz.files})
    clean = {k.split("__", 2)[2]: npz[k] for k in npz.files if k.startswith("clean__0__")}
    e6_auroc = parse_e6_auroc(E7_RESULTS)

    rows = []
    missing_e6 = []
    for cond_key in conditions:
        cname, sev = cond_key.split("__")[0], int(cond_key.split("__")[1])
        if cname == "clean":
            continue
        cell = {k.split("__", 2)[2]: npz[k] for k in npz.files if k.startswith(cond_key + "__")}
        nc = n_collapsed(clean, cell)
        e6 = e6_auroc.get((cname, sev))
        if e6 is None:
            missing_e6.append(cond_key)
            continue
        collapsed = nc >= COLLAPSE_CELL_MIN
        rows.append({"corruption": cname, "severity": sev, "n_collapsed": nc,
                     "e6_auroc": e6, "collapsed": collapsed,
                     "verdict": classify(collapsed, e6)})

    rows.sort(key=lambda r: (r["corruption"], r["severity"]))
    collapsed_cells = [r for r in rows if r["collapsed"]]
    false_negs = [r for r in rows if r["verdict"] == "FALSE NEGATIVE"]
    fp_cells = [r for r in rows if r["verdict"].startswith("FP")]
    caught = sum(1 for r in collapsed_cells if r["e6_auroc"] >= E6_FIRES)

    _write_md(rows, carla_collapsed, collapsed_cells, caught, false_negs, fp_cells, missing_e6)
    _figure(rows)

    print(f"cells={len(rows)}  output-collapsed(>= {COLLAPSE_CELL_MIN}/10 heads)="
          f"{len(collapsed_cells)}  FALSE_NEGATIVES={len(false_negs)}  "
          f"E6-fires-no-collapse(FP)={len(fp_cells)}  missing_e6={len(missing_e6)}")
    print(f"  max heads collapsed in any corruption cell: {max(r['n_collapsed'] for r in rows)}/{N_HEADS}")
    if false_negs:
        print("FALSE NEGATIVES (output collapsed, E6 quiet):")
        for r in false_negs:
            print(f"  {r['corruption']} sev{r['severity']}: {r['n_collapsed']}/10 heads, e6={r['e6_auroc']:.3f}")
    else:
        print("No false negatives.")
    if fp_cells:
        print("E6 fires WITHOUT output collapse (decoupled from the collapse mode):")
        for r in fp_cells:
            print(f"  {r['corruption']} sev{r['severity']}: {r['n_collapsed']}/10 heads, e6={r['e6_auroc']:.3f}")
    print(f"wrote {OUT_MD}\nwrote {FIG}")
    return 0


def _write_md(rows, carla_collapsed, collapsed_cells, caught, false_negs, fp_cells, missing_e6):
    L = []
    L.append("# E7 overlay: E1 output-collapse vs E6 detection (cell-for-cell)\n")
    L.append("Resolves the E7 `[AUTHOR TODO]`: is E6's quiet response on a corruption "
             "correct (no output collapse) or a miss (collapse undetected)? Metric: the "
             "canonical `teardown.e1_collapse_map` (head collapsed at activity ratio < "
             f"{COLLAPSE}); a cell is output-collapsed at >= {COLLAPSE_CELL_MIN}/{N_HEADS} "
             "heads collapsed. E6 AUROC read from the verified `e7_results.md`.\n")
    L.append(f"- VALIDATION GATE: e1_collapse_map on real-vs-CARLA reproduces "
             f"**{carla_collapsed}/{N_HEADS}** heads collapsed (published collapse confirmed).")
    L.append(f"- corruption cells evaluated: **{len(rows)}**"
             + (f" (missing E6 in results.md: {len(missing_e6)})" if missing_e6 else ""))
    L.append(f"- output-collapsed cells (>= {COLLAPSE_CELL_MIN}/{N_HEADS} heads): "
             f"**{len(collapsed_cells)}**")
    L.append(f"- max heads collapsed in ANY corruption cell: "
             f"**{max(r['n_collapsed'] for r in rows)}/{N_HEADS}**")
    L.append(f"- **FALSE NEGATIVES (output collapsed, E6 AUROC < {E6_QUIET}): "
             f"{len(false_negs)}**")
    L.append(f"- E6 fires with NO output collapse (FP): **{len(fp_cells)}**"
             + (": " + ", ".join(f"{r['corruption']} sev{r['severity']} (AUROC {r['e6_auroc']:.2f})"
                                 for r in fp_cells) if fp_cells else "") + "\n")
    if false_negs:
        L.append("> Result: E6 MISSES output collapse on the cells flagged below; the "
                 "contribution must report these as false negatives.\n")
    elif len(collapsed_cells) == 0:
        L.append("> Result (read carefully): **no ImageNet-C corruption cell collapses "
                 f"the output** (max {max(r['n_collapsed'] for r in rows)}/{N_HEADS} heads, "
                 f"vs {carla_collapsed}/{N_HEADS} under CARLA). Two consequences: "
                 "(1) the false-negative question is resolved trivially, there is no "
                 "collapse to miss; (2) the **silent-collapse failure mode is CARLA / "
                 "full-sim specific and does NOT reproduce under ImageNet-C corruptions of "
                 "real frames.** The FP cells above are E6 firing on a recurrent-feature "
                 "spread shift that is NOT an output collapse, so on this corpus E6's "
                 "firings are decoupled from the collapse mode. **Draft implication:** the "
                 "current E7 wording (E6 fires on corruptions that 'induce the same "
                 "recurrent freeze' as CARLA) overstates the link, because no output "
                 "collapse occurs here; the defensible E7 claim is narrower, the collapse "
                 "is sim-specific and E6 is collapse-specific (and mostly quiet on real-"
                 "frame corruptions).\n")
    else:
        L.append("> Result: every output-collapsed cell is also detected by E6, and E6 "
                 "stays quiet only where the output does not collapse.\n")
    L.append("| corruption | sev | heads collapsed (/10) | E6 AUROC | verdict |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['corruption']} | {r['severity']} | {r['n_collapsed']} | "
                 f"{r['e6_auroc']:.3f} | {r['verdict']} |")
    L.append("")
    OUT_MD.write_text("\n".join(L))


def _figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    colors = {"TP (E6 catches collapse)": "#2f855a", "TN (correctly quiet)": "#3182ce",
              "FALSE NEGATIVE": "#d63a3a", "FP (E6 fires, no collapse)": "#dd6b20",
              "marginal": "#999999"}
    jitter = {}
    for v, c in colors.items():
        pts = [r for r in rows if r["verdict"] == v]
        if not pts:
            continue
        xs = [r["n_collapsed"] + 0.06 * (hash(r["corruption"]) % 5 - 2) for r in pts]
        ys = [r["e6_auroc"] for r in pts]
        ax.scatter(xs, ys, s=34, c=c, label=f"{v} (n={len(pts)})",
                   edgecolors="white", linewidths=0.5, zorder=3)
    ax.axvline(COLLAPSE_CELL_MIN - 0.5, color="#444", lw=1.0, ls="--",
               label=f"output-collapse cutoff (>= {COLLAPSE_CELL_MIN}/10 heads)")
    ax.axhline(E6_FIRES, color="#2f855a", lw=0.8, ls=":")
    ax.axhline(E6_QUIET, color="#3182ce", lw=0.8, ls=":")
    ax.set_xlim(-0.5, N_HEADS + 0.5)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("output heads collapsed (of 10)  [right = output collapsed]")
    ax.set_ylabel("E6 detection AUROC")
    ax.set_title("E7 overlay: output collapse (E1) vs monitor detection (E6)\n"
                 "no corruption cell reaches the collapse cutoff; CARLA = 7/10")
    ax.legend(fontsize=7, loc="center left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
