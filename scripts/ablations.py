"""CLI entry point for hyperparameter-sensitivity ablation sweeps.

Sweeps KNN k and E6 window size, writing results to
report/ablations_results.md and report/ablations_collected.npz.

All logic lives in src/ablations.py; this is the thin runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.ablations import (
    REPORT,
    KNN_K_VALUES,
    E6_WINDOW_VALUES,
    fmt_ci,
    load_id_hidden,
    load_ood_hidden,
    sweep_knn_k,
    sweep_e6_window,
    write_cache,
    write_report,
)


def main() -> int:
    print("[1/4] loading data...")
    id_hidden = load_id_hidden()
    ood_hidden = load_ood_hidden(alpha=1.0)
    print(f"  ID: {id_hidden.shape}, OOD: {ood_hidden.shape}")

    print(f"[2/4] KNN k sweep: {KNN_K_VALUES}")
    knn_results = sweep_knn_k(id_hidden, ood_hidden)
    for r in knn_results:
        print(f"  k={r['k']:3d}  AUROC={r['auroc_point']:.4f}  "
              f"CI={fmt_ci(r['auroc_ci'])}")

    print(f"[3/4] E6 window sweep: {E6_WINDOW_VALUES}")
    e6_results = sweep_e6_window(id_hidden, ood_hidden)
    for r in e6_results:
        fa = f"{r['fires_at_alpha']:.3f}" if np.isfinite(r['fires_at_alpha']) else "n/a"
        print(f"  window={r['window']:3d}  AUROC={r['auroc_point']:.4f}  "
              f"CI={fmt_ci(r['auroc_ci'])}  fires@={fa}")

    print("[4/4] writing outputs...")
    write_report(knn_results, e6_results, REPORT / "ablations_results.md")
    write_cache(knn_results, e6_results, REPORT / "ablations_collected.npz")
    print(f"  -> {REPORT / 'ablations_results.md'}")
    print(f"  -> {REPORT / 'ablations_collected.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
