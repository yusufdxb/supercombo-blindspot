"""H1 + H2 offline diagnostic for the daytime-control near-zero recurrent attractor.

H1: Does the low-norm regime lag a near-zero predicted curvature in the
    prev_desired_curv feedback loop?  (no GPU needed — uses cached scalars)

H2: Do the 512-D daytime low-norm states share a basin with the CARLA collapse?
    (k=2 clustering + cosine similarity of cluster centroids)

Reads:
  report/real_weather_collected.npz   — daytime_control hidden_state + desired_curv
  report/teardown_collected.npz       — carla__hidden_state reference

Writes:
  report/attractor_diagnostic_results.md
  report/figures/attractor_norm_trajectory.png
  report/figures/attractor_cluster.png

Usage::

    env -u PYTHONPATH .venv/bin/python scripts/attractor_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
RW_CACHE = ROOT / "report" / "real_weather_collected.npz"
TD_CACHE = ROOT / "report" / "teardown_collected.npz"
RESULTS_OUT = ROOT / "report" / "attractor_diagnostic_results.md"
FIG_NORM = ROOT / "report" / "figures" / "attractor_norm_trajectory.png"
FIG_CLUSTER = ROOT / "report" / "figures" / "attractor_cluster.png"

NORM_THRESHOLD = 0.1  # hard gap at [0.05, 0.5]; 0.1 cleanly separates regimes


def _regime_labels(norms: np.ndarray) -> np.ndarray:
    """0 = low-norm, 1 = high-norm, -1 = gap (should be empty by construction)."""
    labels = np.full(len(norms), -1, dtype=np.int8)
    labels[norms < NORM_THRESHOLD] = 0
    labels[norms >= 0.5] = 1
    return labels


def h1_curvature_lag(
    desired_curv: np.ndarray,
    labels: np.ndarray,
    max_lag: int = 5,
) -> dict:
    """Test whether low-norm onset lags a near-zero desired_curv event.

    For each lag k in [0 .. max_lag], split frames by regime and compute
    mean |desired_curv[t-k]| for low vs high regime at time t.  If H1 holds,
    mean |curv[t-k]| should be smaller (nearer zero) for low-norm frames,
    and the effect should peak at some lag > 0.
    """
    n = len(desired_curv)
    results = {}
    abs_curv = np.abs(desired_curv)
    for lag in range(max_lag + 1):
        valid = slice(lag, n)
        shifted_curv = abs_curv[: n - lag]
        lab = labels[lag:]
        low_mean = float(np.nanmean(shifted_curv[lab == 0]))
        high_mean = float(np.nanmean(shifted_curv[lab == 1]))
        results[lag] = {"low_mean_abs_curv": low_mean, "high_mean_abs_curv": high_mean,
                        "ratio": low_mean / high_mean if high_mean > 0 else float("nan")}
    return results


def h2_basin_similarity(
    dc_states: np.ndarray,
    labels: np.ndarray,
    carla_states: np.ndarray,
) -> dict:
    """K=2 clustering of daytime states + cosine similarity to CARLA collapse mean.

    Returns cosine similarity of the low-norm centroid vs the CARLA mean,
    and vs the high-norm centroid, to discriminate shared-basin vs same-sign.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(dc_states)

    # Map cluster IDs to regime: the cluster whose centroid norm is lower is "low".
    c0_norm = float(np.linalg.norm(km.cluster_centers_[0]))
    c1_norm = float(np.linalg.norm(km.cluster_centers_[1]))
    low_cluster = 0 if c0_norm < c1_norm else 1
    high_cluster = 1 - low_cluster

    low_centroid = km.cluster_centers_[low_cluster]
    high_centroid = km.cluster_centers_[high_cluster]
    carla_mean = carla_states.mean(axis=0)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

    # Agreement with ground-truth low-norm labels (purity check)
    assigned_low = km.labels_ == low_cluster
    gt_low = labels == 0
    agreement = float((assigned_low == gt_low).mean())

    return {
        "low_centroid_norm": c0_norm if low_cluster == 0 else c1_norm,
        "high_centroid_norm": c1_norm if low_cluster == 0 else c0_norm,
        "carla_mean_norm": float(np.linalg.norm(carla_mean)),
        "cosine_low_vs_carla": cosine(low_centroid, carla_mean),
        "cosine_high_vs_carla": cosine(high_centroid, carla_mean),
        "cosine_low_vs_high": cosine(low_centroid, high_centroid),
        "label_purity": agreement,
    }


def _figure_norm(norms: np.ndarray, labels: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
    frames = np.arange(len(norms))
    colors = np.where(labels == 0, "tab:red", np.where(labels == 1, "tab:blue", "tab:grey"))
    ax.scatter(frames, norms, c=colors, s=4, zorder=3)
    ax.axhline(NORM_THRESHOLD, color="grey", lw=0.8, ls="--", label=f"threshold={NORM_THRESHOLD}")
    ax.set_xlabel("frame")
    ax.set_ylabel("hidden-state L2 norm")
    ax.set_title("Daytime control — recurrent hidden-state norm (red=low, blue=high)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _figure_cluster(dc_states: np.ndarray, carla_states: np.ndarray,
                    labels: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    from sklearn.decomposition import PCA

    combined = np.vstack([dc_states, carla_states])
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(combined)
    dc_proj = proj[:len(dc_states)]
    carla_proj = proj[len(dc_states):]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.scatter(dc_proj[labels == 1, 0], dc_proj[labels == 1, 1],
               c="tab:blue", s=6, alpha=0.6, label="daytime high-norm")
    ax.scatter(dc_proj[labels == 0, 0], dc_proj[labels == 0, 1],
               c="tab:red", s=6, alpha=0.6, label="daytime low-norm")
    ax.scatter(carla_proj[:, 0], carla_proj[:, 1],
               c="tab:orange", s=6, alpha=0.4, label="CARLA")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Hidden-state PCA: daytime regimes vs CARLA collapse")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _write_results(norms: np.ndarray, labels: np.ndarray,
                   h1: dict, h2: dict, out: Path) -> None:
    n_low = int((labels == 0).sum())
    n_high = int((labels == 1).sum())
    n_gap = int((labels == -1).sum())
    lines = [
        "# Daytime Attractor Diagnostic: H1 + H2",
        "",
        "## Regime statistics",
        f"- Frames total: {len(norms)}",
        f"- Low-norm (< {NORM_THRESHOLD}): {n_low} ({n_low/len(norms):.1%})",
        f"- High-norm (>= 0.5): {n_high} ({n_high/len(norms):.1%})",
        f"- Gap (0.1–0.5, should be 0): {n_gap}",
        f"- Norm range: [{norms.min():.4f}, {norms.max():.4f}]",
        "",
        "## H1 — prev_desired_curv latch",
        "",
        "Mean |desired_curv| at lag k for low-norm vs high-norm frames at time t.",
        "H1 predicts: low-norm frames should show lower |curv| at some lag > 0.",
        "",
        "| lag | low-norm mean|curv| | high-norm mean|curv| | ratio (low/high) |",
        "|-----|---------------------|----------------------|-----------------|",
    ]
    for lag, v in h1.items():
        lines.append(
            f"| {lag} | {v['low_mean_abs_curv']:.6f} | "
            f"{v['high_mean_abs_curv']:.6f} | {v['ratio']:.4f} |"
        )

    # Interpret H1
    min_ratio_lag = min(h1, key=lambda k: h1[k]["ratio"])
    min_ratio = h1[min_ratio_lag]["ratio"]
    if min_ratio < 0.7:
        h1_verdict = (
            f"**SUPPORTED** — low-norm frames show {(1-min_ratio)*100:.0f}% lower "
            f"|desired_curv| at lag={min_ratio_lag} (ratio={min_ratio:.3f} < 0.7 threshold). "
            "The prev_desired_curv feedback loop likely reinforces the low-activity basin."
        )
    elif min_ratio < 0.9:
        h1_verdict = (
            f"**WEAKLY SUPPORTED** — low-norm frames show modestly lower |desired_curv| "
            f"at lag={min_ratio_lag} (ratio={min_ratio:.3f}). Effect exists but modest."
        )
    else:
        h1_verdict = (
            f"**NOT SUPPORTED** — |desired_curv| is similar for both regimes "
            f"(min ratio={min_ratio:.3f} at lag={min_ratio_lag}). "
            "Curvature is not the trigger."
        )
    lines += ["", f"**H1 verdict:** {h1_verdict}", ""]

    lines += [
        "## H2 — Shared CARLA basin",
        "",
        f"| metric | value |",
        "|--------|-------|",
        f"| low-norm centroid norm | {h2['low_centroid_norm']:.4f} |",
        f"| high-norm centroid norm | {h2['high_centroid_norm']:.4f} |",
        f"| CARLA mean norm | {h2['carla_mean_norm']:.4f} |",
        f"| cosine(low-norm centroid, CARLA mean) | {h2['cosine_low_vs_carla']:.4f} |",
        f"| cosine(high-norm centroid, CARLA mean) | {h2['cosine_high_vs_carla']:.4f} |",
        f"| cosine(low-norm centroid, high-norm centroid) | {h2['cosine_low_vs_high']:.4f} |",
        f"| k=2 label purity (agreement with norm labels) | {h2['label_purity']:.3f} |",
        "",
    ]

    cos_lc = h2["cosine_low_vs_carla"]
    cos_hc = h2["cosine_high_vs_carla"]
    if cos_lc > 0.9:
        h2_verdict = (
            f"**SHARED BASIN CONFIRMED** — cosine(low-norm, CARLA) = {cos_lc:.4f} > 0.9. "
            "The daytime low-norm states are geometrically inside the CARLA collapse attractor. "
            f"High-norm centroid has cosine {cos_hc:.4f} vs CARLA (expected near zero or negative)."
        )
    elif cos_lc > 0.7:
        h2_verdict = (
            f"**SHARED BASIN LIKELY** — cosine(low-norm, CARLA) = {cos_lc:.4f}, "
            "high overlap but not identical. The daytime collapse is in the same general "
            "neighborhood as the CARLA basin."
        )
    elif cos_lc > 0.4:
        h2_verdict = (
            f"**PARTIAL OVERLAP** — cosine(low-norm, CARLA) = {cos_lc:.4f}. "
            "Some directional similarity but separate basins. H2 partially supported."
        )
    else:
        h2_verdict = (
            f"**DIFFERENT BASIN** — cosine(low-norm, CARLA) = {cos_lc:.4f}. "
            "The daytime attractor is geometrically distinct from the CARLA collapse. "
            "H2 not supported; a different mechanism drives the daytime collapse."
        )
    lines += [f"**H2 verdict:** {h2_verdict}", ""]

    lines += [
        "## Conclusion",
        "",
        "See `report/figures/attractor_norm_trajectory.png` (norm per frame) and",
        "`report/figures/attractor_cluster.png` (PCA of hidden states).",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    if not RW_CACHE.exists():
        print(f"ERROR: {RW_CACHE} not found", file=sys.stderr)
        return 1
    if not TD_CACHE.exists():
        print(f"ERROR: {TD_CACHE} not found", file=sys.stderr)
        return 1

    rw = np.load(RW_CACHE)
    td = np.load(TD_CACHE)

    dc_states = rw["daytime_control__hidden_state"].astype(np.float32)
    dc_curv = rw["daytime_control__desired_curv"]
    carla_states = td["carla__hidden_state"].astype(np.float32)

    norms = np.linalg.norm(dc_states, axis=1)
    labels = _regime_labels(norms)

    print(f"Regime: {(labels==0).sum()} low-norm, {(labels==1).sum()} high-norm, "
          f"{(labels==-1).sum()} gap")

    h1 = h1_curvature_lag(dc_curv, labels)
    h2 = h2_basin_similarity(dc_states, labels, carla_states)

    _figure_norm(norms, labels, FIG_NORM)
    _figure_cluster(dc_states, carla_states, labels, FIG_CLUSTER)
    _write_results(norms, labels, h1, h2, RESULTS_OUT)

    print(f"Written: {RESULTS_OUT}")
    print(f"H1 min ratio: {min(v['ratio'] for v in h1.values()):.4f}")
    print(f"H2 cosine(low, CARLA): {h2['cosine_low_vs_carla']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
