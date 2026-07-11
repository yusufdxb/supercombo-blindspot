"""E5: layer-localized collapse along the real-to-sim interpolation sweep.

We add one tensor per vision-encoder stage to the ONNX graph's outputs, then
run the same alpha sweep as E4 and measure per-layer activity ratio
CARLA / real. The output is the layer-by-alpha map that says where the
collapse cliff lives in the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LayerProbe:
    name: str
    tensor: str


LAYER_PROBES: list[LayerProbe] = [
    LayerProbe("stem",        "/supercombo/vision/_en/stem/stem.2/act/Mul_5_output_0"),
    LayerProbe("stage0_blk0", "/supercombo/vision/_en/stages.0/blocks/blocks.0/Add_output_0"),
    LayerProbe("stage0_blk1", "/supercombo/vision/_en/stages.0/blocks/blocks.1/Add_output_0"),
    LayerProbe("stage1_blk0", "/supercombo/vision/_en/stages.1/blocks/blocks.0/Add_output_0"),
    LayerProbe("stage1_blk1", "/supercombo/vision/_en/stages.1/blocks/blocks.1/Add_output_0"),
    LayerProbe("stage2_blk0", "/supercombo/vision/_en/stages.2/blocks/blocks.0/Add_output_0"),
    LayerProbe("stage2_blk1", "/supercombo/vision/_en/stages.2/blocks/blocks.1/Add_output_0"),
    LayerProbe("stage2_blk2", "/supercombo/vision/_en/stages.2/blocks/blocks.2/Add_output_0"),
    LayerProbe("stage2_blk3", "/supercombo/vision/_en/stages.2/blocks/blocks.3/Add_output_0"),
    LayerProbe("stage2_blk4", "/supercombo/vision/_en/stages.2/blocks/blocks.4/Add_output_0"),
    LayerProbe("stage2_blk5", "/supercombo/vision/_en/stages.2/blocks/blocks.5/Add_output_0"),
    LayerProbe("stage3_blk0", "/supercombo/vision/_en/stages.3/blocks/blocks.0/Add_output_0"),
    LayerProbe("stage3_blk1", "/supercombo/vision/_en/stages.3/blocks/blocks.1/Add_output_0"),
    LayerProbe("head",        "/supercombo/vision/_en/head/global_pool/flatten/Flatten_output_0"),
]


def intermediates_to_outputs(src: Path, dst: Path, tensor_names: list[str]) -> None:
    """Copy `src` to `dst` with the given intermediate tensors added as graph outputs.
    Names that do not exist in the graph raise.
    Uses ONNX shape inference to obtain the correct element dtype for each tensor."""
    import onnx
    m = onnx.load(str(src))
    # Run shape inference so value_info gets populated with element types.
    m = onnx.shape_inference.infer_shapes(m)
    existing = {o.name for o in m.graph.output}
    vi_by_name = {vi.name: vi for vi in m.graph.value_info}
    # Fall back: collect node outputs not yet in value_info (unknown dtype -> FLOAT).
    for node in m.graph.node:
        for out in node.output:
            vi_by_name.setdefault(out, onnx.helper.make_tensor_value_info(
                out, onnx.TensorProto.FLOAT, None))
    for name in tensor_names:
        if name in existing:
            continue
        if name not in vi_by_name:
            raise KeyError(f"tensor not in graph: {name}")
        m.graph.output.append(vi_by_name[name])
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))


def per_layer_activity_ratio(real: np.ndarray, carla: np.ndarray) -> float:
    """Sum of per-element temporal std, CARLA / real. Matches the
    src.teardown.e1_collapse_map convention so the two numbers are comparable."""
    r = real.reshape(len(real), -1)
    c = carla.reshape(len(carla), -1)
    rstd = r.std(axis=0).sum()
    cstd = c.std(axis=0).sum()
    return float(cstd / rstd) if rstd > 1e-12 else float("nan")


def per_layer_mean_shift(real: np.ndarray, carla: np.ndarray) -> float:
    """Ratio of absolute-mean activations, CARLA / real.
    Companion to per_layer_activity_ratio: activity_ratio captures temporal
    variation only; this captures DC shift. A layer can have activity_ratio
    ~ 1.0 while its features distribute around a very different mean."""
    r = real.reshape(len(real), -1)
    c = carla.reshape(len(carla), -1)
    rm = float(np.abs(r.mean(axis=0)).sum())
    cm = float(np.abs(c.mean(axis=0)).sum())
    return float(cm / rm) if rm > 1e-12 else float("nan")


def save_cache(path: Path, alphas: np.ndarray, per_layer: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, alphas=alphas,
                        **{f"layer__{k}": v for k, v in per_layer.items()})


def load_cache(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    d = np.load(path)
    alphas = d["alphas"]
    per_layer = {k.removeprefix("layer__"): d[k] for k in d.files
                 if k.startswith("layer__")}
    return alphas, per_layer


def save_summary_cache(path: Path, alphas: np.ndarray,
                       ratios: dict[str, np.ndarray],
                       cliffs: dict[str, float],
                       mean_shifts: dict[str, float]) -> None:
    """Persist only the pre-computed analysis metrics (not raw activations).
    Output is ~hundreds of bytes vs the ~3 GB full cache, safe to commit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"alphas": alphas}
    for name in ratios:
        payload[f"ratio__{name}"] = ratios[name]
        payload[f"cliff__{name}"] = np.array([cliffs[name]])
        payload[f"mshift__{name}"] = np.array([mean_shifts[name]])
    np.savez_compressed(path, **payload)


def load_summary_cache(path: Path) -> tuple[np.ndarray, dict, dict, dict]:
    """Load the pre-computed summary cache (alphas, ratios, cliffs, mean_shifts)."""
    d = np.load(path)
    alphas = d["alphas"]
    names = {k.removeprefix("ratio__") for k in d.files if k.startswith("ratio__")}
    ratios = {n: d[f"ratio__{n}"] for n in names}
    cliffs = {n: float(d[f"cliff__{n}"][0]) for n in names}
    mean_shifts = {n: float(d[f"mshift__{n}"][0]) for n in names}
    return alphas, ratios, cliffs, mean_shifts


def cliff_alpha(alphas: np.ndarray, ratios: np.ndarray, threshold: float = 0.5) -> float:
    """Smallest alpha at which the activity ratio first drops below `threshold`.
    Returns NaN if no crossing in range."""
    below = ratios < threshold
    if not below.any():
        return float("nan")
    return float(alphas[np.argmax(below)])


import argparse
import sys


def _build_probed_session():
    """Materialise the probed ONNX (if needed) and open an ORT session over it."""
    import onnxruntime as ort
    src_path = Path("models/supercombo.onnx")
    dst_path = Path("models/supercombo_probed.onnx")
    if not dst_path.exists():
        intermediates_to_outputs(src_path, dst_path, [p.tensor for p in LAYER_PROBES])
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return (ort.InferenceSession(str(dst_path), so,
                                 providers=[("CUDAExecutionProvider", {"device_id": 0}),
                                            "CPUExecutionProvider"]),
            dst_path)


def collect_per_layer(alphas: np.ndarray, n_frames: int,
                      real_six: list[np.ndarray],
                      carla_six: list[np.ndarray],
                      *,
                      session=None,
                      slices=None,
                      probes: list[LayerProbe] | None = None) -> dict[str, np.ndarray]:
    """For each alpha, blend frame-by-frame and run the probed model.
    Returns {layer_name: array of shape (n_alphas, n_frames - 1, *layer_shape)}.

    Pass ``session``, ``slices``, and ``probes`` to run against an alternate
    model (e.g. v0.9.6); otherwise the default v0.9.7 probed session is used."""
    from src.e4_interp import blend
    from src.state import ModelStateMirror, load_output_slices
    from src.warped_preprocessor import stack_pair

    if session is None:
        session, _ = _build_probed_session()
    if slices is None:
        slices = load_output_slices()
    active_probes = probes if probes is not None else LAYER_PROBES
    per_layer: dict[str, list[np.ndarray]] = {p.name: [] for p in active_probes}

    for a in alphas:
        state = ModelStateMirror(session=session, output_slices=slices)
        prev = None
        rows: dict[str, list[np.ndarray]] = {p.name: [] for p in active_probes}
        for k in range(n_frames):
            six = blend(real_six[k], carla_six[k], float(a))
            if prev is not None:
                inp = stack_pair(prev, six)
                _ = state.run(inp, inp)
                raw = state.last_raw_outputs
                for probe in active_probes:
                    rows[probe.name].append(
                        np.asarray(raw[probe.tensor], dtype=np.float32).ravel()
                    )
            prev = six
        for name in rows:
            per_layer[name].append(np.array(rows[name]))
    return {k: np.array(v) for k, v in per_layer.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _analyse(alphas: np.ndarray, per_layer: dict[str, np.ndarray]) -> dict:
    """Per-layer activity ratio + mean shift at alpha=1, plus cliff alpha
    per layer. Activity ratio = temporal std(carla)/std(real); mean shift =
    |mean|(carla) / |mean|(real). The first captures variation, the second
    captures DC offset."""
    real_idx = 0
    ratios = {}
    cliffs = {}
    mean_shifts = {}
    for name, arr in per_layer.items():
        r0 = arr[real_idx]
        per_alpha = [per_layer_activity_ratio(r0, arr[i]) for i in range(len(alphas))]
        ratios[name] = np.array(per_alpha)
        cliffs[name] = cliff_alpha(alphas, ratios[name])
        mean_shifts[name] = per_layer_mean_shift(r0, arr[-1])
    return {"ratios": ratios, "cliffs": cliffs, "mean_shifts": mean_shifts}


# Canonical per-stage view: one representative block per encoder stage, matching
# the originally published figure. Collection probes all 14 blocks (finer
# localization), but the figure shows the same 6 stage curves as the paper.
_STAGE_VIEW = [
    ("head", "head"),
    ("stage0_blk1", "stage0"),
    ("stage1_blk1", "stage1"),
    ("stage2_blk5", "stage2"),
    ("stage3_blk1", "stage3"),
    ("stem", "stem"),
]


def _figure(alphas: np.ndarray, ratios: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    # Reduce the per-block probes to the canonical per-stage representatives when
    # present; fall back to whatever curves are given (e.g. a 6-probe cache).
    view = [(k, lbl) for k, lbl in _STAGE_VIEW if k in ratios]
    if not view:
        view = [(k, k) for k in ratios]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    colors = _physx_style.cmap_cycle(len(view))  # physx editorial palette (blue/green/neutral-gray)
    for (key, label), c in zip(view, colors):
        ax.plot(alphas, ratios[key], marker="o", lw=1.6, color=c, label=label)
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("activity ratio (CARLA / real)")
    ax.set_title("E5: per-stage activity ratio CARLA / real (the cliff is NOT in the encoder)")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="0.5 collapse threshold (never crossed)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def _write_results(alphas: np.ndarray, ratios: dict[str, np.ndarray],
                   cliffs: dict[str, float], mean_shifts: dict[str, float],
                   out: Path) -> None:
    lines = ["# E5 Results: Layer-Localized Collapse", ""]
    lines.append("Activity ratio = sum of per-element temporal std, CARLA / real "
                 "(captures temporal variation).")
    lines.append("Mean shift = sum |mean|(CARLA) / sum |mean|(real) at alpha=1 "
                 "(captures DC offset).")
    lines.append("")
    lines.append("| layer | cliff alpha | activity ratio @ alpha=1 | mean shift @ alpha=1 |")
    lines.append("|---|---|---|---|")
    for name in ratios:
        ca = cliffs[name]
        ca_s = "N/A" if np.isnan(ca) else f"{ca:.3f}"
        lines.append(f"| {name} | {ca_s} | "
                     f"{ratios[name][-1]:.4f} | {mean_shifts[name]:.4f} |")
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true",
                   help="run the heavy GPU collection path; otherwise read cache")
    p.add_argument("--alphas", type=int, default=11)
    p.add_argument("--frames", type=int, default=320)
    args = p.parse_args(argv)

    cache = Path("report/e5_collected.npz")
    summary = Path("report/e5_summary.npz")
    alphas = np.linspace(0.0, 1.0, args.alphas)

    if args.collect:
        from src.e4_interp import CARLA_NPY, SUBARU_HEVC, SUBARU_RLOG
        from src.probe_model import load_carla_six, load_real_six
        real = load_real_six(SUBARU_HEVC, SUBARU_RLOG, args.frames)
        carla = load_carla_six(CARLA_NPY, args.frames)
        per_layer = collect_per_layer(alphas, args.frames, real, carla)
        save_cache(cache, alphas, per_layer)
        a = _analyse(alphas, per_layer)
        save_summary_cache(summary, alphas, a["ratios"], a["cliffs"], a["mean_shifts"])
    elif summary.exists() and not cache.exists():
        alphas, ratios, cliffs, mean_shifts = load_summary_cache(summary)
        a = {"ratios": ratios, "cliffs": cliffs, "mean_shifts": mean_shifts}
        _figure(alphas, a["ratios"], Path("report/figures/e5_layer_localization.png"))
        _write_results(alphas, a["ratios"], a["cliffs"], a["mean_shifts"],
                       Path("report/e5_results.md"))
        print("E5 done (from summary):", a["cliffs"])
        return 0
    else:
        alphas, per_layer = load_cache(cache)
        a = _analyse(alphas, per_layer)

    _figure(alphas, a["ratios"], Path("report/figures/e5_layer_localization.png"))
    _write_results(alphas, a["ratios"], a["cliffs"], a["mean_shifts"],
                   Path("report/e5_results.md"))
    print("E5 done:", a["cliffs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
