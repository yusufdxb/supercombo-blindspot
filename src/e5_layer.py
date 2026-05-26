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
import onnx


@dataclass(frozen=True)
class LayerProbe:
    name: str
    tensor: str


LAYER_PROBES: list[LayerProbe] = [
    LayerProbe("head",   "/supercombo/vision/_en/head/global_pool/flatten/Flatten_output_0"),
    LayerProbe("stage0", "/supercombo/vision/_en/stages.0/blocks/blocks.1/Add_output_0"),
    LayerProbe("stage1", "/supercombo/vision/_en/stages.1/blocks/blocks.1/Add_output_0"),
    LayerProbe("stage2", "/supercombo/vision/_en/stages.2/blocks/blocks.5/Add_output_0"),
    LayerProbe("stage3", "/supercombo/vision/_en/stages.3/blocks/blocks.1/Add_output_0"),
    LayerProbe("stem",   "/supercombo/vision/_en/stem/stem.2/act/Mul_5_output_0"),
]


def intermediates_to_outputs(src: Path, dst: Path, tensor_names: list[str]) -> None:
    """Copy `src` to `dst` with the given intermediate tensors added as graph outputs.
    Names that do not exist in the graph raise.
    Uses ONNX shape inference to obtain the correct element dtype for each tensor."""
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


def cliff_alpha(alphas: np.ndarray, ratios: np.ndarray, threshold: float = 0.5) -> float:
    """Smallest alpha at which the activity ratio first drops below `threshold`.
    Returns NaN if no crossing in range."""
    below = ratios < threshold
    if not below.any():
        return float("nan")
    return float(alphas[np.argmax(below)])


import argparse
import sys

import onnxruntime as ort  # noqa: E402


def _build_probed_session() -> tuple[ort.InferenceSession, Path]:
    """Materialise the probed ONNX (if needed) and open an ORT session over it."""
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
                      carla_six: list[np.ndarray]) -> dict[str, np.ndarray]:
    """For each alpha, blend frame-by-frame and run the probed model.
    Returns {layer_name: array of shape (n_alphas, n_frames - 1, *layer_shape)}."""
    from src.e4_interp import blend
    from src.state import ModelStateMirror, load_output_slices
    from src.warped_preprocessor import stack_pair

    sess, _ = _build_probed_session()
    slices = load_output_slices()
    per_layer: dict[str, list[np.ndarray]] = {p.name: [] for p in LAYER_PROBES}

    for a in alphas:
        state = ModelStateMirror(session=sess, output_slices=slices)
        prev = None
        rows: dict[str, list[np.ndarray]] = {p.name: [] for p in LAYER_PROBES}
        for k in range(n_frames):
            six = blend(real_six[k], carla_six[k], float(a))
            if prev is not None:
                inp = stack_pair(prev, six)
                _ = state.run(inp, inp)
                raw = state.last_raw_outputs
                for probe in LAYER_PROBES:
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
    """Per-layer activity ratio at each alpha (vs the alpha=0 / real baseline),
    plus cliff alpha per layer."""
    real_idx = 0
    ratios = {}
    cliffs = {}
    for name, arr in per_layer.items():
        r0 = arr[real_idx]
        per_alpha = [per_layer_activity_ratio(r0, arr[i]) for i in range(len(alphas))]
        ratios[name] = np.array(per_alpha)
        cliffs[name] = cliff_alpha(alphas, ratios[name])
    return {"ratios": ratios, "cliffs": cliffs}


def _figure(alphas: np.ndarray, ratios: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ratios)))
    for (name, r), c in zip(ratios.items(), colors):
        ax.plot(alphas, r, marker="o", lw=1.6, color=c, label=name)
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("activity ratio (CARLA / real)")
    ax.set_title("E5: where the collapse cliff lives in supercombo")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="cliff threshold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def _write_results(alphas: np.ndarray, ratios: dict[str, np.ndarray],
                   cliffs: dict[str, float], out: Path) -> None:
    lines = ["# E5 Results: Layer-Localized Collapse", ""]
    lines.append("| layer | cliff alpha | ratio @ alpha=1 |")
    lines.append("|---|---|---|")
    for name in ratios:
        lines.append(f"| {name} | {cliffs[name]:.3f} | {ratios[name][-1]:.4f} |")
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true",
                   help="run the heavy GPU collection path; otherwise read cache")
    p.add_argument("--alphas", type=int, default=11)
    p.add_argument("--frames", type=int, default=320)
    args = p.parse_args(argv)

    cache = Path("report/e5_collected.npz")
    alphas = np.linspace(0.0, 1.0, args.alphas)

    if args.collect:
        from src.e4_interp import CARLA_NPY, SUBARU_HEVC, SUBARU_RLOG
        from src.probe_model import load_carla_six, load_real_six
        real = load_real_six(SUBARU_HEVC, SUBARU_RLOG, args.frames)
        carla = load_carla_six(CARLA_NPY, args.frames)
        per_layer = collect_per_layer(alphas, args.frames, real, carla)
        save_cache(cache, alphas, per_layer)
    else:
        alphas, per_layer = load_cache(cache)

    a = _analyse(alphas, per_layer)
    _figure(alphas, a["ratios"], Path("report/figures/e5_layer_localization.png"))
    _write_results(alphas, a["ratios"], a["cliffs"], Path("report/e5_results.md"))
    print("E5 done:", a["cliffs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
