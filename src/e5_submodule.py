"""E5 submodule localization: pin the collapse to a specific block inside
the recurrent / policy stack, downstream of the vision encoder.

Builds on src/e5_layer.py (which already established that the cliff is NOT in
the encoder: all 6 stages keep activity > 0.96 at alpha=1). This script probes
8 tensors between summarizer and the per-head Gemms to localise the drift to
one structural role:

    vision_post -> summarizer -> [transformer attention + MLP] -> ReduceSum
                                -> {action_block, hydra_trunk, temporal_hydra_trunk}
                                -> per-head Gemms -> Concat -> outputs

Notes:
- Reuses `intermediates_to_outputs` from src.e5_layer to build a probed ONNX
  with the 8 internal tensors exposed as graph outputs (do NOT edit e5_layer.py).
- Uses the same blend + ModelStateMirror loop as e5_layer.collect_per_layer,
  with the new probe list.
- The original e5 cache is corrupt (truncated zip, ~4 GB). This script writes
  to a separate cache: report/e5_submodule_collected.npz.

Run (collect, heavy GPU path):
    cd ~/Projects/phantom-braking && source .venv/bin/activate
    python -m src.e5_submodule --collect --alphas 11 --frames 320

Analyse only (from cache):
    python -m src.e5_submodule
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.e5_layer import (
    cliff_alpha,
    intermediates_to_outputs,
    per_layer_activity_ratio,
    per_layer_mean_shift,
)


@dataclass(frozen=True)
class SubProbe:
    name: str
    tensor: str
    role: str


SUBMODULE_PROBES: list[SubProbe] = [
    SubProbe(
        "vision_post",
        "/supercombo/vision/Flatten_output_0",
        "post-encoder FC (1024 to 2048)",
    ),
    SubProbe(
        "summarizer_div",
        "/summarizer/Div_output_0",
        "summarizer VAE-mu, == hidden_state",
    ),
    SubProbe(
        "attention_block_out",
        "/Add_1_output_0",
        "transformer self-attention output + residual (1, 10, 512)",
    ),
    SubProbe(
        "transformer_block_out",
        "/Add_2_output_0",
        "transformer FFN output + residual (1, 10, 512)",
    ),
    SubProbe(
        "reduce_sum",
        "/ReduceSum_output_0",
        "temporal aggregation, 10 tokens to 512",
    ),
    SubProbe(
        "action_block_body",
        "/action_block/resblocks.1/final_relu/Relu_output_0",
        "action-block last resblock output (1, 128), pre-curvature head",
    ),
    SubProbe(
        "hydra_trunk",
        "/supercombo/no_bottleneck_policy/hydra/resblock/final_relu/Relu_output_0",
        "non-temporal hydra trunk (1, 512), feeds meta/pose/desire_pred/etc.",
    ),
    SubProbe(
        "temporal_hydra_trunk",
        "/temporal_hydra/resblock/final_relu/Relu_output_0",
        "temporal hydra trunk (1, 512), feeds plan/lane_lines/lead/etc.",
    ),
]


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "report" / "e5_submodule_collected.npz"
FIG = ROOT / "report" / "figures" / "e5_submodule_localization.png"
RESULTS = ROOT / "report" / "e5_submodule_results.md"
PROBED_ONNX = ROOT / "models" / "supercombo_submodule_probed.onnx"


def save_cache(path: Path, alphas: np.ndarray,
               per_probe: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        alphas=alphas,
        **{f"probe__{k}": v for k, v in per_probe.items()},
    )


def load_cache(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    d = np.load(path)
    alphas = d["alphas"]
    per_probe = {
        k.removeprefix("probe__"): d[k]
        for k in d.files
        if k.startswith("probe__")
    }
    return alphas, per_probe


def _build_probed_session():
    """Build / open the submodule-probed ONNX session. Mirrors the e5_layer
    helper but uses a separate probed file so it does not clobber the
    encoder-only probed model."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    src_path = ROOT / "models" / "supercombo.onnx"
    if not PROBED_ONNX.exists():
        intermediates_to_outputs(
            src_path, PROBED_ONNX, [p.tensor for p in SUBMODULE_PROBES]
        )
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(PROBED_ONNX),
        so,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ],
    )


def collect_per_probe(
    alphas: np.ndarray,
    n_frames: int,
    real_six: list[np.ndarray],
    carla_six: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """For each alpha, blend frame-by-frame and run the probed model. Returns
    {probe_name: array of shape (n_alphas, n_frames - 1, flat_dim)}."""
    from src.e4_interp import blend
    from src.state import ModelStateMirror, load_output_slices
    from src.warped_preprocessor import stack_pair

    sess = _build_probed_session()
    slices = load_output_slices()
    per_probe: dict[str, list[np.ndarray]] = {p.name: [] for p in SUBMODULE_PROBES}

    for ai, a in enumerate(alphas):
        state = ModelStateMirror(session=sess, output_slices=slices)
        prev = None
        rows: dict[str, list[np.ndarray]] = {p.name: [] for p in SUBMODULE_PROBES}
        for k in range(n_frames):
            six = blend(real_six[k], carla_six[k], float(a))
            if prev is not None:
                inp = stack_pair(prev, six)
                _ = state.run(inp, inp)
                raw = state.last_raw_outputs
                for probe in SUBMODULE_PROBES:
                    rows[probe.name].append(
                        np.asarray(raw[probe.tensor], dtype=np.float32).ravel()
                    )
            prev = six
        for name in rows:
            per_probe[name].append(np.array(rows[name], dtype=np.float32))
        print(f"  alpha={float(a):.4f} collected ({ai + 1}/{len(alphas)})")
    return {k: np.array(v, dtype=np.float32) for k, v in per_probe.items()}


def analyse(alphas: np.ndarray,
            per_probe: dict[str, np.ndarray]) -> dict:
    """Per-probe activity ratio over the alpha sweep + cliff alpha + mean
    shift at alpha=1. Re-uses the same metrics e5_layer.py uses, so the two
    reports are directly comparable."""
    real_idx = 0
    ratios: dict[str, np.ndarray] = {}
    cliffs: dict[str, float] = {}
    mean_shifts: dict[str, float] = {}
    for name, arr in per_probe.items():
        r0 = arr[real_idx]
        per_alpha = [
            per_layer_activity_ratio(r0, arr[i]) for i in range(len(alphas))
        ]
        ratios[name] = np.array(per_alpha)
        cliffs[name] = cliff_alpha(alphas, ratios[name])
        mean_shifts[name] = per_layer_mean_shift(r0, arr[-1])
    return {"ratios": ratios, "cliffs": cliffs, "mean_shifts": mean_shifts}


def figure(alphas: np.ndarray, ratios: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=140)
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(ratios)))
    for (name, r), c in zip(ratios.items(), colors):
        ax.plot(alphas, r, marker="o", lw=1.6, color=c, label=name)
    ax.axhline(
        0.5,
        color="grey",
        lw=0.8,
        ls="--",
        label="0.5 collapse threshold",
    )
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("activity ratio (CARLA / real)")
    ax.set_title(
        "E5 submodule: localising the cliff downstream of the encoder"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def write_results(
    alphas: np.ndarray,
    ratios: dict[str, np.ndarray],
    cliffs: dict[str, float],
    mean_shifts: dict[str, float],
    out: Path,
) -> None:
    lines = [
        "# E5 Submodule Results: where the cliff actually lives",
        "",
        "Extends `report/e5_results.md` (vision encoder, no cliff found) by",
        "probing 8 tensors between summarizer and the per-head Gemms.",
        "",
        "Metrics match `src/e5_layer.py`:",
        "",
        "- activity ratio = sum of per-element temporal std, CARLA / real",
        "- mean shift = sum |mean|(CARLA) / sum |mean|(real) at alpha=1",
        "- cliff alpha = smallest alpha at which activity ratio first drops",
        "  below 0.5; N/A if it never crosses",
        "",
        "## Probe list (8 points, chosen from"
        " `report/e5_submodule_enumeration.md`)",
        "",
        "| probe | role | tensor |",
        "|---|---|---|",
    ]
    for p in SUBMODULE_PROBES:
        lines.append(f"| `{p.name}` | {p.role} | `{p.tensor}` |")
    lines.append("")
    lines.append("## Per-probe activity along the alpha sweep")
    lines.append("")
    lines.append(
        "| probe | cliff alpha | activity ratio @ alpha=1 | mean shift @ alpha=1 |"
    )
    lines.append("|---|---|---|---|")
    for name in ratios:
        ca = cliffs[name]
        ca_s = "no cliff" if np.isnan(ca) else f"{ca:.3f}"
        lines.append(
            f"| `{name}` | {ca_s} | {ratios[name][-1]:.4f} | "
            f"{mean_shifts[name]:.4f} |"
        )
    lines.append("")
    lines.append("## Per-alpha activity ratio, by probe")
    lines.append("")
    header = "| alpha | " + " | ".join(ratios.keys()) + " |"
    sep = "|---" * (len(ratios) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for i, a in enumerate(alphas):
        row = [f"{float(a):.4f}"] + [
            f"{ratios[k][i]:.3f}" for k in ratios
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Figure: `report/figures/e5_submodule_localization.png`.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--collect",
        action="store_true",
        help="Run the model across the alpha sweep (heavy GPU path). "
             "Otherwise read from cache.",
    )
    p.add_argument(
        "--alphas",
        type=int,
        default=11,
        help="Number of alphas in linspace(0, 1, alphas).",
    )
    p.add_argument("--frames", type=int, default=320)
    args = p.parse_args(argv)

    alphas = np.linspace(0.0, 1.0, args.alphas)

    if args.collect:
        from src.e4_interp import CARLA_NPY, SUBARU_HEVC, SUBARU_RLOG
        from src.probe_model import load_carla_six, load_real_six

        print(f"Loading {args.frames} frames of real + CARLA inputs...")
        real = load_real_six(SUBARU_HEVC, SUBARU_RLOG, args.frames)
        carla = load_carla_six(CARLA_NPY, args.frames)
        print(f"Collecting across {len(alphas)} alphas...")
        per_probe = collect_per_probe(alphas, args.frames, real, carla)
        save_cache(CACHE, alphas, per_probe)
        print(f"  cached -> {CACHE.relative_to(ROOT)}")
    else:
        if not CACHE.exists():
            print(
                f"cache {CACHE} not found; run with --collect first",
                file=sys.stderr,
            )
            return 2
        alphas, per_probe = load_cache(CACHE)

    a = analyse(alphas, per_probe)
    figure(alphas, a["ratios"], FIG)
    write_results(alphas, a["ratios"], a["cliffs"], a["mean_shifts"], RESULTS)
    print("E5 submodule done. Cliffs:")
    for k, v in a["cliffs"].items():
        v_s = "no cliff" if np.isnan(v) else f"{v:.3f}"
        print(f"  {k:24s}  cliff_alpha={v_s}  "
              f"activity@1={a['ratios'][k][-1]:.4f}  "
              f"mean_shift@1={a['mean_shifts'][k]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
