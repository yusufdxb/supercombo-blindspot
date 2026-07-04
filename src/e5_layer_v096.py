"""E5: layer-localized collapse for the v0.9.6 supercombo.

Same alpha-sweep / activity-ratio methodology as e5_layer.py but uses
supercombo_v096.onnx.  Two probe names differ from v0.9.7:

  stem  : stem.2/act/Mul_5_output_0 (v0.9.7, SiLU)
        → stem.2/act/Mul_3_output_0 (v0.9.6, GELU last-Mul)
  head  : head/global_pool/flatten/Flatten_output_0 (v0.9.7)
        → head/Conv_output_0 (v0.9.6, pre-flatten Conv head)

Stage-block probes are identical between the two versions.

Usage::

    # heavy GPU collection
    env -u PYTHONPATH .venv/bin/python -m src.e5_layer_v096 --collect

    # plot + results from summary cache (no GPU needed)
    env -u PYTHONPATH .venv/bin/python -m src.e5_layer_v096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.e5_layer import (
    LayerProbe,
    cliff_alpha,
    collect_per_layer,
    load_summary_cache,
    per_layer_activity_ratio,
    per_layer_mean_shift,
    save_cache,
    save_summary_cache,
)

LAYER_PROBES_V096: list[LayerProbe] = [
    LayerProbe("stem",        "/supercombo/vision/_en/stem/stem.2/act/Mul_3_output_0"),
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
    LayerProbe("head",        "/supercombo/vision/_en/head/Conv_output_0"),
]

_REPO = Path(__file__).parent.parent
_MODEL_SRC = _REPO / "models" / "supercombo_v096.onnx"
_MODEL_PROBED = _REPO / "models" / "supercombo_v096_probed.onnx"
_CACHE = _REPO / "report" / "e5_v096_collected.npz"
_SUMMARY = _REPO / "report" / "e5_v096_summary.npz"
_FIG_OUT = _REPO / "report" / "figures" / "e5_v096_layer_localization.png"
_RESULTS_OUT = _REPO / "report" / "e5_v096_results.md"


def _build_probed_session_v096():
    from src.e5_layer import intermediates_to_outputs
    import onnxruntime as ort

    if not _MODEL_PROBED.exists():
        intermediates_to_outputs(_MODEL_SRC, _MODEL_PROBED,
                                 [p.tensor for p in LAYER_PROBES_V096])
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(_MODEL_PROBED), so,
                                providers=[("CUDAExecutionProvider", {"device_id": 0}),
                                           "CPUExecutionProvider"])


def _analyse_v096(alphas: np.ndarray,
                  per_layer: dict[str, np.ndarray]) -> dict:
    real_idx = 0
    ratios: dict[str, np.ndarray] = {}
    cliffs: dict[str, float] = {}
    mean_shifts: dict[str, float] = {}
    for name, arr in per_layer.items():
        r0 = arr[real_idx]
        per_alpha = [per_layer_activity_ratio(r0, arr[i]) for i in range(len(alphas))]
        ratios[name] = np.array(per_alpha)
        cliffs[name] = cliff_alpha(alphas, ratios[name])
        mean_shifts[name] = per_layer_mean_shift(r0, arr[-1])
    return {"ratios": ratios, "cliffs": cliffs, "mean_shifts": mean_shifts}


def _figure_v096(alphas: np.ndarray, ratios: dict[str, np.ndarray],
                 out: Path) -> None:
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ratios)))
    for (name, r), c in zip(ratios.items(), colors):
        ax.plot(alphas, r, marker="o", lw=1.6, color=c, label=name)
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("activity ratio (CARLA / real)")
    ax.set_title("E5 v0.9.6: per-stage activity ratio CARLA / real")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="0.5 collapse threshold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def _write_results_v096(alphas: np.ndarray, ratios: dict[str, np.ndarray],
                        cliffs: dict[str, float], mean_shifts: dict[str, float],
                        out: Path) -> None:
    lines = ["# E5 v0.9.6 Results: Layer-Localized Collapse", ""]
    lines.append("Activity ratio = sum of per-element temporal std, CARLA / real.")
    lines.append("Mean shift = sum |mean|(CARLA) / sum |mean|(real) at alpha=1.")
    lines.append("")
    lines.append("| layer | cliff alpha | activity ratio @ alpha=1 | mean shift @ alpha=1 |")
    lines.append("|---|---|---|---|")
    for name in ratios:
        lines.append(f"| {name} | {cliffs[name]:.3f} | "
                     f"{ratios[name][-1]:.4f} | {mean_shifts[name]:.4f} |")
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true",
                   help="run the heavy GPU collection path; otherwise read summary cache")
    p.add_argument("--alphas", type=int, default=11)
    p.add_argument("--frames", type=int, default=320)
    args = p.parse_args(argv)

    alphas = np.linspace(0.0, 1.0, args.alphas)

    if args.collect:
        from src.e4_interp import CARLA_NPY, SUBARU_HEVC, SUBARU_RLOG
        from src.probe_model import load_carla_six, load_real_six
        from src.state import load_output_slices
        sess = _build_probed_session_v096()
        slices = load_output_slices(_MODEL_PROBED)
        real = load_real_six(SUBARU_HEVC, SUBARU_RLOG, args.frames)
        carla = load_carla_six(CARLA_NPY, args.frames)
        per_layer = collect_per_layer(alphas, args.frames, real, carla,
                                      session=sess, slices=slices,
                                      probes=LAYER_PROBES_V096)
        save_cache(_CACHE, alphas, per_layer)
        a = _analyse_v096(alphas, per_layer)
        save_summary_cache(_SUMMARY, alphas, a["ratios"], a["cliffs"], a["mean_shifts"])
    elif _SUMMARY.exists():
        alphas, ratios, cliffs, mean_shifts = load_summary_cache(_SUMMARY)
        a = {"ratios": ratios, "cliffs": cliffs, "mean_shifts": mean_shifts}
    else:
        print("No summary cache found. Run with --collect to generate.", file=sys.stderr)
        return 1

    _figure_v096(alphas, a["ratios"], _FIG_OUT)
    _write_results_v096(alphas, a["ratios"], a["cliffs"], a["mean_shifts"], _RESULTS_OUT)
    print("E5 v0.9.6 done:", a["cliffs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
