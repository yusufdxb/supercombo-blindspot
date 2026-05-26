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
    Names that do not exist in the graph raise."""
    m = onnx.load(str(src))
    existing = {o.name for o in m.graph.output}
    vi_by_name = {vi.name: vi for vi in m.graph.value_info}
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
