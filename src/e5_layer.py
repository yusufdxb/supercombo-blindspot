"""E5: layer-localized collapse along the real-to-sim interpolation sweep.

We add one tensor per vision-encoder stage to the ONNX graph's outputs, then
run the same alpha sweep as E4 and measure per-layer activity ratio
CARLA / real. The output is the layer-by-alpha map that says where the
collapse cliff lives in the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
