"""Extract output_slices dict from supercombo.onnx metadata_props (the
authoritative slice boundaries comma uses at runtime). Prints the byte budget
and confirms it sums to the model's flat output length."""

from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path

import onnx

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "supercombo.onnx"


def extract_output_slices(model_path: Path) -> dict[str, slice]:
    model = onnx.load(str(model_path))
    for p in model.metadata_props:
        if p.key == "output_slices":
            return pickle.loads(base64.b64decode(p.value.encode()))
    raise RuntimeError("output_slices not found in model metadata_props")


def get_output_size(model_path: Path) -> int:
    model = onnx.load(str(model_path))
    for o in model.graph.output:
        if o.name == "outputs":
            return int(o.type.tensor_type.shape.dim[1].dim_value)
    raise RuntimeError("no 'outputs' tensor")


def main() -> int:
    slices = extract_output_slices(MODEL_PATH)
    flat_len = get_output_size(MODEL_PATH)

    print(f"flat output length: {flat_len}")
    print(f"slices found: {len(slices)}\n")

    header = f"{'name':<28s} {'start':>8s} {'stop':>8s} {'len':>8s}"
    print(header)
    print("-" * len(header))
    total = 0
    for name, sl in slices.items():
        if not isinstance(sl, slice):
            print(f"  WARN non-slice entry: {name} = {sl!r}")
            continue
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else flat_len
        length = stop - start
        total += length
        print(f"{name:<28s} {start:>8d} {stop:>8d} {length:>8d}")

    print("-" * len(header))
    print(f"{'TOTAL':<28s} {'':>8s} {'':>8s} {total:>8d}")
    print(f"{'EXPECTED':<28s} {'':>8s} {'':>8s} {flat_len:>8d}")
    print()
    if total == flat_len:
        print(f"OK: slice budget sums to {flat_len} exactly.")
        return 0
    print(f"FAIL: slice budget {total} != flat output {flat_len} (delta {total - flat_len})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
