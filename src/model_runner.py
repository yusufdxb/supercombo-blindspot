"""Load openpilot v0.9.7 supercombo.onnx, print I/O spec, run zero-input
inference, benchmark CUDA latency."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Load NVIDIA pip-wheel CUDA/cuDNN libs so CUDAExecutionProvider can find them
# without system-level installs. ORT 1.18+ ships this helper.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "supercombo.onnx"


def build_session(model_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3  # WARN+ only
    # ORT 1.26 Level2 SimplifiedLayerNormFusion is incompatible with this
    # supercombo export (torch 2.2.2). Disable all fusions; perf comes from EP kernels.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(str(model_path), so, providers=providers)
    return sess


def describe_io(sess: ort.InferenceSession) -> tuple[list, list]:
    print("\n=== INPUTS ===")
    inputs = []
    for inp in sess.get_inputs():
        print(f"  {inp.name:30s} shape={inp.shape} dtype={inp.type}")
        inputs.append(inp)
    print("\n=== OUTPUTS ===")
    outputs = []
    for out in sess.get_outputs():
        print(f"  {out.name:30s} shape={out.shape} dtype={out.type}")
        outputs.append(out)
    return inputs, outputs


def _np_dtype_for(ort_type: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
    }
    if ort_type not in mapping:
        raise ValueError(f"unsupported ORT dtype: {ort_type}")
    return mapping[ort_type]


def build_zero_inputs(inputs) -> dict[str, np.ndarray]:
    feed = {}
    for inp in inputs:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feed[inp.name] = np.zeros(shape, dtype=_np_dtype_for(inp.type))
    return feed


def benchmark(sess, feed, n_iter: int = 100, warmup: int = 5) -> dict:
    output_names = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(output_names, feed)
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(output_names, feed)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "iterations": n_iter,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(0.95 * len(samples)) - 1],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> int:
    print(f"onnxruntime: {ort.__version__}")
    print(f"available providers: {ort.get_available_providers()}")
    print(f"model: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")

    sess = build_session(MODEL_PATH)
    print(f"\nactive session providers (in priority order): {sess.get_providers()}")

    inputs, outputs = describe_io(sess)
    feed = build_zero_inputs(inputs)

    print("\n=== SINGLE INFERENCE (zeros) ===")
    t0 = time.perf_counter()
    out_list = sess.run([o.name for o in outputs], feed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  ran in {elapsed_ms:.2f} ms (cold)")
    for o, arr in zip(outputs, out_list):
        print(f"  {o.name:30s} returned shape={arr.shape} dtype={arr.dtype}")

    print("\n=== BENCHMARK (100 iter, 5 warmup) ===")
    stats = benchmark(sess, feed)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:12s} {v:.2f}")
        else:
            print(f"  {k:12s} {v}")

    gpu_threshold_ms = 100.0
    if stats["median_ms"] > gpu_threshold_ms:
        print(
            f"\nWARN: median latency {stats['median_ms']:.1f} ms > {gpu_threshold_ms} ms — "
            f"GPU may not be active. Investigate."
        )
        return 2
    print(f"\nOK: median latency {stats['median_ms']:.1f} ms <= {gpu_threshold_ms} ms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
