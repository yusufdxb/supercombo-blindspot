"""ModelStateMirror — direct port of openpilot ModelState (modeld.py:45-111) at v0.9.7.

Owns the recurrent state buffers (features_buffer, prev_desired_curv, desire)
and threads them across frames. **Zero-init applies only on the very first
frame of a run** — after that the buffers roll forward via shift-and-append.

The slicing dict is loaded directly from supercombo.onnx metadata_props, so
boundaries match comma's runtime exactly (verified to sum to 6504)."""

from __future__ import annotations

import base64
import pickle
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

from src.constants import ModelConstants
from src.parser import Parser

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "supercombo.onnx"


def load_output_slices(model_path: Path = MODEL_PATH) -> dict[str, slice]:
    """Pull the canonical output_slices dict from supercombo.onnx metadata."""
    model = onnx.load(str(model_path))
    for p in model.metadata_props:
        if p.key == "output_slices":
            return pickle.loads(base64.b64decode(p.value.encode()))
    raise RuntimeError("output_slices not present in model metadata_props")


def build_session(model_path: Path = MODEL_PATH) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # ORT 1.26 Level2 SimplifiedLayerNormFusion is incompatible with this export
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), so, providers=providers)


class ModelStateMirror:
    """Mirror of openpilot ModelState. State buffers are float32 (matches reference);
    cast to float16 happens at the ONNX feed boundary."""

    def __init__(self, session: ort.InferenceSession | None = None,
                 output_slices: dict[str, slice] | None = None):
        c = ModelConstants
        self.session = session if session is not None else build_session()
        self.output_slices = output_slices if output_slices is not None else load_output_slices()
        self.parser = Parser()

        # zero-init state — only first frame
        self.prev_desire = np.zeros(c.DESIRE_LEN, dtype=np.float32)
        self.state = {
            "desire": np.zeros(c.DESIRE_LEN * (c.HISTORY_BUFFER_LEN + 1), dtype=np.float32),
            "traffic_convention": np.zeros(c.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
            "lateral_control_params": np.zeros(c.LATERAL_CONTROL_PARAMS_LEN, dtype=np.float32),
            "prev_desired_curv": np.zeros(c.PREV_DESIRED_CURV_LEN * (c.HISTORY_BUFFER_LEN + 1),
                                          dtype=np.float32),
            "features_buffer": np.zeros(c.HISTORY_BUFFER_LEN * c.FEATURE_LEN, dtype=np.float32),
        }

        # cached for shape feed
        self._output_names = [o.name for o in self.session.get_outputs()]

    # --- helpers ---

    @staticmethod
    def _input_shapes() -> dict[str, tuple[int, ...]]:
        c = ModelConstants
        return {
            "input_imgs": (1, 12, 128, 256),
            "big_input_imgs": (1, 12, 128, 256),
            "desire": (1, c.HISTORY_BUFFER_LEN + 1, c.DESIRE_LEN),
            "traffic_convention": (1, c.TRAFFIC_CONVENTION_LEN),
            "lateral_control_params": (1, c.LATERAL_CONTROL_PARAMS_LEN),
            "prev_desired_curv": (1, c.HISTORY_BUFFER_LEN + 1, c.PREV_DESIRED_CURV_LEN),
            "features_buffer": (1, c.HISTORY_BUFFER_LEN, c.FEATURE_LEN),
        }

    def _build_feed(self, input_imgs: np.ndarray, big_input_imgs: np.ndarray) -> dict[str, np.ndarray]:
        shapes = self._input_shapes()
        feed = {
            "input_imgs": input_imgs.astype(np.float16, copy=False).reshape(shapes["input_imgs"]),
            "big_input_imgs": big_input_imgs.astype(np.float16, copy=False).reshape(shapes["big_input_imgs"]),
            "desire": self.state["desire"].astype(np.float16).reshape(shapes["desire"]),
            "traffic_convention": self.state["traffic_convention"].astype(np.float16).reshape(shapes["traffic_convention"]),
            "lateral_control_params": self.state["lateral_control_params"].astype(np.float16).reshape(shapes["lateral_control_params"]),
            "prev_desired_curv": self.state["prev_desired_curv"].astype(np.float16).reshape(shapes["prev_desired_curv"]),
            "features_buffer": self.state["features_buffer"].astype(np.float16).reshape(shapes["features_buffer"]),
        }
        return feed

    def _slice_outputs(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Mirror ModelState.slice_outputs (modeld.py:79-83). Add newaxis batch dim."""
        return {k: flat[np.newaxis, v] for k, v in self.output_slices.items()}

    # --- main entry point ---

    def run(self,
            input_imgs: np.ndarray,
            big_input_imgs: np.ndarray,
            desire: np.ndarray | None = None,
            traffic_convention: np.ndarray | None = None,
            lateral_control_params: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Run one frame, threading state forward. Pre-inference: update non-recurrent
        inputs + roll desire ring (matches modeld.py:87-94). Post-inference: roll
        features_buffer + prev_desired_curv from this frame's outputs (matches
        modeld.py:107-110).
        """
        c = ModelConstants

        # ---- pre-inference state update (lines 87-94 in modeld.py) ----
        if desire is None:
            desire = np.zeros(c.DESIRE_LEN, dtype=np.float32)
        else:
            desire = desire.astype(np.float32).reshape(c.DESIRE_LEN)
            desire[0] = 0  # per modeld.py:88

        # rising-edge pulse + shift-and-append
        self.state["desire"][:-c.DESIRE_LEN] = self.state["desire"][c.DESIRE_LEN:]
        self.state["desire"][-c.DESIRE_LEN:] = np.where(desire - self.prev_desire > 0.99, desire, 0)
        self.prev_desire[:] = desire

        if traffic_convention is not None:
            self.state["traffic_convention"][:] = traffic_convention
        if lateral_control_params is not None:
            self.state["lateral_control_params"][:] = lateral_control_params

        # ---- inference ----
        feed = self._build_feed(input_imgs, big_input_imgs)
        raw_list = self.session.run(self._output_names, feed)
        self.last_raw_outputs = dict(zip(self._output_names, raw_list))
        flat = self.last_raw_outputs["outputs"]  # shape (1, 6504), fp16
        flat = flat[0].astype(np.float32)  # match modeld.py's fp32 buffer

        # slice -> parse
        sliced = self._slice_outputs(flat)
        parsed = self.parser.parse_outputs(sliced)

        # ---- post-inference state update (lines 107-110 in modeld.py) ----
        # features_buffer rolls in hidden_state
        self.state["features_buffer"][:-c.FEATURE_LEN] = self.state["features_buffer"][c.FEATURE_LEN:]
        self.state["features_buffer"][-c.FEATURE_LEN:] = parsed["hidden_state"][0, :]

        # prev_desired_curv rolls in the model's predicted curvature
        self.state["prev_desired_curv"][:-c.PREV_DESIRED_CURV_LEN] = (
            self.state["prev_desired_curv"][c.PREV_DESIRED_CURV_LEN:]
        )
        self.state["prev_desired_curv"][-c.PREV_DESIRED_CURV_LEN:] = parsed["desired_curvature"][0, :]

        return parsed


def long_accel_t0(parsed: dict[str, np.ndarray]) -> float:
    """Per-frame longitudinal accel at t≈0 from the top plan hypothesis."""
    # plan shape: (1, 33, 15); channel 6 = ax (Plan.ACCELERATION slice 6:9 -> ax,ay,az)
    return float(parsed["plan"][0, 0, 6])
