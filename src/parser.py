"""Verbatim port of openpilot/selfdrive/modeld/parse_model_outputs.py at v0.9.7.

Kept self-contained (no openpilot import) so the package runs without the
openpilot working tree on disk. If you touch this file, diff against
references/openpilot-v0.9.7/parse_model_outputs.py to verify byte-for-byte
behavioral equivalence."""

from __future__ import annotations

import numpy as np

from src.constants import ModelConstants


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x) if x.dtype in (np.float32, np.float64) else None
    if x.dtype not in (np.float32, np.float64):
        x = np.exp(x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


class Parser:
    def __init__(self, ignore_missing: bool = False):
        self.ignore_missing = ignore_missing

    def _check_missing(self, outs: dict, name: str) -> bool:
        if name not in outs and not self.ignore_missing:
            raise ValueError(f"Missing output {name}")
        return name not in outs

    def parse_categorical_crossentropy(self, name: str, outs: dict, out_shape=None) -> None:
        if self._check_missing(outs, name):
            return
        raw = outs[name]
        if out_shape is not None:
            raw = raw.reshape((raw.shape[0],) + out_shape)
        outs[name] = _softmax(raw.astype(np.float32), axis=-1)

    def parse_binary_crossentropy(self, name: str, outs: dict) -> None:
        if self._check_missing(outs, name):
            return
        outs[name] = _sigmoid(outs[name].astype(np.float32))

    def parse_mdn(self, name: str, outs: dict, in_N: int = 0, out_N: int = 1, out_shape=None) -> None:
        if self._check_missing(outs, name):
            return
        raw = outs[name].astype(np.float32)
        raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))

        n_values = (raw.shape[2] - out_N) // 2
        pred_mu = raw[:, :, :n_values]
        pred_std = np.exp(raw[:, :, n_values:2 * n_values])

        if in_N > 1:
            weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
            for i in range(out_N):
                weights[:, :, i - out_N] = _softmax(raw[:, :, i - out_N].copy(), axis=-1)

            if out_N == 1:
                for fidx in range(weights.shape[0]):
                    idxs = np.argsort(weights[fidx][:, 0])[::-1]
                    weights[fidx] = weights[fidx][idxs]
                    pred_mu[fidx] = pred_mu[fidx][idxs]
                    pred_std[fidx] = pred_std[fidx][idxs]
            full_shape = tuple([raw.shape[0], in_N] + list(out_shape))
            outs[name + "_weights"] = weights
            outs[name + "_hypotheses"] = pred_mu.reshape(full_shape)
            outs[name + "_stds_hypotheses"] = pred_std.reshape(full_shape)

            pred_mu_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
            pred_std_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
            for fidx in range(weights.shape[0]):
                for hidx in range(out_N):
                    idxs = np.argsort(weights[fidx, :, hidx])[::-1]
                    pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
                    pred_std_final[fidx, hidx] = pred_std[fidx, idxs[0]]
        else:
            pred_mu_final = pred_mu
            pred_std_final = pred_std

        if out_N > 1:
            final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
        else:
            final_shape = tuple([raw.shape[0]] + list(out_shape))
        outs[name] = pred_mu_final.reshape(final_shape)
        outs[name + "_stds"] = pred_std_final.reshape(final_shape)

    def parse_outputs(self, outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        c = ModelConstants
        self.parse_mdn("plan", outs, in_N=c.PLAN_MHP_N, out_N=c.PLAN_MHP_SELECTION,
                       out_shape=(c.IDX_N, c.PLAN_WIDTH))
        self.parse_mdn("lane_lines", outs, in_N=0, out_N=0,
                       out_shape=(c.NUM_LANE_LINES, c.IDX_N, c.LANE_LINES_WIDTH))
        self.parse_mdn("road_edges", outs, in_N=0, out_N=0,
                       out_shape=(c.NUM_ROAD_EDGES, c.IDX_N, c.LANE_LINES_WIDTH))
        self.parse_mdn("pose", outs, in_N=0, out_N=0, out_shape=(c.POSE_WIDTH,))
        self.parse_mdn("road_transform", outs, in_N=0, out_N=0, out_shape=(c.POSE_WIDTH,))
        self.parse_mdn("sim_pose", outs, in_N=0, out_N=0, out_shape=(c.POSE_WIDTH,))
        self.parse_mdn("wide_from_device_euler", outs, in_N=0, out_N=0,
                       out_shape=(c.WIDE_FROM_DEVICE_WIDTH,))
        self.parse_mdn("lead", outs, in_N=c.LEAD_MHP_N, out_N=c.LEAD_MHP_SELECTION,
                       out_shape=(c.LEAD_TRAJ_LEN, c.LEAD_WIDTH))
        if "desired_curvature" in outs:
            self.parse_mdn("desired_curvature", outs, in_N=0, out_N=0,
                           out_shape=(c.DESIRED_CURV_WIDTH,))
        for k in ("lead_prob", "lane_lines_prob", "meta"):
            self.parse_binary_crossentropy(k, outs)
        self.parse_categorical_crossentropy("desire_state", outs,
                                            out_shape=(c.DESIRE_PRED_WIDTH,))
        self.parse_categorical_crossentropy("desire_pred", outs,
                                            out_shape=(c.DESIRE_PRED_LEN, c.DESIRE_PRED_WIDTH))
        return outs
