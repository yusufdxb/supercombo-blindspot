"""Mirror of openpilot/selfdrive/modeld/constants.py at v0.9.7.

We re-export ModelConstants + Plan slices here so the rest of the package
doesn't have to import from references/. Kept in sync deliberately."""

from __future__ import annotations


def _index_function(idx: int, max_val: float = 192.0, max_idx: int = 32) -> float:
    return max_val * ((idx / max_idx) ** 2)


class ModelConstants:
    IDX_N = 33
    T_IDXS = [_index_function(i, max_val=10.0) for i in range(IDX_N)]
    X_IDXS = [_index_function(i, max_val=192.0) for i in range(IDX_N)]

    MODEL_FREQ = 20
    FEATURE_LEN = 512
    HISTORY_BUFFER_LEN = 99
    DESIRE_LEN = 8
    TRAFFIC_CONVENTION_LEN = 2
    LATERAL_CONTROL_PARAMS_LEN = 2
    PREV_DESIRED_CURV_LEN = 1

    POSE_WIDTH = 6
    WIDE_FROM_DEVICE_WIDTH = 3
    SIM_POSE_WIDTH = 6
    LEAD_WIDTH = 4
    LANE_LINES_WIDTH = 2
    PLAN_WIDTH = 15
    DESIRE_PRED_WIDTH = 8
    DESIRED_CURV_WIDTH = 1

    NUM_LANE_LINES = 4
    NUM_ROAD_EDGES = 2
    LEAD_TRAJ_LEN = 6
    DESIRE_PRED_LEN = 4

    PLAN_MHP_N = 5
    LEAD_MHP_N = 2
    PLAN_MHP_SELECTION = 1
    LEAD_MHP_SELECTION = 3


class Plan:
    # parsed plan tensor channel layout (shape: [batch, IDX_N=33, PLAN_WIDTH=15])
    POSITION_X, POSITION_Y, POSITION_Z = 0, 1, 2
    VELOCITY_X, VELOCITY_Y, VELOCITY_Z = 3, 4, 5
    ACCELERATION_X, ACCELERATION_Y, ACCELERATION_Z = 6, 7, 8
    # 9..14: orientation, orientation_rate
