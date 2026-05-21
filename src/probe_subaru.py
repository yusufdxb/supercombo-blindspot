"""Extract Subaru source-rlog metadata for parity setup."""

from __future__ import annotations

import sys
from pathlib import Path

from src.rlog import iter_events

DATA = Path(__file__).resolve().parents[1] / "data"
SRC = DATA / "subaru_source" / "rlog.bz2"
REGEN = DATA / "subaru_regen" / "rlog.bz2"


def main() -> int:
    device_type = None
    sensor = None
    rpy_calibs = []
    encode_idx_count = 0
    first_road_cam_frameid = None
    last_road_cam_frameid = None

    for ev in iter_events(SRC):
        which = ev.which()
        if which == "deviceState" and device_type is None:
            device_type = str(ev.deviceState.deviceType)
        elif which == "roadCameraState":
            if sensor is None:
                sensor = str(ev.roadCameraState.sensor)
            fid = ev.roadCameraState.frameId
            if first_road_cam_frameid is None:
                first_road_cam_frameid = fid
            last_road_cam_frameid = fid
        elif which == "liveCalibration":
            rpy_calibs.append(list(ev.liveCalibration.rpyCalib))
        elif which == "roadEncodeIdx":
            encode_idx_count += 1

    print(f"=== SUBARU source rlog ===")
    print(f"  deviceType      : {device_type!r}")
    print(f"  road sensor     : {sensor!r}")
    print(f"  roadEncodeIdx   : {encode_idx_count}")
    print(f"  roadCameraState : {first_road_cam_frameid}..{last_road_cam_frameid} "
          f"({(last_road_cam_frameid - first_road_cam_frameid + 1) if first_road_cam_frameid is not None else 0} frames)")
    print(f"  liveCalibration : {len(rpy_calibs)} updates")
    if rpy_calibs:
        first = rpy_calibs[0]
        last = rpy_calibs[-1]
        print(f"    first rpy (rad): [{first[0]:+.4f}, {first[1]:+.4f}, {first[2]:+.4f}]")
        print(f"    last  rpy (rad): [{last[0]:+.4f}, {last[1]:+.4f}, {last[2]:+.4f}]")

    # also probe regen modelV2 count
    n_mv2 = 0
    first_mv2_fid = None
    last_mv2_fid = None
    first_accel = None
    for ev in iter_events(REGEN):
        if ev.which() == "modelV2":
            mv = ev.modelV2
            if len(mv.acceleration.x) >= 33:
                if first_mv2_fid is None:
                    first_mv2_fid = mv.frameId
                    first_accel = mv.acceleration.x[0]
                last_mv2_fid = mv.frameId
                n_mv2 += 1
    print(f"\n=== SUBARU regen rlog ===")
    print(f"  modelV2          : {n_mv2} frames, frameId {first_mv2_fid}..{last_mv2_fid}")
    print(f"  first accel.x[0] : {first_accel}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
