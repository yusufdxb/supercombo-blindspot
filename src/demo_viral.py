"""Viral demo render: openpilot supercombo driving on real dashcam footage,
then progressively blinded until its predicted path collapses while its own
confidence stays pinned high.

Story: "This is the brain of an open-source self-driving car. Watch me slowly
blind it, and it never notices." The real Subaru highway clip plays with the
model's live predicted path + lane lines overlaid. Across the timeline the input
is alpha-blended toward the CARLA / out-of-distribution render (the SAME real
corruption the E4 sweep uses, not an ad-hoc filter). The predicted trajectory
flattens / collapses while the model's predicted plan uncertainty barely moves.
An internal-feature OOD monitor (E6 rolling spread on the recurrent hidden
state) is shown alongside as the detector that actually catches it.

Correctness (these are known traps in THIS model, see repo memory):
  * Input construction reuses the parity-verified path exactly:
    load_real_six / load_carla_six (calibrated warp + yuv_to_6ch), blend() from
    src.e4_interp, stack_pair, and ModelStateMirror.run. Nothing is reinvented.
  * YUV is UNNORMALIZED (uint8 -> float, 0..255). yuv_to_6ch owns that; we do
    not touch it.
  * Recurrent state ROLLS: ModelStateMirror threads features_buffer and
    prev_desired_curv shift-and-append; zero-init only on frame 1. We drive one
    continuous warm run, never re-instantiating per frame.

A parity gate runs FIRST: the alpha=0 (clean real) hidden_state from this
script is compared against the committed teardown_collected.npz subaru cache
(same collect() path). It must match to < PARITY_TOL max-abs before any frame is
rendered.

    env -u PYTHONPATH .venv/bin/python -m src.demo_viral            # full render
    env -u PYTHONPATH .venv/bin/python -m src.demo_viral --parity-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from src.constants import ModelConstants
from src.decode_hevc import yuv_frame_iter
from src.e4_interp import blend
from src.probe_model import _calib_warps, load_carla_six, load_real_six
from src.state import ModelStateMirror
from src.transformations import (_ar_ox_config, rot_from_euler,
                                 view_frame_from_device_frame)
from src.warped_preprocessor import stack_pair

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEMO_DIR = ROOT / "demo"
SUBARU_HEVC = DATA / "subaru_source" / "fcamera.hevc"
SUBARU_RLOG = DATA / "subaru_source" / "rlog.bz2"
CARLA_NPY = DATA / "domain_gap" / "carla_rgb.npy"
TEARDOWN_CACHE = ROOT / "report" / "teardown_collected.npz"

CAM_W, CAM_H = 1928, 1208
N_FRAMES = 300            # clip length fed to the model (matches teardown N range)
FPS = 20                  # supercombo runs at 20 Hz
PARITY_TOL = 1e-3         # max-abs hidden_state diff alpha=0 vs cache (fp16 model)

# alpha ramp across the timeline: hold clean, then slowly blind toward CARLA.
HOLD_CLEAN = 70           # frames of untouched real driving up front
ALPHA_MAX = 0.85          # peak blindness (full collapse well before 1.0)
RAMP_END = 250            # frame at which ALPHA_MAX is reached, then hold

# E6 rolling-spread monitor
WINDOW = 30

# colours (BGR for cv2)
C_PATH = (90, 230, 90)      # green predicted path
C_LANE = (240, 200, 70)     # cyan-ish lane lines
C_EDGE = (70, 130, 240)     # orange road edges
C_GOOD = (90, 220, 90)
C_BAD = (70, 70, 235)
C_WARN = (60, 200, 250)
C_TEXT = (235, 235, 235)
C_DIM = (150, 150, 150)


# --------------------------------------------------------------------------
# alpha schedule
# --------------------------------------------------------------------------

def alpha_at(frame_idx: int) -> float:
    if frame_idx <= HOLD_CLEAN:
        return 0.0
    if frame_idx >= RAMP_END:
        return ALPHA_MAX
    t = (frame_idx - HOLD_CLEAN) / (RAMP_END - HOLD_CLEAN)
    # smoothstep for a cinematic ease
    s = t * t * (3 - 2 * t)
    return ALPHA_MAX * s


# --------------------------------------------------------------------------
# device-frame -> camera-image projection (vendored openpilot geometry)
# --------------------------------------------------------------------------

def camera_from_device(rpy_calib: np.ndarray) -> np.ndarray:
    """3x3 mapping device-frame XYZ -> camera pixel homogeneous coords.

    camera_from_calib = K @ view_frame_from_device_frame @ device_from_calib
    (same composition src.transformations.get_warp_matrix uses). The model's
    plan / lane_lines live in the calibrated device frame; this projects them
    onto the full-res fcamera image we display.
    """
    K = _ar_ox_config.fcam.intrinsics
    device_from_calib = rot_from_euler(rpy_calib)
    return K @ view_frame_from_device_frame @ device_from_calib


def project_points(xyz: np.ndarray, cam_from_dev: np.ndarray) -> np.ndarray:
    """xyz (N,3) in device frame -> (N,2) pixel coords. Points behind the
    camera (x<=0 forward) or out of frame are returned as NaN."""
    pts = (cam_from_dev @ xyz.T).T          # (N,3) homogeneous
    z = pts[:, 2:3]
    valid = (xyz[:, 0] > 1.0) & (z[:, 0] > 1e-3)
    uv = np.full((len(xyz), 2), np.nan)
    uv[valid] = pts[valid, :2] / z[valid]
    return uv


# --------------------------------------------------------------------------
# E6 rolling spread (location-invariant collapse monitor, lower = more OOD)
# --------------------------------------------------------------------------

def rolling_spread(hidden: np.ndarray, window: int) -> np.ndarray:
    T = len(hidden)
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T + 1):
        out[t - 1] = float(np.var(hidden[t - window:t], axis=0).sum())
    return out


# --------------------------------------------------------------------------
# model run with rolling state (mirrors probe_model.collect, but keeps the
# full plan/lane_lines/road_edges tensors and the RGB display frames)
# --------------------------------------------------------------------------

def _decode_rgb(hevc_path: Path, n: int) -> list[np.ndarray]:
    """Decode first n frames as BGR uint8 for display (separate ffmpeg pass)."""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(hevc_path),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_size = CAM_W * CAM_H * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 * frame_size)
    frames = []
    try:
        for _ in range(n):
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frames.append(np.frombuffer(buf, np.uint8).reshape(CAM_H, CAM_W, 3).copy())
    finally:
        proc.stdout.close()
        proc.wait()
    return frames


def run_model_sweep(real_six, carla_six, rpy_calib):
    """One continuous warm run. Per frame: blend real->carla by alpha_at(k),
    stack with previous, run the mirror. Returns per-frame parsed tensors +
    hidden states + the alpha used.

    State threads through ModelStateMirror across the whole loop (rolling
    features_buffer / prev_desired_curv); we instantiate it ONCE.
    """
    state = ModelStateMirror()
    n = min(len(real_six), len(carla_six))
    rec = {"plan": [], "lane_lines": [], "road_edges": [], "plan_std": [],
           "hidden_state": [], "alpha": []}
    prev = None
    for k in range(n):
        a = alpha_at(k)
        six = blend(real_six[k], carla_six[k], a)
        if prev is not None:
            inp = stack_pair(prev, six)
            p = state.run(inp, inp)
            rec["plan"].append(np.asarray(p["plan"][0], np.float32))         # (1,33,15)->(33,15) after [0]? see below
            rec["lane_lines"].append(np.asarray(p["lane_lines"][0], np.float32))
            rec["road_edges"].append(np.asarray(p["road_edges"][0], np.float32))
            rec["plan_std"].append(np.asarray(p["plan_stds"][0], np.float32).ravel())
            rec["hidden_state"].append(np.asarray(p["hidden_state"][0], np.float32))
            rec["alpha"].append(a)
        prev = six
    return rec


# --------------------------------------------------------------------------
# parity gate
# --------------------------------------------------------------------------

def parity_gate(real_six, rpy_calib) -> float:
    """Run the clean (alpha=0) real path and compare hidden_state to the
    committed teardown subaru cache. Returns max-abs diff."""
    state = ModelStateMirror()
    hs = []
    prev = None
    n = min(len(real_six), 320)
    for k in range(n):
        if prev is not None:
            inp = stack_pair(prev, real_six[k])
            p = state.run(inp, inp)
            hs.append(np.asarray(p["hidden_state"][0], np.float32))
        prev = real_six[k]
    hs = np.array(hs)
    cache = np.load(TEARDOWN_CACHE)["subaru__hidden_state"]
    m = min(len(hs), len(cache))
    return float(np.max(np.abs(hs[:m] - cache[:m])))


# --------------------------------------------------------------------------
# overlay rendering
# --------------------------------------------------------------------------

def _poly(img, uv, color, thick):
    pts = uv[np.all(np.isfinite(uv), axis=1)]
    if len(pts) < 2:
        return
    cv2.polylines(img, [pts.astype(np.int32)], False, color, thick, cv2.LINE_AA)


def draw_overlay(frame, plan, lane_lines, road_edges, cam_from_dev):
    """Draw predicted path (filled corridor + center line), lane lines, edges."""
    c = ModelConstants
    img = frame
    xs = np.array(c.X_IDXS)

    # plan: (33,15) device-frame trajectory; ch0=x fwd, ch1=y left, ch2=z up
    px, py, pz = plan[:, 0], plan[:, 1], plan[:, 2]
    center = project_points(np.stack([px, py, pz], 1), cam_from_dev)

    # corridor edges at +-0.9m lateral (visual lane half-width)
    left = project_points(np.stack([px, py + 0.9, pz], 1), cam_from_dev)
    right = project_points(np.stack([px, py - 0.9, pz], 1), cam_from_dev)
    good_l = np.all(np.isfinite(left), 1)
    good_r = np.all(np.isfinite(right), 1)
    if good_l.sum() > 2 and good_r.sum() > 2:
        poly = np.concatenate([left[good_l], right[good_r][::-1]]).astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], C_PATH)
        cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)
    _poly(img, center, (255, 255, 255), 6)
    _poly(img, center, C_PATH, 4)

    # lane lines: (4,33,2) -> (y,z) at X_IDXS (dimmer, secondary to the path)
    for i in range(lane_lines.shape[0]):
        y, z = lane_lines[i, :, 0], lane_lines[i, :, 1]
        uv = project_points(np.stack([xs, y, z], 1), cam_from_dev)
        _poly(img, uv, C_LANE, 2)
    # road edges
    for i in range(road_edges.shape[0]):
        y, z = road_edges[i, :, 0], road_edges[i, :, 1]
        uv = project_points(np.stack([xs, y, z], 1), cam_from_dev)
        _poly(img, uv, C_EDGE, 2)
    return img


def _bar(img, x, y, w, h, frac, color, label, val_txt):
    frac = float(np.clip(frac, 0, 1))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (110, 110, 110), 1)
    cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, val_txt, (x + w + 12, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)


def draw_hud(img, alpha, conf_frac, conf_label, ood_frac, ood_fires, collapsed, reach_m):
    h, w = img.shape[:2]
    panel = img.copy()
    cv2.rectangle(panel, (0, 0), (w, 150), (18, 18, 22), -1)
    cv2.addWeighted(panel, 0.78, img, 0.22, 0, img)

    cv2.putText(img, "openpilot supercombo  -  live predicted path",
                (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, C_TEXT, 2, cv2.LINE_AA)

    # blindness meter + predicted reach
    bl = int(100 * alpha / ALPHA_MAX)
    cv2.putText(img, f"input degradation: {bl:3d}%", (24, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WARN, 2, cv2.LINE_AA)
    rc = C_BAD if collapsed else C_DIM
    cv2.putText(img, f"predicted reach: {reach_m:3.0f} m", (470, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc, 2, cv2.LINE_AA)

    # confidence (model's own) bar - stays HIGH
    _bar(img, 24, 104, 260, 22, conf_frac, C_GOOD if conf_frac > 0.5 else C_WARN,
         "model confidence", conf_label)
    # internal OOD monitor bar - the thing that actually catches it
    oc = C_BAD if ood_fires else C_GOOD
    _bar(img, 24 + 470, 104, 260, 22, ood_frac, oc,
         "internal OOD monitor", "OOD" if ood_fires else "ok")

    if collapsed:
        txt = "PATH COLLAPSED   -   model still reports high confidence"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.putText(img, txt, ((w - tw) // 2, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_BAD, 3, cv2.LINE_AA)
    return img


def end_card(w, h, finding_lines):
    card = np.full((h, w, 3), 14, np.uint8)
    blocks = [("openpilot supercombo  /  distribution-shift teardown", 0.95, C_DIM, 2),
              ("", 0, None, 0)]
    for fl in finding_lines:
        blocks.append((fl, 1.25, C_WARN, 3))
    blocks += [("", 0, None, 0),
               ("github.com/yusufdxb/supercombo-blindspot", 0.9, C_DIM, 2)]
    # measure total height to vertically center
    total = 0
    for ln, scale, _, thick in blocks:
        if not ln:
            total += 40
        else:
            (_, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            total += th + 38
    y = (h - total) // 2
    for ln, scale, col, thick in blocks:
        if not ln:
            y += 40
            continue
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(card, ln, ((w - tw) // 2, y + th), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, col, thick, cv2.LINE_AA)
        y += th + 38
    return card


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

FINDING = ("The path collapsed. The model's confidence did not. "
           "Output-side monitors miss it; an internal-feature monitor catches it.")
FINDING_LINES = [
    "The path collapsed. The model's confidence did not.",
    "Output-side monitors miss it.",
    "An internal-feature monitor catches it.",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="viral supercombo blindspot demo")
    ap.add_argument("--parity-only", action="store_true",
                    help="run the parity gate and exit (no render)")
    ap.add_argument("--out", default=str(DEMO_DIR / "supercombo_blindspot.mp4"))
    ap.add_argument("--n", type=int, default=N_FRAMES)
    args = ap.parse_args(argv)

    print("Building parity-verified real + CARLA model inputs ...", flush=True)
    rpy = _calib_rpy()
    real_six = load_real_six(SUBARU_HEVC, SUBARU_RLOG, args.n)
    carla_six = load_carla_six(CARLA_NPY, args.n)

    print("PARITY GATE: alpha=0 real path vs teardown subaru cache ...", flush=True)
    diff = parity_gate(real_six, rpy)
    print(f"  max-abs hidden_state diff = {diff:.3e}  (tol {PARITY_TOL:.0e})", flush=True)
    if diff > PARITY_TOL:
        print("PARITY FAILED. Refusing to render a faked overlay.", file=sys.stderr)
        return 2
    print("  PARITY OK.", flush=True)
    if args.parity_only:
        return 0

    print("Decoding RGB display frames ...", flush=True)
    rgb = _decode_rgb(SUBARU_HEVC, args.n)

    print("Running supercombo across the blinding sweep (rolling state) ...", flush=True)
    rec = run_model_sweep(real_six, carla_six, rpy)

    plan = np.array(rec["plan"])            # (T,33,15)
    lanes = np.array(rec["lane_lines"])     # (T,4,33,2)
    edges = np.array(rec["road_edges"])     # (T,2,33,2)
    hidden = np.array(rec["hidden_state"])  # (T,512)
    plan_std = np.array(rec["plan_std"])    # (T, ...)
    alphas = np.array(rec["alpha"])
    T = len(plan)

    # --- detectors ---
    # model's own confidence: invert mean plan uncertainty. We anchor the scale
    # to the clean baseline and a 3x-clean "this would be alarming" ceiling, NOT
    # to the sweep's own max. This is the honest point: the model's predicted
    # uncertainty only creeps to ~1.7x clean at full path collapse (see repo E3),
    # far below any level a safety gate would act on, so the confidence bar
    # stays high while the path is already dead.
    unc = plan_std.mean(1)
    clean_unc = unc[:max(HOLD_CLEAN - 1, 10)]
    u_base = float(np.median(clean_unc))
    u_ceil = 3.0 * u_base                       # uncertainty that WOULD read as low-confidence
    conf = 1.0 - np.clip((unc - u_base) / max(u_ceil - u_base, 1e-9), 0, 1)

    # internal OOD monitor: E6 rolling spread, calibrated on the clean hold.
    spread = rolling_spread(hidden, WINDOW)
    clean_sp = spread[WINDOW:HOLD_CLEAN]
    clean_sp = clean_sp[np.isfinite(clean_sp)]
    sp_ref = float(np.median(clean_sp)) if len(clean_sp) else float(np.nanmedian(spread))
    sp_thr = 0.5 * sp_ref                      # collapse = spread below half clean ref
    # OOD score for display: higher = more OOD (spread falling below ref)
    ood_frac = np.clip((sp_ref - np.nan_to_num(spread, nan=sp_ref)) / max(sp_ref, 1e-9), 0, 1)

    # collapse flag for the headline: forward reach shrinks AND the lateral
    # trajectory flattens to a dead-straight stub (path stops tracking the road).
    reach = plan[:, :, 0].max(1)               # furthest predicted x (m)
    lat_std = plan[:, :, 1].std(1)             # lateral spread of the path
    clean_reach = float(np.median(reach[:max(HOLD_CLEAN - 1, 10)]))
    clean_lat = float(np.median(lat_std[:max(HOLD_CLEAN - 1, 10)]))
    collapse_flag = (reach < 0.55 * clean_reach) & (lat_std < 0.5 * clean_lat)

    cam_from_dev = camera_from_device(rpy)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (CAM_W, CAM_H))
    print(f"Rendering {T} frames ...", flush=True)
    for t in range(T):
        frame = rgb[t + 1].copy() if (t + 1) < len(rgb) else rgb[-1].copy()
        ood_fires = bool(np.isfinite(spread[t]) and spread[t] < sp_thr)
        collapsed = bool(collapse_flag[t])
        draw_overlay(frame, plan[t], lanes[t], edges[t], cam_from_dev)
        draw_hud(frame, alphas[t],
                 conf_frac=float(conf[t]),
                 conf_label=f"{100*conf[t]:.0f}%",
                 ood_frac=float(ood_frac[t]),
                 ood_fires=ood_fires,
                 collapsed=collapsed,
                 reach_m=float(reach[t]))
        vw.write(frame)
    # hold the last collapse frame, then end card
    for _ in range(int(1.2 * FPS)):
        vw.write(frame)
    card = end_card(CAM_W, CAM_H, FINDING_LINES)
    for _ in range(int(3.0 * FPS)):
        vw.write(card)
    vw.release()

    # re-encode to h264 (smaller, web-playable)
    print("Encoding h264 ...", flush=True)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
        "-vf", "scale=1280:-2", "-movflags", "+faststart", str(out_path),
    ], check=True)
    tmp_path.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / 1e6
    dur = (T + int(1.2 * FPS) + int(3.0 * FPS)) / FPS
    n_collapse = int(np.sum(collapse_flag))
    print(f"\nDONE")
    print(f"  clip: subaru_source (real highway dashcam)")
    print(f"  parity max-abs hidden_state diff: {diff:.3e}")
    print(f"  mp4: {out_path}")
    print(f"  size: {size_mb:.2f} MB   duration: {dur:.1f} s   frames: {T}")
    print(f"  collapse frames (reach < 0.45x clean): {n_collapse}/{T}")
    print(f"  clean fwd reach {clean_reach:.0f} m -> min reach {reach.min():.0f} m")
    print(f"  finding: {FINDING}")
    return 0


def _calib_rpy() -> np.ndarray:
    """liveCalibration rpy for the Subaru segment (same source _calib_warps uses)."""
    from src.rlog import iter_events
    rpy = None
    for ev in iter_events(SUBARU_RLOG):
        if ev.which() == "liveCalibration":
            rpy = np.array(list(ev.liveCalibration.rpyCalib), dtype=np.float64)
    if rpy is None:
        raise RuntimeError("no liveCalibration in subaru rlog")
    return rpy


if __name__ == "__main__":
    sys.exit(main())
