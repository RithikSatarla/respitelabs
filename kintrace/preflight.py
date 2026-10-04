"""kintrace preflight: run every check that applies to one recording, then GO or NO-GO.

    python -m kintrace preflight <recording> [--episode N] [--camera SERIAL] [--config cfg.json]
                                 [--arm NAME | --urdf robot.urdf] [--reference K]

A recording is a DROID episode folder (trajectory.h5) or a LeRobot dataset folder.
Each check prints one line: ok, problem, or skipped and why.

  timing    command-to-motion delay (diagnose.estimate_delay) against a known-good value:
            config "known_good": {"delay_ms": ...}, or the median of --reference K other
            episodes of the same LeRobot dataset. Problem if more than 4 ms apart.
  encoders  each joint's median of measured minus commanded against known-good offsets
            (config "known_good": {"offsets_deg": [...]}, or the --reference episodes).
            Problem if any joint moved more than 1 deg.
  camera    DROID only, needs a fixed camera, its intrinsics and the gripper detector:
            median residual between the detected gripper and the FK tip projected through
            the configured extrinsic (or config "camera_extrinsic"), over confident frames.
            Problem above the alarm level stored with the detector (detector_final.json).

Exit code: 0 GO, 1 NO-GO, 2 not enough data (no check could run).
Use it in front of a data-collection or shift-start script to block a run when the body is off.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

TIMING_TOL_MS = 4.0
OFFSET_TOL_DEG = 1.0


class Result:
    def __init__(self, name, status, detail):
        self.name, self.status, self.detail = name, status, detail

    def line(self):
        return f"  [{self.status:7s}] {self.name:9s} {self.detail}"


def _delay_ms(t, cmd, meas, search_s=1.5):
    from types import SimpleNamespace

    from .diagnose import estimate_delay
    return estimate_delay(SimpleNamespace(t_joint=t, q_cmd=cmd, q_meas=meas), search_s) * 1000


def _offsets_deg(cmd, meas, to_deg):
    moving = np.linalg.norm(np.gradient(cmd, axis=0), axis=1) > 1e-6
    sel = moving if moving.sum() > 20 else np.ones(len(cmd), bool)
    return np.median((meas - cmd)[sel], 0) * to_deg


def timing_and_encoders(rec: dict, ref: dict | None) -> list:
    out = []
    d = _delay_ms(rec["t"], rec["cmd"], rec["meas"])
    if ref and ref.get("delay_ms") is not None:
        gap = d - ref["delay_ms"]
        st = "problem" if abs(gap) > TIMING_TOL_MS else "ok"
        out.append(Result("timing", st, f"{d:.1f} ms vs {ref['delay_ms']:.1f} ms known good ({gap:+.1f} ms)"))
    else:
        out.append(Result("timing", "skipped", f"{d:.1f} ms measured; no known-good delay to compare"))
    if rec.get("to_deg") is None:
        out.append(Result("encoders", "skipped", "joint units unknown"))
    elif ref and ref.get("offsets_deg") is not None:
        off = _offsets_deg(rec["cmd"], rec["meas"], rec["to_deg"])
        dlt = off - np.asarray(ref["offsets_deg"])
        j = int(np.argmax(np.abs(dlt)))
        st = "problem" if abs(dlt[j]) > OFFSET_TOL_DEG else "ok"
        name = rec["joints"][j] if rec.get("joints") else f"joint {j + 1}"
        out.append(Result("encoders", st, f"largest change {dlt[j]:+.2f} deg ({name}); limit {OFFSET_TOL_DEG:g} deg"))
    else:
        out.append(Result("encoders", "skipped", "no known-good joint offsets to compare"))
    return out


# --------------------------------------------------------------------------
# LeRobot
# --------------------------------------------------------------------------
def lerobot_recording(root: str, episode: int, preset: tuple | None, cols: tuple | None) -> list:
    """List of per-episode dicts (t, cmd, meas, to_deg, joints) for all episodes found; episode first."""
    from .crossrobot import ROBOTS, _cols, episodes
    name = os.path.basename(os.path.normpath(root))
    if cols is None:
        if name not in ROBOTS:
            raise SystemExit(f"no column preset for {name}; pass --state and --action column names")
        state, action, idx, units = ROBOTS[name][2][0][1:]
    else:
        state, action, idx, units = cols
    eps = episodes(name) if os.path.dirname(os.path.normpath(root)).endswith("lerobot") else episodes(name)
    out = []
    for g in eps:
        t = g["timestamp"].to_numpy(float)
        out.append(dict(episode=int(g["episode_index"].iloc[0]), t=t - t[0], cmd=_cols(g, action)[:, idx],
                        meas=_cols(g, state)[:, idx], to_deg={"deg": 1.0, "rad": float(np.degrees(1.0))}.get(units)))
    out.sort(key=lambda r: r["episode"] != episode)
    return out


# --------------------------------------------------------------------------
# DROID
# --------------------------------------------------------------------------
def droid_camera(ep_dir: str, serial: str | None, cfg: dict, data: str = "data/droid") -> tuple:
    from .droid import episode as ep_mod
    from .droid import labels_fk
    from .droid.convert import DEFAULT_MODEL

    ep = ep_mod.load(ep_dir)
    serial = serial or ep.serials.get("ext1")
    rec = dict(t=ep.t, cmd=ep.q_cmd, meas=ep.q, to_deg=float(np.degrees(1.0)), joints=[f"J{i + 1}" for i in range(7)])
    model = cfg.get("detector", DEFAULT_MODEL)
    video = ep.video(serial)
    if video is None:
        return rec, Result("camera", "skipped", f"no video for camera {serial}")
    if not os.path.exists(model):
        return rec, Result("camera", "skipped", f"no detector at {model}")
    intr = json.load(open(os.path.join(data, "intrinsics.json"), encoding="utf-8"))
    key = os.path.basename(next(iter(sorted(__import__("glob").glob(os.path.join(ep_dir, "metadata_*.json")))), ""))[9:-5]
    small, (w, h) = labels_fk.read_small(video)
    K = labels_fk.intrinsics_for(intr, key, serial, (w, h))
    if K is None:
        return rec, Result("camera", "skipped", f"no intrinsics for camera {serial}")
    T_belief = np.asarray(cfg["camera_extrinsic"]) if "camera_extrinsic" in cfg else ep.configured[serial]
    det = detect_frames(small, model, w)
    n = min(len(small), len(ep.t))
    tc = ep.t_cam.get(serial)
    t = tc[:n] if tc is not None and len(tc) >= n else ep.t[:n]
    q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(7)], 1)
    keep = det["keep"][:n]
    if keep.sum() < 12:
        return rec, Result("camera", "skipped", f"only {int(keep.sum())} confident gripper detections")
    from .droid.check2d import fk_tips, project
    r = float(np.median(np.linalg.norm(project(T_belief, K, fk_tips(q[keep])) - det["uv"][:n][keep], axis=1)))
    alarm = det["alarm_px"]
    st = "problem" if r > alarm else "ok"
    return rec, Result("camera", st, f"camera {serial}: median residual {r:.1f} px under the calibration over "
                                     f"{int(keep.sum())} frames; alarm {alarm:.1f} px")


def droid_camera_vs_baseline(ep_dir: str, base_dir: str, serial: str, cfg: dict, data: str = "data/droid") -> tuple:
    """Per-camera alarm (droid/alarm.py): this session's median residual vs a healthy earlier session of the
    same camera. Reliability gate first; thresholds from data/droid/phase2/alarm_calibration.json."""
    import glob

    from .droid import alarm
    from .droid import episode as ep_mod
    from .droid import labels_fk
    from .droid.check2d import fk_tips, project
    from .droid.convert import DEFAULT_MODEL

    cal = json.load(open(alarm.CAL))
    model = cfg.get("detector", DEFAULT_MODEL)
    intr = json.load(open(os.path.join(data, "intrinsics.json"), encoding="utf-8"))

    def session(d, T_override=None):
        ep = ep_mod.load(d)
        key = os.path.basename(sorted(glob.glob(os.path.join(d, "metadata_*.json")))[0])[9:-5]
        small, (w, h) = labels_fk.read_small(ep.video(serial))
        K = labels_fk.intrinsics_for(intr, key, serial, (w, h))
        det = detect_frames(small, model, w)
        n = min(len(small), len(ep.t))
        tc = ep.t_cam.get(serial)
        t = tc[:n] if tc is not None and len(tc) >= n else ep.t[:n]
        q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(7)], 1)
        keep = det["vis"][:n] >= alarm.MIN_VISIBLE
        T = T_override if T_override is not None else ep.configured[serial]
        r = np.linalg.norm(project(T, K, fk_tips(q[keep])) - det["uv"][:n][keep], axis=1)
        return ep, r, keep.mean()

    rec_ep = ep_mod.load(ep_dir)
    rec = dict(t=rec_ep.t, cmd=rec_ep.q_cmd, meas=rec_ep.q, to_deg=float(np.degrees(1.0)), joints=[f"J{i + 1}" for i in range(7)])
    _, base, frac = session(base_dir)
    if len(base) < 10 or not alarm.reliable(base, frac, cal["gate"]):
        return rec, Result("camera", "can't tell", f"camera {serial}: detector unreliable on this view in the baseline "
                                                   f"(median {np.median(base):.1f} px, {100 * frac:.0f}% frames confident; "
                                                   f"gate {cal['gate']['X']} px, {100 * cal['gate']['Y']:.0f}%)")
    T_belief = np.asarray(cfg["camera_extrinsic"]) if "camera_extrinsic" in cfg else None
    _, now, _ = session(ep_dir, T_belief)
    z = alarm.z_score(base, now)
    k = cal["k_between"]["all"]["k"]
    st = "problem" if z > k else "ok"
    return rec, Result("camera", st, f"camera {serial}: median residual {np.median(now):.1f} px vs {np.median(base):.1f} px "
                                     f"in the baseline session (change {np.median(now) - np.median(base):+.1f} px, z {z:.1f}, "
                                     f"alarm at z > {k:.1f})")


def detect_frames(small, model_path, full_w):
    """Detector on 320x180 frames -> full-res uv and a confident-frame mask (thresholds from the model's JSON)."""
    import torch

    from .droid import detector as D
    ck_json = os.path.splitext(model_path)[0] + ".json"
    thr = json.load(open(ck_json)) if os.path.exists(ck_json) else {}
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    if "cfg" in ck and ck["cfg"].get("keypoints", 1) > 1:
        # three-keypoint detector: the alarm is calibrated on the fingertip, so use that keypoint
        from .droid.experiments import load_model
        from .droid.multikp import KP_NAMES, predict as predict_multi
        m, _ = load_model(model_path)
        uv3, p3, peak3 = predict_multi(m, small, None, "preflight")
        j = KP_NAMES.index("fingertip_centre")
        uv, vis, peak = uv3[:, j], p3[:, j], peak3[:, j]
    elif "cfg" in ck:
        from .droid.experiments import load_model, predict
        m, _ = load_model(model_path)
        p = predict(m, small, None, "preflight")
        uv, vis, peak = p["uv"], p["p"], p["peak"]
    else:
        uv, vis, peak, _ = D.predict_conf(D.load(model_path), small)
    keep = (vis >= thr.get("min_visible", 0.5)) & (peak >= thr.get("min_peak", 0.0))
    return dict(uv=uv * (full_w / D.SMALL_W), keep=keep, vis=vis, alarm_px=thr.get("camera_alarm_px", float("inf")))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="kintrace preflight", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--episode", type=int, default=0, help="LeRobot: episode to check")
    ap.add_argument("--camera", help="DROID: fixed camera serial (default ext1)")
    ap.add_argument("--config", help="json with known_good delay_ms / offsets_deg, camera_extrinsic, detector")
    ap.add_argument("--arm", help="named arm (body only; checks here use joints)")
    ap.add_argument("--urdf", help="robot body from a URDF")
    ap.add_argument("--reference", type=int, default=0, help="LeRobot: use K other episodes as the known-good reference")
    ap.add_argument("--baseline", help="DROID: a healthy earlier episode of the same camera (per-camera alarm)")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config)) if a.config else {}
    results = []
    if a.urdf:
        from . import urdf
        body = urdf.load(a.urdf)
        print(f"body: {body.name}, {body.n} joints from {a.urdf}")
    if os.path.exists(os.path.join(a.recording, "trajectory.h5")):
        if a.baseline:
            rec, cam = droid_camera_vs_baseline(a.recording, a.baseline, a.camera, cfg)
        else:
            rec, cam = droid_camera(a.recording, a.camera, cfg)
        ref = cfg.get("known_good")
        results += timing_and_encoders(rec, ref)
        results.append(cam)
    elif os.path.exists(os.path.join(a.recording, "meta", "info.json")):
        recs = lerobot_recording(a.recording, a.episode, None, None)
        rec = recs[0]
        ref = cfg.get("known_good")
        if ref is None and a.reference:
            others = recs[1:1 + a.reference]
            if others:
                ref = dict(delay_ms=float(np.median([_delay_ms(r["t"], r["cmd"], r["meas"]) for r in others])),
                           offsets_deg=np.median([_offsets_deg(r["cmd"], r["meas"], r["to_deg"]) for r in others], 0).tolist()
                           if rec["to_deg"] else None)
                print(f"reference: {len(others)} other episodes of this dataset")
        if cfg.get("inject_delay_ms"):
            d = cfg["inject_delay_ms"] / 1000
            rec["meas"] = np.stack([np.interp(rec["t"] - d, rec["t"], rec["meas"][:, j]) for j in range(rec["meas"].shape[1])], 1)
            print(f"injected: measured stream delayed by {cfg['inject_delay_ms']} ms")
        results += timing_and_encoders(rec, ref)
        results.append(Result("camera", "skipped", "no calibrated fixed camera in this recording"))
    else:
        raise SystemExit(f"{a.recording}: not a DROID episode folder or a LeRobot dataset")
    print(f"preflight: {a.recording}" + (f" episode {a.episode}" if os.path.exists(os.path.join(a.recording, 'meta')) else ""))
    for r in results:
        print(r.line())
    ran = [r for r in results if r.status not in ("skipped", "can't tell")]
    if any(r.status == "can't tell" for r in results) and not any(r.status == "problem" for r in ran):
        print("CAN'T TELL")
        return 2
    if not ran:
        print("NOT ENOUGH DATA")
        return 2
    go = all(r.status == "ok" for r in ran)
    print("GO" if go else "NO-GO")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
