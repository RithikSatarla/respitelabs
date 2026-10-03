"""Free 2D labels for a gripper detector, from DROID GT episodes.

On a GT episode DROID kept the configured calibration, so the configured
extrinsic is the best pose we have. Projecting the fingertip from forward
kinematics through it gives a label for every frame, with no hand labelling:

  tip_base = PANDA fk(q at the frame's capture time) applied to the tip point
  tip_cam  = inverse(cam2base) @ tip_base
  (u, v)   = K @ tip_cam, divided by depth

The raw DROID MP4s hold one view per fixed camera (1280x720), not a
side-by-side stereo pair; the pair is only in the ZED SVO files. So labels
are made for that one view. Frames behind the camera or outside the image
are kept but marked not visible.

Run: python -m kintrace.droid fklabels --data data/droid
Writes data/droid/fk_labels/<episode key>_<serial>.npz and labels_check.png.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from .. import arms
from . import episode as ep_mod
from .extrinsics import Corrected

SMALL = (320, 180)  # (w, h) the detector trains at
TIP = arms.PANDA_HAND_TIP


def intrinsics_for(intr: dict, key: str, serial: str, size=None) -> np.ndarray | None:
    """3x3 K for one camera, scaled to size=(w, h) if the video is not the calibrated size."""
    cam = (intr.get(key) or {}).get(str(serial))
    if not cam or "cameraMatrix" not in cam:
        return None
    fx, cx, fy, cy = cam["cameraMatrix"]  # DROID order: fx, cx, fy, cy
    if fx <= 0 or fy <= 0 or not cam.get("width") or not cam.get("height"):
        return None  # ~2% of DROID entries are all zeros
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    if size is not None:
        sx, sy = size[0] / cam.get("width", size[0]), size[1] / cam.get("height", size[1])
        K = np.diag([sx, sy, 1.0]) @ K
    return K


def tip_in_camera(ep, serial: str, T_cam2base: np.ndarray, n_frames: int, tip=TIP) -> tuple:
    """Fingertip in the camera frame for frame i = 0..n_frames-1, at the camera's capture time."""
    tc = ep.t_cam.get(str(serial))
    t = tc[:n_frames] if tc is not None and len(tc) >= n_frames else ep.t[:n_frames]
    q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(ep.q.shape[1])], 1)
    F = arms.PANDA.fk(q)
    p = np.einsum("nij,j->ni", F[:, :3, :3], tip) + F[:, :3, 3]
    Tinv = np.linalg.inv(T_cam2base)
    return p @ Tinv[:3, :3].T + Tinv[:3, 3], t


def project(K: np.ndarray, pts_cam: np.ndarray, size) -> tuple:
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pts_cam @ K.T)[:, :2] / z[:, None]
    vis = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
    return np.nan_to_num(uv), vis


def read_frames(path: str) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return np.asarray(out)


def gt_cameras(data: str):
    """(manifest entry, Episode, serial, configured cam2base, meta) for each fixed camera of each GT episode."""
    with open(os.path.join(data, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    corrected = Corrected.load(data)
    for m in manifest:
        meta = corrected.meta(m["key"])
        if meta.get("source") != "GT":
            continue
        try:
            ep = ep_mod.load(m["dir"], key=m["path"])
        except Exception as e:
            print(f"skip {m['path']}: {e}", file=sys.stderr)
            continue
        for name in ("ext1", "ext2"):
            serial = ep.serials.get(name)
            if serial and serial in ep.configured:
                yield m, ep, serial, ep.configured[serial], meta


def build(data: str, out_dir: str | None = None) -> list:
    import cv2

    out_dir = out_dir or os.path.join(data, "fk_labels")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(data, "intrinsics.json"), encoding="utf-8") as f:
        intr = json.load(f)
    written, skipped = [], []
    for m, ep, serial, T, meta in gt_cameras(data):
        video = ep.video(serial)
        if video is None:
            skipped.append((m["path"], serial, "no MP4"))
            continue
        frames = read_frames(video)
        if not len(frames):
            skipped.append((m["path"], serial, "MP4 decoded to 0 frames"))
            continue
        h, w = frames.shape[1:3]
        K = intrinsics_for(intr, m["key"], serial, (w, h))
        if K is None:
            skipped.append((m["path"], serial, "no intrinsics"))
            continue
        n = min(len(frames), len(ep.t))
        pc, t = tip_in_camera(ep, serial, T, n)
        uv, vis = project(K, pc, (w, h))
        s = SMALL[0] / w
        small = np.stack([cv2.resize(f, SMALL, interpolation=cv2.INTER_AREA) for f in frames[:n]])
        path = os.path.join(out_dir, f"{m['key']}_{serial}.npz")
        np.savez_compressed(
            path, frames=small, uv=(uv * s).astype(np.float32), uv_full=uv.astype(np.float32), visible=vis,
            tip_cam=pc.astype(np.float32), t=t, K=K, full_size=np.array([w, h]), serial=str(serial),
            key=m["key"], path=m["path"], tip=TIP)
        written.append(dict(file=path, key=m["key"], serial=str(serial), frames=int(n), visible=int(vis.sum())))
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(dict(written=written, skipped=skipped, tip_flange_m=TIP.tolist(),
                       note="one view per camera; DROID MP4s are not stereo"), f, indent=2)
    print(f"FK labels (DROID GT episodes): {len(written)} cameras, "
          f"{sum(r['frames'] for r in written)} frames, {sum(r['visible'] for r in written)} with the tip in view")
    for p, s, why in skipped:
        print(f"  skipped {p} {s}: {why}")
    print(f"wrote {out_dir}")
    return written


def contact_sheet(data: str, out: str | None = None, n: int = 12, seed: int = 0) -> str:
    """n frames from n different cameras, the label drawn on each. Green: label point.
    Blue: FK flange. Red: 0.17 m along the tool axis (a Robotiq 2F-85 fingertip guess)."""
    import cv2

    out = out or os.path.join(data, "labels_check.png")
    with open(os.path.join(data, "fk_labels", "index.json"), encoding="utf-8") as f:
        index = json.load(f)["written"]
    with open(os.path.join(data, "manifest.json"), encoding="utf-8") as f:
        dirs = {x["key"]: x for x in json.load(f)}
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(index), size=min(n, len(index)), replace=False)
    tiles = []
    for i in picks:
        z = np.load(index[i]["file"])
        vis = np.flatnonzero(z["visible"])
        if not len(vis):
            continue
        k = int(vis[rng.integers(len(vis))])
        img = cv2.resize(z["frames"][k], (640, 360), interpolation=cv2.INTER_LINEAR)
        sc = 640 / int(z["full_size"][0])
        # reference marks along the tool axis, through the same camera model
        K, kk = z["K"], str(z["key"])
        ep = ep_mod.load(dirs[kk]["dir"], key=dirs[kk]["path"])
        for off, col in ((0.0, (255, 120, 0)), (0.17, (0, 0, 255))):
            pc, _ = tip_in_camera(ep, str(z["serial"]), ep.configured[str(z["serial"])], k + 1, tip=np.array([0, 0, off]))
            uv, ok = project(K, pc[k:k + 1], tuple(z["full_size"]))
            if ok[0]:
                cv2.circle(img, tuple(int(c * sc) for c in uv[0]), 4, col, -1)
        u, v = (z["uv_full"][k] * sc).astype(int)
        cv2.circle(img, (int(u), int(v)), 7, (0, 255, 0), 2)
        cv2.putText(img, f"{kk[:28]} {z['serial']} f{k}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]
    cv2.imwrite(out, np.vstack(rows))
    print(f"wrote {out}")
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m kintrace.droid fklabels")
    ap.add_argument("--data", default="data/droid")
    ap.add_argument("--sheet-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.sheet_only:
        build(a.data)
    contact_sheet(a.data)


if __name__ == "__main__":
    main()
