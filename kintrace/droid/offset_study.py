"""Where does the per-camera offset between detections and FK labels come from? Validation only, CPU.

1. offset in the gripper's frame: each detection is back-projected to the FK tip's depth, the
   3D difference to the FK tip is rotated into the flange frame, and a constant tip-point
   correction is fitted (per camera, and one shared by all cameras) by least squares on the
   2D reprojection error. A shared correction that removes most of the offset means the
   keypoint is defined in the wrong place.
2. extrinsic refit: per camera, a 6-DoF camera pose is fitted (solvePnP, refined, starting from
   the label pose) on the first half of its confident frames in time; the second half is scored
   under the label pose and under the refit pose.

Run: python -m kintrace.droid.offset_study --preds data/droid/exp/F_E3_unfreeze_val_preds.npz
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares

from .. import arms
from . import episode as ep_mod

DATA = "data/droid"


def _cameras(preds: str):
    z = np.load(preds)
    index = {(r["key"], r["serial"]): r["file"] for r in
             json.load(open(os.path.join(DATA, "fk_labels", "index.json"), encoding="utf-8"))["written"]}
    manifest = {m["key"]: m for m in json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))}
    out = []
    for key, serial in sorted(set(zip(z["key"].tolist(), z["serial"].tolist()))):
        sel = np.flatnonzero((z["key"] == key) & (z["serial"] == serial))
        sel = sel[np.argsort(z["frame"][sel])]
        keep = sel[z["rule"][sel].astype(bool) & (z["p"][sel] >= 0.5)]
        if len(keep) < 30:
            continue
        lab = np.load(index[(key, serial)])
        fr = z["frame"][keep]
        m = manifest[key]
        ep = ep_mod.load(m["dir"], key=m["path"])
        T = ep.configured[serial]
        t = lab["t"][fr]
        q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(7)], 1)
        F = arms.PANDA.fk(q)                                   # flange in base
        out.append(dict(key=key, serial=serial, uv=z["uv_full"][keep], label=z["label_full"][keep], K=lab["K"], T=T, F=F,
                        t=t, ep_t=ep.t, ep_q=ep.q, grip=np.interp(t, ep.t, ep.gripper)))
    return out


def timing_shift(c, shifts_ms=np.arange(-200, 205, 5)):
    """Video-to-joints shift (ms) that minimises the median detector-vs-FK error, and the offsets before/after."""
    errs = []
    for s in shifts_ms:
        q = np.stack([np.interp(c["t"] + s / 1000, c["ep_t"], c["ep_q"][:, j]) for j in range(7)], 1)
        uv = _project(c["T"], c["K"], arms.PANDA.fk(q), arms.ROBOTIQ_2F85_TIP)
        errs.append((np.median(np.linalg.norm(uv - c["uv"], axis=1)), np.linalg.norm(np.median(uv - c["uv"], 0))))
    errs = np.array(errs)
    i0 = int(np.flatnonzero(shifts_ms == 0)[0])
    best = int(np.argmin(errs[:, 0]))
    return dict(shift_ms=float(shifts_ms[best]), median_at_0=float(errs[i0, 0]), median_at_best=float(errs[best, 0]),
                offset_at_0=float(errs[i0, 1]), offset_at_best=float(errs[best, 1]), at_edge=best in (0, len(shifts_ms) - 1))


def pose_effects(cams):
    """Error against gripper opening and against the tool axis angle to the camera, within and across episodes."""
    from scipy.stats import spearmanr
    within_grip, within_ang, rows = [], [], []
    for c in cams:
        e = np.linalg.norm(_project(c["T"], c["K"], c["F"], arms.ROBOTIQ_2F85_TIP) - c["uv"], axis=1)
        Ti = np.linalg.inv(c["T"])
        z_cam = np.einsum("ij,nj->ni", Ti[:3, :3], c["F"][:, :3, 2])      # tool axis in the camera frame
        ang = np.degrees(np.arccos(np.clip(np.abs(z_cam[:, 2]), 0, 1)))   # 0 = pointing along the view ray
        if np.ptp(c["grip"]) > 0.1:
            within_grip.append(spearmanr(c["grip"], e).correlation)
        within_ang.append(spearmanr(ang, e).correlation)
        rows.append((np.linalg.norm(np.median(_project(c["T"], c["K"], c["F"], arms.ROBOTIQ_2F85_TIP) - c["uv"], 0)),
                     float(np.mean(c["grip"] > 0.5)), float(np.median(ang)),
                     float(np.median(e[c["grip"] <= 0.5])) if (c["grip"] <= 0.5).sum() > 10 else np.nan,
                     float(np.median(e[c["grip"] > 0.5])) if (c["grip"] > 0.5).sum() > 10 else np.nan))
    r = np.array(rows)
    return dict(
        within_episode_spearman_grip=dict(median=float(np.nanmedian(within_grip)), episodes=len(within_grip)),
        within_episode_spearman_angle=dict(median=float(np.nanmedian(within_ang)), episodes=len(within_ang)),
        across_episodes_spearman_offset_vs_closed_fraction=float(spearmanr(r[:, 0], r[:, 1]).correlation),
        across_episodes_spearman_offset_vs_angle=float(spearmanr(r[:, 0], r[:, 2]).correlation),
        median_error_open=float(np.nanmedian(r[:, 3])), median_error_closed=float(np.nanmedian(r[:, 4])))


def _project(T_cam2base, K, F, tip):
    p = np.einsum("nij,j->ni", F[:, :3, :3], tip) + F[:, :3, 3]
    Ti = np.linalg.inv(T_cam2base)
    pc = p @ Ti[:3, :3].T + Ti[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3]


def gripper_frame_offsets(cams, tip=arms.ROBOTIQ_2F85_TIP):
    """Per-frame 3D offset (detection minus FK tip, at the tip's depth) in the flange frame."""
    res = []
    for c in cams:
        Ti = np.linalg.inv(c["T"])
        p = np.einsum("nij,j->ni", c["F"][:, :3, :3], tip) + c["F"][:, :3, 3]
        pc = p @ Ti[:3, :3].T + Ti[:3, 3]
        ray = np.c_[c["uv"], np.ones(len(pc))] @ np.linalg.inv(c["K"]).T
        det_c = ray * (pc[:, 2:3] / ray[:, 2:3])
        d_base = (det_c - pc) @ c["T"][:3, :3].T                  # camera -> base (rotation only)
        d_grip = np.einsum("nji,nj->ni", c["F"][:, :3, :3], d_base)  # base -> flange
        res.append(d_grip)
    return res


def fit_tip(cams, tip0=arms.ROBOTIQ_2F85_TIP):
    def r(x, cs):
        return np.concatenate([(_project(c["T"], c["K"], c["F"], tip0 + x) - c["uv"]).ravel() for c in cs])
    x = least_squares(r, np.zeros(3), args=(cams,), loss="soft_l1", f_scale=10.0).x
    return x


def refit_extrinsic(c, half: str):
    import cv2
    n = len(c["uv"])
    a, b = (slice(0, n // 2), slice(n // 2, n))
    P = np.einsum("nij,j->ni", c["F"][:, :3, :3], arms.ROBOTIQ_2F85_TIP) + c["F"][:, :3, 3]
    Ti = np.linalg.inv(c["T"])
    rv, _ = cv2.Rodrigues(Ti[:3, :3])
    tv = Ti[:3, 3].reshape(3, 1).copy()
    ok, rv, tv, inl = cv2.solvePnPRansac(P[a], c["uv"][a].astype(np.float64), c["K"], None, rv.copy(), tv,
                                         useExtrinsicGuess=True, reprojectionError=30.0, iterationsCount=300)
    if not ok or inl is None or len(inl) < 8:
        return None
    inl = inl.ravel()
    rv, tv = cv2.solvePnPRefineLM(P[a][inl], c["uv"][a][inl].astype(np.float64), c["K"], None, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    Tb2c = np.eye(4)
    Tb2c[:3, :3], Tb2c[:3, 3] = R, tv.ravel()
    Tfit = np.linalg.inv(Tb2c)
    e_before = np.linalg.norm(_project(c["T"], c["K"], c["F"][b], arms.ROBOTIQ_2F85_TIP) - c["uv"][b], axis=1)
    e_after = np.linalg.norm(_project(Tfit, c["K"], c["F"][b], arms.ROBOTIQ_2F85_TIP) - c["uv"][b], axis=1)
    off_before = np.linalg.norm(np.median(_project(c["T"], c["K"], c["F"][b], arms.ROBOTIQ_2F85_TIP) - c["uv"][b], 0))
    off_after = np.linalg.norm(np.median(_project(Tfit, c["K"], c["F"][b], arms.ROBOTIQ_2F85_TIP) - c["uv"][b], 0))
    from .extrinsics import pose_error
    pe = pose_error(c["T"], Tfit)
    return dict(median_before=float(np.median(e_before)), median_after=float(np.median(e_after)),
                offset_before=float(off_before), offset_after=float(off_after),
                refit_mm=pe["translation_mm"], refit_deg=pe["rotation_deg"])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.offset_study")
    ap.add_argument("--preds", required=True)
    a = ap.parse_args(argv)
    cams = _cameras(a.preds)
    print(f"validation cameras with >= 30 confident, visible frames: {len(cams)}")

    # 1. offset in the gripper frame
    g = gripper_frame_offsets(cams)
    per_cam = np.array([np.median(x, 0) * 1000 for x in g])         # mm, flange frame
    print("per-camera median offset in the flange frame (mm, x y z; z = tool axis):")
    for c, v in zip(cams, per_cam):
        print(f"  {c['serial']} {c['key'][:34]:34s} {v[0]:7.1f} {v[1]:7.1f} {v[2]:7.1f}")
    unit = per_cam / np.linalg.norm(per_cam, axis=1, keepdims=True)
    mean_dir = unit.mean(0)
    print(f"  median over cameras: {np.median(per_cam, 0).round(1)} mm; consistency of direction "
          f"(length of mean unit vector, 1 = all the same way, ~0 = random): {np.linalg.norm(mean_dir):.2f}")
    shared = fit_tip(cams)
    base = np.concatenate([np.linalg.norm(_project(c["T"], c["K"], c["F"], arms.ROBOTIQ_2F85_TIP) - c["uv"], axis=1) for c in cams])
    after = np.concatenate([np.linalg.norm(_project(c["T"], c["K"], c["F"], arms.ROBOTIQ_2F85_TIP + shared) - c["uv"], axis=1) for c in cams])
    print(f"  one shared tip correction fitted on all cameras: {np.round(shared * 1000, 1)} mm; "
          f"median error {np.median(base):.1f} -> {np.median(after):.1f} px (fitted and scored on the same frames)")
    # shared correction, honest version: fit on half the cameras, score the other half
    idx = np.arange(len(cams))
    A, B = idx[::2], idx[1::2]
    sA = fit_tip([cams[i] for i in A])
    eB0 = np.concatenate([np.linalg.norm(_project(cams[i]["T"], cams[i]["K"], cams[i]["F"], arms.ROBOTIQ_2F85_TIP) - cams[i]["uv"], axis=1) for i in B])
    eB1 = np.concatenate([np.linalg.norm(_project(cams[i]["T"], cams[i]["K"], cams[i]["F"], arms.ROBOTIQ_2F85_TIP + sA) - cams[i]["uv"], axis=1) for i in B])
    print(f"  shared correction fitted on half the cameras ({np.round(sA * 1000, 1)} mm), scored on the other half: "
          f"median {np.median(eB0):.1f} -> {np.median(eB1):.1f} px")

    # 2. extrinsic refit per camera, first half -> second half
    rows = [r for r in (refit_extrinsic(c, "first") for c in cams) if r]
    mb = np.array([r["median_before"] for r in rows]); ma = np.array([r["median_after"] for r in rows])
    ob = np.array([r["offset_before"] for r in rows]); oa = np.array([r["offset_after"] for r in rows])
    print(f"extrinsic refit on the first half, scored on the second half ({len(rows)} cameras):")
    print(f"  median error per camera: before {np.median(mb):.1f} px, after {np.median(ma):.1f} px")
    print(f"  constant offset per camera: before {np.median(ob):.1f} px, after {np.median(oa):.1f} px "
          f"({100 * (1 - np.median(oa) / np.median(ob)):.0f}% of the median offset removed)")
    print(f"  size of the refit: median {np.median([r['refit_mm'] for r in rows]):.1f} mm, "
          f"{np.median([r['refit_deg'] for r in rows]):.2f} deg")
    out = dict(cameras=len(cams), flange_offsets_mm=per_cam.tolist(), direction_consistency=float(np.linalg.norm(mean_dir)),
               shared_tip_correction_mm=(shared * 1000).tolist(), shared_split_mm=(sA * 1000).tolist(),
               shared_split_median_before=float(np.median(eB0)), shared_split_median_after=float(np.median(eB1)),
               refit=rows)
    json.dump(out, open(os.path.join(DATA, "exp", "offset_study.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
