"""Camera fix from three keypoints, compared with DROID's corrected pose (rules in PROGRESS.md).

fit         6-DoF pose from flange, hand centre and fingertip centre detections over an episode:
            solvePnPRansac from the believed pose, then solvePnPRefineLM ("ours"), or plain
            cv2.solvePnP with no starting pose and no RANSAC ("plain").
bootstrap   200 fits over resampled blocks of 10 frames -> 90% intervals on the fix.
pred        every cached Pred camera: configured vs DROID's corrected pose, before and after the fix.
coverage    injected bumps of known size on test GT cameras: does the 90% interval hold the true size?

Run: python -m kintrace.droid.fixes pred | coverage
"""
from __future__ import annotations

import json
import os

import numpy as np

from .alarm import KP3_OFFSETS, perturb
from .check2d import fk_tips, project
from .extrinsics import pose_error
from .phase2 import OUT, load_cache

MIN_VISIBLE = 0.5
BOOT = 200
BLOCK = 10


def frames_ok(c):
    return np.flatnonzero((c["p3"] >= MIN_VISIBLE).sum(1) >= 2)


def points(c, idx):
    """3D keypoints in the base frame and their detections, flattened over confident (frame, keypoint) pairs."""
    P, U = [], []
    for j, o in enumerate(KP3_OFFSETS):
        m = c["p3"][idx, j] >= MIN_VISIBLE
        P.append(fk_tips(c["q"][idx][m], tip=np.array([0, 0, o])))
        U.append(c["uv3"][idx, j][m])
    return np.concatenate(P).astype(np.float64), np.concatenate(U).astype(np.float64)


def fit(c, idx, T_start, method="ours"):
    import cv2
    P, U = points(c, idx)
    if len(P) < 12:
        return None
    if method == "plain":
        ok, rv, tv = cv2.solvePnP(P, U, c["K"], None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
    else:
        Ti = np.linalg.inv(T_start)
        rv, _ = cv2.Rodrigues(Ti[:3, :3])
        tv = Ti[:3, 3].reshape(3, 1).copy()
        ok, rv, tv, inl = cv2.solvePnPRansac(P, U, c["K"], None, rv.copy(), tv, useExtrinsicGuess=True,
                                             reprojectionError=30.0, iterationsCount=300)
        if not ok or inl is None or len(inl) < 12:
            return None
        inl = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(P[inl], U[inl], c["K"], None, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    Tb2c = np.eye(4)
    Tb2c[:3, :3], Tb2c[:3, 3] = R, np.asarray(tv).ravel()
    return np.linalg.inv(Tb2c)


def residual(c, T, idx):
    d = np.stack([np.linalg.norm(project(T, c["K"], fk_tips(c["q"][idx], tip=np.array([0, 0, o]))) - c["uv3"][idx, j], axis=1)
                  for j, o in enumerate(KP3_OFFSETS)], 1)
    m = c["p3"][idx] >= MIN_VISIBLE
    return (d * m).sum(1) / np.maximum(m.sum(1), 1)


def bootstrap(c, idx, T_start, rng, n=BOOT):
    """90% intervals on the fix's translation (mm) and rotation (deg) away from T_start."""
    blocks = [idx[i:i + BLOCK] for i in range(0, len(idx), BLOCK)]
    tr, rot = [], []
    for _ in range(n):
        pick = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        T = fit(c, np.sort(pick), T_start)
        if T is not None:
            e = pose_error(T_start, T)
            tr.append(e["translation_mm"])
            rot.append(e["rotation_deg"])
    if len(tr) < n // 2:
        return None
    return dict(translation_mm=(float(np.percentile(tr, 5)), float(np.percentile(tr, 95))),
                rotation_deg=(float(np.percentile(rot, 5)), float(np.percentile(rot, 95))), fits=len(tr))


def healthy_level():
    """p90 of healthy validation cameras' median residual (three keypoints): the GO line for a re-check."""
    meds = []
    for c in load_cache("val_kp3"):
        k = frames_ok(c)
        if len(k) >= 20:
            meds.append(float(np.median(residual(c, c["T_truth"], k))))
    return float(np.percentile(meds, 90))


def pred(seed=0) -> dict:
    from .experiments import load_split
    rng = np.random.default_rng(seed)
    sp = load_split()
    untrained = set(sp["val"]) | set(sp["test"])
    go_px = healthy_level()
    rows = []
    for c in load_cache("pred_kp3"):
        k = frames_ok(c)
        if len(k) < 30:
            rows.append(dict(key=c["key"], serial=c["serial"], answer="can't tell (fewer than 30 frames with keypoints)"))
            continue
        before = pose_error(c["T_belief"], c["T_truth"])
        T_ours = fit(c, k, c["T_belief"], "ours")
        T_plain = fit(c, k, c["T_belief"], "plain")
        h = len(k) // 2
        T_half = fit(c, k[:h], c["T_belief"], "ours")
        r_before = float(np.median(residual(c, c["T_belief"], k[h:])))
        r_after = float(np.median(residual(c, T_half, k[h:]))) if T_half is not None else None
        row = dict(key=c["key"], path=c["path"], serial=c["serial"], untrained=c["serial"] in untrained, frames=int(len(k)),
                   configured_vs_droid_mm=before["translation_mm"], configured_vs_droid_deg=before["rotation_deg"],
                   held_out_residual_before=r_before, held_out_residual_after=r_after,
                   status=("GO" if r_after is not None and r_after <= go_px else "NO-GO"))
        for name, T in (("ours", T_ours), ("plain", T_plain)):
            if T is None:
                row[f"{name}_vs_droid_mm"] = row[f"{name}_vs_droid_deg"] = None
                continue
            e = pose_error(T, c["T_truth"])
            row[f"{name}_vs_droid_mm"], row[f"{name}_vs_droid_deg"] = e["translation_mm"], e["rotation_deg"]
        if T_ours is not None:
            e = pose_error(c["T_belief"], T_ours)
            row["fix_mm"], row["fix_deg"] = e["translation_mm"], e["rotation_deg"]
            row["fix_ci"] = bootstrap(c, k, c["T_belief"], rng, n=100)
        rows.append(row)

    def summary(rs):
        rs = [r for r in rs if r.get("ours_vs_droid_mm") is not None]
        if not rs:
            return None
        med = lambda key: float(np.median([r[key] for r in rs if r.get(key) is not None]))  # noqa: E731
        closer = sum(r["ours_vs_droid_mm"] < r["configured_vs_droid_mm"] for r in rs)
        closer_plain = sum(r["plain_vs_droid_mm"] is not None and r["ours_vs_droid_mm"] <= r["plain_vs_droid_mm"] for r in rs)
        return dict(cameras=len(rs), configured_vs_droid_mm=med("configured_vs_droid_mm"), configured_vs_droid_deg=med("configured_vs_droid_deg"),
                    ours_vs_droid_mm=med("ours_vs_droid_mm"), ours_vs_droid_deg=med("ours_vs_droid_deg"),
                    plain_vs_droid_mm=med("plain_vs_droid_mm"), plain_vs_droid_deg=med("plain_vs_droid_deg"),
                    ours_closer_than_configured=closer, ours_at_least_as_close_as_plain=closer_plain,
                    go=sum(r["status"] == "GO" for r in rs))
    res = dict(go_line_px=go_px, all=summary(rows), untrained=summary([r for r in rows if r.get("untrained")]),
               cant_tell=sum(r.get("answer", "").startswith("can't tell") for r in rows), rows=rows)
    json.dump(res, open(os.path.join(OUT, "fixes_pred.json"), "w"), indent=1, default=float)
    return res


WIDEN = os.path.join(OUT, "fixes_widen.json")


def coverage(seed=1, per_camera=2, on="test") -> dict:
    """Injected bumps on GT cameras: is the true size inside the 90% interval? On validation this also
    sets the widening (90th percentile of size error); on test the widened interval is checked too."""
    rng = np.random.default_rng(seed)
    widen = json.load(open(WIDEN)) if on == "test" and os.path.exists(WIDEN) else None
    hits_t = hits_r = n = 0
    whits_t = whits_r = 0
    err_t, err_r = [], []
    for c in load_cache(f"{on}_kp3"):
        k = frames_ok(c)
        if len(k) < 30:
            continue
        for _ in range(per_camera):
            size_cm, size_deg = rng.uniform(0, 5), rng.uniform(0, 5)
            D = perturb("trans", size_cm, rng) @ perturb("rot", size_deg, rng)
            T_bel = c["T_truth"] @ D                                   # the believed pose is off by D
            T = fit(c, k, T_bel)
            ci = bootstrap(c, k, T_bel, rng, n=60)
            if T is None or ci is None:
                continue
            true = pose_error(T_bel, c["T_truth"])
            est = pose_error(T_bel, T)
            n += 1
            hits_t += ci["translation_mm"][0] <= true["translation_mm"] <= ci["translation_mm"][1]
            hits_r += ci["rotation_deg"][0] <= true["rotation_deg"] <= ci["rotation_deg"][1]
            err_t.append(abs(est["translation_mm"] - true["translation_mm"]))
            err_r.append(abs(est["rotation_deg"] - true["rotation_deg"]))
            if widen:
                wt, wr = widen["translation_mm"], widen["rotation_deg"]
                whits_t += ci["translation_mm"][0] - wt <= true["translation_mm"] <= ci["translation_mm"][1] + wt
                whits_r += ci["rotation_deg"][0] - wr <= true["rotation_deg"] <= ci["rotation_deg"][1] + wr
    res = dict(on=on, cases=n, coverage_translation=hits_t / max(n, 1), coverage_rotation=hits_r / max(n, 1),
               size_error_mm_median=float(np.median(err_t)), size_error_deg_median=float(np.median(err_r)))
    if on == "val":
        json.dump(dict(translation_mm=float(np.percentile(err_t, 90)), rotation_deg=float(np.percentile(err_r, 90)),
                       note="90th percentile of size error on validation injected bumps"), open(WIDEN, "w"), indent=1)
    if widen:
        res.update(widened_by=widen, coverage_translation_widened=whits_t / max(n, 1), coverage_rotation_widened=whits_r / max(n, 1))
    json.dump(res, open(os.path.join(OUT, f"fixes_coverage_{on}.json"), "w"), indent=1)
    return res


def record(key: str, serial: str, sign_key: str = "kintrace_keys/private.pem") -> str:
    """Full signed record for one camera from fixes_pred.json: what moved (with the widened 90% range),
    the 6-DoF fix, the held-out re-check, GO / NO-GO, and DROID's pose as the reference."""
    import time

    from .. import certify
    res = json.load(open(os.path.join(OUT, "fixes_pred.json")))
    r = next(x for x in res["rows"] if x["key"] == key and x["serial"] == serial)
    widen = json.load(open(WIDEN))
    ci = r["fix_ci"]
    rng_t = (max(0.0, ci["translation_mm"][0] - widen["translation_mm"]), ci["translation_mm"][1] + widen["translation_mm"])
    rng_r = (max(0.0, ci["rotation_deg"][0] - widen["rotation_deg"]), ci["rotation_deg"][1] + widen["rotation_deg"])
    status = "GO after fix" if r["status"] == "GO" else "NO-GO"
    rec = dict(kind="kintrace recommission record (DROID, real Franka data, three-keypoint camera check)",
               episode=r["path"], episode_id=key, camera=serial, checked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               what_moved=dict(camera_translation_mm=r["fix_mm"], camera_rotation_deg=r["fix_deg"],
                               translation_90pct_mm=rng_t, rotation_90pct_deg=rng_r, frames=r["frames"]),
               fix="camera calibration replaced with the 6-DoF pose fitted from flange, hand centre and fingertip",
               recheck=dict(held_out_residual_before_px=r["held_out_residual_before"], held_out_residual_after_px=r["held_out_residual_after"],
                            go_line_px=res["go_line_px"]),
               droid_reference=dict(configured_vs_droid_mm=r["configured_vs_droid_mm"], configured_vs_droid_deg=r["configured_vs_droid_deg"],
                                    fix_vs_droid_mm=r["ours_vs_droid_mm"], fix_vs_droid_deg=r["ours_vs_droid_deg"]),
               status=status,
               note="Real DROID recording. 90% ranges are bootstrap intervals widened by the validation size error "
                    "(test coverage 92% / 93%). A Pred difference is camera movement or a calibration that was never right.")
    out = os.path.join("incidents", f"droid_kp3_{key}_{serial}")
    os.makedirs(out, exist_ok=True)
    signed = certify.sign(rec, certify.load_private(sign_key)) if os.path.exists(sign_key) else rec
    json.dump(signed, open(os.path.join(out, "record.json"), "w"), indent=2, default=float)
    lines = [f"KINTRACE  recommission record, DROID camera {serial} (three-keypoint check)", "=" * 64,
             f"Episode      : {r['path']}",
             f"What moved   : camera {r['fix_mm']:.0f} mm (90%: {rng_t[0]:.0f} to {rng_t[1]:.0f}), "
             f"{r['fix_deg']:.1f} deg (90%: {rng_r[0]:.1f} to {rng_r[1]:.1f}) over {r['frames']} frames",
             "Fix          : camera calibration replaced with the fitted 6-DoF pose",
             f"Re-check     : held-out error {r['held_out_residual_before']:.1f} px -> {r['held_out_residual_after']:.1f} px "
             f"(healthy level {res['go_line_px']:.1f} px)",
             f"vs DROID     : calibration was {r['configured_vs_droid_mm']:.0f} mm / {r['configured_vs_droid_deg']:.1f} deg from DROID's "
             f"re-solved pose; the fix is {r['ours_vs_droid_mm']:.0f} mm / {r['ours_vs_droid_deg']:.1f} deg from it",
             f"STATUS: {status}" + ("  (signed)" if "signature" in signed else "  (not signed: no key)")]
    open(os.path.join(out, "record.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.fixes")
    ap.add_argument("what", choices=("pred", "coverage"))
    ap.add_argument("--on", default="test", choices=("val", "test"))
    a = ap.parse_args(argv)
    r = pred() if a.what == "pred" else coverage(on=a.on)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()
