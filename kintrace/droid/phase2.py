"""Phase 2: the camera check on real DROID frames, with the final detector.

cache       run the detector once (gentle GPU) on the validation, test and Pred cameras and save
            per frame: uv, visibility, heatmap peak, joint angles at capture time; per camera: K,
            the configured extrinsic (belief) and DROID's corrected one.
analyse     everything else, on the CPU:
  1. confidence threshold picked on validation (lowest threshold whose kept median is lowest,
     same rule as before), coverage and error reported on test
  2. injected drift on test GT cameras: belief = truth @ D for known D; per camera score = median
     residual over confident frames; alarm level = the validation healthy score at 1 in 100
  3. real drift: median residual under the configured pose vs the healthy validation level, on every test GT and Pred camera;
     recall on Pred cameras DROID calls moved, false alarms on test GT cameras, size error
     (fitted pose vs DROID's corrected pose); Pred serials seen in training are flagged
  4. fix and re-check: for each camera called moved, the fitted pose is the proposed fix; the
     check is re-run with it as the belief; one signed record is written to incidents/

Run: python -m kintrace.droid.phase2 cache --model data/droid/detector_final.pt
     python -m kintrace.droid.phase2 analyse --model data/droid/detector_final.pt
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from . import detector as D
from . import experiments as X
from .check2d import fk_tips, project
from .extrinsics import Corrected, pose_error

DATA = "data/droid"
OUT = os.path.join(DATA, "phase2")
FA_RATE = 0.01
TRANS_CM = (0.5, 1.0, 2.0, 5.0)
ROT_DEG = (0.5, 1.0, 2.0, 5.0)
DIRECTIONS = 20
FRAME_COUNTS = (5, 10, 20, 50, 100)


def _cameras(split: dict) -> dict:
    """{group: [(manifest entry, serial)]} for val, test (GT, own camera) and pred (Pred, own camera)."""
    corrected = Corrected.load(DATA)
    manifest = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))
    groups = dict(val=[], test=[], pred=[])
    for m in manifest:
        meta = corrected.meta(m["key"])
        cams = corrected.lookup(m["key"]) or {}
        for serial in cams:
            if serial in X.load_split().get("excluded", []) or serial == "23960472":
                continue
            if meta.get("source") == "Pred":
                groups["pred"].append((m, serial))
            elif serial in split["test"]:
                groups["test"].append((m, serial))
            elif serial in split["val"]:
                groups["val"].append((m, serial))
    return groups


def cache(model_path: str, logfile: str):
    from . import episode as ep_mod
    from . import labels_fk

    os.makedirs(OUT, exist_ok=True)
    split = X.load_split()
    corrected = Corrected.load(DATA)
    intr = json.load(open(os.path.join(DATA, "intrinsics.json"), encoding="utf-8"))
    model = D.load(model_path)
    gentle = X.Gentle(logfile)
    for group, items in _cameras(split).items():
        path = os.path.join(OUT, f"cache_{group}.npz")
        rows, skipped = [], []
        for m, serial in items:
            try:
                ep = ep_mod.load(m["dir"], key=m["path"])
            except Exception as e:
                skipped.append((m["path"], serial, str(e)))
                continue
            video = ep.video(serial)
            if video is None or serial not in ep.configured:
                skipped.append((m["path"], serial, "no video or no configured pose"))
                continue
            small, (w, h) = labels_fk.read_small(video)
            K = labels_fk.intrinsics_for(intr, m["key"], serial, (w, h))
            if K is None or not len(small):
                skipped.append((m["path"], serial, "no intrinsics or no frames"))
                continue
            n = min(len(small), len(ep.t))
            uv, p, peak, _ = D.predict_conf(model, small[:n], gentle)  # gentle.tick() rests after every batch
            tc = ep.t_cam.get(serial)
            t = tc[:n] if tc is not None and len(tc) >= n else ep.t[:n]
            q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(7)], 1)
            rows.append(dict(key=m["key"], path=m["path"], serial=serial, uv=uv * (w / D.SMALL_W), p=p, peak=peak, q=q,
                             K=K, T_belief=ep.configured[serial], T_truth=(corrected.lookup(m["key"]) or {})[serial],
                             size=(w, h), lab=ep.meta.get("lab", "")))
        np.save(path.replace(".npz", ".npy"), np.array(rows, dtype=object), allow_pickle=True)
        X.log(f"cache {group}: {len(rows)} cameras, {len(skipped)} skipped", logfile)
        json.dump(skipped, open(os.path.join(OUT, f"skipped_{group}.json"), "w"), indent=1)
    X.log(f"heat readings: {gentle.readings}", logfile)


def load_cache(group: str) -> list:
    return list(np.load(os.path.join(OUT, f"cache_{group}.npy"), allow_pickle=True))


def _residual(c, T, keep, n=None):
    P = fk_tips(c["q"][keep])
    r = np.linalg.norm(project(T, c["K"], P) - c["uv"][keep], axis=1)
    return r[:n] if n else r


def pick_threshold(val: list) -> dict:
    """Peak threshold on validation: same rule as before (lowest kept median)."""
    err, peak, vis = [], [], []
    for c in val:
        fk = project(c["T_belief"], c["K"], fk_tips(c["q"]))
        w, h = c["size"]
        inside = (fk[:, 0] >= 0) & (fk[:, 0] < w) & (fk[:, 1] >= 0) & (fk[:, 1] < h)
        err.append(np.linalg.norm(c["uv"] - fk, axis=1)[inside])
        peak.append(c["peak"][inside])
        vis.append(c["p"][inside] >= 0.5)
    e, pk, v = np.concatenate(err), np.concatenate(peak), np.concatenate(vis)
    best = None
    for t in np.unique(np.quantile(pk, np.linspace(0, 0.98, 99))):
        k = v & (pk >= t)
        if k.sum() < 50:
            continue
        r = dict(min_peak=float(t), coverage=float(k.mean()), median=float(np.median(e[k])), p90=float(np.percentile(e[k], 90)))
        if best is None or r["median"] < best["median"]:
            best = r
    return dict(best, min_visible=0.5)


def _keep(c, thr):
    return (c["p"] >= thr["min_visible"]) & (c["peak"] >= thr["min_peak"])


def _perturb(kind, size, rng):
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    Dm = np.eye(4)
    if kind == "trans":
        Dm[:3, 3] = v * size / 100
    else:
        Dm[:3, :3] = Rotation.from_rotvec(v * np.deg2rad(size)).as_matrix()
    return Dm


def analyse(model_path: str, logfile: str, visibility_only: bool = False) -> dict:
    """visibility_only: no heatmap-peak filter (the picked threshold keeps ~10% of validation frames,
    which leaves many cameras with too few frames); written to phase2_visibility_only.json."""
    split = X.load_split()
    val, test, pred = load_cache("val"), load_cache("test"), load_cache("pred")
    thr = pick_threshold(val)
    if visibility_only:
        thr = dict(thr, min_peak=0.0, variant="visibility head only, no peak filter")
    cfg_path = os.path.splitext(model_path)[0] + ".json"
    if visibility_only:
        cfg_path = os.devnull  # detect() and preflight read min_peak / min_visible here
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) and not visibility_only else {}
    cfg.update(min_peak=thr["min_peak"], min_visible=thr["min_visible"],
               confidence_on_validation=dict(coverage=thr["coverage"], kept_median=thr["median"], kept_p90=thr["p90"]))
    cfg.pop("note", None)
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    res = dict(threshold=thr)

    # 1. confidence on test
    e_all, e_kept, n_in = [], [], 0
    for c in test:
        fk = project(c["T_truth"], c["K"], fk_tips(c["q"]))
        w, h = c["size"]
        inside = (fk[:, 0] >= 0) & (fk[:, 0] < w) & (fk[:, 1] >= 0) & (fk[:, 1] < h)
        e = np.linalg.norm(c["uv"] - fk, axis=1)
        e_all.append(e[inside])
        e_kept.append(e[inside & _keep(c, thr)])
        n_in += int(inside.sum())
    ea, ek = np.concatenate(e_all), np.concatenate(e_kept)
    res["test_confidence"] = dict(frames_in_view=n_in, kept=int(len(ek)), coverage=len(ek) / max(n_in, 1),
                                  kept_median=float(np.median(ek)), kept_p90=float(np.percentile(ek, 90)),
                                  all_median=float(np.median(ea)))

    # 2. injected drift on test, alarm level from validation
    rng = np.random.default_rng(0)

    def cam_score(c, Dm=np.eye(4), n=None):
        k = _keep(c, thr)
        if k.sum() < 5:
            return None
        return float(np.median(_residual(c, c["T_truth"] @ Dm, k, n)))

    drift = {}
    for n in list(FRAME_COUNTS) + [None]:
        hv = np.array([s for s in (cam_score(c, n=n) for c in val) if s is not None])
        ht = np.array([s for s in (cam_score(c, n=n) for c in test) if s is not None])
        alarm = float(np.quantile(hv, 1 - FA_RATE))
        per = {}
        for kind, sizes in (("trans", TRANS_CM), ("rot", ROT_DEG)):
            for s in sizes:
                sc = np.array([x for c in test for _ in range(DIRECTIONS) if (x := cam_score(c, _perturb(kind, s, rng), n)) is not None])
                per[f"{kind}_{s}"] = dict(median_px=float(np.median(sc)), caught=float((sc > alarm).mean()))
        drift["all" if n is None else str(n)] = dict(alarm_px=alarm, val_cameras=len(hv), test_cameras=len(ht),
                                                      healthy_test_median_px=float(np.median(ht)),
                                                      healthy_test_false_alarms=float((ht > alarm).mean()),
                                                      healthy_test_quantiles={str(q): float(np.percentile(ht, q)) for q in (50, 75, 90, 95, 99)},
                                                      by_drift=per)
    res["injected_drift"] = drift

    # 2b. same drift test against each camera's own healthy history: first half of its confident frames
    #     is the baseline under the true pose; the score is how much the second half's median residual
    #     rises. A fixed per-camera offset (calibration or tip point) cancels out.
    def change(c, Dm=np.eye(4)):
        k = np.flatnonzero(_keep(c, thr))
        if len(k) < 10:
            return None
        h = len(k) // 2
        a, b = np.zeros(len(c["p"]), bool), np.zeros(len(c["p"]), bool)
        a[k[:h]], b[k[h:]] = True, True
        return float(np.median(_residual(c, c["T_truth"] @ Dm, b)) - np.median(_residual(c, c["T_truth"], a)))

    hv = np.array([s for s in (change(c) for c in val) if s is not None])
    ht = np.array([s for s in (change(c) for c in test) if s is not None])
    alarm_c = float(np.quantile(hv, 1 - FA_RATE))
    per_c = {}
    for kind, sizes in (("trans", TRANS_CM), ("rot", ROT_DEG)):
        for s in sizes:
            sc = np.array([x for c in test for _ in range(DIRECTIONS) if (x := change(c, _perturb(kind, s, rng))) is not None])
            per_c[f"{kind}_{s}"] = dict(median_rise_px=float(np.median(sc)), caught=float((sc > alarm_c).mean()))
    res["injected_drift_per_camera"] = dict(alarm_px=alarm_c, val_cameras=len(hv), test_cameras=len(ht),
                                            healthy_test_false_alarms=float((ht > alarm_c).mean()),
                                            healthy_test_rise_median=float(np.median(ht)), by_drift=per_c)

    # 2c. frames to decide, per camera: baseline = first half of the confident frames (all of it),
    #     decision window = the first n confident frames after it. Alarm level from validation at each n.
    def change_n(c, n, Dm=np.eye(4)):
        k = np.flatnonzero(_keep(c, thr))
        h = len(k) // 2
        if h < 5 or len(k) - h < n:
            return None
        a, b = np.zeros(len(c["p"]), bool), np.zeros(len(c["p"]), bool)
        a[k[:h]], b[k[h:h + n]] = True, True
        return float(np.median(_residual(c, c["T_truth"] @ Dm, b)) - np.median(_residual(c, c["T_truth"], a)))

    by_n = {}
    for n in FRAME_COUNTS:
        hv_n = np.array([s for s in (change_n(c, n) for c in val) if s is not None])
        ht_n = np.array([s for s in (change_n(c, n) for c in test) if s is not None])
        if len(hv_n) < 5 or len(ht_n) < 5:
            continue
        al = float(np.quantile(hv_n, 1 - FA_RATE))
        caught = {}
        for kind, sizes in (("trans", TRANS_CM), ("rot", ROT_DEG)):
            for s in sizes:
                sc = np.array([x for c in test for _ in range(DIRECTIONS) if (x := change_n(c, n, _perturb(kind, s, rng))) is not None])
                caught[f"{kind}_{s}"] = float((sc > al).mean())
        by_n[str(n)] = dict(alarm_px=al, test_cameras=len(ht_n), healthy_test_false_alarms=float((ht_n > al).mean()),
                            caught=caught)
    res["frames_to_decide_per_camera"] = by_n

    # 2d. between sessions: the baseline is an earlier episode on the same camera (same serial), the test
    #     is a later one. Healthy: later minus earlier median residual, both under their own true pose.
    #     Drifted: the later episode's belief is off by D. Alarm level from validation pairs at 1 in 100.
    def pairs(cams):
        by = {}
        for c in cams:
            by.setdefault(c["serial"], []).append(c)
        out = []
        for ser, cs in by.items():
            cs = sorted(cs, key=lambda c: c["key"])        # DROID ids embed the date and time
            out += [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
        return out

    def ep_median(c, Dm=np.eye(4)):
        k = _keep(c, thr)
        return float(np.median(_residual(c, c["T_truth"] @ Dm, k))) if k.sum() >= 5 else None

    def between(pair, Dm=np.eye(4)):
        a0, b1 = ep_median(pair[0]), ep_median(pair[1], Dm)
        return None if a0 is None or b1 is None else b1 - a0

    pv, pt = pairs(val), pairs(test)
    hv_b = np.array([x for x in (between(p) for p in pv) if x is not None])
    ht_b = np.array([x for x in (between(p) for p in pt) if x is not None])
    alarm_b = float(np.quantile(hv_b, 1 - FA_RATE))
    per_b = {}
    for kind, sizes in (("trans", TRANS_CM), ("rot", ROT_DEG)):
        for s_ in sizes:
            sc = np.array([x for p in pt for _ in range(DIRECTIONS) if (x := between(p, _perturb(kind, s_, rng))) is not None])
            per_b[f"{kind}_{s_}"] = dict(median_rise_px=float(np.median(sc)), caught=float((sc > alarm_b).mean()))
    res["between_sessions"] = dict(
        val_pairs=len(hv_b), test_pairs=len(ht_b), alarm_px=alarm_b,
        noise_floor_test={str(q): float(np.percentile(np.abs(ht_b), q)) for q in (50, 90, 99)},
        healthy_test_false_alarms=float((ht_b > alarm_b).mean()), by_drift=per_b)

    # 3. real drift on test GT and Pred cameras. Decision: the camera's median residual over confident frames
    #    under its configured pose, against the healthy validation level at 1 in 100 (absolute, one episode,
    #    no baseline: a Pred camera has no healthy history in the data). Size: a translation-only fit of the
    #    camera position (robust loss), which is weakly constrained with one keypoint; reported as such.
    from scipy.optimize import least_squares

    alarm_abs = drift["all"]["alarm_px"]

    def fit_translation(c, k):
        P = fk_tips(c["q"][k])

        def r(x):
            T = c["T_belief"].copy()
            T[:3, 3] = T[:3, 3] + x
            return (project(T, c["K"], P) - c["uv"][k]).ravel()
        x = least_squares(r, np.zeros(3), loss="soft_l1", f_scale=10.0).x
        T = c["T_belief"].copy()
        T[:3, 3] = T[:3, 3] + x
        return T, x

    train_serials = set(split.get("train_final", split["train"]))
    rows = []
    for group, items in (("test GT", test), ("Pred", pred)):
        for c in items:
            k = _keep(c, thr)
            label = pose_error(c["T_belief"], c["T_truth"])
            row = dict(group=group, key=c["key"], path=c["path"], serial=c["serial"], frames=int(k.sum()),
                       label_mm=label["translation_mm"], label_deg=label["rotation_deg"],
                       label_moved=bool(label["translation_mm"] > 10 or label["rotation_deg"] > 1),
                       serial_in_training=c["serial"] in train_serials)
            if k.sum() < 12:
                row["called"] = None
                rows.append(row)
                continue
            score = float(np.median(_residual(c, c["T_belief"], k)))
            row.update(score_px=score, called=bool(score > alarm_abs))
            if row["called"]:
                # fix fitted on the first half of the confident frames, re-checked on the second half
                kk = np.flatnonzero(k)
                k1, k2 = np.zeros_like(k), np.zeros_like(k)
                k1[kk[:len(kk) // 2]], k2[kk[len(kk) // 2:]] = True, True
                T_fix, x = fit_translation(c, k1)
                after = float(np.median(_residual(c, T_fix, k2)))
                row.update(fix_translation_mm=float(np.linalg.norm(x) * 1000),
                           fix_vs_droid_mm=float(np.linalg.norm(T_fix[:3, 3] - c["T_truth"][:3, 3]) * 1000),
                           droid_translation_mm=float(np.linalg.norm(c["T_belief"][:3, 3] - c["T_truth"][:3, 3]) * 1000),
                           recheck_px=after, recheck_pass=bool(after <= alarm_abs), recheck_note="fit on the first half, re-checked on the second")
            rows.append(row)
    scored = [r for r in rows if r["called"] is not None]
    p = [r for r in scored if r["group"] == "Pred"]
    g = [r for r in scored if r["group"] == "test GT"]
    pm = [r for r in p if r["label_moved"]]
    tp = sum(r["called"] for r in pm)
    fp_gt = sum(r["called"] for r in g)
    fp_pred = sum(r["called"] for r in p if not r["label_moved"])
    caught = [r for r in pm if r["called"]]
    res["real_drift"] = dict(
        decision="median residual under the configured pose > validation healthy level at 1 in 100",
        alarm_px=alarm_abs, pred_cameras=len(p), pred_moved_per_droid=len(pm), caught=tp, recall=tp / max(len(pm), 1),
        test_gt_cameras=len(g), false_alarms_test_gt=fp_gt, false_alarm_rate=fp_gt / max(len(g), 1),
        precision=tp / max(tp + fp_gt + fp_pred, 1),
        size_note="translation-only fit, one keypoint: weakly constrained",
        fix_vs_droid_mm_median=float(np.median([r["fix_vs_droid_mm"] for r in caught])) if caught else None,
        droid_translation_mm_median=float(np.median([r["droid_translation_mm"] for r in caught])) if caught else None,
        pred_serials_seen_in_training=sorted({r["serial"] for r in p if r["serial_in_training"]}),
        caught_serial_in_training=sum(r["called"] for r in pm if r["serial_in_training"]),
        moved_serial_in_training=sum(1 for r in pm if r["serial_in_training"]),
        not_scored=len(rows) - len(scored))

    # 4. fix and re-check: the translation fix is applied and the same decision is re-run
    called = [r for r in rows if r.get("called")]
    res["fix_recheck"] = dict(cameras=len(called), pass_recheck=sum(r["recheck_pass"] for r in called),
                              rows=[{k: r[k] for k in ("group", "key", "serial", "score_px", "recheck_px", "recheck_pass",
                                                       "fix_vs_droid_mm", "droid_translation_mm")} for r in called])

    # 5. encoder drift, injected into real joint streams of test GT cameras. The 2D check cannot tell a
    #    joint that reads wrong from a camera that moved: both shift the fingertip in the image. So this
    #    reports how often something is flagged, and that it is flagged as "camera moved".
    from .. import arms
    from ..diagnose import _tip_shift_from_bias
    enc = {}
    rng_e = np.random.default_rng(1)
    for mm in (10.0, 20.0):
        flagged = n = 0
        for c in test:
            k = _keep(c, thr)
            if k.sum() < 12:
                continue
            j = int(rng_e.integers(0, 6))
            per_deg = _tip_shift_from_bias(c["q"][k], j, np.deg2rad(1.0), arms.ROBOTIQ_2F85_TIP, arms.PANDA) * 1000
            qb = c["q"][k].copy()
            qb[:, j] += np.deg2rad(mm / max(per_deg, 1e-6))
            score = float(np.median(np.linalg.norm(project(c["T_belief"], c["K"], fk_tips(qb)) - c["uv"][k], axis=1)))
            n += 1
            flagged += bool(score > alarm_abs)
        enc[str(mm)] = dict(cameras=n, flagged=flagged, named_as="camera (the 2D check has no encoder hypothesis)")
    res["encoder_injected"] = enc
    res["tool_offset_edit"] = "settings comparison against the known-good config; does not depend on the recording, not run on frames"

    res["rows"] = rows
    name = "phase2_visibility_only.json" if visibility_only else "phase2.json"
    json.dump(res, open(os.path.join(OUT, name), "w"), indent=1, default=float)
    return res


def write_record(model_path: str, key: str, serial: str, sign_key: str = "keys/private.pem") -> str:
    """One full record for a camera the check called moved: when, what changed, fix, re-check, GO/NO-GO; signed.
    Uses the analysis results (phase2.json): the residual decision and the translation-only fix."""
    from .. import certify

    res = json.load(open(os.path.join(OUT, "phase2.json")))
    r = next(r for r in res["rows"] if r["key"] == key and r["serial"] == serial)
    c = next(c for g in ("pred", "test") for c in load_cache(g) if c["key"] == key and c["serial"] == serial)
    alarm = res["real_drift"]["alarm_px"]
    status = "GO" if r.get("recheck_pass") else "NO-GO"
    rec = dict(
        kind="kintrace recommission record (DROID, real Franka data, 2D camera check)",
        robot=f"DROID {c['lab']} Franka Panda", episode=c["path"], episode_id=key, camera=serial,
        checked_at=time.strftime("%Y-%m-%d %H:%M:%S"), detector=model_path,
        what_changed=dict(camera_moved=bool(r["called"]), residual_px=r["score_px"], alarm_px=alarm, frames_used=r["frames"]),
        fix=dict(camera_translation_mm=r["fix_translation_mm"],
                 note="translation-only fit of the camera position from one gripper keypoint: weakly constrained"),
        recheck=dict(residual_px=r["recheck_px"], passed=bool(r["recheck_pass"])),
        droid_reference=dict(source="KarlP/droid cam2base_extrinsics.json (Pred)",
                             configured_vs_droid_mm=r["label_mm"], configured_vs_droid_deg=r["label_deg"],
                             fix_vs_droid_mm=r["fix_vs_droid_mm"]),
        status=status,
        note="A real DROID recording; DROID's re-solved camera pose is the reference. A Pred difference is camera "
             "movement or a calibration that was never right; the data cannot tell which.")
    out = os.path.join("incidents", f"droid_{key}_{serial}")
    os.makedirs(out, exist_ok=True)
    signed = certify.sign(rec, certify.load_private(sign_key)) if os.path.exists(sign_key) else rec
    json.dump(signed, open(os.path.join(out, "record.json"), "w"), indent=2, default=float)
    txt = [f"KINTRACE  recommission record, DROID {c['lab']} Franka, camera {serial}",
           "=" * 64, f"Episode      : {c['path']}", f"Checked at   : {rec['checked_at']}",
           f"What changed : camera off its calibration: residual {r['score_px']:.1f} px over {r['frames']} frames "
           f"(alarm {alarm:.1f} px)",
           f"Fix          : camera position moved {r['fix_translation_mm']:.1f} mm (translation only, weakly constrained)",
           f"Re-check     : residual {r['recheck_px']:.1f} px, " + ("passes" if r["recheck_pass"] else "still above the alarm"),
           f"vs DROID     : configured pose was {r['label_mm']:.1f} mm / {r['label_deg']:.2f} deg from DROID's re-solved pose; "
           f"after the fix the camera position is {r['fix_vs_droid_mm']:.1f} mm from it",
           f"STATUS: {status}" + ("  (signed)" if "signature" in signed else "  (not signed: no key)")]
    open(os.path.join(out, "record.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.phase2")
    ap.add_argument("what", choices=("cache", "analyse", "record"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--visibility-only", action="store_true")
    ap.add_argument("--key")
    ap.add_argument("--serial")
    a = ap.parse_args(argv)
    logfile = os.path.join(DATA, "runs", f"phase2_{time.strftime('%Y%m%d')}.log")
    if a.what == "cache":
        if not X.gpu_allowed():
            raise SystemExit("2 heat stops this session: no GPU jobs")
        cache(a.model, logfile)
    elif a.what == "analyse":
        r = analyse(a.model, logfile, a.visibility_only)
        print(json.dumps({k: v for k, v in r.items() if k not in ("rows",)}, indent=1, default=float)[:6000])
    else:
        print(write_record(a.model, a.key, a.serial))


if __name__ == "__main__":
    main()
