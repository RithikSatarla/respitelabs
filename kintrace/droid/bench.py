"""The two DROID result levels.

  labels   python -m kintrace.droid labels --data data/droid
           Reads every episode in the manifest, compares the extrinsic each
           cell was configured with against the corrected one, and reports
           how often cameras were off and by how much. Real data, no vision.

  detect   python -m kintrace.droid detect --data data/droid [--synthetic]
           Builds a kintrace Log per (episode, fixed camera) with the
           configured extrinsic as the belief, runs diagnose(), and scores the
           "camera moved" call against the label. --synthetic swaps the (not
           yet written) gripper detector for FK-through-the-corrected-pose
           plus noise; the report says so in its first line.

  faults   python -m kintrace.droid faults --data data/droid [--synthetic]
           The other faults, injected into real DROID logs. Latency: shift the
           measured joint stream against the commanded one (needs no vision,
           runs on real data today). Encoder drift: add a bias to one joint's
           measured angle. Tool offset edit: change the config. Each is scored
           on whether diagnose() names it and sizes it. Camera moved comes
           from the real labels (see detect). A bent tool cannot be injected
           into DROID: nothing in the data records the tool, so that one
           waits for the desk arm.

All write a JSON next to the data and print a short table.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from .. import arms
from ..diagnose import _tip_shift_from_bias, diagnose, estimate_delay, self_baseline
from . import episode as ep_mod
from .convert import episode_to_log
from .extrinsics import Corrected, labels_for

MOVED_MM = 10.0   # a camera this far from its configured pose counts as "moved"
MOVED_DEG = 1.0
GT_PRED_NOTE = ("GT = DROID kept the original calibration. Pred = DROID re-solved it. "
                "Only Pred entries can show a moved camera, and a Pred difference is camera movement "
                "or a calibration that was never right; the data cannot tell which.")


def _load_all(data: str):
    with open(os.path.join(data, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    corrected = Corrected.load(data)
    out = []
    for m in manifest:
        try:
            ep = ep_mod.load(m["dir"], key=m.get("path") or m.get("key"))
        except Exception as e:
            print(f"skip {m.get('path')}: {e}", file=sys.stderr)
            continue
        corr = corrected.lookup(m.get("path", "")) or corrected.lookup(m.get("key", ""))
        if corr is None:
            print(f"skip {m.get('path')}: no corrected extrinsics", file=sys.stderr)
            continue
        ep.meta["_corrected"] = corrected.meta(m.get("path", "")) or corrected.meta(m.get("key", ""))
        ep.meta["_id"] = m.get("key", "")
        fixed = {s: T for s, T in ep.configured.items() if s != ep.serials.get("wrist")}
        labels = labels_for(fixed, corr)
        if labels:
            out.append((ep, labels))
    return out


def _moved(lbl) -> bool:
    return lbl["translation_mm"] > MOVED_MM or lbl["rotation_deg"] > MOVED_DEG


# --------------------------------------------------------------------------
def run_labels(data: str) -> dict:
    rows = []
    for ep, labels in _load_all(data):
        for serial, l in labels.items():
            rows.append(dict(episode=ep.key, lab=ep.meta.get("lab", ""), camera=serial,
                             source=ep.meta["_corrected"].get("source", ""),
                             translation_mm=l["translation_mm"], rotation_deg=l["rotation_deg"], moved=_moved(l)))
    if not rows:
        raise SystemExit("no labelled cameras found. Run `python -m kintrace.droid download` first.")
    t = np.array([r["translation_mm"] for r in rows])
    r_ = np.array([r["rotation_deg"] for r in rows])
    moved = np.array([r["moved"] for r in rows])
    summary = dict(
        source="DROID raw 1.0.1 metadata vs KarlP/droid cam2base_extrinsics.json",
        cameras=len(rows), episodes=len({r["episode"] for r in rows}),
        moved_threshold=dict(translation_mm=MOVED_MM, rotation_deg=MOVED_DEG),
        fraction_moved=float(moved.mean()),
        translation_mm=dict(median=float(np.median(t)), p90=float(np.percentile(t, 90)), max=float(t.max())),
        rotation_deg=dict(median=float(np.median(r_)), p90=float(np.percentile(r_, 90)), max=float(r_.max())),
    )
    pred = [r for r in rows if r["source"] == "Pred"]
    summary["by_source"] = {"GT": sum(r["source"] == "GT" for r in rows), "Pred": len(pred)}
    if pred:
        tp = np.array([r["translation_mm"] for r in pred])
        rp = np.array([r["rotation_deg"] for r in pred])
        summary["pred_only"] = dict(
            cameras=len(pred), fraction_moved=float(np.mean([r["moved"] for r in pred])),
            translation_mm=dict(median=float(np.median(tp)), p90=float(np.percentile(tp, 90)), max=float(tp.max())),
            rotation_deg=dict(median=float(np.median(rp)), p90=float(np.percentile(rp, 90)), max=float(rp.max())),
        )
    summary["note"] = GT_PRED_NOTE
    out = dict(summary=summary, rows=rows)
    path = os.path.join(data, "labels.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"DROID camera labels: {summary['cameras']} fixed cameras over {summary['episodes']} episodes")
    print(f"  by source: GT {summary['by_source']['GT']}, Pred {summary['by_source']['Pred']}")
    print(f"  all entries (GT are identical by construction), off by > {MOVED_MM:.0f} mm or > {MOVED_DEG:.0f} deg:  {100*summary['fraction_moved']:.0f}%")
    print(f"  translation error  median {summary['translation_mm']['median']:.1f} mm   p90 {summary['translation_mm']['p90']:.1f} mm   max {summary['translation_mm']['max']:.1f} mm")
    print(f"  rotation error     median {summary['rotation_deg']['median']:.2f} deg  p90 {summary['rotation_deg']['p90']:.2f} deg  max {summary['rotation_deg']['max']:.2f} deg")
    if pred:
        po = summary["pred_only"]
        print(f"  Pred only ({po['cameras']} cameras), off by > {MOVED_MM:.0f} mm or > {MOVED_DEG:.0f} deg:  {100*po['fraction_moved']:.0f}%")
        print(f"    translation error  median {po['translation_mm']['median']:.1f} mm   p90 {po['translation_mm']['p90']:.1f} mm   max {po['translation_mm']['max']:.1f} mm")
        print(f"    rotation error     median {po['rotation_deg']['median']:.2f} deg  p90 {po['rotation_deg']['p90']:.2f} deg  max {po['rotation_deg']['max']:.2f} deg")
    print(GT_PRED_NOTE)
    print(f"wrote {path}")
    return out


# --------------------------------------------------------------------------
def run_survey(data: str) -> dict:
    """The whole corrected set, no episodes needed: how many entries DROID kept vs re-solved."""
    corrected = Corrected.load(data)
    metas = [corrected.meta(k) for k in corrected.keys()]
    groups = {}
    for m in metas:
        groups.setdefault(m.get("source", "?"), []).append(m)
    summary = dict(source="KarlP/droid cam2base_extrinsics.json", entries=len(metas), groups={})
    print(f"DROID corrected extrinsics: {len(metas)} entries")
    for src in sorted(groups):
        ms = groups[src]
        q = np.array([m["quality_metric"] for m in ms if isinstance(m.get("quality_metric"), (int, float))])
        types = sorted({str(m.get("metric_type", "?")) for m in ms})
        g = dict(n=len(ms), percent=100 * len(ms) / max(len(metas), 1), metric_type=types[0] if len(types) == 1 else types)
        if q.size:
            g["quality_metric"] = {f"p{p}": float(np.percentile(q, p)) for p in (10, 50, 90)}
        summary["groups"][src] = g
        line = f"  {src:5s} {len(ms):6d} ({g['percent']:.1f}%)  quality_metric ({', '.join(types)})"
        if q.size:
            line += "  " + "  ".join(f"{k} {v:.3f}" for k, v in g["quality_metric"].items())
        print(line)
    print("  quality_metric is a different measure in each group, so compare within a group only")
    summary["note"] = GT_PRED_NOTE
    print(GT_PRED_NOTE)
    path = os.path.join(data, "survey.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {path}")
    return summary


# --------------------------------------------------------------------------
def _baseline_for(log, sigma_mm: float) -> dict:
    """On DROID every episode is its own cell: single-session mode."""
    return self_baseline(log, sigma_mm)


def run_detect(data: str, synthetic: bool = False, every: int = 1, sigma_mm: float = 4.0, seed: int = 0) -> dict:
    rows = []
    for ep, labels in _load_all(data):
        for serial, l in labels.items():
            try:
                log = episode_to_log(ep, serial, l["belief"], l["truth"], synthetic=synthetic, every=every, seed=seed)
            except NotImplementedError as e:
                raise SystemExit(str(e))
            if log.visible.sum() < 20:
                print(f"skip {ep.key} {serial}: only {int(log.visible.sum())} frames with the gripper located", file=sys.stderr)
                continue
            d = diagnose(log, _baseline_for(log, sigma_mm))
            cam = next((f for f in d.findings if f.fault == "camera_moved"), None)
            geo_check = next((c for c in d.checks if c.name == "Camera vs joint encoders"), None)
            est_mm = float(np.linalg.norm(cam.params["translation_mm"])) if cam else 0.0
            rows.append(dict(
                episode=ep.key, camera=serial, label_moved=_moved(l), label_mm=l["translation_mm"], label_deg=l["rotation_deg"],
                called_moved=cam is not None, disagree_score=geo_check.score if geo_check else None,
                estimated_mm=est_mm, estimated_deg=float(cam.params["rotation_deg"]) if cam else 0.0,
                primary=d.primary, frames_used=int(log.visible.sum()),
            ))
    if not rows:
        raise SystemExit("nothing scored")
    lm = np.array([r["label_moved"] for r in rows])
    cm = np.array([r["called_moved"] for r in rows])
    tp, fp, fn, tn = int((lm & cm).sum()), int((~lm & cm).sum()), int((lm & ~cm).sum()), int((~lm & ~cm).sum())
    err = [abs(r["estimated_mm"] - r["label_mm"]) for r in rows if r["label_moved"] and r["called_moved"]]
    summary = dict(
        mode="synthetic detections (pipeline test, not a real-image result)" if synthetic else "detect_gripper() on real frames",
        cameras=len(rows), moved_in_labels=int(lm.sum()),
        recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), tp=tp, fp=fp, fn=fn, tn=tn,
        size_error_mm=dict(median=float(np.median(err)), p90=float(np.percentile(err, 90))) if err else None,
        sigma_mm=sigma_mm, moved_threshold=dict(translation_mm=MOVED_MM, rotation_deg=MOVED_DEG),
    )
    out = dict(summary=summary, rows=rows)
    path = os.path.join(data, "detect_synthetic.json" if synthetic else "detect.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"DROID detect run: {summary['mode']}")
    print(f"  {len(rows)} cameras, {int(lm.sum())} moved per the labels")
    print(f"  caught {tp}/{tp+fn} moved cameras (recall {100*summary['recall']:.0f}%), {fp} false alarms on {fp+tn} fine cameras")
    if err:
        print(f"  size error on caught cameras: median {summary['size_error_mm']['median']:.1f} mm, p90 {summary['size_error_mm']['p90']:.1f} mm")
    print(f"wrote {path}")
    return out


# --------------------------------------------------------------------------
def run_detect_2d(data: str, model_path: str, gt_only: bool = False) -> dict:
    """Real DROID frames: detector -> 2D camera check (check2d.py). diagnose() is not used here."""
    import time

    import cv2
    import torch

    from . import check2d, detector, labels_fk
    from .extrinsics import pose_error

    heat = detector.Heat()
    with open(os.path.join(data, "intrinsics.json"), encoding="utf-8") as f:
        intr = json.load(f)
    model = detector.load(model_path)
    test_serials = set(torch.load(model_path, map_location="cpu", weights_only=False).get("test_serials", []))
    rows, skipped = [], []
    try:
        heat.settle("before inference (idle)")
        for ep, labels in _load_all(data):
            src = ep.meta["_corrected"].get("source", "")
            if gt_only and src != "GT":
                continue
            for serial, l in labels.items():
                if serial in labels_fk.EXCLUDED:
                    skipped.append((ep.key, serial, "excluded: " + labels_fk.EXCLUDED[serial]))
                    continue
                group = "Pred" if src == "Pred" else ("GT held-out" if serial in test_serials else "GT training")
                if gt_only and group != "GT held-out":
                    continue
                video = ep.video(serial)
                if video is None:
                    skipped.append((ep.key, serial, "no MP4"))
                    continue
                small, (w, h) = labels_fk.read_small(video, (detector.SMALL_W, detector.SMALL_H))
                K = labels_fk.intrinsics_for(intr, ep.meta["_id"], serial, (w, h))
                if K is None:
                    skipped.append((ep.key, serial, "no intrinsics"))
                    continue
                n = min(len(small), len(ep.t))
                small = small[:n]
                uv, p = detector.predict(model, small, heat)
                uv = uv * (w / detector.SMALL_W)
                tc = ep.t_cam.get(serial)
                t = tc[:n] if tc is not None and len(tc) >= n else ep.t[:n]
                q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(ep.q.shape[1])], 1)
                det = p > 0.5
                # 2D error against FK projected through the belief (the label, on GT cameras)
                fk_uv = check2d.project(l["belief"], K, check2d.fk_tips(q))
                in_img = (fk_uv[:, 0] >= 0) & (fk_uv[:, 0] < w) & (fk_uv[:, 1] >= 0) & (fk_uv[:, 1] < h)
                e2d = np.linalg.norm(uv - fk_uv, axis=1)[det & in_img]
                res = check2d.camera_check_2d(q[det], uv[det], K, l["belief"])
                if res is None:
                    skipped.append((ep.key, serial, f"only {int(det.sum())} frames with a detection"))
                    continue
                fit_truth = pose_error(res.T_fit, l["truth"]) if res.T_fit is not None else None
                rows.append(dict(
                    episode=ep.key, camera=serial, group=group, label_moved=_moved(l),
                    label_mm=l["translation_mm"], label_deg=l["rotation_deg"], called_moved=res.moved,
                    frames=int(n), detections=int(det.sum()), inliers=res.inliers,
                    err2d_px_median=float(np.median(e2d)) if len(e2d) else None, err2d_px=e2d.tolist(),
                    residual_belief_px=res.residual_belief_px, residual_fit_px=res.residual_fit_px,
                    fit_vs_belief_mm=res.fit_vs_belief_mm, fit_vs_belief_deg=res.fit_vs_belief_deg,
                    fit_vs_droid_mm=fit_truth["translation_mm"] if fit_truth else None,
                    fit_vs_droid_deg=fit_truth["rotation_deg"] if fit_truth else None))
                print(f"  {group:11s} {ep.key[:44]:44s} {serial}  det {int(det.sum()):4d}/{n:<4d} "
                      f"resid belief {res.residual_belief_px:6.1f} fit {res.residual_fit_px:6.1f} px  "
                      f"fit-belief {res.fit_vs_belief_mm:7.1f} mm  moved={res.moved}", flush=True)
    except detector.Overheat as e:
        detector.write_progress(
            "# Progress\n\n"
            f"Stopped {time.strftime('%Y-%m-%d %H:%M:%S')}: {e}.\n\n"
            "Phase 4/5, `python -m kintrace.droid detect` (real frames, inference). Stopped by the heat rule.\n"
            "Do not resume without checking the laptop.\n\nTemperature log:\n\n"
            + "\n".join(f"- {t} {lbl}: {c} C" for t, lbl, c in heat.log) + "\n")
        print(f"STOPPED: {e}. Wrote PROGRESS.md")
        sys.exit(3)
    if not rows:
        raise SystemExit("nothing scored")

    def grp(g):
        return [r for r in rows if r["group"] == g]

    summary = dict(mode="detector on real DROID frames, 2D camera check (kintrace/droid/check2d.py), not diagnose()",
                   model=model_path, test_serials=sorted(test_serials), skipped=skipped, heat_log=heat.log)
    print(f"DROID detect run: {summary['mode']}")
    print(f"  model {model_path}")
    gh = grp("GT held-out")
    if gh:
        errs = [np.asarray(r["err2d_px"]) for r in gh if r["err2d_px"]]
        e = np.concatenate(errs) if errs else np.zeros(0)
        fb = np.array([r["fit_vs_belief_mm"] for r in gh])
        fbd = np.array([r["fit_vs_belief_deg"] for r in gh])
        fa = sum(r["called_moved"] for r in gh)
        summary["gt_held_out"] = dict(
            cameras=len(gh), false_alarms=fa,
            err2d_px=dict(median=float(np.median(e)), p90=float(np.percentile(e, 90))) if len(e) else None,
            fit_vs_belief_mm=dict(median=float(np.nanmedian(fb)), p90=float(np.nanpercentile(fb, 90))),
            fit_vs_belief_deg=dict(median=float(np.nanmedian(fbd)), p90=float(np.nanpercentile(fbd, 90))))
        s = summary["gt_held_out"]
        print(f"  GT held-out cameras ({len(gh)}, serials never seen in training):")
        if s["err2d_px"]:
            print(f"    2D error, detector vs FK projection   median {s['err2d_px']['median']:.1f} px   "
                  f"p90 {s['err2d_px']['p90']:.1f} px (full res)")
        print(f"    fitted pose vs belief                 median {s['fit_vs_belief_mm']['median']:.1f} mm / "
              f"{s['fit_vs_belief_deg']['median']:.2f} deg   p90 {s['fit_vs_belief_mm']['p90']:.1f} mm / "
              f"{s['fit_vs_belief_deg']['p90']:.2f} deg")
        print(f"    false alarms (called moved)           {fa}/{len(gh)}")
    gp = grp("Pred")
    if gp:
        lm = [r for r in gp if r["label_moved"]]
        caught = sum(r["called_moved"] for r in lm)
        ft = np.array([r["fit_vs_droid_mm"] for r in gp if r["fit_vs_droid_mm"] is not None])
        ftd = np.array([r["fit_vs_droid_deg"] for r in gp if r["fit_vs_droid_deg"] is not None])
        summary["pred"] = dict(
            cameras=len(gp), moved_in_labels=len(lm), caught=caught, recall=caught / max(len(lm), 1),
            fit_vs_droid_pred_mm=dict(median=float(np.median(ft)), p90=float(np.percentile(ft, 90))) if len(ft) else None,
            fit_vs_droid_pred_deg=dict(median=float(np.median(ftd))) if len(ftd) else None)
        s = summary["pred"]
        print(f"  Pred cameras ({len(gp)}; {len(lm)} off by > {MOVED_MM:.0f} mm or > {MOVED_DEG:.0f} deg per DROID):")
        print(f"    called moved {caught}/{len(lm)} (recall {100 * s['recall']:.0f}%)")
        if s["fit_vs_droid_pred_mm"]:
            print(f"    fitted pose vs DROID's re-solved pose  median {s['fit_vs_droid_pred_mm']['median']:.1f} mm / "
                  f"{s['fit_vs_droid_pred_deg']['median']:.2f} deg   p90 {s['fit_vs_droid_pred_mm']['p90']:.1f} mm")
    gtr = grp("GT training")
    if gtr:
        print(f"  GT training cameras ({len(gtr)}): not counted, the detector trained on these serials")
    print(f"  excluded or skipped: {len(skipped)}")
    for k, s_, why in skipped:
        print(f"    {k} {s_}: {why}")
    print("  " + GT_PRED_NOTE)
    out = dict(summary=summary, rows=rows)
    path = os.path.join(data, "detect_gt_held_out.json" if gt_only else "detect.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("  temperature log: " + ", ".join(f"{lbl} {c} C" for _, lbl, c in heat.log))
    print(f"wrote {path}")
    return out


# --------------------------------------------------------------------------
def _interp_rows(t_src, q_src, t):
    return np.stack([np.interp(t, t_src, q_src[:, i]) for i in range(q_src.shape[1])], 1)


def _shift_measured(log, delay_s: float):
    """Make the encoders report each pose delay_s later than they did."""
    import copy
    out = copy.copy(log)
    out.q_meas = _interp_rows(log.t_joint, log.q_meas, log.t_joint - delay_s)
    return out


def run_faults(data: str, synthetic: bool = False, every: int = 1, sigma_mm: float = 4.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    any_located = False  # did any episode get a gripper located in its frames
    delays_ms = (10.0, 20.0, 50.0)
    tip_mm = (5.0, 10.0, 20.0)
    tcp_mm = (3.0, 6.0, 12.0)
    for ep, labels in _load_all(data):
        for serial, l in labels.items():
            # --- latency: real joint streams, no vision needed ------------------
            base_log = episode_to_log(ep, serial, l["belief"], l["truth"], synthetic=True, seed=seed) if synthetic else None
            try:
                plain = base_log or episode_to_log(ep, serial, l["belief"], l["truth"], synthetic=False, every=every)
                have_vision = True
            except (NotImplementedError, FileNotFoundError):
                # no detector: still run the latency fault, which needs only joints
                from .convert import to_log
                plain = to_log(ep, serial, l["belief"], np.zeros((len(ep.t), 1, 3)), np.zeros(len(ep.t), bool))
                have_vision = False
            any_located = any_located or (have_vision and bool(plain.visible.any()))
            tau0 = estimate_delay(plain, plain.config.get("delay_search_s", 0.2))
            for d in delays_ms:
                tau1 = estimate_delay(_shift_measured(plain, d / 1000), plain.config.get("delay_search_s", 0.2))
                est = (tau1 - tau0) * 1000
                rows.append(dict(fault="latency", episode=ep.key, camera=serial, injected=d, estimated=est,
                                 caught=abs(est) >= 4.0, size_error=abs(est - d), vision=False,
                                 note="real joint streams, delay injected"))
            if not have_vision or _moved(l) or plain.visible.sum() < 20:
                continue  # the next two need a clean camera and located gripper
            baseline = _baseline_for(plain, sigma_mm)
            # --- encoder drift: bias one joint's measured angle --------------------
            # sized by how far it moves the fingertip, like the simulation, so a
            # fault is "N mm at the tip" whichever joint carries it. J7 is the
            # wrist roll: the fingertip centre sits on its axis, so a J7 bias is
            # invisible to a single tip point by construction and is left out.
            for mm in tip_mm:
                j = int(rng.integers(0, 6))
                per_deg = _tip_shift_from_bias(plain.q_meas, j, np.deg2rad(1.0), np.array(plain.config["tcp_offset"]), arms.PANDA) * 1000
                b = mm / max(per_deg, 1e-6)
                log = _shift_measured(plain, 0.0)
                log.q_meas = plain.q_meas.copy()
                log.q_meas[:, j] += np.deg2rad(b)
                dg = diagnose(log, baseline)
                f = next((x for x in dg.findings if x.fault == "encoder_bias"), None)
                right_joint = f is not None and f.params.get("joint") == j + 1
                rows.append(dict(fault="encoder_bias", episode=ep.key, camera=serial, injected=b, injected_tip_mm=mm, joint=j + 1,
                                 estimated=float(f.params["bias_deg"]) if f else 0.0, primary=dg.primary,
                                 caught=dg.primary == "encoder_bias" and right_joint,
                                 size_error=abs(f.params["bias_deg"] - b) if right_joint else None, vision=True,
                                 note="real joints and camera, bias injected into one joint"))
            # --- tool offset edited in the controller ------------------------------
            for mm in tcp_mm:
                log = _shift_measured(plain, 0.0)
                cfg = dict(plain.config)
                v = rng.normal(size=3)
                cfg["tcp_offset"] = (np.array(plain.config["tcp_offset"]) + v / np.linalg.norm(v) * mm / 1000).tolist()
                log.config = cfg
                dg = diagnose(log, baseline)
                f = next((x for x in dg.findings if x.fault == "tcp_config"), None)
                est = float(np.linalg.norm(np.array(f.params["tcp_offset_now_mm"]) - np.array(f.params["tcp_offset_good_mm"]))) if f else 0.0
                rows.append(dict(fault="tcp_config", episode=ep.key, camera=serial, injected=mm, estimated=est,
                                 primary=dg.primary, caught=dg.primary == "tcp_config", size_error=abs(est - mm) if f else None,
                                 vision=True, note="config edit; the check is a settings diff, data only has to stay quiet"))
    if not rows:
        raise SystemExit("nothing scored")
    if synthetic:
        mode = "synthetic detections (pipeline test, not a real-image result)"
    elif any_located:
        mode = "real frames where vision was needed"
    else:
        mode = "latency on real joint streams; vision checks need a gripper detector"
    summary = {"mode": mode,
               "not_possible": {"tool_bent": "DROID records nothing about the tool; waits for the desk arm"}}
    print("DROID injected faults:", summary["mode"])
    for fault in ("latency", "encoder_bias", "tcp_config"):
        rs = [r for r in rows if r["fault"] == fault]
        if not rs:
            summary[fault] = None
            print(f"  {fault:13s} not run (needs a located gripper)")
            continue
        caught = sum(r["caught"] for r in rs)
        errs = [r["size_error"] for r in rs if r["size_error"] is not None]
        summary[fault] = dict(n=len(rs), caught=caught, rate=caught / len(rs),
                              size_error_median=float(np.median(errs)) if errs else None, note=rs[0]["note"])
        unit = {"latency": "ms", "encoder_bias": "deg", "tcp_config": "mm"}[fault]
        print(f"  {fault:13s} {caught}/{len(rs)} named and sized"
              + (f", size error median {np.median(errs):.2f} {unit}" if errs else "") + f"   ({rs[0]['note']})")
        key = "injected_tip_mm" if fault == "encoder_bias" else "injected"
        by = {}
        for r in rs:
            by.setdefault(r[key], []).append(r["caught"])
        summary[fault]["by_size"] = {str(k): dict(n=len(v), caught=int(sum(v))) for k, v in sorted(by.items())}
        print("                " + "   ".join(f"{k:g} {'mm' if fault != 'latency' else 'ms'}: {sum(v)}/{len(v)}" for k, v in sorted(by.items())))
    print("  tool_bent     not possible on DROID (nothing in the data records the tool)")
    out = dict(summary=summary, rows=rows)
    path = os.path.join(data, "faults_synthetic.json" if synthetic else "faults.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    return out


def main(argv=None):
    import argparse

    from . import download

    ap = argparse.ArgumentParser(prog="python -m kintrace.droid", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", add_help=False)
    sub.add_parser("fklabels", add_help=False)
    sub.add_parser("train-detector", add_help=False)
    sub.add_parser("train_detector", add_help=False)
    p = sub.add_parser("survey")
    p.add_argument("--data", default="data/droid")
    p = sub.add_parser("labels")
    p.add_argument("--data", default="data/droid")
    p = sub.add_parser("detect")
    p.add_argument("--data", default="data/droid")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--model", default="data/droid/detector_wide.pt", help="real frames: detector checkpoint")
    p.add_argument("--gt-only", action="store_true", help="real frames: held-out GT cameras only (the detector test)")
    p.add_argument("--every", type=int, default=1, help="use every Nth video frame")
    p.add_argument("--sigma-mm", type=float, default=4.0, help="expected detector noise")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("faults")
    p.add_argument("--data", default="data/droid")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--sigma-mm", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "download":
        return download.main(argv[1:])
    if argv and argv[0] == "fklabels":
        from . import labels_fk
        return labels_fk.main(argv[1:])
    if argv and argv[0] == "train-detector":
        from . import detector
        return detector.main(argv[1:])
    if argv and argv[0] == "train_detector":
        from . import train_detector_cmd
        return train_detector_cmd.main(argv[1:])
    a = ap.parse_args(argv)
    if a.cmd == "survey":
        run_survey(a.data)
    elif a.cmd == "labels":
        run_labels(a.data)
    elif a.cmd == "detect":
        if a.synthetic:
            run_detect(a.data, True, a.every, a.sigma_mm, a.seed)
        else:
            run_detect_2d(a.data, a.model, a.gt_only)
    elif a.cmd == "faults":
        run_faults(a.data, a.synthetic, a.every, a.sigma_mm, a.seed)


if __name__ == "__main__":
    main()
