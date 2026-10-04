"""Confidence and drift studies on cached detector outputs (no GPU, no network).

Both read the prediction cache written by eval_detector.py (held-out DROID GT
cameras only) and the FK label files (fingertip in the camera frame, K).

confidence
    Error against heatmap peak. Picks the lowest peak threshold whose kept
    detections have a median error at or under the target (12 px full res), and
    reports coverage and the kept error. Also counts confident detections on
    frames where the fingertip is out of view.

drift
    On each held-out GT camera the camera did not move. We pretend the belief
    is off by a known amount: T_belief = T_true @ D, with D a translation or a
    rotation about the camera centre in a random direction. Projecting the FK
    fingertip through T_belief and comparing with the detector gives the
    residual a real camera move of D would produce. The healthy residual uses
    D = identity. Per episode the score is the median residual over its
    confident frames (taken in time order, first N). The alarm threshold is the
    healthy score at the chosen false-alarm rate.

Run: python -m kintrace.droid.study confidence|drift --preds data/droid/eval_epoch60_preds.npz
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

DATA = "data/droid"
TARGET_PX = 12.0
MIN_VISIBLE = 0.5
TRANS_CM = (0.5, 1.0, 2.0, 5.0)
ROT_DEG = (0.5, 1.0, 2.0, 5.0)
DIRECTIONS = 20          # random directions per size per camera
FALSE_ALARM = 0.01       # 1 in 100 healthy episodes
FRAME_COUNTS = (5, 10, 20, 50, 100, None)


def _load(preds: str):
    z = np.load(preds)
    err = np.linalg.norm(z["uv_full"] - z["label_full"], axis=1)
    return z, err


def confidence(preds: str, out_dir: str = os.path.join(DATA, "figures")) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, err = _load(preds)
    vis_label = z["visible"].astype(bool)
    said_vis = z["p"] >= MIN_VISIBLE
    peak = z["peak"]
    e = err[vis_label]
    pk = peak[vis_label]
    sv = said_vis[vis_label]

    rows = []
    for t in np.unique(np.quantile(pk, np.linspace(0, 0.98, 99))):
        keep = sv & (pk >= t)
        if keep.sum() < 50:
            continue
        rows.append(dict(threshold=float(t), coverage=float(keep.mean()), median_px=float(np.median(e[keep])),
                         p90_px=float(np.percentile(e[keep], 90))))
    ok = [r for r in rows if r["median_px"] <= TARGET_PX]
    # lowest threshold that meets the target; if none does, the one with the lowest kept median
    chosen = min(ok, key=lambda r: r["threshold"]) if ok else min(rows, key=lambda r: r["median_px"])
    chosen = dict(chosen, meets_target=bool(ok))
    # out-of-view frames that still pass both filters (false detections)
    oov = ~vis_label
    fp = None
    if chosen:
        fp = int((oov & said_vis & (peak >= chosen["threshold"])).sum())

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hexbin(pk, np.clip(e, 0.5, None), gridsize=60, yscale="log", bins="log", cmap="viridis")
    ax[0].set_xlabel("heatmap peak (confidence)")
    ax[0].set_ylabel("error, full-res px (log)")
    ax[0].axhline(TARGET_PX, color="w", ls="--", lw=1)
    if chosen:
        ax[0].axvline(chosen["threshold"], color="r", lw=1)
    ax[1].plot([r["coverage"] * 100 for r in rows], [r["median_px"] for r in rows], label="median")
    ax[1].plot([r["coverage"] * 100 for r in rows], [r["p90_px"] for r in rows], label="p90")
    ax[1].axhline(TARGET_PX, color="k", ls="--", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("frames kept (%)")
    ax[1].set_ylabel("error of kept frames, full-res px")
    ax[1].legend()
    fig.suptitle("DROID held-out GT cameras: error against detector confidence")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "confidence.png")
    fig.savefig(fig_path, dpi=120)

    res = dict(preds=preds, frames_in_view=int(vis_label.sum()), frames_out_of_view=int(oov.sum()),
               target_px=TARGET_PX, chosen=chosen, out_of_view_passing=fp, curve=rows, figure=fig_path)
    print(f"confidence study ({preds}): {res['frames_in_view']} in-view frames, {res['frames_out_of_view']} out of view")
    if not chosen["meets_target"]:
        print(f"  no threshold reaches a kept median of {TARGET_PX:.0f} px; using the lowest kept median instead")
    print(f"  peak >= {chosen['threshold']:.4f} and visibility >= {MIN_VISIBLE}: keeps {100 * chosen['coverage']:.1f}% "
          f"of in-view frames, kept median {chosen['median_px']:.1f} px, p90 {chosen['p90_px']:.1f} px")
    print(f"  out-of-view frames that still pass: {fp} of {res['frames_out_of_view']}")
    print(f"  wrote {fig_path}")
    return res


def _perturb(kind: str, size: float, rng) -> np.ndarray:
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    D = np.eye(4)
    if kind == "trans":
        D[:3, 3] = v * size / 100.0
    else:
        D[:3, :3] = Rotation.from_rotvec(v * np.deg2rad(size)).as_matrix()
    return D


def drift(preds: str, threshold: float, out_dir: str = os.path.join(DATA, "figures"), seed: int = 0) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, _ = _load(preds)
    index = {(r["key"], r["serial"]): r["file"] for r in
             json.load(open(os.path.join(DATA, "fk_labels", "index.json"), encoding="utf-8"))["written"]}
    rng = np.random.default_rng(seed)
    cams = []
    for key, serial in sorted(set(zip(z["key"].tolist(), z["serial"].tolist()))):
        sel = np.flatnonzero((z["key"] == key) & (z["serial"] == serial))
        sel = sel[np.argsort(z["frame"][sel])]
        good = sel[(z["p"][sel] >= MIN_VISIBLE) & (z["peak"][sel] >= threshold)]
        if len(good) < 5:
            continue
        lab = np.load(index[(key, serial)])
        frames = z["frame"][good]
        cams.append(dict(key=key, serial=serial, uv=z["uv_full"][good], tip=lab["tip_cam"][frames], K=lab["K"]))

    def score(c, D, n=None):
        Dinv = np.linalg.inv(D)
        pc = c["tip"] @ Dinv[:3, :3].T + Dinv[:3, 3]
        uv = pc @ c["K"].T
        uv = uv[:, :2] / uv[:, 2:3]
        r = np.linalg.norm(uv - c["uv"], axis=1)
        return float(np.median(r[:n] if n else r))

    results = {}
    for n in FRAME_COUNTS:
        usable = [c for c in cams if n is None or len(c["uv"]) >= n]
        healthy = np.array([score(c, np.eye(4), n) for c in usable])
        thr = float(np.quantile(healthy, 1 - FALSE_ALARM))
        per = {}
        for kind, sizes in (("trans", TRANS_CM), ("rot", ROT_DEG)):
            for s in sizes:
                sc = np.array([score(c, _perturb(kind, s, rng), n) for c in usable for _ in range(DIRECTIONS)])
                per[f"{kind}_{s}"] = dict(median_score_px=float(np.median(sc)), detected=float((sc > thr).mean()))
        results["all" if n is None else str(n)] = dict(cameras=len(usable), healthy_median_px=float(np.median(healthy)),
                                                       healthy_p99_px=thr, by_drift=per, healthy=healthy.tolist())

    # Per-camera version (how Kintrace runs: against the same camera's own healthy history).
    # First half of the confident frames = baseline under the true calibration; the score is how much the
    # median residual on the second half rises over that baseline.
    def change(c, D):
        h = len(c["uv"]) // 2
        base = {k: (v[:h] if k in ("uv", "tip") else v) for k, v in c.items()}
        test = {k: (v[h:] if k in ("uv", "tip") else v) for k, v in c.items()}
        return score(test, D) - score(base, np.eye(4))

    # per-frame tail of the healthy residual (confident frames only), reported separately from episode medians
    pf = np.concatenate([np.linalg.norm((lambda uv: uv[:, :2] / uv[:, 2:3])(c["tip"] @ c["K"].T) - c["uv"], axis=1)
                         for c in cams])
    per_frame = dict(frames=int(len(pf)), median_px=float(np.median(pf)), p90_px=float(np.percentile(pf, 90)),
                     p99_px=float(np.percentile(pf, 99)), over_50px=float((pf > 50).mean()), over_100px=float((pf > 100).mean()))

    paired = [c for c in cams if len(c["uv"]) >= 10]
    h_change = np.array([change(c, np.eye(4)) for c in paired])
    thr_c = float(np.quantile(h_change, 1 - FALSE_ALARM))
    per_c = {}
    for kind, sizes in (("trans", TRANS_CM), ("rot", ROT_DEG)):
        for s in sizes:
            sc = np.array([change(c, _perturb(kind, s, rng)) for c in paired for _ in range(DIRECTIONS)])
            per_c[f"{kind}_{s}"] = dict(median_rise_px=float(np.median(sc)), detected=float((sc > thr_c).mean()))
    per_camera = dict(cameras=len(paired), healthy_rise_median_px=float(np.median(h_change)),
                      healthy_rise_p99_px=thr_c, by_drift=per_c)

    full = results["all"]

    def smallest(kind, sizes, res, rate=0.9):
        hit = [s for s in sizes if res["by_drift"][f"{kind}_{s}"]["detected"] >= rate]
        return min(hit) if hit else None

    frames_needed = {}
    for kind, sizes in (("trans", TRANS_CM), ("rot", ROT_DEG)):
        for s in sizes:
            ok = [n for n in FRAME_COUNTS[:-1] if results[str(n)]["by_drift"][f"{kind}_{s}"]["detected"] >= 0.9]
            frames_needed[f"{kind}_{s}"] = min(ok) if ok else None

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (kind, sizes, unit) in enumerate((("trans", TRANS_CM, "cm"), ("rot", ROT_DEG, "deg"))):
        data = [full["healthy"]] + [[score(c, _perturb(kind, s, rng)) for c in cams for _ in range(5)] for s in sizes]
        ax[i].boxplot(data, tick_labels=["healthy"] + [f"{s} {unit}" for s in sizes], showfliers=False)
        ax[i].axhline(full["healthy_p99_px"], color="r", ls="--", lw=1, label="alarm (1% false alarms)")
        ax[i].set_yscale("log")
        ax[i].set_ylabel("episode median residual, full-res px")
        ax[i].set_title("camera translated" if kind == "trans" else "camera rotated")
        ax[i].legend(loc="upper left")
    fig.suptitle("DROID held-out GT cameras: residual with a known calibration error applied")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "drift.png")
    fig.savefig(fig_path, dpi=120)

    res = dict(preds=preds, threshold_peak=threshold, cameras=len(cams), false_alarm=FALSE_ALARM, directions=DIRECTIONS,
               smallest_detected_90pct=dict(trans_cm=smallest("trans", TRANS_CM, full), rot_deg=smallest("rot", ROT_DEG, full)),
               frames_needed_90pct=frames_needed, by_frames=results, figure=fig_path, per_camera=per_camera,
               healthy_per_frame=per_frame,
               per_camera_smallest_90pct=dict(
                   trans_cm=min([s for s in TRANS_CM if per_c[f"trans_{s}"]["detected"] >= 0.9], default=None),
                   rot_deg=min([s for s in ROT_DEG if per_c[f"rot_{s}"]["detected"] >= 0.9], default=None)),
               note="healthy alarm threshold is the 99th percentile of healthy episode scores; with this many cameras "
                    "the 1-in-100 rate is an estimate, not a measured rate")
    print(f"drift study ({preds}): {len(cams)} held-out GT cameras with >= 5 confident frames, "
          f"{DIRECTIONS} random directions per size")
    print(f"  healthy per-frame residual (confident frames, {per_frame['frames']}): median {per_frame['median_px']:.1f} px, "
          f"p90 {per_frame['p90_px']:.1f}, p99 {per_frame['p99_px']:.1f}; over 50 px {100 * per_frame['over_50px']:.1f}%, "
          f"over 100 px {100 * per_frame['over_100px']:.1f}%")
    print(f"  healthy episode median residual: median {full['healthy_median_px']:.1f} px, "
          f"alarm threshold (99th pct) {full['healthy_p99_px']:.1f} px")
    for k, v in full["by_drift"].items():
        kind, s = k.split("_")
        unit = "cm" if kind == "trans" else "deg"
        print(f"  {('moved ' if kind == 'trans' else 'turned ')}{s} {unit}: median residual {v['median_score_px']:.1f} px, "
              f"caught {100 * v['detected']:.0f}%, frames needed for 90%: {frames_needed[k]}")
    print(f"  smallest drift caught in >= 90% of episodes: {res['smallest_detected_90pct']['trans_cm']} cm, "
          f"{res['smallest_detected_90pct']['rot_deg']} deg")
    print(f"  per camera, against its own first-half baseline ({per_camera['cameras']} cameras): healthy rise median "
          f"{per_camera['healthy_rise_median_px']:.1f} px, alarm at {per_camera['healthy_rise_p99_px']:.1f} px")
    for k, v in per_c.items():
        kind, s = k.split("_")
        unit = "cm" if kind == "trans" else "deg"
        print(f"    {('moved ' if kind == 'trans' else 'turned ')}{s} {unit}: median rise {v['median_rise_px']:.1f} px, "
              f"caught {100 * v['detected']:.0f}%")
    print(f"    smallest drift caught in >= 90%: {res['per_camera_smallest_90pct']['trans_cm']} cm, "
          f"{res['per_camera_smallest_90pct']['rot_deg']} deg")
    print(f"  wrote {fig_path}")
    return res


def overlays(preds: str, out_dir: str = os.path.join(DATA, "figures"), n: int = 4) -> list:
    """Predicted (magenta x) vs FK label (green ring) on n best, n median and n worst held-out in-view frames."""
    import cv2

    z, err = _load(preds)
    index = {(r["key"], r["serial"]): r["file"] for r in
             json.load(open(os.path.join(DATA, "fk_labels", "index.json"), encoding="utf-8"))["written"]}
    vis = np.flatnonzero(z["visible"].astype(bool))
    order = vis[np.argsort(err[vis])]

    def distinct(seq):
        """First n frames from n different cameras, so the pictures are not one camera four times."""
        out, seen = [], set()
        for r in seq:
            cam = (str(z["key"][r]), str(z["serial"][r]))
            if cam not in seen:
                seen.add(cam)
                out.append(r)
            if len(out) == n:
                break
        return np.array(out)

    mid = len(order) // 2
    around_mid = order[np.argsort(np.abs(np.arange(len(order)) - mid))]
    groups = {"best": distinct(order), "median": distinct(around_mid), "worst": distinct(order[::-1])}
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, rows in groups.items():
        tiles = []
        for r in rows:
            key, serial, k = str(z["key"][r]), str(z["serial"][r]), int(z["frame"][r])
            img = cv2.resize(np.load(index[(key, serial)])["frames"][k], (640, 360), interpolation=cv2.INTER_LINEAR)
            sc = 640 / ((z["scale"][r] if "scale" in z.files else 4.0) * 320)
            lu, lv = (z["label_full"][r] * sc).astype(int)
            pu, pv = (z["uv_full"][r] * sc).astype(int)
            cv2.circle(img, (int(lu), int(lv)), 9, (0, 255, 0), 2)
            cv2.drawMarker(img, (int(pu), int(pv)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(img, f"{serial} f{k}  error {err[r]:.1f} px", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        cv2.putText(sheet, "green ring: FK label   magenta x: detector", (8, sheet.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        path = os.path.join(out_dir, f"overlay_{name}.png")
        cv2.imwrite(path, sheet)
        written.append(path)
        print(f"  {name}: errors " + ", ".join(f"{err[r]:.1f}" for r in rows) + f" px -> {path}")
    return written


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.study")
    ap.add_argument("what", choices=("confidence", "drift", "overlays"))
    ap.add_argument("--preds", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="drift: heatmap peak threshold (from confidence)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.what == "confidence":
        res = confidence(a.preds)
    elif a.what == "overlays":
        res = overlays(a.preds)
    else:
        if a.threshold is None:
            raise SystemExit("--threshold is required for drift (run confidence first)")
        res = drift(a.preds, a.threshold)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
