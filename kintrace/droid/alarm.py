"""Per-camera drift alarm with a detector-reliability gate (rules in PROGRESS.md, set on validation).

Residual per frame: distance between the detected fingertip and the FK fingertip projected through
the believed camera pose. Frames count if the visibility head says >= 0.5.

gate      reliable if the baseline's median residual <= X px and at least Y of its frames are
          confident; otherwise "can't tell (detector unreliable on this view)".
alarm     z = (median now - median at baseline) / sqrt(SE_b^2 + SE_n^2), SE = 1.2533 sigma / sqrt(n / 5),
          sigma = 1.4826 MAD of the baseline residuals. Alarm if z > k (k from healthy validation
          decisions at 1 in 100).

Run: python -m kintrace.droid.alarm calibrate   (validation only)
     python -m kintrace.droid.alarm test        (test, once)
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from .check2d import fk_tips, project
from .phase2 import load_cache as _load_cache

KP3 = False                                    # three-keypoint mode (multikp.py caches)
KP3_OFFSETS = (0.0, 0.11, 0.17)


def load_cache(group):
    return _load_cache(group + ("_kp3" if KP3 else ""))

DATA = "data/droid"
OUT = os.path.join(DATA, "phase2")
CAL = os.path.join(OUT, "alarm_calibration.json")


def _cal_path():
    return CAL.replace(".json", "_kp3.json") if KP3 else CAL
MIN_VISIBLE = 0.5
BLOCK = 5
FA = 0.01
WINDOWS = (5, 10, 20, 50, 100)
X_GRID = (8, 10, 12, 15, 20, 30, 40, 60)   # 8, 10, 12 added 18:15 on validation: nothing in the first grid passed with three keypoints
Y_GRID = (0.3, 0.5, 0.7)
HALF_CHANGE_MAX = 20.0
TRANS_CM = (0.5, 1.0, 2.0, 5.0)
ROT_DEG = (0.5, 1.0, 2.0, 5.0)
DIRECTIONS = 20


# --------------------------------------------------------------------------
def conf(c):
    if KP3:
        return (c["p3"] >= MIN_VISIBLE).sum(1) >= 2
    return c["p"] >= MIN_VISIBLE


def resid(c, T, idx):
    """Residuals (px) on frame indices idx under camera pose T (three-keypoint mode: mean over confident keypoints)."""
    if not KP3:
        return np.linalg.norm(project(T, c["K"], fk_tips(c["q"][idx])) - c["uv"][idx], axis=1)
    d = np.stack([np.linalg.norm(project(T, c["K"], fk_tips(c["q"][idx], tip=np.array([0, 0, o]))) - c["uv3"][idx, j], axis=1)
                  for j, o in enumerate(KP3_OFFSETS)], 1)
    m = c["p3"][idx] >= MIN_VISIBLE
    return (d * m).sum(1) / np.maximum(m.sum(1), 1)


def sigma(r):
    return 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6


def z_score(base, now):
    s = sigma(base)
    se = lambda n: 1.2533 * s / np.sqrt(max(n / BLOCK, 1.0))  # noqa: E731
    return float((np.median(now) - np.median(base)) / np.sqrt(se(len(base)) ** 2 + se(len(now)) ** 2))


def reliable(base, frac, gate):
    return bool(np.median(base) <= gate["X"] and frac >= gate["Y"])


def perturb(kind, size, rng):
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    D = np.eye(4)
    if kind == "trans":
        D[:3, 3] = v * size / 100
    else:
        D[:3, :3] = Rotation.from_rotvec(v * np.deg2rad(size)).as_matrix()
    return D


def _splits(k):
    """Baseline / test index pairs within one episode: split at 1/3, 1/2, 2/3, both directions."""
    out = []
    for f in (1 / 3, 1 / 2, 2 / 3):
        h = int(len(k) * f)
        if h >= 10 and len(k) - h >= 10:
            out += [(k[:h], k[h:]), (k[h:], k[:h])]
    return out


def _windows(k, n):
    """Baseline = first half; every non-overlapping window of n frames in the second half."""
    h = len(k) // 2
    if h < 10:
        return []
    rest = k[h:]
    return [(k[:h], rest[i:i + n]) for i in range(0, len(rest) - n + 1, n)]


def _pairs(cams, ordered=True):
    by = {}
    for c in cams:
        by.setdefault(c["serial"], []).append(c)
    out = []
    for cs in by.values():
        cs = sorted(cs, key=lambda c: c["key"].split("+")[-1])  # ids are LAB+hash+date: sort by the date part
        for i in range(len(cs)):
            for j in range(len(cs)):
                if i != j and (not ordered or i < j):
                    out.append((cs[i], cs[j]))
    return out


# --------------------------------------------------------------------------
def calibrate() -> dict:
    val = load_cache("val")
    # gate: pick X, Y on validation
    rows = []
    for c in val:
        k = np.flatnonzero(conf(c))
        frac = len(k) / len(c["p"])
        if len(k) < 20:
            rows.append((np.inf, frac, np.inf))
            continue
        h = len(k) // 2
        base = resid(c, c["T_truth"], k[:h])
        rest = resid(c, c["T_truth"], k[h:])
        rows.append((float(np.median(base)), frac, abs(float(np.median(rest) - np.median(base)))))
    rows = np.array(rows)
    best = None
    for X in X_GRID:
        for Y in Y_GRID:
            keep = (rows[:, 0] <= X) & (rows[:, 1] >= Y)
            if keep.sum() < 3:
                continue
            p99 = float(np.percentile(rows[keep, 2], 99))
            if p99 > HALF_CHANGE_MAX:
                continue
            cand = (int(keep.sum()), -X, Y, p99)
            if best is None or cand[:3] > best[:3]:
                best = cand
    gate = dict(X=-best[1], Y=best[2], val_cameras_kept=best[0], val_cameras=len(val), half_change_p99=best[3])

    # within-session k per window size
    def within_z(c, n):
        k = np.flatnonzero(conf(c))
        if len(k) < 20:
            return []
        out = []
        pairs = _splits(k) if n is None else _windows(k, n)
        for b, t in pairs:
            base = resid(c, c["T_truth"], b)
            if not reliable(base, len(k) / len(c["p"]), gate):
                continue
            now = resid(c, c["T_truth"], t)
            out.append((z_score(base, now), float(np.median(now) - np.median(base))))
        return out

    k_within = {}
    for n in list(WINDOWS) + [None]:
        zs = [z for c in val for z in within_z(c, n)]
        if len(zs) >= 20:
            zz, dd = np.array([z for z, _ in zs]), np.array([d for _, d in zs])
            k_within["all" if n is None else str(n)] = dict(k=float(np.percentile(zz, 100 * (1 - FA))),
                                                             d_px=float(np.percentile(dd, 100 * (1 - FA))), decisions=len(zs))

    # between-session k: every ordered pair of episodes on the same validation camera
    k_between = {}
    for n in list(WINDOWS) + [None]:
        zs = []
        for a, b in _pairs(val, ordered=False):
            ka, kb = np.flatnonzero(conf(a)), np.flatnonzero(conf(b))
            if len(ka) < 10 or len(kb) < (n or 10):
                continue
            base = resid(a, a["T_truth"], ka)
            if not reliable(base, len(ka) / len(a["p"]), gate):
                continue
            now = resid(b, b["T_truth"], kb[:n] if n else kb)
            zs.append((z_score(base, now), float(np.median(now) - np.median(base))))
        if len(zs) >= 20:
            zz, dd = np.array([z for z, _ in zs]), np.array([d for _, d in zs])
            k_between["all" if n is None else str(n)] = dict(k=float(np.percentile(zz, 100 * (1 - FA))),
                                                              d_px=float(np.percentile(dd, 100 * (1 - FA))), decisions=len(zs))
    cal = dict(gate=gate, k_within=k_within, k_between=k_between, min_visible=MIN_VISIBLE, block=BLOCK, false_alarm=FA)
    json.dump(cal, open(_cal_path(), "w"), indent=1)
    return cal


# --------------------------------------------------------------------------
def test() -> dict:
    cal = json.load(open(_cal_path()))
    gate = cal["gate"]
    test_cams, pred, val = load_cache("test"), load_cache("pred"), load_cache("val")
    rng = np.random.default_rng(0)
    res = dict(gate=gate)

    # within a session: baseline = first half, the bump applies to the second half
    within = {}
    for n in list(WINDOWS) + [None]:
        key = "all" if n is None else str(n)
        if key not in cal["k_within"]:
            continue
        kk, dd = cal["k_within"][key]["k"], cal["k_within"][key]["d_px"]
        cant, healthy, caught = 0, [], {f"{kd}_{s}": [] for kd, ss in (("trans", TRANS_CM), ("rot", ROT_DEG)) for s in ss}
        healthy_b, caught_b = [], {a: [] for a in caught}
        for c in test_cams:
            k = np.flatnonzero(conf(c))
            h = len(k) // 2
            if h < 10 or (n and len(k) - h < n):
                cant += 1
                continue
            b, t = k[:h], (k[h:h + n] if n else k[h:])
            base = resid(c, c["T_truth"], b)
            if not reliable(base, len(k) / len(c["p"]), gate):
                cant += 1
                continue
            now = resid(c, c["T_truth"], t)
            healthy.append(z_score(base, now) > kk)
            healthy_b.append(np.median(now) - np.median(base) > dd)
            for kd, ss in (("trans", TRANS_CM), ("rot", ROT_DEG)):
                for s in ss:
                    for _ in range(DIRECTIONS):
                        now = resid(c, c["T_truth"] @ perturb(kd, s, rng), t)
                        caught[f"{kd}_{s}"].append(z_score(base, now) > kk)
                        caught_b[f"{kd}_{s}"].append(np.median(now) - np.median(base) > dd)
        within[key] = dict(k=kk, d_px=dd, decided=len(healthy), cant_tell=cant,
                           rule2=dict(false_alarms=int(sum(healthy)), caught={a: float(np.mean(v)) if v else None for a, v in caught.items()}),
                           rule2b=dict(false_alarms=int(sum(healthy_b)), caught={a: float(np.mean(v)) if v else None for a, v in caught_b.items()}))
    res["within_session"] = within

    # between sessions: earlier episode on the same camera = baseline
    between = {}
    for n in list(WINDOWS) + [None]:
        key = "all" if n is None else str(n)
        if key not in cal["k_between"]:
            continue
        kk, dd = cal["k_between"][key]["k"], cal["k_between"][key]["d_px"]
        cant, healthy, deltas = 0, [], []
        caught = {f"{kd}_{s}": [] for kd, ss in (("trans", TRANS_CM), ("rot", ROT_DEG)) for s in ss}
        healthy_b, caught_b = [], {a: [] for a in caught}
        for a, b in _pairs(test_cams, ordered=True):
            ka, kb = np.flatnonzero(conf(a)), np.flatnonzero(conf(b))
            if len(ka) < 10 or len(kb) < (n or 10):
                cant += 1
                continue
            base = resid(a, a["T_truth"], ka)
            if not reliable(base, len(ka) / len(a["p"]), gate):
                cant += 1
                continue
            tb = kb[:n] if n else kb
            now = resid(b, b["T_truth"], tb)
            deltas.append(float(np.median(now) - np.median(base)))
            healthy.append(z_score(base, now) > kk)
            healthy_b.append(np.median(now) - np.median(base) > dd)
            for kd, ss in (("trans", TRANS_CM), ("rot", ROT_DEG)):
                for s in ss:
                    for _ in range(DIRECTIONS // 4):
                        nw = resid(b, b["T_truth"] @ perturb(kd, s, rng), tb)
                        caught[f"{kd}_{s}"].append(z_score(base, nw) > kk)
                        caught_b[f"{kd}_{s}"].append(np.median(nw) - np.median(base) > dd)
        d = np.abs(np.array(deltas)) if deltas else np.zeros(0)
        between[key] = dict(k=kk, d_px=dd, decided=len(healthy), cant_tell=cant,
                            noise_floor_px={str(q): float(np.percentile(d, q)) for q in (50, 90, 99)} if len(d) else None,
                            rule2=dict(false_alarms=int(sum(healthy)), caught={a: float(np.mean(v)) if v else None for a, v in caught.items()}),
                            rule2b=dict(false_alarms=int(sum(healthy_b)), caught={a: float(np.mean(v)) if v else None for a, v in caught_b.items()}))
    res["between_sessions"] = between

    # real drift: Pred episode vs a healthy GT episode on the same camera (val/test serials only)
    kk, dd = cal["k_between"]["all"]["k"], cal["k_between"]["all"]["d_px"]
    gt = val + test_cams
    by = {}
    for c in gt:
        by.setdefault(c["serial"], []).append(c)
    rows = []
    for c in pred:
        if c["serial"] not in by:
            rows.append(dict(key=c["key"], serial=c["serial"], answer="not scored (camera seen in training)"))
            continue
        cands = sorted(by[c["serial"]], key=lambda g: g["key"])
        earlier = [g for g in cands if g["key"] < c["key"]]
        g = earlier[-1] if earlier else cands[0]
        kg, kp = np.flatnonzero(conf(g)), np.flatnonzero(conf(c))
        if len(kg) < 10 or len(kp) < 10:
            rows.append(dict(key=c["key"], serial=c["serial"], answer="can't tell (too few confident frames)"))
            continue
        base = resid(g, g["T_truth"], kg)
        if not reliable(base, len(kg) / len(g["p"]), gate):
            rows.append(dict(key=c["key"], serial=c["serial"], answer="can't tell (detector unreliable on this view)"))
            continue
        now = resid(c, c["T_belief"], kp)
        z = z_score(base, now)
        delta = float(np.median(now) - np.median(base))
        rows.append(dict(key=c["key"], serial=c["serial"], baseline=g["key"], baseline_earlier=bool(earlier), z=z, delta_px=delta,
                         answer="moved" if z > kk else "no change", answer_2b="moved" if delta > dd else "no change"))
    decided = [r for r in rows if r["answer"] in ("moved", "no change")]
    tp = sum(r["answer"] == "moved" for r in decided)
    tp_b = sum(r["answer_2b"] == "moved" for r in decided)
    fp = between["all"]["rule2"]["false_alarms"]
    fp_b = between["all"]["rule2b"]["false_alarms"]
    res["real_drift"] = dict(
        pred_cameras=len(pred), pred_on_untrained_serials=sum(not r["answer"].startswith("not scored") for r in rows),
        decided=len(decided), moved=tp, recall_of_decided=tp / max(len(decided), 1),
        cant_tell=sum(r["answer"].startswith("can't tell") for r in rows),
        healthy_gt_pairs=between["all"]["decided"], false_alarms_gt=fp, precision=tp / max(tp + fp, 1),
        rule2b=dict(moved=tp_b, recall_of_decided=tp_b / max(len(decided), 1), false_alarms_gt=fp_b, precision=tp_b / max(tp_b + fp_b, 1)),
        rows=rows)
    json.dump(res, open(os.path.join(OUT, "alarm_test_kp3.json" if KP3 else "alarm_test.json"), "w"), indent=1, default=float)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.alarm")
    ap.add_argument("what", choices=("calibrate", "test"))
    ap.add_argument("--kp3", action="store_true", help="three-keypoint detector caches")
    a = ap.parse_args(argv)
    global KP3
    KP3 = a.kp3
    r = calibrate() if a.what == "calibrate" else test()
    print(json.dumps({k: v for k, v in r.items() if k != "real_drift"} | (
        {"real_drift": {k: v for k, v in r["real_drift"].items() if k != "rows"}} if "real_drift" in r else {}), indent=1))


if __name__ == "__main__":
    main()
