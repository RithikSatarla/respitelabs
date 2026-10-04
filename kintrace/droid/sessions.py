"""Real drift over time: track each camera's pose session by session (rules in PROGRESS.md).

Per serial (validation and test serials, never trained on), sessions in date order (DROID ids carry
the date and time). Kintrace's pose per session = the three-keypoint fix (fixes.fit) started from
that session's configured pose. DROID's pose per session = its corrected pose (GT: kept, Pred:
re-solved). Kintrace flags a session if its pose moved from the previous one by more than the
threshold set on validation; DROID says moved if its pose changed by >= 10 mm or >= 1 deg.

Run: python -m kintrace.droid.sessions
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import fixes
from .experiments import load_split
from .extrinsics import pose_error
from .phase2 import OUT

FIG = os.path.join("data", "droid", "figures", "final")


def sessions(serials):
    cams = [c for g in ("val_kp3", "test_kp3", "pred_kp3") for c in fixes.load_cache(g) if c["serial"] in serials]
    by = {}
    for c in cams:
        k = fixes.frames_ok(c)
        if len(k) < 30:
            continue
        T = fixes.fit(c, k, c["T_belief"])
        if T is None:
            continue
        by.setdefault(c["serial"], []).append(dict(key=c["key"], T_fit=T, T_droid=c["T_truth"]))
    for s in by:
        by[s].sort(key=lambda r: r["key"].split("+")[-1])   # ids are LAB+hash+date: sort by the date part
    return by


def steps(seq):
    out = []
    for a, b in zip(seq[:-1], seq[1:]):
        k = pose_error(a["T_fit"], b["T_fit"])
        d = pose_error(a["T_droid"], b["T_droid"])
        out.append(dict(prev=a["key"], key=b["key"], kintrace_mm=k["translation_mm"], kintrace_deg=k["rotation_deg"],
                        droid_mm=d["translation_mm"], droid_deg=d["rotation_deg"],
                        droid_moved=bool(d["translation_mm"] >= 10 or d["rotation_deg"] >= 1)))
    return out


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = load_split()
    val, test = sessions(set(sp["val"])), sessions(set(sp["test"]))
    still = [s for seq in val.values() for s in steps(seq) if not s["droid_moved"]]
    thr_mm = float(np.percentile([s["kintrace_mm"] for s in still], 99))
    thr_deg = float(np.percentile([s["kintrace_deg"] for s in still], 99))
    rows = []
    for serial, seq in test.items():
        for s in steps(seq):
            s["serial"] = serial
            s["kintrace_moved"] = bool(s["kintrace_mm"] > thr_mm or s["kintrace_deg"] > thr_deg)
            rows.append(s)
    agree = sum(r["kintrace_moved"] == r["droid_moved"] for r in rows)
    tp = sum(r["kintrace_moved"] and r["droid_moved"] for r in rows)
    miss = sum(not r["kintrace_moved"] and r["droid_moved"] for r in rows)
    false = sum(r["kintrace_moved"] and not r["droid_moved"] for r in rows)
    track = [pose_error(r["T_fit"], r["T_droid"]) for seq in test.values() for r in seq]
    res = dict(threshold=dict(mm=thr_mm, deg=thr_deg, from_val_pairs=len(still)), test_serials=len(test),
               sessions=sum(len(v) for v in test.values()),
               tracking_error_mm=dict(median=float(np.median([t["translation_mm"] for t in track])),
                                      p90=float(np.percentile([t["translation_mm"] for t in track], 90))),
               tracking_error_deg=dict(median=float(np.median([t["rotation_deg"] for t in track])),
                                       p90=float(np.percentile([t["rotation_deg"] for t in track], 90))),
               session_pairs=len(rows), droid_moved=sum(r["droid_moved"] for r in rows), agree=agree,
               both_moved=tp, missed=miss, false_flags=false, rows=rows)
    json.dump(res, open(os.path.join(OUT, "sessions.json"), "w"), indent=1)
    # one plot per camera for the 3 serials with the most sessions
    os.makedirs(FIG, exist_ok=True)
    top = sorted(test, key=lambda s: -len(test[s]))[:3]
    fig, ax = plt.subplots(len(top), 1, figsize=(10, 3.2 * len(top)), squeeze=False)
    for i, serial in enumerate(top):
        seq = test[serial]
        ref_f, ref_d = seq[0]["T_fit"], seq[0]["T_droid"]
        kf = [np.linalg.norm(r["T_fit"][:3, 3] - ref_f[:3, 3]) * 1000 for r in seq]
        kd = [np.linalg.norm(r["T_droid"][:3, 3] - ref_d[:3, 3]) * 1000 for r in seq]
        x = np.arange(len(seq))
        ax[i, 0].plot(x, kd, "s-", label="DROID's corrected pose")
        ax[i, 0].plot(x, kf, "o-", label="Kintrace's fitted pose")
        flags = [j + 1 for j, s in enumerate(steps(seq)) if s["kintrace_mm"] > thr_mm or s["kintrace_deg"] > thr_deg]
        dm = [j + 1 for j, s in enumerate(steps(seq)) if s["droid_moved"]]
        ax[i, 0].plot(flags, [kf[j] for j in flags], "rx", ms=10, label="Kintrace flags a move")
        ax[i, 0].plot(dm, [kd[j] for j in dm], "k^", mfc="none", ms=10, label="DROID's pose changed")
        ax[i, 0].set_ylabel("camera position vs first session (mm)")
        ax[i, 0].set_title(f"camera {serial}: {len(seq)} sessions in date order")
        ax[i, 0].set_xticks(x)
        ax[i, 0].set_xticklabels([r["key"].split("+")[-1][:10] for r in seq], rotation=60, fontsize=6)
        ax[i, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sessions_top3.png"), dpi=110)
    return res


def main(argv=None):
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
