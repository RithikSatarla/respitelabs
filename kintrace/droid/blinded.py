"""Blinded drift test on held-out DROID cameras. Rerun it yourself:

    python -m kintrace.droid.blinded make  [--n 40]   # picks cases with a fresh random seed, seals the answers
    python -m kintrace.droid.blinded run              # Kintrace decides from cases.json only (no answers read)
    python -m kintrace.droid.blinded score            # opens the sealed answers and scores

make   random test GT episodes that have an earlier healthy episode on the same camera; about a
       quarter healthy, the rest with a random camera bump: 0 to 5 cm in a random direction plus
       0 to 5 deg about a random axis. cases.json holds what Kintrace may see (episode, camera,
       the believed pose, the baseline episode). answers.json is sealed: its SHA-256 goes in
       cases.json and is checked before scoring.
run    decision: the per-camera alarm (alarm.py, single-keypoint rule 2, between sessions, gate and
       k from validation). Size: the three-keypoint fix (fixes.py) with the widened 90% interval.
score  caught, missed, false alarms, can't tell; size error; interval coverage.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import alarm, fixes
from .extrinsics import pose_error

DIR = os.path.join("data", "droid", "blinded")
CASES = os.path.join(DIR, "cases.json")
ANSWERS = os.path.join(DIR, "answers.json")
RUN = os.path.join(DIR, "kintrace_answers.json")


def _cams():
    alarm.KP3 = False
    single = {(c["key"], c["serial"]): c for c in alarm.load_cache("test")}
    kp3 = {(c["key"], c["serial"]): c for c in fixes.load_cache("test_kp3")}
    return single, kp3


def make(n=40):
    os.makedirs(DIR, exist_ok=True)
    seed = int.from_bytes(os.urandom(8), "little")
    rng = np.random.default_rng(seed)
    single, _ = _cams()
    by = {}
    for (key, serial) in single:
        by.setdefault(serial, []).append(key)
    pool = [(sorted(ks)[i], serial, sorted(ks)[i - 1]) for serial, ks in by.items() for i in range(1, len(ks))]
    picks = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    cases, answers = [], []
    for i, j in enumerate(picks):
        key, serial, base = pool[j]
        c = single[(key, serial)]
        healthy = rng.random() < 0.25
        if healthy:
            D, cm, deg = np.eye(4), 0.0, 0.0
        else:
            cm, deg = float(rng.uniform(0, 5)), float(rng.uniform(0, 5))
            D = alarm.perturb("trans", cm, rng) @ alarm.perturb("rot", deg, rng)
        T_bel = c["T_truth"] @ D
        e = pose_error(T_bel, c["T_truth"])
        cases.append(dict(case=i, episode=key, camera=serial, baseline_episode=base, believed_pose=T_bel.tolist()))
        answers.append(dict(case=i, healthy=healthy, size_cm=cm, size_deg=deg,
                            true_translation_mm=e["translation_mm"], true_rotation_deg=e["rotation_deg"]))
    blob = json.dumps(dict(seed=seed, answers=answers), indent=1).encode()
    open(ANSWERS, "wb").write(blob)
    json.dump(dict(answers_sha256=hashlib.sha256(blob).hexdigest(), cases=cases), open(CASES, "w"), indent=1)
    return len(cases)


def run():
    """Reads cases.json only."""
    cases = json.load(open(CASES))["cases"]
    single, kp3 = _cams()
    alarm.KP3 = False
    cal = json.load(open(alarm.CAL))
    k = cal["k_between"]["all"]["k"]
    widen = json.load(open(fixes.WIDEN))
    rng = np.random.default_rng(0)
    out = []
    for cs in cases:
        T_bel = np.array(cs["believed_pose"])
        c, b = single[(cs["episode"], cs["camera"])], single[(cs["baseline_episode"], cs["camera"])]
        kb, kc = np.flatnonzero(alarm.conf(b)), np.flatnonzero(alarm.conf(c))
        row = dict(case=cs["case"])
        base = alarm.resid(b, b["T_truth"], kb) if len(kb) >= 10 else None
        if base is None or len(kc) < 10 or not alarm.reliable(base, len(kb) / len(b["p"]), cal["gate"]):
            row["answer"] = "can't tell"
        else:
            row["z"] = alarm.z_score(base, alarm.resid(c, T_bel, kc))
            row["answer"] = "moved" if row["z"] > k else "no change"
        c3 = kp3[(cs["episode"], cs["camera"])]
        k3 = fixes.frames_ok(c3)
        T = fixes.fit(c3, k3, T_bel) if len(k3) >= 30 else None
        if T is not None:
            e = pose_error(T_bel, T)
            ci = fixes.bootstrap(c3, k3, T_bel, rng, n=60)
            row.update(size_mm=e["translation_mm"], size_deg=e["rotation_deg"])
            if ci:
                row["ci_mm"] = (max(0.0, ci["translation_mm"][0] - widen["translation_mm"]), ci["translation_mm"][1] + widen["translation_mm"])
                row["ci_deg"] = (max(0.0, ci["rotation_deg"][0] - widen["rotation_deg"]), ci["rotation_deg"][1] + widen["rotation_deg"])
        out.append(row)
    json.dump(out, open(RUN, "w"), indent=1)
    return out


def score():
    cases = json.load(open(CASES))
    blob = open(ANSWERS, "rb").read()
    assert hashlib.sha256(blob).hexdigest() == cases["answers_sha256"], "answers file changed after sealing"
    ans = {a["case"]: a for a in json.loads(blob)["answers"]}
    got = json.load(open(RUN))
    caught = missed = fa = tn = cant = 0
    by_size = {"0-1 cm": [0, 0], "1-2 cm": [0, 0], "2-5 cm": [0, 0]}
    err_mm, err_deg, cov_t, cov_r = [], [], [], []
    for g in got:
        a = ans[g["case"]]
        if g["answer"] == "can't tell":
            cant += 1
        elif a["healthy"]:
            fa += g["answer"] == "moved"
            tn += g["answer"] != "moved"
        else:
            hit = g["answer"] == "moved"
            caught += hit
            missed += not hit
            band = "0-1 cm" if a["size_cm"] < 1 else "1-2 cm" if a["size_cm"] < 2 else "2-5 cm"
            by_size[band][0] += hit
            by_size[band][1] += 1
        if "size_mm" in g:
            err_mm.append(abs(g["size_mm"] - a["true_translation_mm"]))
            err_deg.append(abs(g["size_deg"] - a["true_rotation_deg"]))
            if "ci_mm" in g:
                cov_t.append(g["ci_mm"][0] <= a["true_translation_mm"] <= g["ci_mm"][1])
                cov_r.append(g["ci_deg"][0] <= a["true_rotation_deg"] <= g["ci_deg"][1])
    res = dict(cases=len(got), healthy=sum(a["healthy"] for a in ans.values()), caught=caught, missed=missed,
               false_alarms=fa, true_negatives=tn, cant_tell=cant, caught_by_size=by_size,
               size_error_mm_median=float(np.median(err_mm)) if err_mm else None,
               size_error_deg_median=float(np.median(err_deg)) if err_deg else None,
               interval_coverage_translation=float(np.mean(cov_t)) if cov_t else None,
               interval_coverage_rotation=float(np.mean(cov_r)) if cov_r else None,
               answers_sha256=cases["answers_sha256"])
    json.dump(res, open(os.path.join(DIR, "score.json"), "w"), indent=1)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.blinded")
    ap.add_argument("what", choices=("make", "run", "score"))
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)
    if a.what == "make":
        print(f"made {make(a.n)} cases; answers sealed in {ANSWERS}")
    elif a.what == "run":
        print(f"Kintrace answered {len(run())} cases -> {RUN} (answers not read)")
    else:
        print(json.dumps(score(), indent=1))


if __name__ == "__main__":
    main()
