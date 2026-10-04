"""Timing and joint-offset checks on public LeRobot recordings from different robots.

No camera ground truth is used: these sets have no calibrated fixed cameras, so the camera
checks are not tested here. What runs is what needs only commanded and measured joints:

  latency   kintrace.diagnose.estimate_delay (the same estimator as the DROID latency test).
            Healthy: each episode's delay vs the median of the robot's other episodes; more
            than TOL_MS apart is a false alarm. Injected: the measured stream is shifted by
            20, 50, 100 ms; caught if the estimated rise is at least TOL_MS, sized by the error.
  offset    (new check, written for this test) each joint's median of measured minus commanded
            over moving frames, vs the robot's other episodes. Healthy: more than the threshold
            apart is a false alarm. Injected: 2 and 5 deg added to one joint's measured angle;
            caught if that joint is flagged and is the largest change.

Run: python -m kintrace.crossrobot
"""
from __future__ import annotations

import glob
import json
import os
from types import SimpleNamespace

import numpy as np

from .diagnose import estimate_delay

ROOT = "data/lerobot"
TOL_MS = 4.0
OFFSET_TOL_DEG = 1.0
DELAYS_MS = (20, 50, 100)
OFFSETS_DEG = (2.0, 5.0)
SEARCH_S = 1.0

# dataset -> (robot, body, [(group, state column(s), action column(s), joint indices, units)])
ROBOTS = {
    "lerobot__aloha_static_coffee": ("ALOHA (bimanual)", "two 6-joint arms", [
        ("arms", "observation.state", "action", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], "rad")]),
    "lerobot__koch_pick_place_5_lego": ("Koch v1.1", "5-joint arm", [
        ("arm", "observation.state", "action", [0, 1, 2, 3, 4], "deg")]),
    "lerobot__svla_so100_pickplace": ("SO-100", "5-joint arm", [
        ("arm", "observation.state", "action", [0, 1, 2, 3, 4], "deg")]),
    "unitreerobotics__G1_Dex1_Stack_Block": ("Unitree G1 (humanoid)", "two 7-joint arms", [
        ("arms", ["observation.left_arm", "observation.right_arm"],  # data files differ from info.json names
         ["action.left_arm", "action.right_arm"], list(range(14)), "rad")]),
    "QianGroup__lekiwi_pick_sponge": ("LeKiwi (mobile manipulator)", "SO-101 arm on a 3-wheel base", [
        ("arm", "observation.state", "action", [0, 2, 3, 4, 5], "deg"),
        ("base wheels", "observation.state", "action", [6, 7, 8], "wheel")]),
}


def _cols(df, names):
    names = [names] if isinstance(names, str) else names
    return np.concatenate([np.stack(df[n].to_numpy()) for n in names], 1).astype(float)


def episodes(name: str) -> list:
    import pandas as pd  # the "readers" extra

    files = sorted(glob.glob(os.path.join(ROOT, name, "data", "**", "*.parquet"), recursive=True))
    out = []
    for f in files:
        df = pd.read_parquet(f)
        for e, g in df.groupby("episode_index"):
            out.append(g.sort_values("frame_index").reset_index(drop=True))
    return out[:10]


def _log(t, cmd, meas):
    return SimpleNamespace(t_joint=t, q_cmd=cmd, q_meas=meas)


def _shift(t, meas, d):
    return np.stack([np.interp(t - d, t, meas[:, j]) for j in range(meas.shape[1])], 1)


def _offsets(cmd, meas, scale):
    moving = np.linalg.norm(np.gradient(cmd, axis=0), axis=1) > 1e-6
    sel = moving if moving.sum() > 20 else np.ones(len(cmd), bool)
    return np.median((meas - cmd)[sel], 0) * scale


def run_group(eps, state, action, idx, units):
    data = []
    for g in eps:
        t = g["timestamp"].to_numpy(float)
        t = t - t[0]
        data.append((t, _cols(g, action)[:, idx], _cols(g, state)[:, idx]))
    tau = np.array([estimate_delay(_log(t, c, m), SEARCH_S) for t, c, m in data]) * 1000
    healthy_fa = 0
    for i in range(len(tau)):
        ref = np.median(np.delete(tau, i))
        healthy_fa += abs(tau[i] - ref) > TOL_MS
    lat = {}
    for d in DELAYS_MS:
        caught, err = 0, []
        for (t, c, m), t0 in zip(data, tau):
            est = estimate_delay(_log(t, c, _shift(t, m, d / 1000)), SEARCH_S) * 1000 - t0
            caught += est >= TOL_MS
            err.append(abs(est - d))
        lat[d] = (int(caught), len(data), float(np.median(err)))
    res = dict(episodes=len(data), delay_ms=dict(median=float(np.median(tau)), min=float(tau.min()), max=float(tau.max())),
               latency_false_alarms=int(healthy_fa), latency=lat)
    if units in ("deg", "rad"):
        scale = 1.0 if units == "deg" else np.degrees(1.0)
        off = np.array([_offsets(c, m, scale) for _, c, m in data])
        fa = 0
        for i in range(len(off)):
            ref = np.median(np.delete(off, i, 0), 0)
            fa += (np.abs(off[i] - ref) > OFFSET_TOL_DEG).any()
        rng = np.random.default_rng(0)
        inj = {}
        for o in OFFSETS_DEG:
            caught = sized = 0
            errs = []
            for i, (t, c, m) in enumerate(data):
                j = int(rng.integers(m.shape[1]))
                m2 = m.copy()
                m2[:, j] += o / scale
                ref = np.median(np.delete(off, i, 0), 0)
                dlt = _offsets(c, m2, scale) - ref
                hit = abs(dlt[j]) > OFFSET_TOL_DEG and int(np.argmax(np.abs(dlt))) == j
                caught += hit
                errs.append(abs(dlt[j] - o))
            inj[o] = (int(caught), len(data), float(np.median(errs)))
        res.update(offset_false_alarms=int(fa), offset=inj)
    return res


def main():
    table = {}
    for name, (robot, body, groups) in ROBOTS.items():
        eps = episodes(name)
        if len(eps) < 3:
            print(f"{robot}: only {len(eps)} episodes, skipped")
            continue
        for gname, state, action, idx, units in groups:
            try:
                r = run_group(eps, state, action, idx, units)
            except Exception as e:
                print(f"{robot} {gname}: failed: {e}")
                continue
            table[f"{robot} | {gname}"] = dict(r, dataset=name.replace("__", "/"), body=body)
            lat = "  ".join(f"{d} ms {c}/{n} (err {e:.1f})" for d, (c, n, e) in r["latency"].items())
            line = (f"{robot:28s} {gname:12s} eps {r['episodes']:2d}  delay {r['delay_ms']['median']:6.1f} ms "
                    f"[{r['delay_ms']['min']:.0f}..{r['delay_ms']['max']:.0f}]  healthy latency alarms {r['latency_false_alarms']}/{r['episodes']}  "
                    f"injected latency: {lat}")
            if "offset" in r:
                off = "  ".join(f"{o:g} deg {c}/{n} (err {e:.2f})" for o, (c, n, e) in r["offset"].items())
                line += f"\n{'':42s}healthy offset alarms {r['offset_false_alarms']}/{r['episodes']}  injected offset: {off}"
            print(line)
    os.makedirs(os.path.join(ROOT), exist_ok=True)
    json.dump(table, open(os.path.join(ROOT, "crossrobot.json"), "w"), indent=1, default=float)
    print(f"wrote {os.path.join(ROOT, 'crossrobot.json')}")


if __name__ == "__main__":
    main()
