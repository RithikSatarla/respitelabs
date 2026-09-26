"""Benchmark: Kintrace vs a standard monitoring dashboard.

Every fault is sized so the robot misses picks by about the same amount
(~20 mm). A standard dashboard watches the usual health numbers (camera
tracking error, joint tracking error, pick success) and matches them to
known fault signatures. Kintrace tests which sensor agreements broke.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import sim
from .diagnose import _interp, diagnose, learn_baseline, _predict_cam
from .logio import Log

CLASSES = sim.FAULTS


def dashboard_features(log: Log) -> np.ndarray:
    """What a typical robot monitoring dashboard tracks."""
    cam = np.array(log.config["camera_extrinsic"])
    markers = np.array(log.config["markers"])
    sel = log.visible
    q = _interp(log.t_joint, log.q_meas, log.t_cam[sel])
    res = log.markers_cam[sel] - _predict_cam(q, np.zeros(6), cam, markers)
    vision_err = np.sqrt(np.mean(np.sum(res**2, -1))) * 1000  # mm
    track_err = np.sqrt(np.mean((log.q_meas - log.q_cmd) ** 2)) * 1000  # mrad
    fail = 1.0 - log.pick_ok.mean()
    return np.array([vision_err, track_err, fail])


class NearestSignature:
    """Standard anomaly scoring: z-score the health metrics, then match to
    the closest known fault signature."""

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.labels = sorted(set(y))
        self.cent = np.array([Z[np.array(y) == c].mean(0) for c in self.labels])
        return self

    def predict(self, x):
        z = (x - self.mu) / self.sd
        return self.labels[int(np.argmin(np.linalg.norm(self.cent - z, axis=1)))]


def _size_error(truth: dict, finding) -> float | None:
    """Absolute error (mm or deg or ms) between true and diagnosed size."""
    p = finding.params
    t = truth["type"]
    if t == "camera_moved" and "translation_mm" in p:
        return float(np.linalg.norm(np.array(p["translation_mm"]) - np.array(truth["translation"]) * 1000))
    if t == "encoder_bias" and "bias_deg" in p:
        return abs(p["bias_deg"] - np.rad2deg(truth["bias"]))
    if t == "latency" and "delay_ms" in p:
        return abs(p["delay_ms"] - p["normal_ms"] - truth["extra_s"] * 1000)
    if t == "tcp_config" and "tcp_offset_now_mm" in p:
        return float(np.linalg.norm(np.array(p["tcp_offset_now_mm"]) - np.array(p["tcp_offset_good_mm"]) - np.array(truth["delta"]) * 1000))
    if t == "tool_bent" and "tcp_measured_mm" in p:
        return float(np.linalg.norm(np.array(p["tcp_measured_mm"]) - np.array(p["tcp_expected_mm"]) - np.array(truth["delta"]) * 1000))
    return None


def run(n_per_class: int = 20, seed: int = 0, progress=print) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    baseline = learn_baseline(sim.simulate({"type": "none"}, seed=10_000, with_probe=False))

    # train the dashboard's signatures on separate runs
    Xtr, ytr = [], []
    for c in CLASSES:
        for i in range(max(n_per_class // 2, 5)):
            log = sim.simulate(sim.random_fault(c, rng), seed=int(rng.integers(1e9)), with_probe=False)
            Xtr.append(dashboard_features(log))
            ytr.append(c)
    dash = NearestSignature().fit(np.array(Xtr), ytr)
    progress(f"  trained dashboard signatures on {len(ytr)} runs")

    rows = []
    for c in CLASSES:
        for i in range(n_per_class):
            fault = sim.random_fault(c, rng)
            log = sim.simulate(fault, seed=int(rng.integers(1e9)))
            passive = diagnose(log, baseline, run_probe=False)
            probed = diagnose(log, baseline, run_probe=True)
            rows.append(dict(
                truth=c,
                dashboard=dash.predict(dashboard_features(log)),
                passive=passive.primary,
                kintrace=probed.primary,
                probe_used=probed.probe_used,
                size_err=_size_error(fault, probed.findings[0]) if probed.findings else None,
                missed_picks=int((~log.pick_ok).sum()),
            ))
        progress(f"  {c:<14} done ({time.time()-t0:.0f}s)")

    # two faults at once: does Kintrace report both?
    multi = []
    for i in range(n_per_class // 2 or 1):
        a = sim.random_fault("latency", rng)
        b = sim.random_fault(str(rng.choice(["camera_moved", "encoder_bias", "tcp_config"])), rng)
        log = sim.simulate({"type": "multi", "faults": [a, b]}, seed=int(rng.integers(1e9)), with_probe=False)
        found = {f.fault for f in diagnose(log, baseline).findings}
        multi.append({"latency", b["type"]} <= found)

    def acc(key, cls=None):
        r = [x for x in rows if cls is None or x["truth"] == cls]
        return float(np.mean([x[key] == x["truth"] for x in r]))

    summary = dict(
        n_runs=len(rows),
        dashboard_acc=acc("dashboard"),
        kintrace_passive_acc=acc("passive"),
        kintrace_acc=acc("kintrace"),
        per_class={c: dict(dashboard=acc("dashboard", c), passive=acc("passive", c), kintrace=acc("kintrace", c)) for c in CLASSES},
        size_err_median={c: float(np.median([x["size_err"] for x in rows if x["truth"] == c and x["size_err"] is not None]))
                         for c in CLASSES if c != "none"},
        two_fault_both_found=float(np.mean(multi)),
        dashboard_confusion=[[sum(1 for x in rows if x["truth"] == t and x["dashboard"] == p) for p in CLASSES] for t in CLASSES],
        kintrace_confusion=[[sum(1 for x in rows if x["truth"] == t and x["kintrace"] == p) for p in CLASSES] for t in CLASSES],
        classes=CLASSES,
        seconds=time.time() - t0,
    )
    return dict(summary=summary, rows=rows)


def chart(summary: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import GRID, INK, INK_2, MUTED, SERIES_1, SURFACE, AXIS

    label = {"none": "Healthy", "camera_moved": "Camera moved", "tool_bent": "Tool bent",
             "tcp_config": "Tool offset edited", "encoder_bias": "Encoder drift", "latency": "Latency"}
    cls = summary["classes"]
    dash = [summary["per_class"][c]["dashboard"] * 100 for c in cls]
    kin = [summary["per_class"][c]["kintrace"] * 100 for c in cls]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(cls))
    ax.bar(x - 0.19, dash, 0.36, color=MUTED, label=f"Standard dashboard ({summary['dashboard_acc']*100:.0f}% overall)")
    ax.bar(x + 0.19, kin, 0.36, color=SERIES_1, label=f"Kintrace ({summary['kintrace_acc']*100:.0f}% overall)")
    for xi, d, k in zip(x, dash, kin):
        ax.text(xi - 0.19, d + 1.5, f"{d:.0f}", ha="center", fontsize=8, color=INK_2)
        ax.text(xi + 0.19, k + 1.5, f"{k:.0f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [label[c] for c in cls], color=INK)
    ax.set_ylim(0, 125)
    ax.set_ylabel("correct cause (%)", color=INK_2)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.04), ncol=2, labelcolor=INK_2)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    fig.text(0.02, 0.955, "Same symptom, different causes: which tool names the right one?", color=INK,
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.885, f"{summary['n_runs']} simulated runs. Every fault sized to cause a ~20 mm pick miss.", color=INK_2,
             fontsize=10.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    print(json.dumps(out["summary"], indent=2))
