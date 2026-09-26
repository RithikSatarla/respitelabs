"""Human-readable output: terminal report and a one-page chart."""
from __future__ import annotations

import numpy as np

from . import robot
from .diagnose import Diagnosis

# reference palette (validated): categorical slots + reserved status colors
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

HYP_LABEL = {"none": "No change", "camera_moved": "Camera moved"}
HYP_LABEL.update({f"encoder_bias:{j}": f"J{j+1} encoder ({n})" for j, n in enumerate(robot.JOINT_NAMES)})


def text_report(d: Diagnosis, name: str = "log") -> str:
    out = [f"KINTRACE  diagnosis of {name}", "=" * 60, "Sensor agreement checks"]
    for c in d.checks:
        mark = "ok  " if c.agrees else "FAIL"
        out.append(f"  [{mark}] {c.name:<26} {c.detail}")
    out.append("")
    if not d.findings:
        out.append("Verdict: no fault. Robot matches its healthy baseline.")
    for i, f in enumerate(d.findings):
        head = "Verdict" if i == 0 else "Also"
        conf = ("   (direct measurement)" if f.confidence >= 1.0 else
                f"   confidence {f.confidence*100:.0f}%" if f.confidence > 0 else "")
        out.append(f"{head}: {f.label.upper()}{conf}")
        out.append(f"  size : {f.size}")
        out.append(f"  fix  : {f.fix}")
    if d.probe_used:
        out.append("  (resolved with the 1.2 s probe motion)")
    if not d.findings or d.next_step != d.findings[0].fix:
        out.append("")
        out.append(f"Next step: {d.next_step}")
    return "\n".join(out)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart(d: Diagnosis, path: str, name: str = "log") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.patch.set_facecolor(SURFACE)

    # --- left: which agreements broke ------------------------------------
    _style(a1)
    names = [c.name for c in d.checks][::-1]
    scores = np.clip([c.score for c in d.checks][::-1], 0.06, None)
    colors = [GOOD if c.agrees else CRITICAL for c in d.checks][::-1]
    y = np.arange(len(names))
    a1.barh(y, scores, color=colors, height=0.5)
    a1.set_xscale("log")
    a1.set_xlim(0.03, max(10.0, scores.max() * 4))
    a1.axvline(1.0, color=INK_2, linestyle="--", linewidth=1)
    a1.text(1.08, -0.62, "tolerance", color=INK_2, fontsize=8)
    a1.set_ylim(-0.75, len(names) - 0.5)
    a1.set_yticks(y, names, color=INK)
    for yi, s, c in zip(y, scores, d.checks[::-1]):
        a1.text(s * 1.15, yi, "ok" if c.agrees else "broken", va="center", fontsize=9,
                color=INK_2, fontweight="normal" if c.agrees else "bold")
    a1.set_xlabel("disagreement / tolerance (log scale)", color=INK_2, fontsize=9)
    a1.set_title("Which sensor agreements broke", loc="left", color=INK, fontsize=11)

    # --- right: evidence for the verdict ---------------------------------
    _style(a2)
    primary = d.primary
    if primary in ("camera_moved", "encoder_bias", "unexplained") and d.hypotheses:
        items = sorted(d.hypotheses.items(), key=lambda kv: kv[1]["chi2_dof"])
        labels = [HYP_LABEL.get(k, k) for k, _ in items][::-1]
        vals = np.array([np.sqrt(v["chi2_dof"]) for _, v in items])[::-1]
        best = items[0][0]
        cols = [SERIES_1 if k == best else "#d8d7d2" for k, _ in items][::-1]
        yy = np.arange(len(labels))
        a2.barh(yy, vals, color=cols, height=0.55)
        a2.set_xscale("log")
        a2.axvline(1.0, color=INK_2, linestyle="--", linewidth=1)
        a2.text(1.05, -0.95, "sensor noise", color=INK_2, fontsize=8)
        a2.set_ylim(-1.1, len(labels) - 0.5)
        a2.set_yticks(yy, labels, color=INK)
        a2.set_xlabel("error left after fitting each explanation (x noise, log scale)", color=INK_2, fontsize=9)
        a2.set_title("Only one physical change explains the data", loc="left", color=INK, fontsize=11)
    elif primary == "latency":
        t = d.plot["timing"]
        a2.grid(axis="y", color=GRID, linewidth=0.8)
        tt = np.array(t["t"])
        tt_ms = (tt - tt[0]) * 1000
        cmd, meas = np.rad2deg(t["cmd"]), np.rad2deg(t["meas"])
        normal = np.interp(tt - t["tau0"], tt, cmd)
        a2.plot(tt_ms, cmd, color=SERIES_1, linewidth=2, label="commanded")
        a2.plot(tt_ms, normal, color=MUTED, linewidth=2, linestyle="--", label=f"expected at normal delay ({t['tau0']*1000:.0f} ms)")
        a2.plot(tt_ms, meas, color=SERIES_2, linewidth=2, label=f"measured today ({t['tau']*1000:.0f} ms)")
        a2.legend(frameon=False, fontsize=9, loc="best", labelcolor=INK_2)
        a2.set_xlabel(f"time (ms), from the log: J{t['joint']+1} during its fastest move", color=INK_2, fontsize=9)
        a2.set_ylabel("joint angle (deg)", color=INK_2, fontsize=9)
        a2.set_title("Motion now lags commands by the measured delay", loc="left", color=INK, fontsize=11)
    elif primary in ("tool_bent",) and "probe" in d.plot:
        e = np.array(d.plot["probe"]["expected"]) * 1000
        m = np.array(d.plot["probe"]["measured"]) * 1000
        a2.grid(axis="y", color=GRID, linewidth=0.8)
        a2.scatter([e[0]], [e[1]], s=90, color=MUTED, zorder=3, label="controller thinks")
        a2.scatter([m[0]], [m[1]], s=90, color=SERIES_2, zorder=3, label="probe measured")
        a2.annotate("", xy=m[:2], xytext=e[:2], arrowprops=dict(arrowstyle="->", color=INK_2))
        r = max(np.abs(m[:2] - e[:2]).max() * 1.8, 10)
        a2.set_xlim(e[0] - r, e[0] + r)
        a2.set_ylim(e[1] - r, e[1] + r)
        a2.set_aspect("equal")
        a2.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        a2.set_xlabel("fingertip x in tool frame (mm)", color=INK_2, fontsize=9)
        a2.set_ylabel("fingertip y (mm)", color=INK_2, fontsize=9)
        a2.set_title("Probe: the fingertip is not where the robot thinks", loc="left", color=INK, fontsize=11)
    elif primary == "tcp_config":
        f = d.findings[0].params
        good = np.array(f["tcp_offset_good_mm"])
        now = np.array(f["tcp_offset_now_mm"])
        x = np.arange(3)
        a2.grid(axis="y", color=GRID, linewidth=0.8)
        a2.grid(axis="x", visible=False)
        a2.bar(x - 0.18, good, 0.34, color=MUTED, label="last known-good")
        a2.bar(x + 0.18, now, 0.34, color=SERIES_2, label="running now")
        a2.set_xticks(x, ["x", "y", "z"])
        a2.axhline(0, color=AXIS, linewidth=1)
        a2.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        a2.set_ylabel("tool offset (mm)", color=INK_2, fontsize=9)
        a2.set_title("Tool offset in the controller was edited", loc="left", color=INK, fontsize=11)
    else:
        p = d.plot.get("picks", {})
        t, ok = np.array(p.get("t", [])), np.array(p.get("ok", []), dtype=bool)
        a2.scatter(t[ok], np.ones(ok.sum()), color=GOOD, s=60, label="pick ok")
        a2.scatter(t[~ok], np.ones((~ok).sum()), color=CRITICAL, marker="x", s=60, label="pick missed")
        a2.set_yticks([])
        a2.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        a2.set_xlabel("time (s)", color=INK_2, fontsize=9)
        a2.set_title("Pick outcomes", loc="left", color=INK, fontsize=11)

    import textwrap

    f0 = d.findings[0] if d.findings else None
    head = f0.label if f0 else "No fault"
    sub = f0.size if f0 else "robot matches its healthy baseline"
    if f0 and 0 < f0.confidence < 1.0:
        sub += f"   |   confidence {f0.confidence*100:.0f}%"
    elif f0 and f0.confidence >= 1.0:
        sub += "   |   direct measurement"
    fig.text(0.02, 0.955, head, color=INK, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.885, textwrap.fill(sub, 150), color=INK_2, fontsize=10.5, va="top")
    fig.text(0.98, 0.955, f"Kintrace  |  {name}", color=MUTED, fontsize=9, ha="right", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
