"""Check, fix, verify, certify.

After a crash, maintenance, a part swap or a config change, Kintrace:

  1. reads the production log and finds WHEN things changed (and which
     logged event lines up with it)
  2. runs a ~6 s check motion and finds WHAT changed and by how much
  3. writes the corrected values (camera calibration, tool offset, joint
     zero offsets) to the controller
  4. runs the check again plus a few test picks to prove the fix worked
  5. issues a go / no-go record for the cell
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from . import arms
from .diagnose import (Check, Finding, FAULT_LABEL, _interp, _predict_cam, _fixed_pred,
                       estimate_delay, explain_geometry)
from .logio import Log

TIP_TOL = 0.0015  # m, fingertip within this of known-good counts as unchanged
CFG_TOL = 0.0005  # m
TAU_TOL = 0.004  # s


# --------------------------------------------------------------------------
# when did it change?
# --------------------------------------------------------------------------
def _split(x):
    """Best single change point in a series: index and strength."""
    n = len(x)
    if n < 8:
        return None, 0.0
    c = np.cumsum(x)
    best, best_k = 0.0, None
    for k in range(3, n - 3):
        m1, m2 = c[k - 1] / k, (c[-1] - c[k - 1]) / (n - k)
        stat = abs(m2 - m1) * np.sqrt(k * (n - k) / n)
        if stat > best:
            best, best_k = stat, k
    return best_k, best


def when_changed(log: Log, baseline: dict) -> dict:
    cfg = log.config
    sigma = np.array(baseline["sigma_cam"])
    cam = np.array(baseline["camera_extrinsic"])
    markers = np.array(cfg["markers"])
    sel = log.visible
    t = log.t_cam[sel]
    signals = []

    # arm vs camera, frame by frame, against the known-good calibration
    arm = arms.for_config(cfg)
    q = _interp(log.t_joint, log.q_meas, t)
    r = (log.markers_cam[sel] - _predict_cam(q, np.zeros(arm.n), cam, markers, arm)) / sigma
    arm_raw = np.mean(r**2, axis=(1, 2))
    arm = np.clip(arm_raw, 0, 50)
    k, stat = _split(arm)
    if k is not None and arm[k:].mean() > 3 and arm[:k].mean() < 2:
        signals.append(dict(signal="camera vs arm", t=float(t[k]), resolution_s=1.0 / 15 * 2))

    if log.fixed_cam is not None and "fixed_markers" in cfg:
        fr = (log.fixed_cam[sel] - _fixed_pred(cam, np.array(cfg["fixed_markers"]))[None]) / sigma
        fx_raw = np.mean(fr**2, axis=(1, 2))
        fx = np.clip(fx_raw, 0, 50)
        k, stat = _split(fx)
        if k is not None and fx[k:].mean() > 3 and fx[:k].mean() < 2:
            signals.append(dict(signal="camera vs table marker", t=float(t[k]), resolution_s=1.0 / 15 * 2))

    # command delay in sliding windows
    win, step = 6.0, 1.0
    starts = np.arange(log.t_joint[0], log.t_joint[-1] - win, step)
    for s0 in starts:
        m = (log.t_joint >= s0) & (log.t_joint < s0 + win)
        d = estimate_delay(SimpleNamespace(t_joint=log.t_joint[m], q_cmd=log.q_cmd[m], q_meas=log.q_meas[m]))
        if d - baseline["tau"] > TAU_TOL:
            signals.append(dict(signal="command timing", t=float(s0 + win / 2), resolution_s=win / 2))
            break

    # picks: first miss after which most picks fail
    ok = log.pick_ok
    for i in range(len(ok)):
        if not ok[i] and (~ok[i:]).mean() > 0.6:
            signals.append(dict(signal="first missed pick", t=float(log.pick_t[i]), resolution_s=0.0))
            break

    events = cfg.get("events", [])
    for e in events:
        if e["kind"] == "config_change" and "new" in e:
            if np.linalg.norm(np.array(cfg.get(e["key"], e["new"])) - np.array(baseline.get(e["key"], e["old"]))) > CFG_TOL:
                signals.append(dict(signal=f"controller setting '{e['key']}' changed", t=float(e["t"]), resolution_s=0.0))
    series = dict(t=t[::2].tolist(), arm=np.round(np.sqrt(arm_raw[::2]), 3).tolist())
    if log.fixed_cam is not None and "fixed_markers" in cfg:
        series["table"] = np.round(np.sqrt(fx_raw[::2]), 3).tolist()
    physical = [s for s in signals if s["signal"] != "first missed pick"]
    onset = min((s["t"] for s in physical), default=None)
    if onset is None and signals:
        onset = signals[0]["t"]
    match = None
    if onset is not None:
        near = [e for e in events if abs(e["t"] - onset) < 6.0]
        match = min(near, key=lambda e: abs(e["t"] - onset)) if near else None
    return dict(onset=onset, signals=sorted(signals, key=lambda x: x["t"]), event=match, events=events,
                series=series, picks=dict(t=log.pick_t.tolist(), ok=[bool(x) for x in log.pick_ok]))


# --------------------------------------------------------------------------
# what changed? (from the check motion)
# --------------------------------------------------------------------------
def analyze_check(check: dict, cfg: dict, good: dict, sigma, tip_tol: float = TIP_TOL,
                  tau_tol: float = TAU_TOL) -> dict:
    """good: last known-good values (tcp_offset, camera_extrinsic, tau).
    tip_tol (m) and tau_tol (s) default to the simulated cell's values; the
    desk rig passes its own from rig.json."""
    checks, findings = [], []
    fix = {}

    # timing
    tau = estimate_delay(SimpleNamespace(t_joint=check["t_joint"], q_cmd=check["q_cmd"], q_meas=check["q_meas"]))
    d_tau = tau - good["tau"]
    checks.append(Check("Command timing", abs(d_tau) < tau_tol, abs(d_tau) / tau_tol,
                        f"{tau*1000:.0f} ms vs {good['tau']*1000:.0f} ms normal"))
    blockers = []
    if abs(d_tau) >= tau_tol:
        findings.append(Finding("latency", FAULT_LABEL["latency"], f"{d_tau*1000:+.0f} ms slower than normal", 1.0,
                                "Not a parameter fix. Roll back the update or fix the network / CPU load, then re-run the check.",
                                {"delay_ms": tau * 1000, "normal_ms": good["tau"] * 1000}))
        blockers.append("commands are arriving late; needs a software or network fix")

    # camera and joints
    q = _interp(check["t_joint"], check["q_meas"], check["t_cam"])
    g_checks, g_find, _, geo = explain_geometry(q, check["markers_cam"], check["fixed_cam"], cfg, sigma)
    checks += g_checks
    findings += g_find
    cam_T = geo["camera"]
    bias = geo["bias"]
    if any(f.fault == "camera_moved" for f in g_find):
        fix["camera_extrinsic"] = cam_T.tolist()
    if np.any(bias != 0):
        fix["joint_zero_offsets_deg"] = np.rad2deg(bias).tolist()
    if any(f.fault == "unexplained" for f in g_find):
        blockers.append("camera and arm disagree in a way Kintrace can't pin to one cause")

    # the tool itself: where is the fingertip, really?
    F = arms.for_config(cfg).fk(q - bias)
    tip_w = (cam_T[:3, :3] @ check["tip_cam"].T).T + cam_T[:3, 3]
    A = F[:, :3, :3].reshape(-1, 3)
    b = (tip_w - F[:, :3, 3]).reshape(-1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    tcp_good = np.array(good["tcp_offset"])
    tcp_cfg = np.array(cfg["tcp_offset"])
    d_phys = x - tcp_good
    d_cfg = tcp_cfg - tcp_good
    tool_ok = np.linalg.norm(d_phys) < tip_tol
    cfg_ok = np.linalg.norm(d_cfg) < CFG_TOL
    checks.append(Check("Fingertip vs known-good", tool_ok, np.linalg.norm(d_phys) / tip_tol,
                        f"fingertip {np.linalg.norm(d_phys)*1000:.1f} mm from its known-good position"))
    checks.append(Check("Tool offset setting", cfg_ok, np.linalg.norm(d_cfg) / CFG_TOL,
                        "matches known-good" if cfg_ok else f"edited by {np.linalg.norm(d_cfg)*1000:.1f} mm"))
    if not tool_ok:
        findings.append(Finding(
            "tool_bent", FAULT_LABEL["tool_bent"],
            f"fingertip moved {np.linalg.norm(d_phys)*1000:.1f} mm (x {d_phys[0]*1000:+.1f}, y {d_phys[1]*1000:+.1f}, z {d_phys[2]*1000:+.1f})",
            0.98, "Tool offset set to the measured fingertip so picks land again. Replace or straighten the finger at the next stop.",
            {"tcp_measured_mm": (x * 1000).tolist(), "tcp_good_mm": (tcp_good * 1000).tolist()}))
        fix["tcp_offset"] = x.tolist()
    if not cfg_ok and tool_ok:
        findings.append(Finding(
            "tcp_config", FAULT_LABEL["tcp_config"],
            f"{np.linalg.norm(d_cfg)*1000:.1f} mm (x {d_cfg[0]*1000:+.1f}, y {d_cfg[1]*1000:+.1f}, z {d_cfg[2]*1000:+.1f})",
            1.0, "Tool offset restored to the known-good value.",
            {"tcp_now_mm": (tcp_cfg * 1000).tolist(), "tcp_good_mm": (tcp_good * 1000).tolist()}))
        fix["tcp_offset"] = tcp_good.tolist()
    return dict(checks=checks, findings=findings, fix=fix, blockers=blockers,
                tip_measured=x.tolist(), tau=tau)


# --------------------------------------------------------------------------
# the whole flow (simulated cell)
# --------------------------------------------------------------------------
def recommission_sim(log: Log, baseline: dict, world, seed: int = 0) -> dict:
    """Run the full check-fix-verify-certify flow on a simulated cell."""
    from . import sim

    rng = np.random.default_rng(seed)
    sigma = baseline["sigma_cam"]
    good = dict(tcp_offset=baseline["tcp_offset"], camera_extrinsic=baseline["camera_extrinsic"], tau=baseline["tau"])
    cfg = copy.deepcopy(log.config)

    when = when_changed(log, baseline)
    check1 = sim.run_check(world, rng)
    before = analyze_check(check1, cfg, good, sigma)

    after = None
    test_picks = None
    fixed_world = world
    if before["fix"]:
        fixed_world = sim.apply_fix(world, before["fix"])
        cfg2 = copy.deepcopy(cfg)
        for k in ("camera_extrinsic", "tcp_offset"):
            if k in before["fix"]:
                cfg2[k] = before["fix"][k]
        # after a fix, the new values are the known-good ones to verify against
        good2 = dict(good, tcp_offset=cfg2["tcp_offset"], camera_extrinsic=cfg2["camera_extrinsic"])
        check2 = sim.run_check(fixed_world, rng)
        after = analyze_check(check2, cfg2, good2, sigma)
    if not before["checks"] or before["fix"] or before["findings"]:
        tp = sim.simulate({"type": "none"}, seed=seed + 1, n_picks=6, with_probe=False, world=fixed_world)
        test_picks = [bool(x) for x in tp.pick_ok]

    verified = after if after is not None else before
    all_ok = all(c.agrees for c in verified["checks"])
    picks_ok = test_picks is None or sum(test_picks) >= len(test_picks) - 1
    blockers = list(before["blockers"]) + ([] if after is None else after["blockers"])
    if after is not None and not all_ok:
        blockers.append("a check still fails after the fix")
    if not picks_ok:
        blockers.append("test picks still missing")
    status = "GO" if not blockers and all_ok and picks_ok else "NO-GO"
    return dict(when=when, before=before, after=after, test_picks=test_picks, status=status,
                blockers=blockers, start_time=cfg.get("start_time"))


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def _clock(start: str | None, s: float | None) -> str:
    if s is None:
        return "unknown"
    if not start:
        return f"t+{s:.1f}s"
    return (datetime.fromisoformat(start) + timedelta(seconds=float(s))).strftime("%H:%M:%S")


def text_record(r: dict, cell: str = "Cell 3") -> str:
    st = r["start_time"]
    w = r["when"]
    out = [f"KINTRACE  recommission record, {cell}", "=" * 64]
    if w["onset"] is not None:
        line = f"Changed at   : {_clock(st, w['onset'])}"
        if w["event"]:
            line += f"  (lines up with {w['event']['kind'].replace('_', ' ')} at {_clock(st, w['event']['t'])}: {w['event']['detail']})"
        out.append(line)
        for s in w["signals"]:
            out.append(f"   seen in    : {s['signal']} at {_clock(st, s['t'])}")
    out.append("")
    out.append("What changed (6 s check motion)")
    if not r["before"]["findings"]:
        out.append("  nothing: cell matches its known-good state")
    for f in r["before"]["findings"]:
        out.append(f"  - {f.label}: {f.size}")
    if r["before"]["fix"]:
        out.append("")
        out.append("Fix written to controller")
        fx = r["before"]["fix"]
        if "camera_extrinsic" in fx:
            out.append("  - camera calibration updated")
        if "tcp_offset" in fx:
            out.append("  - tool offset set to [" + ", ".join(f"{v*1000:.1f}" for v in fx["tcp_offset"]) + "] mm")
        if "joint_zero_offsets_deg" in fx:
            for j, v in enumerate(fx["joint_zero_offsets_deg"]):
                if v:
                    out.append(f"  - J{j+1} zero offset {-v:+.3f} deg")
    if r["after"] is not None:
        out.append("")
        out.append("Verification (check motion re-run)")
        for c in r["after"]["checks"]:
            out.append(f"  [{'ok  ' if c.agrees else 'FAIL'}] {c.name:<28} {c.detail}")
    if r["test_picks"] is not None:
        out.append(f"  test picks: {sum(r['test_picks'])}/{len(r['test_picks'])} ok")
    out.append("")
    out.append(f"STATUS: {r['status']}" + ("" if not r["blockers"] else "  (" + "; ".join(r["blockers"]) + ")"))
    return "\n".join(out)


def to_dict(r: dict) -> dict:
    def fs(fl):
        return [dict(fault=f.fault, label=f.label, size=f.size, fix=f.fix) for f in fl]

    def cs(cl):
        return [dict(name=c.name, ok=c.agrees, detail=c.detail) for c in cl]

    st = r["start_time"]
    w = r["when"]
    clock = lambda x: _clock(st, x)
    return dict(
        start_time=st,
        series=dict(clock=[clock(x) for x in w["series"]["t"]], **w["series"]),
        picks=dict(clock=[clock(x) for x in w["picks"]["t"]], **w["picks"]),
        events=[dict(e, at=clock(e["t"])) for e in w["events"]],
        onset_s=w["onset"],
        status=r["status"],
        blockers=r["blockers"],
        changed_at=_clock(st, w["onset"]),
        event=None if not w["event"] else dict(kind=w["event"]["kind"], at=_clock(st, w["event"]["t"]), detail=w["event"]["detail"]),
        signals=[dict(signal=s["signal"], at=_clock(st, s["t"])) for s in w["signals"]],
        findings=fs(r["before"]["findings"]),
        checks_before=cs(r["before"]["checks"]),
        fix=r["before"]["fix"],
        checks_after=None if r["after"] is None else cs(r["after"]["checks"]),
        test_picks=r["test_picks"],
    )
