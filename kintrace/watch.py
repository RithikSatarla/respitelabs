"""kintrace watch: a physical health score from passive production logs.

No check motion. While the robot works, the camera already sees the table
markers and the wrist plate. Every window (10 s by default) Kintrace asks:

  * where does the camera think the table is?  A drift here means the
    camera itself is moving (a sagging mount).
  * after that, does the arm agree with the camera?  A drift here that one
    joint explains means that joint's encoder zero is creeping.

Each one is turned into the share of the gripper's tolerance it uses up at
the grasp poses in the log. Health = 100 x (1 - share used). A warning fires
when the share stays above a threshold for two windows in a row, with the
likely cause, the drift rate and the projected time until picks miss.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares

from . import arms
from .diagnose import _cam_pose, _fixed_pred, _interp, _predict_cam

WINDOW_S = 10.0
WARN_AT = 0.15  # share of tolerance used before warning
FAIL_AT = 0.70  # share at which picks start to miss (each pick also scatters)
TOL_XY_MM = 8.0
TOL_Z_MM = 15.0
TREND_WINDOWS = 6


def _used(shift_m, tol_xy_mm, tol_z_mm):
    s = np.atleast_2d(shift_m) * 1000
    return float(np.mean(np.maximum(np.linalg.norm(s[:, :2], axis=1) / tol_xy_mm, np.abs(s[:, 2]) / tol_z_mm)))


def _tip(F, tcp):
    return np.einsum("nij,j->ni", F[:, :3, :3], tcp) + F[:, :3, 3]


def score_window(q, obs, fixed_mean, q_grasp, cfg, baseline, tol_xy_mm=TOL_XY_MM, tol_z_mm=TOL_Z_MM):
    """One window of passive data -> health numbers."""
    arm = arms.for_config(cfg)
    sigma = np.asarray(baseline["sigma_cam"])
    cam = np.array(baseline["camera_extrinsic"])
    tcp = np.array(baseline["tcp_offset"])
    fpts = np.array(cfg["fixed_markers"])
    markers = np.array(cfg["markers"])
    w = 1.0 / sigma

    # 1. camera vs table: fit the camera pose to the averaged table markers
    x = least_squares(lambda x: ((fixed_mean - _fixed_pred(_cam_pose(cam, x), fpts)) * w).ravel(),
                      np.zeros(6), x_scale=[0.01] * 6).x
    T = _cam_pose(cam, x)
    p = np.c_[fpts, np.ones(len(fpts))].T
    cam_shift = (cam @ np.linalg.inv(T) @ p)[:3].T - fpts
    cam_used = _used(cam_shift, tol_xy_mm, tol_z_mm)

    # 2. camera vs arm, with the camera where the table says it is
    best = (np.inf, 0, 0.0)
    for j in range(arm.n):
        def res(b, j=j):
            bb = np.zeros(arm.n)
            bb[j] = b[0]
            return ((obs - _predict_cam(q, bb, T, markers, arm)).reshape(-1, 3) * w).ravel()
        s = least_squares(res, [0.0], x_scale=[0.01])
        c = float(np.mean(s.fun**2))
        if c < best[0]:
            best = (c, j, float(s.x[0]))
    chi, j, b = best
    bb = np.zeros(arm.n)
    bb[j] = b
    tip_shift = _tip(arm.fk(q_grasp - bb), tcp) - _tip(arm.fk(q_grasp), tcp)
    arm_used = _used(tip_shift, tol_xy_mm, tol_z_mm)
    used = cam_used + arm_used
    return dict(
        health=round(100.0 * float(np.clip(1.0 - used, 0.0, 1.0)), 1),
        used=used,
        cam_used=cam_used,
        arm_used=arm_used,
        cam_mm=float(np.mean(np.linalg.norm(cam_shift, axis=1)) * 1000),
        cam_deg=float(np.rad2deg(np.linalg.norm(x[:3]))),
        arm_mm=float(np.mean(np.linalg.norm(tip_shift, axis=1)) * 1000),
        joint=int(j),
        bias_deg=float(np.rad2deg(b)),
        chi=chi,
    )


def _slope(t, y):
    if len(t) < 2:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def watch(log, baseline: dict, window_s: float = WINDOW_S, warn_at: float = WARN_AT, fail_at: float = FAIL_AT,
          tol_xy_mm: float = TOL_XY_MM, tol_z_mm: float = TOL_Z_MM) -> dict:
    """Score a log window by window, in time order, the way it would run
    live: each window only sees data up to its own end."""
    cfg = log.config
    if log.fixed_cam is None or "fixed_markers" not in cfg:
        raise ValueError("watch needs fixed table markers in view of the camera")
    arm = arms.for_config(cfg)
    windows = []
    warning = None
    t_end = log.t_cam[-1]
    starts = np.arange(log.t_cam[0], t_end - window_s + 1e-9, window_s)
    for s0 in starts:
        s1 = s0 + window_s
        sel = log.visible & (log.t_cam >= s0) & (log.t_cam < s1)
        if sel.sum() < 10:
            continue
        q = _interp(log.t_joint, log.q_meas, log.t_cam[sel])
        # grasp poses seen so far (fall back to this window's poses)
        pt = log.pick_t[log.pick_t < s1]
        q_grasp = _interp(log.t_joint, log.q_meas, pt) if len(pt) else q[::10]
        r = score_window(q, log.markers_cam[sel], log.fixed_cam[sel].mean(0), q_grasp, cfg, baseline,
                         tol_xy_mm, tol_z_mm)
        r["t"] = float(s1)
        windows.append(r)
        if warning is None and len(windows) >= 2 and windows[-1]["used"] >= warn_at and windows[-2]["used"] >= warn_at:
            warning = _make_warning(windows, fail_at, arm, cfg)
    return dict(windows=windows, warning=warning, window_s=window_s, warn_at=warn_at, fail_at=fail_at,
                tol_mm=[tol_xy_mm, tol_z_mm], start_time=cfg.get("start_time"))


def _make_warning(windows, fail_at, arm, cfg):
    # trend over the recent windows where the drift is visible above noise
    rec = []
    for w in reversed(windows):
        if w["used"] < WARN_AT / 3 or len(rec) == TREND_WINDOWS:
            break
        rec.insert(0, w)
    rec = rec if len(rec) >= 2 else windows[-2:]
    t = np.array([w["t"] for w in rec])
    now = windows[-1]
    camera = now["cam_used"] >= now["arm_used"]
    rate = _slope(t, np.array([w["used"] for w in rec]))  # share per s
    eta = (fail_at - now["used"]) / rate if rate > 1e-6 else None
    if camera:
        mm_rate = _slope(t, np.array([w["cam_mm"] for w in rec])) * 60
        cause = "camera_sag"
        text = (f"Camera is moving (sagging mount). Parts now appear {now['cam_mm']:.1f} mm off, "
                f"growing {mm_rate:.2f} mm/min.")
        size = dict(pick_area_mm=now["cam_mm"], camera_deg=now["cam_deg"], mm_per_min=mm_rate)
    else:
        j = now["joint"]
        deg_rate = _slope(t, np.array([w["bias_deg"] if w["joint"] == j else 0.0 for w in rec])) * 60
        mm_rate = _slope(t, np.array([w["arm_mm"] for w in rec])) * 60
        cause = "joint_creep"
        text = (f"{arm.joint_names[j].capitalize()} joint (J{j+1}) encoder zero is creeping. "
                f"Off by {now['bias_deg']:+.3f} deg ({now['arm_mm']:.1f} mm at the fingertip), "
                f"changing {deg_rate:+.4f} deg/min.")
        size = dict(joint=j + 1, bias_deg=now["bias_deg"], fingertip_mm=now["arm_mm"],
                    deg_per_min=deg_rate, mm_per_min=mm_rate)
    return dict(t=now["t"], cause=cause, health=now["health"], used=now["used"], text=text, size=size,
                share_per_min=rate * 60, eta_s=eta, fail_at=fail_at)


def _clock(start, s):
    from .recommission import _clock as c
    return c(start, s)


def text_report(r: dict, name: str = "log", every: int = 1) -> str:
    st = r["start_time"]
    out = [f"KINTRACE WATCH  {name}", "=" * 64,
           f"window {r['window_s']:.0f} s, tolerance {r['tol_mm'][0]:.0f} mm sideways / {r['tol_mm'][1]:.0f} mm "
           f"up-down, warn at {r['warn_at']*100:.0f}% used", "",
           "  time      health  camera  arm    cause-so-far"]
    for i, w in enumerate(r["windows"]):
        is_warn = r["warning"] is not None and abs(w["t"] - r["warning"]["t"]) < 1e-6
        if i % every and i != len(r["windows"]) - 1 and not is_warn:
            continue
        src = "camera" if w["cam_used"] >= w["arm_used"] else f"J{w['joint']+1}"
        flag = "  <- warning" if is_warn else ""
        out.append(f"  {_clock(st, w['t'])}  {w['health']:5.1f}  {w['cam_mm']:5.2f}mm {w['arm_mm']:5.2f}mm  "
                   f"{src if w['used'] >= r['warn_at'] / 2 else '-'}{flag}")
    out.append("")
    wn = r["warning"]
    if wn is None:
        out.append("No warning. The cell matches its known-good state.")
    else:
        out.append(f"WARNING at {_clock(st, wn['t'])}  health {wn['health']:.0f}/100")
        out.append(f"  {wn['text']}")
        if wn["eta_s"] is not None:
            out.append(f"  Picks likely start missing in about {wn['eta_s']/60:.1f} min "
                       f"(at {wn['fail_at']*100:.0f}% of tolerance), at the current rate.")
        fix = ("Run the check and fix at the next stop (writes a new camera calibration), and tighten the camera mount."
               if wn["cause"] == "camera_sag" else
               f"Run the check and fix at the next stop (writes a new J{wn['size']['joint']} zero). "
               "Check the encoder battery and cable.")
        out.append(f"  Next: {fix}")
    return "\n".join(out)


def warning_record(r: dict, name: str = "") -> dict | None:
    """The warning as a small JSON record (what `kintrace ledger add` reads)."""
    wn = r["warning"]
    if wn is None:
        return None
    return dict(kind="watch_warning", source=name, start_time=r["start_time"], at=_clock(r["start_time"], wn["t"]),
                t=wn["t"], cause=wn["cause"], health=wn["health"], text=wn["text"], size=wn["size"],
                eta_s=wn["eta_s"])


def to_json(r: dict) -> str:
    return json.dumps(r, indent=2, default=float)


# --------------------------------------------------------------------------
# benchmark: how early does the warning fire, compared with the first miss?
# --------------------------------------------------------------------------
def run_bench(n: int = 10, seed: int = 0, n_picks: int = 60, progress=print) -> dict:
    import time

    from . import sim
    from .diagnose import learn_baseline

    b = learn_baseline(sim.simulate({"type": "none"}, seed=1, with_probe=False))
    rows = []
    t0 = time.time()
    for k, kind in enumerate(["camera_sag", "joint_creep", "none"]):
        for i in range(n):
            s = seed + 1000 * k + i
            d = sim.random_drift(kind, np.random.default_rng(s))
            log = sim.simulate_drift(d, seed=s, n_picks=n_picks)
            r = watch(log, b)
            wn = r["warning"]
            miss = np.where(~log.pick_ok)[0]
            first_miss = float(log.pick_t[miss[0]]) if len(miss) else None
            row = dict(kind=kind, seed=s, drift=d, log_s=float(log.t_joint[-1]), n_windows=len(r["windows"]),
                       first_miss_s=first_miss, warned=wn is not None)
            if wn is not None:
                row.update(warn_s=wn["t"], cause=wn["cause"], health=wn["health"], eta_s=wn["eta_s"],
                           warn_mm=wn["size"].get("pick_area_mm", wn["size"].get("fingertip_mm")),
                           joint=wn["size"].get("joint"))
                cause_ok = wn["cause"] == kind
                if kind == "joint_creep":
                    cause_ok = cause_ok and wn["size"]["joint"] == d["joint"] + 1
                row["cause_ok"] = cause_ok
                if first_miss is not None:
                    row["lead_s"] = first_miss - wn["t"]
                    row["lead_share"] = (first_miss - wn["t"]) / (first_miss - d["start_s"])
                    if wn["eta_s"] is not None:
                        row["eta_err_s"] = wn["t"] + wn["eta_s"] - first_miss
            if kind != "none" and first_miss is not None:
                cams, bias = sim.drift_state(d, first_miss)
                if kind == "camera_sag":
                    row["miss_mm"] = sim.pick_area_shift(cams[0], sim.CAM_NOMINAL) * 1000
                else:
                    row["miss_mm"] = float(np.mean(np.linalg.norm(sim.tip_shift_from_bias(bias[0]), axis=1)) * 1000)
            rows.append(row)
        progress(f"  {kind:<12} done ({time.time()-t0:.0f}s)")

    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None

    summ = {}
    for kind in ["camera_sag", "joint_creep"]:
        rr = [x for x in rows if x["kind"] == kind]
        leads = [x.get("lead_s") for x in rr]
        summ[kind] = dict(
            n=len(rr),
            warned=sum(x["warned"] for x in rr),
            warned_before_first_miss=sum(1 for x in rr if x.get("lead_s") is not None and x["lead_s"] > 0),
            picks_missed_in_log=sum(1 for x in rr if x["first_miss_s"] is not None),
            cause_right=sum(1 for x in rr if x.get("cause_ok")),
            median_lead_s=med(leads),
            min_lead_s=min((x for x in leads if x is not None), default=None),
            median_lead_share=med([x.get("lead_share") for x in rr]),
            median_warn_mm=med([x.get("warn_mm") for x in rr]),
            median_error_at_first_miss_mm=med([x.get("miss_mm") for x in rr]),
            median_eta_error_s=med([x.get("eta_err_s") for x in rr]),
            median_abs_eta_error_s=med([abs(x["eta_err_s"]) for x in rr if x.get("eta_err_s") is not None]),
            drift_speed_pick_mm_per_min=[round(x["drift"]["pick_mm_per_min"], 2) for x in rr],
        )
    h = [x for x in rows if x["kind"] == "none"]
    summ["healthy"] = dict(n=len(h), false_alarms=sum(x["warned"] for x in h),
                           windows=sum(x["n_windows"] for x in h), missed_picks=sum(1 for x in h if x["first_miss_s"]))
    summ["settings"] = dict(window_s=WINDOW_S, warn_at=WARN_AT, fail_at=FAIL_AT, tol_mm=[TOL_XY_MM, TOL_Z_MM],
                            n_picks=n_picks, seconds=round(time.time() - t0, 1),
                            note=("Simulated cell. Drift is sped up so the tolerance is used up 100-160 s after "
                                  "the drift starts. Real sag and creep are far slower. Lead time scales with "
                                  "the drift rate, so also read it as a share of the time from drift start to "
                                  "first miss, and as mm of error when the warning fires."))
    return dict(summary=summ, rows=rows)
