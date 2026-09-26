"""Kintrace diagnosis.

Idea: when nothing is wrong, the robot's sensors agree with each other.
  * command timing:  measured joints follow commanded joints after a fixed delay
  * camera vs kinematics: where the camera sees the wrist plate matches where
    the joint encoders + arm model say it is
  * controller settings: the tool offset and camera calibration the robot is
    running with match the last known-good values
Each physical fault breaks a different subset of these agreements, and the
*shape* of the disagreement (constant in the world vs changing with arm pose)
tells which one and how big. Faults that break no agreement at all (a bent
fingertip the camera can't normally see) get resolved with a 1.2 s probe.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import arms, robot
from .logio import Log

FAULT_LABEL = {
    "none": "No fault",
    "camera_moved": "Camera moved",
    "encoder_bias": "Joint encoder zero drifted",
    "latency": "Commands arriving late",
    "tcp_config": "Tool offset changed in controller",
    "tool_bent": "Gripper / tool bent",
    "unexplained": "Unexplained sensor disagreement",
    "needs_probe": "Tool change suspected, run probe",
}


# --------------------------------------------------------------------------
# baseline: learn "normal" from one healthy log
# --------------------------------------------------------------------------
def _interp(t_src, q_src, t):
    return np.stack([np.interp(t, t_src, q_src[:, i]) for i in range(q_src.shape[1])], 1)


def estimate_delay(log: Log) -> float:
    """Command-to-motion delay (s), robust to constant encoder offsets."""
    moving = np.linalg.norm(np.gradient(log.q_cmd, axis=0), axis=1) > 1e-5

    def cost(tau):
        q = _interp(log.t_joint, log.q_cmd, log.t_joint - tau)
        d = (log.q_meas - q)[moving]
        d = d - d.mean(0)
        return np.mean(d**2)

    grid = np.arange(0.0, 0.2, 0.001)
    c = np.array([cost(x) for x in grid])
    i = int(np.argmin(c))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 41)
    return float(fine[np.argmin([cost(x) for x in fine])])


def _marker_data(log: Log, markers_cam, t_cam, visible, t_joint, q_meas):
    sel = visible.astype(bool)
    q = _interp(t_joint, q_meas, t_cam[sel])
    return q, markers_cam[sel]


def _predict_cam(q, bias, cam_T, markers, arm=None):
    F = (arm or arms.UR5E).fk(q - bias)
    world = robot.transform_points(F, markers)
    Tinv = np.linalg.inv(cam_T)
    return np.einsum("ij,nkj->nki", Tinv[:3, :3], world) + Tinv[:3, 3]


def learn_baseline(log: Log) -> dict:
    cfg = log.config
    arm = arms.for_config(cfg)
    cam = np.array(cfg["camera_extrinsic"])
    markers = np.array(cfg["markers"])
    q, obs = _marker_data(log, log.markers_cam, log.t_cam, log.visible, log.t_joint, log.q_meas)
    res = obs - _predict_cam(q, np.zeros(arm.n), cam, markers, arm)
    sigma = res.reshape(-1, 3).std(0)
    return dict(
        tau=estimate_delay(log),
        sigma_cam=np.maximum(sigma, 1e-4).tolist(),
        tcp_offset=cfg["tcp_offset"],
        camera_extrinsic=cfg["camera_extrinsic"],
        # logs from the desk rig have no picks
        pick_success=float(log.pick_ok.mean()) if len(log.pick_ok) else None,
        n_picks=int(len(log.pick_ok)),
    )


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    agrees: bool
    score: float  # observed / tolerance; > 1 means the agreement broke
    detail: str

    def __post_init__(self):
        self.agrees, self.score = bool(self.agrees), float(self.score)


@dataclass
class Finding:
    fault: str
    label: str
    size: str
    confidence: float
    fix: str
    params: dict = field(default_factory=dict)


@dataclass
class Diagnosis:
    findings: list
    checks: list
    hypotheses: dict
    next_step: str
    probe_used: bool = False
    plot: dict = field(default_factory=dict)

    @property
    def primary(self) -> str:
        return self.findings[0].fault if self.findings else "none"

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("plot")
        return json.dumps(d, indent=2)


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------
def _fit_marker_hypotheses(q, obs, cam, markers, sigma, arm=None):
    """Compare explanations for camera-vs-kinematics disagreement."""
    arm = arm or arms.UR5E
    nj = arm.n
    n = obs.size
    w = 1.0 / np.asarray(sigma)

    def chi2_of(res):
        return float(np.sum((res.reshape(-1, 3) * w) ** 2))

    out = {}
    r0 = obs - _predict_cam(q, np.zeros(nj), cam, markers, arm)
    out["none"] = dict(chi2=chi2_of(r0), k=0, params={})

    # camera moved: 6-dof correction of the camera pose
    def cam_res(x):
        T = cam.copy()
        T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix() @ cam[:3, :3]
        T[:3, 3] = cam[:3, 3] + x[3:]
        return ((obs - _predict_cam(q, np.zeros(nj), T, markers, arm)).reshape(-1, 3) * w).ravel()

    s = least_squares(cam_res, np.zeros(6), x_scale=[0.01] * 3 + [0.01] * 3)
    out["camera_moved"] = dict(chi2=float(np.sum(s.fun**2)), k=6, params={"rotvec": s.x[:3].tolist(), "translation": s.x[3:].tolist()})

    # one joint's encoder zero drifted
    for j in range(nj):
        def enc_res(x, j=j):
            b = np.zeros(nj)
            b[j] = x[0]
            return ((obs - _predict_cam(q, b, cam, markers, arm)).reshape(-1, 3) * w).ravel()

        s = least_squares(enc_res, [0.0], x_scale=[0.01])
        out[f"encoder_bias:{j}"] = dict(chi2=float(np.sum(s.fun**2)), k=1, params={"joint": j, "bias": float(s.x[0])})

    for h in out.values():
        h["bic"] = h["chi2"] + h["k"] * np.log(n)
        h["chi2_dof"] = h["chi2"] / max(n - h["k"], 1)
    bics = np.array([h["bic"] for h in out.values()])
    p = np.exp(-(bics - bics.min()) / 2)
    p /= p.sum()
    for h, pi in zip(out.values(), p):
        h["prob"] = float(pi)
    return out


def _tip_shift_from_bias(q, j, bias, tcp, arm=None):
    arm = arm or arms.UR5E
    b = np.zeros(arm.n)
    b[j] = bias
    F0, F1 = arm.fk(q), arm.fk(q - b)
    t0 = np.einsum("nij,j->ni", F0[:, :3, :3], tcp) + F0[:, :3, 3]
    t1 = np.einsum("nij,j->ni", F1[:, :3, :3], tcp) + F1[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum((t1 - t0) ** 2, 1))))



def _cam_pose(cam, x):
    T = cam.copy()
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix() @ cam[:3, :3]
    T[:3, 3] = cam[:3, 3] + x[3:]
    return T


def _fixed_pred(cam_T, pts):
    Tinv = np.linalg.inv(cam_T)
    return (Tinv[:3, :3] @ pts.T).T + Tinv[:3, 3]


def _camera_finding(cam, T, conf, arm=None):
    x_t = T[:3, 3] - cam[:3, 3]
    rv = Rotation.from_matrix(T[:3, :3] @ cam[:3, :3].T).as_rotvec()
    pts = (arm or arms.UR5E).work_pts
    shift = np.mean(np.linalg.norm((cam @ np.linalg.inv(T) @ np.c_[pts, np.ones(3)].T)[:3].T - pts, axis=1))
    return Finding(
        "camera_moved", FAULT_LABEL["camera_moved"],
        f"{np.linalg.norm(x_t)*1000:.1f} mm, {np.rad2deg(np.linalg.norm(rv)):.2f} deg "
        f"(parts appear {shift*1000:.0f} mm off in the pick area)",
        conf,
        "Write the corrected camera extrinsic below to the controller, then re-tighten the camera mount.",
        {"camera_extrinsic_corrected": T.tolist(), "translation_mm": (x_t * 1000).tolist(),
         "rotation_deg": float(np.rad2deg(np.linalg.norm(rv)))},
    )


def _encoder_finding(q, j, b, tcp, conf, arm=None):
    arm = arm or arms.UR5E
    tip = _tip_shift_from_bias(q, j, b, tcp, arm)
    return Finding(
        "encoder_bias", FAULT_LABEL["encoder_bias"],
        f"{arm.joint_names[j]} joint (J{j+1}) off by {np.rad2deg(b):+.3f} deg "
        f"(moves the fingertip about {tip*1000:.0f} mm)",
        conf,
        f"Add {np.rad2deg(-b):+.3f} deg to the J{j+1} zero offset (or remaster J{j+1}). Check the encoder battery and cable if it happens again.",
        {"joint": j + 1, "bias_deg": float(np.rad2deg(b))},
    )


def explain_geometry(q, obs, fixed, cfg, sigma):
    """Camera vs the cell, and camera vs the arm.

    A fixed marker on the table tells a moved camera apart from a drifted
    base joint: both shift the arm in the image, only a moved camera shifts
    the table. Returns checks, findings, hypothesis scores for the chart and
    the corrected camera pose / joint biases.
    """
    arm = arms.for_config(cfg)
    nj = arm.n
    cam = np.array(cfg["camera_extrinsic"])
    markers = np.array(cfg["markers"])
    tcp = np.array(cfg["tcp_offset"])
    w = 1.0 / np.asarray(sigma)
    checks, findings = [], []
    geo = {"camera": cam, "bias": np.zeros(nj)}

    hyps = _fit_marker_hypotheses(q, obs, cam, markers, sigma, arm)
    scores = {k: dict(chi2_dof=v["chi2_dof"], prob=v["prob"]) for k, v in hyps.items()}

    if fixed is None or "fixed_markers" not in cfg:
        # no fixed reference: fall back to comparing explanations on the arm only
        h0 = hyps["none"]
        agree = h0["chi2_dof"] < 1.6
        checks.append(Check("Camera vs joint encoders", agree, h0["chi2_dof"] / 1.6,
                            f"disagreement {np.sqrt(h0['chi2_dof']):.1f}x sensor noise"))
        if agree:
            return checks, findings, scores, geo
        best_key = max(hyps, key=lambda k: hyps[k]["prob"])
        best = hyps[best_key]
        if best["chi2_dof"] > 2.5:
            findings.append(Finding("unexplained", FAULT_LABEL["unexplained"],
                                    f"best single explanation leaves {np.sqrt(best['chi2_dof']):.1f}x noise", 0.5,
                                    "More than one thing may have changed. Run the check motion or send the log for a manual look."))
        elif best_key == "camera_moved":
            T = _cam_pose(cam, np.r_[best["params"]["rotvec"], best["params"]["translation"]])
            geo["camera"] = T
            findings.append(_camera_finding(cam, T, min(best["prob"], 0.999), arm))
        else:
            j, b = best["params"]["joint"], best["params"]["bias"]
            geo["bias"][j] = b
            findings.append(_encoder_finding(q, j, b, tcp, min(best["prob"], 0.999), arm))
        return checks, findings, scores, geo

    # --- with a fixed reference ------------------------------------------
    fpts = np.array(cfg["fixed_markers"])
    rf = (fixed - _fixed_pred(cam, fpts)[None]).reshape(-1, 3) * w
    f_chi = float(np.mean(rf**2))
    cam_ok = f_chi < 1.6
    checks.append(Check("Camera vs fixed cell marker", cam_ok, f_chi / 1.6,
                        f"table marker {'where it should be' if cam_ok else 'shifted'} ({np.sqrt(f_chi):.1f}x noise)"))
    T = cam
    if not cam_ok:
        def fres(x):
            return ((fixed - _fixed_pred(_cam_pose(cam, x), fpts)[None]).reshape(-1, 3) * w).ravel()
        x = least_squares(fres, np.zeros(6), x_scale=[0.01] * 6).x
        T = _cam_pose(cam, x)
        geo["camera"] = T
        findings.append(_camera_finding(cam, T, 0.999, arm))

    # arm vs camera, using the (corrected) camera
    r0 = obs - _predict_cam(q, np.zeros(nj), T, markers, arm)
    a_chi = float(np.mean((r0.reshape(-1, 3) * w) ** 2))
    arm_ok = a_chi < 1.6
    checks.append(Check("Camera vs joint encoders", arm_ok, a_chi / 1.6,
                        f"disagreement {np.sqrt(a_chi):.1f}x sensor noise"
                        + (" (after correcting the camera)" if not cam_ok else "")))
    if not arm_ok:
        fits = {}
        for j in range(nj):
            def enc_res(xb, j=j):
                bb = np.zeros(nj)
                bb[j] = xb[0]
                return ((obs - _predict_cam(q, bb, T, markers, arm)).reshape(-1, 3) * w).ravel()
            sfit = least_squares(enc_res, [0.0], x_scale=[0.01])
            fits[j] = (float(np.mean(sfit.fun**2)), float(sfit.x[0]))
        j = min(fits, key=lambda k: fits[k][0])
        chi, b = fits[j]
        chis = np.array([fits[k][0] for k in range(nj)]) * obs.size
        p = np.exp(-(chis - chis.min()) / 2)
        conf = float(p[j] / p.sum())
        if chi > 2.5:
            findings.append(Finding("unexplained", FAULT_LABEL["unexplained"],
                                    f"the arm disagrees with the camera ({np.sqrt(chi):.1f}x noise) in a way no single joint explains",
                                    0.5, "More than one joint, or a change to the arm itself. Send the log for a manual look."))
        else:
            geo["bias"][j] = b
            findings.append(_encoder_finding(q, j, b, tcp, min(conf, 0.999), arm))
    return checks, findings, scores, geo


def diagnose(log: Log, baseline: dict, run_probe: bool = False) -> Diagnosis:
    cfg = log.config
    cam = np.array(cfg["camera_extrinsic"])
    markers = np.array(cfg["markers"])
    tcp = np.array(cfg["tcp_offset"])
    sigma = np.array(baseline["sigma_cam"])
    checks, findings = [], []
    plot: dict = {}
    probe_used = False

    # 1. command timing ---------------------------------------------------
    tau = estimate_delay(log)
    d_tau = tau - baseline["tau"]
    tau_tol = 0.004
    checks.append(Check("Command timing", abs(d_tau) < tau_tol, abs(d_tau) / tau_tol,
                        f"{tau*1000:.0f} ms delay vs {baseline['tau']*1000:.0f} ms normal"))
    # a real snippet of the log for the chart: the fastest move of the busiest joint
    vel = np.abs(np.gradient(log.q_cmd, axis=0))
    j = int(np.argmax(vel.max(0)))
    i = int(np.argmax(vel[:, j]))
    w = (log.t_joint > log.t_joint[i] - 0.45) & (log.t_joint < log.t_joint[i] + 0.45)
    plot["timing"] = dict(tau=tau, tau0=baseline["tau"], joint=j, t=log.t_joint[w].tolist(),
                          cmd=log.q_cmd[w, j].tolist(), meas=log.q_meas[w, j].tolist())
    if abs(d_tau) >= tau_tol:
        v = cfg.get("conveyor_speed", 0.0)
        findings.append(Finding(
            "latency", FAULT_LABEL["latency"],
            f"{d_tau*1000:+.0f} ms slower than normal" + (f" (about {abs(d_tau)*v*1000:.0f} mm of conveyor travel)" if v else ""),
            1.0,
            "Not a parameter fix. Check network load, controller CPU and the command path (driver, bridge, DDS settings) since the last good run.",
            {"delay_ms": tau * 1000, "normal_ms": baseline["tau"] * 1000},
        ))

    # 2. controller settings ---------------------------------------------
    d_tcp = tcp - np.array(baseline["tcp_offset"])
    d_cam = np.abs(cam - np.array(baseline["camera_extrinsic"])).max()
    tcp_tol = 0.0005
    cfg_ok = np.linalg.norm(d_tcp) < tcp_tol and d_cam < 1e-6
    checks.append(Check("Controller settings", cfg_ok, max(np.linalg.norm(d_tcp) / tcp_tol, d_cam / 1e-6),
                        "match last known-good" if cfg_ok else "tool offset or camera calibration was edited"))
    if np.linalg.norm(d_tcp) >= tcp_tol:
        findings.append(Finding(
            "tcp_config", FAULT_LABEL["tcp_config"],
            f"{np.linalg.norm(d_tcp)*1000:.1f} mm (x {d_tcp[0]*1000:+.1f}, y {d_tcp[1]*1000:+.1f}, z {d_tcp[2]*1000:+.1f})",
            1.0,
            f"Restore tool offset to [{', '.join(f'{x*1000:.1f}' for x in baseline['tcp_offset'])}] mm, unless the tool was changed on purpose.",
            {"tcp_offset_now_mm": (tcp * 1000).tolist(), "tcp_offset_good_mm": (np.array(baseline["tcp_offset"]) * 1000).tolist()},
        ))

    # 3. camera vs the cell, and camera vs joint encoders ----------------
    q, obs = _marker_data(log, log.markers_cam, log.t_cam, log.visible, log.t_joint, log.q_meas)
    fixed = log.fixed_cam[log.visible] if log.fixed_cam is not None else None
    g_checks, g_findings, hyps, geo = explain_geometry(q, obs, fixed, cfg, sigma)
    checks += g_checks
    findings += g_findings
    plot["hypotheses"] = hyps

    # 4. task outcome -----------------------------------------------------
    has_picks = len(log.pick_ok) > 0 and baseline.get("pick_success") is not None
    if has_picks:
        rate = float(log.pick_ok.mean())
        base = baseline["pick_success"]
        drop = base - rate
        outcome_ok = drop < 0.2
        checks.append(Check("Pick success", outcome_ok, max(drop, 0) / 0.2,
                            f"{rate*100:.0f}% vs {base*100:.0f}% normal"))
    else:
        # desk rig logs have no picks. The check motion shows the tool tip to
        # the camera, so measure the tool directly when asked to.
        outcome_ok = True
        if run_probe and log.probe is not None and "tip_cam" in log.probe and not findings:
            f, pdata = _probe_tool(log, cam, markers, tcp, sigma)
            if f.fault == "tool_bent":
                probe_used = True
                plot["probe"] = pdata
                findings.append(f)
    plot["picks"] = dict(t=log.pick_t.tolist(), ok=log.pick_ok.tolist())

    next_step = "Nothing to do. Robot matches its healthy baseline." if outcome_ok and not findings else ""
    if not outcome_ok and not findings:
        # every sensor agrees but the robot misses: something the robot can't
        # see in normal operation changed. Resolve with the probe.
        if run_probe and log.probe is not None:
            probe_used = True
            f, pdata = _probe_tool(log, cam, markers, tcp, sigma)
            plot["probe"] = pdata
            findings.append(f)
        else:
            findings.append(Finding(
                "needs_probe", FAULT_LABEL["needs_probe"],
                "sensors all agree, so the change is somewhere the camera can't normally see",
                0.0,
                "Run the 1.2 s probe (kintrace diagnose --probe): the arm shows its fingertip to the camera and Kintrace measures the tool directly.",
            ))
    if findings and not next_step:
        next_step = findings[0].fix
    elif not next_step:
        next_step = "Picks are fine but a sensor disagreement was found. Fix it before it causes misses."

    if findings and findings[0].fault == "needs_probe":
        next_step = findings[0].fix

    return Diagnosis(findings, checks, {k: dict(prob=v["prob"], chi2_dof=v["chi2_dof"]) for k, v in hyps.items()},
                     next_step, probe_used, plot)


def _probe_tool(log: Log, cam, markers, tcp, sigma):
    p = log.probe
    q = _interp(p["t_joint"], p["q_meas"], p["t_cam"])
    F = arms.for_config(log.config).fk(q)
    w = (cam[:3, :3] @ p["tip_cam"].T).T + cam[:3, 3]  # fingertip in world
    # solve R_i x + p_i = w_i for the fingertip position x in the flange frame
    A = F[:, :3, :3].reshape(-1, 3)
    b = (w - F[:, :3, 3]).reshape(-1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ x - b
    se = np.sqrt(np.mean(resid**2) * np.diag(np.linalg.inv(A.T @ A)))
    d = x - tcp
    dist = np.linalg.norm(d)
    pdata = dict(expected=tcp.tolist(), measured=x.tolist())
    if dist < max(3 * np.linalg.norm(se), 0.002):
        return Finding("unexplained", "Tool is fine, cause is elsewhere",
                       f"fingertip within {dist*1000:.1f} mm of where the controller expects it", 0.5,
                       "Look at the parts and the gripper grasp itself (new part type, worn pads, suction).",
                       {"tip_mm": (x * 1000).tolist()}), pdata
    return Finding(
        "tool_bent", FAULT_LABEL["tool_bent"],
        f"fingertip {dist*1000:.1f} mm from where the controller thinks it is "
        f"(x {d[0]*1000:+.1f}, y {d[1]*1000:+.1f}, z {d[2]*1000:+.1f} in the tool frame)",
        0.98,
        f"Straighten or replace the gripper finger. Short-term: set tool offset to [{', '.join(f'{v*1000:.1f}' for v in x)}] mm so picks land again.",
        {"tcp_measured_mm": (x * 1000).tolist(), "tcp_expected_mm": (tcp * 1000).tolist()},
    ), pdata
