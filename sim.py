"""Simulated robot cell that writes Kintrace logs.

A UR5e-class arm picks parts off a moving conveyor. A fixed camera finds
the parts and also tracks a fiducial plate on the robot's wrist. Faults
are injected into the physical world or the controller settings, and the
log only contains what a real robot would record.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from . import robot
from .logio import Log

# ---- nominal cell ---------------------------------------------------------
TCP_NOMINAL = np.array([0.0, 0.0, 0.18])  # fingertip, flange frame (m)
MARKERS = np.array(  # fiducial plate on the wrist, flange frame (m)
    [[0.045, 0.0, 0.035], [-0.025, 0.04, 0.035], [-0.025, -0.04, 0.035]]
)
FIXED = np.array(  # fixed fiducials on the table/conveyor frame, world (m)
    [[0.30, -0.45, 0.0], [0.62, -0.45, 0.0], [0.62, 0.20, 0.0], [0.30, 0.20, 0.0]]
)
CAM_NOMINAL = robot.look_at(eye=[0.95, -0.75, 1.15], target=[0.45, 0.0, 0.25])
CONVEYOR_V = 0.25  # m/s along +y
TAU0 = 0.008  # s, normal command-to-motion delay
JOINT_HZ = 125
CAM_HZ = 15
CAM_NOISE = np.array([0.0006, 0.0006, 0.0015])  # m, x/y in image, z = depth
ENC_NOISE = 2e-5  # rad
GRASP_TOL = 0.008  # m, gripper tolerance before a pick fails
SEED_Q = np.array([0.0, -1.9, 1.9, -1.57, -1.57, 0.0])

FAULTS = ["none", "camera_moved", "tool_bent", "tcp_config", "encoder_bias", "latency"]

# Typical lever arm (m) from each joint to the fingertip in this cell, used
# to size encoder faults so every fault causes roughly the same miss.
LEVER = np.array([0.55, 0.70, 0.40, 0.22, 0.18])


def base_config() -> dict:
    return dict(
        robot="UR5e-class",
        tcp_offset=TCP_NOMINAL.tolist(),
        camera_extrinsic=CAM_NOMINAL.tolist(),
        markers=MARKERS.tolist(),
        fixed_markers=FIXED.tolist(),
        conveyor_speed=CONVEYOR_V,
    )


def random_fault(kind: str, rng: np.random.Generator, miss_mm: float = 20.0) -> dict:
    """A fault of the given kind, sized so the pick miss is about miss_mm."""
    m = miss_mm / 1000.0 * rng.uniform(0.75, 1.25)
    if kind == "none":
        return {"type": "none"}
    if kind == "camera_moved":
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        r = rng.normal(size=3)
        r = r / np.linalg.norm(r) * np.deg2rad(rng.uniform(0.0, 0.25))
        return {"type": kind, "translation": (d * m).tolist(), "rotvec": r.tolist()}
    if kind in ("tool_bent", "tcp_config"):
        d = rng.normal(size=3) * np.array([1.0, 1.0, 0.4])
        d /= np.linalg.norm(d)
        return {"type": kind, "delta": (d * m).tolist()}
    if kind == "encoder_bias":
        j = int(rng.integers(0, 5))
        return {"type": kind, "joint": j, "bias": float(rng.choice([-1, 1]) * m / LEVER[j])}
    if kind == "latency":
        return {"type": kind, "extra_s": float(m / CONVEYOR_V)}
    raise ValueError(kind)


class _World:
    """True physical state vs what the controller believes."""

    def __init__(self, fault: dict):
        self.fault = fault
        self.cam_param = CAM_NOMINAL.copy()
        self.cam_true = CAM_NOMINAL.copy()
        self.tcp_param = TCP_NOMINAL.copy()
        self.tcp_true = TCP_NOMINAL.copy()
        self.bias = np.zeros(6)
        self.tau = TAU0
        for f in fault.get("faults", [fault]):
            self._apply(f)

    def _apply(self, fault: dict):
        t = fault["type"]
        if t == "camera_moved":
            self.cam_true = CAM_NOMINAL.copy()
            self.cam_true[:3, :3] = (
                Rotation.from_rotvec(fault["rotvec"]).as_matrix() @ CAM_NOMINAL[:3, :3]
            )
            self.cam_true[:3, 3] += np.array(fault["translation"])
        elif t == "tool_bent":
            self.tcp_true = TCP_NOMINAL + np.array(fault["delta"])
        elif t == "tcp_config":
            self.tcp_param = TCP_NOMINAL + np.array(fault["delta"])
        elif t == "encoder_bias":
            self.bias[fault["joint"]] = fault["bias"]
        elif t == "latency":
            self.tau = TAU0 + fault["extra_s"]


def _min_jerk(q0, q1, n):
    s = np.linspace(0.0, 1.0, n)[:, None]
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    return q0 + (q1 - q0) * s


def _build_trajectory(waypoints, dt):
    """waypoints: list of (q, move_s, dwell_s). Returns t, q_cmd."""
    qs = [waypoints[0][0][None]]
    for (q0, _, _), (q1, move, dwell) in zip(waypoints[:-1], waypoints[1:]):
        qs.append(_min_jerk(q0, q1, max(int(move / dt), 2))[1:])
        if dwell > 0:
            qs.append(np.repeat(q1[None], int(dwell / dt), axis=0))
    q = np.concatenate(qs)
    return np.arange(len(q)) * dt, q


def _interp(t_src, q_src, t):
    return np.stack([np.interp(t, t_src, q_src[:, i]) for i in range(q_src.shape[1])], 1)


def _observe_fixed(w: _World, n, rng):
    Tinv = np.linalg.inv(w.cam_true)
    cam = (Tinv[:3, :3] @ FIXED.T).T + Tinv[:3, 3]
    return cam[None] + rng.normal(size=(n,) + cam.shape) * CAM_NOISE


def _observe(w: _World, T_flange, pts, rng):
    """Camera observation (camera frame) of points attached to the flange."""
    world = robot.transform_points(T_flange, pts)
    Tinv = np.linalg.inv(w.cam_true)
    cam = np.einsum("ij,nkj->nki", Tinv[:3, :3], world) + Tinv[:3, 3]
    return cam + rng.normal(size=cam.shape) * CAM_NOISE


EVENT_FOR_FAULT = {
    "tool_bent": ("protective_stop", "collision detected, arm stopped"),
    "camera_moved": ("protective_stop", "collision detected, arm stopped"),
    "tcp_config": ("config_change", "installation file reloaded"),
    "encoder_bias": ("maintenance", "joint service, arm power-cycled"),
    "latency": ("software_update", "cell PC updated"),
}


def _fault_types(fault: dict) -> list:
    return [f["type"] for f in fault.get("faults", [fault])]


def simulate(fault: dict, seed: int = 0, n_picks: int = 14, with_probe: bool = True,
             onset_pick: int = 0, world: "_World | None" = None) -> Log:
    """Simulate a production run.

    onset_pick: the fault appears at the start of this pick (0 = present the
    whole run). Picks before it run on a healthy cell, and an event (a
    collision stop, a config reload...) is logged at the moment it happens.
    world: run on this exact cell state instead (used to test a fix).
    """
    rng = np.random.default_rng(seed)
    w1 = world if world is not None else _World(fault)
    w0 = world if world is not None else _World({"type": "none"})
    dt = 1.0 / JOINT_HZ

    # --- plan the picks the way the controller would, from its beliefs -----
    waypoints = [(SEED_Q.copy(), 0.0, 0.3)]
    q_prev = SEED_Q.copy()
    t_cursor = 0.3
    picks = []
    t_onset = 0.0
    for i in range(n_picks):
        if i == onset_pick and onset_pick > 0:
            t_onset = t_cursor
        w = w1 if i >= onset_pick else w0
        p_obj = np.array([rng.uniform(0.35, 0.6), rng.uniform(-0.55, -0.3), 0.04])
        t_detect = t_cursor
        yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        tilt = rng.uniform(-0.3, 0.3, size=2)
        move = 0.9
        t_grasp = t_detect + 2 * move  # pre-grasp then grasp
        # camera locates the part: true -> camera -> believed world
        p_cam = np.linalg.inv(w.cam_true) @ np.append(p_obj, 1.0)
        p_seen = (w.cam_param @ p_cam)[:3] + rng.normal(size=3) * 0.0008
        p_plan = p_seen + np.array([0.0, CONVEYOR_V * (t_grasp - t_detect), 0.0])
        T_grasp = robot.tool_target(p_plan, yaw, *tilt)
        T_pre = T_grasp.copy()
        T_pre[2, 3] += 0.12
        q_pre = robot.ik(T_pre, w.tcp_param, q_prev)
        q_grasp = robot.ik(T_grasp, w.tcp_param, q_pre)
        place = np.array([rng.uniform(0.2, 0.35), rng.uniform(0.3, 0.45), 0.15])
        T_place = robot.tool_target(place, rng.uniform(-1.2, 1.2), *rng.uniform(-0.25, 0.25, 2))
        q_place = robot.ik(T_place, w.tcp_param, q_grasp)
        waypoints += [(q_pre, move, 0.0), (q_grasp, move, 0.25), (q_pre, 0.6, 0.0), (q_place, 1.0, 0.25)]
        t_cursor = t_grasp + 0.25 + 0.6 + 1.0 + 0.25

        # --- what physically happens at the grasp ---------------------------
        q_true = q_grasp - w.bias
        F = robot.fk(q_true)
        tip = F[:3, :3] @ w.tcp_true + F[:3, 3]
        t_actual = t_grasp + (w.tau - TAU0)  # controller already compensates TAU0
        obj = p_obj + np.array([0.0, CONVEYOR_V * (t_actual - t_detect), 0.0])
        miss = tip - obj + rng.normal(size=3) * 0.002
        ok = np.linalg.norm(miss[:2]) < GRASP_TOL and abs(miss[2]) < 0.015
        picks.append((t_grasp, ok))
        q_prev = q_place

    t_joint, q_cmd = _build_trajectory(waypoints, dt)
    after_j = t_joint >= t_onset
    tau = np.where(after_j, w1.tau, w0.tau)
    q_meas = _interp(t_joint, q_cmd, t_joint - tau) + rng.normal(size=q_cmd.shape) * ENC_NOISE

    t_cam = np.arange(0.0, t_joint[-1], 1.0 / CAM_HZ) + rng.uniform(0, 1.0 / CAM_HZ)
    t_cam = t_cam[t_cam < t_joint[-1]]
    markers_cam = np.zeros((len(t_cam), len(MARKERS), 3))
    fixed_cam = np.zeros((len(t_cam), len(FIXED), 3))
    for wk, sel in ((w0, t_cam < t_onset), (w1, t_cam >= t_onset)):
        if sel.any():
            q_true = _interp(t_joint, q_cmd, t_cam[sel] - wk.tau) - wk.bias
            markers_cam[sel] = _observe(wk, robot.fk(q_true), MARKERS, rng)
            fixed_cam[sel] = _observe_fixed(wk, int(sel.sum()), rng)
    visible = rng.random(len(t_cam)) > 0.1

    config = base_config()
    config["tcp_offset"] = w1.tcp_param.tolist()
    config["camera_extrinsic"] = w1.cam_param.tolist()
    config["start_time"] = "2026-09-24T13:40:00"
    events = []
    if onset_pick > 0 and world is None:
        seen = set()
        for ft in _fault_types(fault):
            ev = EVENT_FOR_FAULT.get(ft)
            if ev and ev[0] not in seen:
                seen.add(ev[0])
                e = {"t": round(float(t_onset) - 0.2, 2), "kind": ev[0], "detail": ev[1]}
                if ft == "tcp_config":
                    e.update(key="tcp_offset", old=w0.tcp_param.tolist(), new=w1.tcp_param.tolist())
                events.append(e)
    config["events"] = events
    log = Log(
        t_joint=t_joint,
        q_cmd=q_cmd,
        q_meas=q_meas,
        t_cam=t_cam,
        markers_cam=markers_cam,
        visible=visible,
        pick_t=np.array([p[0] for p in picks]),
        pick_ok=np.array([p[1] for p in picks]),
        config=config,
        truth={"fault": fault, "seed": seed, "onset_s": float(t_onset) if onset_pick > 0 else 0.0},
        fixed_cam=fixed_cam,
    )
    if with_probe:
        log.probe = run_probe(w1, rng)
    return log


def run_probe(w: _World, rng) -> dict:
    """The 1.2 s probe: show the fingertip to the camera while the wrist
    sweeps. Fingertip marker becomes visible only during this motion."""
    cam_pos = w.cam_param[:3, 3]
    center = np.array([0.45, -0.05, 0.35])
    z_tool = cam_pos - center
    z_tool /= np.linalg.norm(z_tool)
    T = robot.look_at(center, center + z_tool)  # tool z points at camera
    q0 = robot.ik(T, w.tcp_param, SEED_Q)
    dt = 1.0 / JOINT_HZ
    t = np.arange(0.0, 1.2, dt)
    phase = 2 * np.pi * t / 1.2
    sweep = np.stack(
        [np.zeros_like(t)] * 3
        + [0.35 * np.sin(phase), 0.35 * np.sin(phase + 2.1), 0.9 * np.sin(phase + 4.2)],
        1,
    )
    q_cmd = q0 + sweep
    q_meas = _interp(t, q_cmd, t - w.tau) + rng.normal(size=q_cmd.shape) * ENC_NOISE
    t_cam = np.arange(0.0, 1.2, 1.0 / CAM_HZ)
    q_true = _interp(t, q_cmd, t_cam - w.tau) - w.bias
    F = robot.fk(q_true)
    return dict(
        t_joint=t,
        q_meas=q_meas,
        t_cam=t_cam,
        markers_cam=_observe(w, F, MARKERS, rng),
        tip_cam=_observe(w, F, w.tcp_true[None], rng)[:, 0],
        fixed_cam=_observe_fixed(w, len(t_cam), rng),
    )


def run_check(w: _World, rng, seconds_per_pose: float = 0.7) -> dict:
    """The recommission check motion (~6 s).

    The arm visits a few poses in front of the camera with the fingertip
    turned toward it and the wrist at different angles, so the camera sees
    the wrist plate, the fingertip and the table markers from many views.
    """
    cam_pos = w.cam_param[:3, 3]
    base = np.array([0.45, -0.05, 0.33])
    offsets = [[0, 0, 0], [0.09, 0.07, 0.04], [-0.08, 0.08, -0.03], [0.07, -0.09, 0.05],
               [-0.07, -0.07, 0.06], [0.0, 0.1, -0.04], [0.05, 0.0, 0.08], [-0.05, 0.02, -0.05]]
    spins = [0.0, 0.9, -0.9, 1.6, -1.6, 0.4, -0.4, 1.2]
    tilts = [[0, 0], [0.35, 0], [0, 0.35], [-0.35, 0], [0, -0.35], [0.25, 0.25], [-0.25, 0.25], [0.2, -0.3]]
    waypoints = [(SEED_Q.copy(), 0.0, 0.1)]
    q = SEED_Q.copy()
    for off, spin, (tx, ty) in zip(offsets, spins, tilts):
        c = base + np.array(off)
        T = robot.look_at(c, c + (cam_pos - c) / np.linalg.norm(cam_pos - c))
        R = Rotation.from_matrix(T[:3, :3]) * Rotation.from_euler("xyz", [tx, ty, spin])
        T[:3, :3] = R.as_matrix()
        q = robot.ik(T, w.tcp_param, q)
        waypoints.append((q, seconds_per_pose, 0.1))
    dt = 1.0 / JOINT_HZ
    t, q_cmd = _build_trajectory(waypoints, dt)
    q_meas = _interp(t, q_cmd, t - w.tau) + rng.normal(size=q_cmd.shape) * ENC_NOISE
    t_cam = np.arange(0.0, t[-1], 1.0 / CAM_HZ)
    q_true = _interp(t, q_cmd, t_cam - w.tau) - w.bias
    F = robot.fk(q_true)
    return dict(
        t_joint=t,
        q_cmd=q_cmd,
        q_meas=q_meas,
        t_cam=t_cam,
        markers_cam=_observe(w, F, MARKERS, rng),
        tip_cam=_observe(w, F, w.tcp_true[None], rng)[:, 0],
        fixed_cam=_observe_fixed(w, len(t_cam), rng),
    )


def apply_fix(w: _World, fix: dict) -> _World:
    """The cell after Kintrace writes its corrections to the controller."""
    import copy

    w2 = copy.deepcopy(w)
    if "camera_extrinsic" in fix:
        w2.cam_param = np.array(fix["camera_extrinsic"])
    if "tcp_offset" in fix:
        w2.tcp_param = np.array(fix["tcp_offset"])
    if "joint_zero_offsets_deg" in fix:
        # the controller subtracts the offset from what the encoder reads
        w2.bias = w2.bias - np.deg2rad(np.array(fix["joint_zero_offsets_deg"]))
    return w2


# --------------------------------------------------------------------------
# slow drift (for `kintrace watch`). Additive: nothing above uses it, so the
# existing faults and their logs are unchanged.
#
# A drift grows linearly from start_s. Real sag or creep takes days or weeks.
# Here it is sped up so a whole failure fits in a few minutes of log. Every
# drift carries its speed ("pick_mm_per_min": how fast the error at the pick
# area grows), so results can be read at any real rate.
# --------------------------------------------------------------------------
DRIFTS = ["camera_sag", "joint_creep"]
PICK_S = 3.9  # s per pick cycle in simulate()


TOL_XY, TOL_Z = GRASP_TOL, 0.015  # m, what the gripper forgives


def tolerance_used(shift) -> float:
    """Share of the grasp tolerance (0..1+) used by a fingertip-to-part
    error vector (m) or a batch of them (mean over the batch)."""
    s = np.atleast_2d(shift)
    return float(np.mean(np.maximum(np.linalg.norm(s[:, :2], axis=1) / TOL_XY, np.abs(s[:, 2]) / TOL_Z)))


def camera_shift(cam_true: np.ndarray, cam_param: np.ndarray) -> np.ndarray:
    """Where the table markers appear minus where they are (m), when the
    camera is really at cam_true but the controller believes cam_param."""
    pts = np.c_[FIXED, np.ones(len(FIXED))].T
    return (cam_param @ np.linalg.inv(cam_true) @ pts)[:3].T - FIXED


def pick_area_shift(cam_true: np.ndarray, cam_param: np.ndarray) -> float:
    """Mean distance (m) a part on the table appears off."""
    return float(np.mean(np.linalg.norm(camera_shift(cam_true, cam_param), axis=1)))


_GRASP_Q = []
GRASP_SEED_Q = np.array([3.0, -1.6, 1.9, -1.8, -1.6, 0.0])  # the arm's working pose in simulate()


def grasp_poses() -> np.ndarray:
    """Joint angles at a 3x3 grid of grasp spots (where the conveyor has
    carried the parts by grasp time), tool pointing down."""
    if not _GRASP_Q:
        for x in (0.38, 0.47, 0.57):
            for y in (-0.08, 0.02, 0.12):
                _GRASP_Q.append(robot.ik(robot.tool_target(np.array([x, y, 0.04]), 0.0), TCP_NOMINAL, GRASP_SEED_Q))
    return np.array(_GRASP_Q)


def tip_shift_from_bias(bias: np.ndarray) -> np.ndarray:
    """Fingertip error (m) at the grasp grid for a joint bias (rad)."""
    q = grasp_poses()
    F0, F1 = robot.fk(q), robot.fk(q - bias)
    return (np.einsum("nij,j->ni", F1[:, :3, :3], TCP_NOMINAL) + F1[:, :3, 3]
            - np.einsum("nij,j->ni", F0[:, :3, :3], TCP_NOMINAL) - F0[:, :3, 3])


def _sag_pose(drift: dict, a: float) -> np.ndarray:
    T = CAM_NOMINAL.copy()
    T[:3, :3] = Rotation.from_rotvec(np.array(drift["rotvec_rate"]) * a).as_matrix() @ CAM_NOMINAL[:3, :3]
    T[:3, 3] += np.array(drift["translation_rate"]) * a
    return T


def random_drift(kind: str, rng: np.random.Generator, fail_s: float | None = None) -> dict:
    """A slow drift that uses up the whole grasp tolerance about fail_s
    seconds after it starts (default 100-160 s, sped up)."""
    if kind == "none":
        return {"type": "none"}
    T = float(fail_s if fail_s is not None else rng.uniform(100, 160))
    start = float(rng.uniform(20, 40))
    if kind == "camera_sag":
        # the mount droops: a tilt about a horizontal axis plus a small drop
        phi = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(phi), np.sin(phi), 0.0])
        tdir = np.array([0.0, 0.0, -1.0]) + 0.3 * rng.normal(size=3)
        tdir /= np.linalg.norm(tdir)
        d = {"type": kind, "start_s": start, "rotvec_rate": (axis * 1e-3).tolist(),
             "translation_rate": (tdir * 0.3e-3).tolist()}
        k = 1.0 / T / tolerance_used(camera_shift(_sag_pose(d, 1.0), CAM_NOMINAL))
        d["rotvec_rate"] = (axis * 1e-3 * k).tolist()
        d["translation_rate"] = (tdir * 0.3e-3 * k).tolist()
        d["pick_mm_per_min"] = pick_area_shift(_sag_pose(d, 60.0), CAM_NOMINAL) * 1000
        d["deg_per_min"] = float(np.rad2deg(np.linalg.norm(d["rotvec_rate"]) * 60))
    elif kind == "joint_creep":
        j = int(rng.integers(0, 5))
        b = np.zeros(6)
        b[j] = 1e-3
        shift = tip_shift_from_bias(b)
        rate = 1e-3 / T / tolerance_used(shift)
        d = {"type": kind, "start_s": start, "joint": j, "rate_rad_s": float(rng.choice([-1, 1]) * rate)}
        d["pick_mm_per_min"] = float(np.mean(np.linalg.norm(shift, axis=1)) / 1e-3 * rate * 60 * 1000)
        d["deg_per_min"] = float(np.rad2deg(rate * 60))
    else:
        raise ValueError(kind)
    d["fail_s"] = T
    return d


def drift_state(drift: dict, t) -> tuple:
    """(true camera pose(s), joint bias(es)) at time(s) t."""
    t = np.atleast_1d(np.asarray(t, float))
    a = np.maximum(t - drift.get("start_s", 0.0), 0.0)
    bias = np.zeros((len(t), 6))
    cams = np.repeat(CAM_NOMINAL[None], len(t), axis=0)
    if drift["type"] == "camera_sag":
        cams = np.stack([_sag_pose(drift, x) for x in a])
    elif drift["type"] == "joint_creep":
        bias[:, drift["joint"]] = drift["rate_rad_s"] * a
    return cams, bias


def simulate_drift(drift: dict, seed: int = 0, n_picks: int = 60) -> Log:
    """A long production run while the cell slowly drifts. The controller's
    settings never change; only the physical camera or joint moves."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / JOINT_HZ
    waypoints = [(SEED_Q.copy(), 0.0, 0.3)]
    q_prev = SEED_Q.copy()
    t_cursor = 0.3
    picks = []
    for i in range(n_picks):
        p_obj = np.array([rng.uniform(0.35, 0.6), rng.uniform(-0.55, -0.3), 0.04])
        t_detect = t_cursor
        yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        tilt = rng.uniform(-0.3, 0.3, size=2)
        move = 0.9
        t_grasp = t_detect + 2 * move
        cam_true, _ = drift_state(drift, t_detect)
        p_cam = np.linalg.inv(cam_true[0]) @ np.append(p_obj, 1.0)
        p_seen = (CAM_NOMINAL @ p_cam)[:3] + rng.normal(size=3) * 0.0008
        p_plan = p_seen + np.array([0.0, CONVEYOR_V * (t_grasp - t_detect), 0.0])
        T_grasp = robot.tool_target(p_plan, yaw, *tilt)
        T_pre = T_grasp.copy()
        T_pre[2, 3] += 0.12
        q_pre = robot.ik(T_pre, TCP_NOMINAL, q_prev)
        q_grasp = robot.ik(T_grasp, TCP_NOMINAL, q_pre)
        place = np.array([rng.uniform(0.2, 0.35), rng.uniform(0.3, 0.45), 0.15])
        T_place = robot.tool_target(place, rng.uniform(-1.2, 1.2), *rng.uniform(-0.25, 0.25, 2))
        q_place = robot.ik(T_place, TCP_NOMINAL, q_grasp)
        waypoints += [(q_pre, move, 0.0), (q_grasp, move, 0.25), (q_pre, 0.6, 0.0), (q_place, 1.0, 0.25)]
        t_cursor = t_grasp + 0.25 + 0.6 + 1.0 + 0.25

        _, bias = drift_state(drift, t_grasp)
        F = robot.fk(q_grasp - bias[0])
        tip = F[:3, :3] @ TCP_NOMINAL + F[:3, 3]
        obj = p_obj + np.array([0.0, CONVEYOR_V * (t_grasp - t_detect), 0.0])
        miss = tip - obj + rng.normal(size=3) * 0.002
        ok = np.linalg.norm(miss[:2]) < GRASP_TOL and abs(miss[2]) < 0.015
        picks.append((t_grasp, ok))
        q_prev = q_place

    t_joint, q_cmd = _build_trajectory(waypoints, dt)
    q_meas = _interp(t_joint, q_cmd, t_joint - TAU0) + rng.normal(size=q_cmd.shape) * ENC_NOISE

    t_cam = np.arange(0.0, t_joint[-1], 1.0 / CAM_HZ) + rng.uniform(0, 1.0 / CAM_HZ)
    t_cam = t_cam[t_cam < t_joint[-1]]
    cams, bias = drift_state(drift, t_cam)
    Tinv = np.linalg.inv(cams)
    q_true = _interp(t_joint, q_cmd, t_cam - TAU0) - bias
    world = robot.transform_points(robot.fk(q_true), MARKERS)
    markers_cam = np.einsum("nij,nkj->nki", Tinv[:, :3, :3], world) + Tinv[:, None, :3, 3]
    markers_cam += rng.normal(size=markers_cam.shape) * CAM_NOISE
    fixed_cam = np.einsum("nij,kj->nki", Tinv[:, :3, :3], FIXED) + Tinv[:, None, :3, 3]
    fixed_cam += rng.normal(size=fixed_cam.shape) * CAM_NOISE
    visible = rng.random(len(t_cam)) > 0.1

    config = base_config()
    config["start_time"] = "2026-09-24T06:00:00"
    config["events"] = []
    return Log(
        t_joint=t_joint, q_cmd=q_cmd, q_meas=q_meas, t_cam=t_cam, markers_cam=markers_cam,
        visible=visible, pick_t=np.array([p[0] for p in picks]),
        pick_ok=np.array([p[1] for p in picks]), config=config,
        truth={"drift": drift, "seed": seed}, fixed_cam=fixed_cam,
    )
