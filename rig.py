"""The desk rig: config file, tag geometry, check poses and commissioning.

Frames:
  world   = the SO-101 base_link frame (from the URDF). Everything is in meters.
  flange  = the SO-101 gripper_link frame. Its -z axis points out through the jaws.
  camera  = OpenCV camera frame (x right, y down, z forward).
  tag     = x right, y up, z out of the printed face.

rig.json plays the role of the robot controller's settings: the tool offset,
the camera calibration and the joint zero offsets live there, and Kintrace
writes its fixes there.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .. import arms, robot

ARM = arms.SO101

# Nominal tag mounting (flange frame). Both tags face the same way, along the
# flange -y axis. The wrist roll joint turns them toward the camera, so which
# side you pick only matters for this first guess; commissioning measures the
# real wrist plate pose and tip position.
_TAG_R = np.stack([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]], axis=1)  # columns: tag x, y, z in flange
NOMINAL_WRIST_TAG = np.r_[Rotation.from_matrix(_TAG_R).as_rotvec(), [0.0, -0.030, -0.045]]
# pointer tip: about 50 mm past the jaw tip (URDF gripper_frame), tip flag centered on it
NOMINAL_TIP = (arms.SO101_GRIPPER_FRAME_XYZ + np.array([0.0, 0.0, -0.050])).round(4)


def look_at_cv(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera pose in the world with OpenCV axes: z forward, x right, y down
    in the image. (robot.look_at gives an upside-down image, which is fine
    for the point-only UR5e sim but not for rendered frames.)"""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def default_config() -> dict:
    guess = dict(eye=[0.20, -0.40, 0.42], target=[0.21, 0.02, 0.06])
    return dict(
        arm="so101",
        robot="SO-101 desk rig",
        port="/dev/ttyACM0",
        robot_id="kintrace_arm",
        camera=dict(index=0, width=1280, height=720, fps=30, intrinsics="camera.json",
                    focus=0, frame_latency_s=0.0),
        tags=dict(wrist_id=0, wrist_size=0.040, tip_id=1, tip_size=0.020,
                  table_ids=[10, 11, 12, 13], table_size=0.050, target_id=20, target_size=0.050),
        wrist_tag_pose=NOMINAL_WRIST_TAG.round(5).tolist(),
        tcp_offset=NOMINAL_TIP.tolist(),
        camera_guess=guess,
        camera_extrinsic=look_at_cv(guess["eye"], guess["target"]).tolist(),
        fixed_markers=None,
        joint_zero_offsets_deg=[0.0] * ARM.n,
        joint_sign=[1] * ARM.n,
        gripper_hold=10.0,
        speed_deg_s=40.0,
        # moves that would bring the flange or tool tip lower than this over
        # the table go through a raised pose instead
        min_tip_height_m=0.04,
        control_hz=50,
        dwell_s=1.0,
        settle_s=0.4,
        check_poses_deg=[],
        # tip_mm: pointer tip vs known-good; tau_ms: command delay change;
        # touch_mm: touch test miss. Wider than the simulated cell because a
        # webcam and hobby servos are noisier. Tune after the first real runs.
        # tag_noise_floor_mm: the smallest tag noise (x, y, depth in the
        # camera frame) the checks will assume, however quiet commissioning
        # was. A webcam tag pose is not better than this, and without a floor
        # a lucky quiet commissioning run makes the checks fire on noise.
        tolerances=dict(tip_mm=3.0, tau_ms=8.0, touch_mm=4.0, tag_noise_floor_mm=[0.5, 0.5, 1.5]),
        touch=dict(hover_mm=15.0),
        commissioned=False,
    )


def load(path: str) -> dict:
    cfg = default_config()
    with open(path) as f:
        user = json.load(f)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def save(cfg: dict, path: str, backup: bool = True) -> None:
    if backup and os.path.exists(path):
        os.replace(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def pose6(x) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    return T


def as_pose6(T) -> np.ndarray:
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def tag_points(size: float) -> np.ndarray:
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def wrist_markers(cfg: dict) -> np.ndarray:
    """Wrist tag corners in the flange frame, in detector order. These are
    the log's config["markers"]."""
    T = pose6(cfg["wrist_tag_pose"])
    return (T[:3, :3] @ tag_points(cfg["tags"]["wrist_size"]).T).T + T[:3, 3]


def controller_config(cfg: dict) -> dict:
    """The settings block written into each Log (what the rig 'controller'
    is running with), in the shape the diagnosis code expects."""
    out = dict(
        arm="so101",
        robot=cfg.get("robot", "SO-101 desk rig"),
        tcp_offset=list(cfg["tcp_offset"]),
        camera_extrinsic=np.asarray(cfg["camera_extrinsic"]).tolist(),
        markers=wrist_markers(cfg).tolist(),
        joint_zero_offsets_deg=list(cfg["joint_zero_offsets_deg"]),
        tags=cfg["tags"],
        events=[],
    )
    if cfg.get("fixed_markers") is not None:
        out["fixed_markers"] = cfg["fixed_markers"]
    return out


# --------------------------------------------------------------------------
# inverse kinematics for planning (5 joints: position + one direction)
# --------------------------------------------------------------------------
def ik_point_dir(p_flange, target, d_flange=None, d_world=None, seed=None, w_dir=0.3,
                 extra=None, rng=None):
    """Joint angles that put flange point p_flange at target (world) and,
    softly, turn flange direction d_flange toward d_world."""
    seed = ARM.home_q if seed is None else np.asarray(seed, float)
    lo, hi = arms.SO101_LIMITS[:, 0] + 0.05, arms.SO101_LIMITS[:, 1] - 0.05
    rng = rng or np.random.default_rng(0)

    def res(q):
        F = ARM.fk(q)
        r = [(F[:3, :3] @ p_flange + F[:3, 3] - target) * 100.0]  # position first
        if d_flange is not None:
            r.append((F[:3, :3] @ d_flange - d_world) * w_dir)
        if extra is not None:
            r.append(extra(F))
        r.append((q - seed) * 0.002)  # tiny pull toward the seed, keeps the solve well posed
        return np.concatenate(r)

    # collect solutions from many starts, then keep the one closest to the
    # seed among those that hit the point and face about as well as the best
    sols = []
    seeds = [np.clip(seed, lo, hi)] + [np.clip(seed + rng.normal(scale=0.5, size=ARM.n), lo, hi)
                                       for _ in range(12)] + [rng.uniform(lo, hi) for _ in range(12)]
    for s0 in seeds:
        x = least_squares(res, s0, bounds=(lo, hi)).x
        r = res(x)
        soft = r[3:-ARM.n]  # facing + extra terms
        sols.append((np.linalg.norm(r[:3]) / 100.0, np.linalg.norm(soft), x))
    ok = [s for s in sols if s[0] < 0.002] or sols
    best_dir = min(s[1] for s in ok)
    ok = [s for s in ok if s[1] <= best_dir + 0.05]
    q = min(ok, key=lambda s: np.linalg.norm(s[2] - seed))[2]
    F = ARM.fk(q)
    if np.linalg.norm(F[:3, :3] @ p_flange + F[:3, 3] - target) > 0.005:
        raise RuntimeError(f"target {np.round(target, 3)} is out of reach")
    return q


def plan_check_poses(cfg: dict, n: int = 8, tilt: float = 0.5) -> np.ndarray:
    """Check poses (rad): the wrist plate at a few spots in front of the
    camera, turned toward it and tilted about 30 deg different ways, so the
    tags are seen at an angle (head-on views give poor depth)."""
    eye = np.array(cfg["camera_guess"]["eye"])
    Tw = pose6(cfg["wrist_tag_pose"])
    n_f = Tw[:3, 2]
    base = np.array([0.21, 0.0, 0.19])  # wrist plate height: keeps the tip 4+ cm off the table
    offs = [[0, 0, 0], [0.05, 0.05, 0.03], [-0.04, 0.05, -0.02], [0.05, -0.04, 0.04],
            [-0.04, -0.04, 0.05], [0.0, 0.06, -0.03], [0.05, 0.0, 0.06], [-0.05, 0.02, -0.03]]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    poses, q = [], ARM.home_q.copy()
    for i in range(n):
        c = base + np.array(offs[i % len(offs)])
        d = eye - c
        d /= np.linalg.norm(d)
        # tilt the facing direction away from the camera axis
        a = np.cross(d, [0, 0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        axis = np.cos(angles[i]) * a + np.sin(angles[i]) * b
        d_t = Rotation.from_rotvec(axis * tilt).apply(d)
        # of all the ways to face the tag at the camera, prefer the pointer
        # hanging as close to straight down as it can: same posture family
        # every time, no wrist flips between poses, tip kept low and safe
        down = np.array([0.0, 0.0, -1.0])
        g = down - (down @ d_t) * d_t
        g /= np.linalg.norm(g)
        q = ik_point_dir(Tw[:3, 3], c, n_f, d_t, seed=q, w_dir=0.3,
                         extra=lambda F, g=g: (F[:3, :3] @ [0.0, 0.0, -1.0] - g) * 0.2)
        poses.append(q)
    return np.array(poses)


def path_min_height(q0, q1, cfg: dict, steps: int = 30) -> float:
    """Lowest height (m) of the flange or the tool tip on a straight joint
    move from q0 to q1. Used to keep the check motion off the table."""
    s = np.linspace(0, 1, steps)[:, None]
    F = ARM.fk(q0 + (q1 - q0) * s)
    tip = np.einsum("nij,j->ni", F[:, :3, :3], np.asarray(cfg["tcp_offset"])) + F[:, :3, 3]
    return float(min(F[:, 2, 3].min(), tip[:, 2].min()))


def check_poses(cfg: dict) -> np.ndarray:
    if cfg.get("check_poses_deg"):
        return np.deg2rad(np.array(cfg["check_poses_deg"], float))
    return plan_check_poses(cfg)


# --------------------------------------------------------------------------
# commissioning: learn the camera pose, tag mounting and joint zeros
# --------------------------------------------------------------------------
def _group_by_pose(q, tol=np.deg2rad(0.5)):
    """Labels for runs of frames taken at the same (static) arm pose."""
    lab = np.zeros(len(q), int)
    for i in range(1, len(q)):
        lab[i] = lab[i - 1] + (np.abs(q[i] - q[i - 1]).max() > tol)
    return lab


def _kabsch(A, B):
    """R, t with R A + t ~= B (points as rows)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _hand_eye_init(q, obs, size, sign):
    """Closed-form first guess from OpenCV's robot-world/hand-eye solver.
    Here the 'camera on the gripper' is our wrist tag and the 'world
    target' is our fixed camera."""
    import cv2

    lab = _group_by_pose(q)
    obj = tag_points(size)
    Rw, tw, Rb, tb = [], [], [], []
    for k in np.unique(lab):
        m = lab == k
        if m.sum() < 2:
            continue
        R_ct, t_ct = _kabsch(obj, obs[m].mean(0))  # tag -> camera
        T_ct = np.eye(4)
        T_ct[:3, :3], T_ct[:3, 3] = R_ct, t_ct
        T_tc = np.linalg.inv(T_ct)  # camera -> tag  ("world2cam" in OpenCV terms)
        F = ARM.fk(q[m].mean(0) * sign)
        T_fb = np.linalg.inv(F)  # base -> flange ("base2gripper")
        Rw.append(T_tc[:3, :3]); tw.append(T_tc[:3, 3])
        Rb.append(T_fb[:3, :3]); tb.append(T_fb[:3, 3])
    if len(Rw) < 3:
        raise RuntimeError("need the tag seen at 3 or more different arm poses")
    R_bw, t_bw, R_gc, t_gc = cv2.calibrateRobotWorldHandEye(Rw, tw, Rb, tb)
    T_cam_base = np.eye(4)
    T_cam_base[:3, :3], T_cam_base[:3, 3] = R_bw, t_bw.ravel()  # base -> camera
    T_tag_flange = np.eye(4)
    T_tag_flange[:3, :3], T_tag_flange[:3, 3] = R_gc, t_gc.ravel()  # flange -> tag
    return np.linalg.inv(T_cam_base), np.linalg.inv(T_tag_flange)


def fit_geometry(q, obs, size, sign=None, fit_zeros=(1, 2, 3), init=None):
    """Least-squares fit of camera pose (world), wrist tag pose (flange) and
    zero offsets for the joints in fit_zeros.

    q: (N, 5) joint readings (rad), obs: (N, 4, 3) wrist tag corners in the
    camera frame. Shoulder pan and wrist roll zeros are left out: a pan
    offset only turns the whole world frame, and a roll offset only turns
    the tag mounting, so both are absorbed by the other unknowns.
    Returns dict with camera (4x4), wrist_tag (4x4), zeros (rad, 5), rms (m).
    """
    sign = np.ones(ARM.n) if sign is None else np.asarray(sign, float)
    fit_zeros = list(fit_zeros)
    obj = tag_points(size)
    if init is None:
        cam0, tag0 = _hand_eye_init(q, obs, size, sign)
    else:
        cam0, tag0 = init
    x0 = np.r_[as_pose6(cam0), as_pose6(tag0), np.zeros(len(fit_zeros))]

    def unpack(x):
        z = np.zeros(ARM.n)
        z[fit_zeros] = x[12:]
        return pose6(x[:6]), pose6(x[6:12]), z

    def res(x):
        C, W, z = unpack(x)
        F = ARM.fk(q * sign - z)
        pw = robot.transform_points(F, (W[:3, :3] @ obj.T).T + W[:3, 3])
        Ci = np.linalg.inv(C)
        pc = np.einsum("ij,nkj->nki", Ci[:3, :3], pw) + Ci[:3, 3]
        r = (pc - obs).ravel()
        return np.r_[r, x[12:] * 0.01]  # weak pull of the zero offsets toward 0

    s = least_squares(res, x0, loss="soft_l1", f_scale=0.003)
    C, W, z = unpack(s.x)
    r = res(s.x)[: obs.size].reshape(-1, 3)
    return dict(camera=C, wrist_tag=W, zeros=z, rms=float(np.sqrt(np.mean(np.sum(r**2, 1)))), sign=sign)


def fit_tip(q, tip_cam, camera):
    """Tool tip position in the flange frame from tip sightings (rad, camera frame)."""
    F = ARM.fk(q)
    w = (camera[:3, :3] @ tip_cam.T).T + camera[:3, 3]
    A = F[:, :3, :3].reshape(-1, 3)
    b = (w - F[:, :3, 3]).reshape(-1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.sqrt(np.mean(np.sum((A @ x - b).reshape(-1, 3) ** 2, 1))))
    return x, rms


def commission(log, cfg: dict, search_signs: bool = False, echo=print) -> tuple[dict, dict]:
    """Learn the rig geometry from one healthy check capture.

    Returns (new rig config, report dict). The log's q_meas already has the
    rig's current zero offsets applied.
    """
    import itertools

    p = log.probe
    size = cfg["tags"]["wrist_size"]
    from ..diagnose import _interp

    q = _interp(log.t_joint, log.q_meas, log.t_cam)
    obs = log.markers_cam
    candidates = [np.ones(ARM.n)]
    if search_signs:
        candidates = [np.array(s, float) for s in itertools.product([1, -1], repeat=ARM.n)]
    best = None
    for sgn in candidates:
        try:
            fit = fit_geometry(q, obs, size, sign=sgn)
        except Exception:
            continue
        if best is None or fit["rms"] < best["rms"]:
            best = fit
    if best is None:
        raise RuntimeError("could not fit the rig geometry; check that the wrist tag was seen at 3+ poses")
    fit = best
    sign_new = (np.array(cfg["joint_sign"], float) * fit["sign"]).astype(int)
    zeros_deg = np.array(cfg["joint_zero_offsets_deg"], float) * fit["sign"] + np.rad2deg(fit["zeros"])

    # tool tip, using frames where the tip flag was seen
    qp = _interp(p["t_joint"], p["q_meas"], p["t_cam"]) * fit["sign"] - fit["zeros"]
    tip, tip_rms = fit_tip(qp, p["tip_cam"], fit["camera"])

    # table tags, in the world frame
    fixed = None
    if log.fixed_cam is not None:
        fc = np.median(log.fixed_cam, axis=0)
        C = fit["camera"]
        fixed = ((C[:3, :3] @ fc.T).T + C[:3, 3]).tolist()

    new = copy.deepcopy(cfg)
    new.update(
        camera_extrinsic=fit["camera"].tolist(),
        wrist_tag_pose=as_pose6(fit["wrist_tag"]).round(6).tolist(),
        tcp_offset=tip.round(6).tolist(),
        joint_zero_offsets_deg=zeros_deg.round(4).tolist(),
        joint_sign=sign_new.tolist(),
        fixed_markers=fixed,
        commissioned=True,
    )
    new["camera_guess"] = dict(eye=fit["camera"][:3, 3].round(4).tolist(), target=cfg["camera_guess"]["target"])
    rep = dict(wrist_rms_mm=fit["rms"] * 1000, tip_rms_mm=tip_rms * 1000,
               zeros_added_deg=np.rad2deg(fit["zeros"]).tolist(), sign=fit["sign"].tolist())
    return new, rep
