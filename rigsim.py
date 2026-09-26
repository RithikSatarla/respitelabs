"""Dry-run stand-ins for the SO-101 and the webcam.

The simulated rig is deliberately a bit different from the nominal rig.json
(camera a few cm off the guess, tag glued slightly crooked, joint zeros off
by a degree or two) so commissioning has something real to find. The same
seed always builds the same rig, so `commission` and later `check` runs
see the same desk.

Faults, same names as the rest of Kintrace:
  camera_moved   the camera clamp got bumped
  tool_bent      the pointer tip got bent or swapped
  tcp_config     someone edited tcp_offset in rig.json (applied by the CLI)
  encoder_bias   a servo's zero shifted (calibration file edit)
  latency        commands delayed (applied by the capture loop's delay line)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .. import arms
from . import render, rig
from .markers import Intrinsics, make_detector, observe

ARM = arms.SO101
TICK = 2 * np.pi / 4096  # STS3215: 12-bit encoder
# where the dry-run arm starts: raised, pointer tip about 9 cm above the table
READY_DEG = [0.0, -80.0, 80.0, 20.0, 0.0]


class SimClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def wait_until(self, t: float) -> float:
        self.t = max(self.t, t)
        return self.t


class SimRig:
    """The physical desk: true geometry, servo behavior and faults."""

    def __init__(self, cfg: dict, seed: int = 0, fault: dict | None = None):
        rng = np.random.default_rng(1000 + seed)
        self.cfg = cfg
        self.fault = fault or {"type": "none"}
        # the desk is built from the default guess, not cfg["camera_guess"],
        # so commissioning (which rewrites the guess) doesn't move the desk
        g = rig.default_config()["camera_guess"]
        cam = rig.look_at_cv(g["eye"], g["target"])
        # true camera: the tape-measure guess was off by a few cm and degrees
        cam[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * 0.03).as_matrix() @ cam[:3, :3]
        cam[:3, 3] += rng.normal(size=3) * 0.015
        self.cam = cam
        self.intr = Intrinsics.guess(cfg["camera"]["width"], cfg["camera"]["height"])
        # tags as actually glued on
        W = rig.pose6(rig.NOMINAL_WRIST_TAG)
        W[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * 0.04).as_matrix() @ W[:3, :3]
        W[:3, 3] += rng.normal(size=3) * 0.003
        self.wrist_tag = W
        self.tip = rig.NOMINAL_TIP + rng.normal(size=3) * 0.002
        self.tip_R = W[:3, :3].copy()
        # table tags, flat, face up (tag z = world z), around the work area
        self.table = {tid: np.array(p) for tid, p in zip(
            cfg["tags"]["table_ids"],
            [[0.10, -0.10, 0.0], [0.34, -0.10, 0.0], [0.34, 0.16, 0.0], [0.10, 0.16, 0.0]])}
        self.target = np.array([0.24, 0.02, 0.0])
        # servo zeros that don't match the URDF zero (what commissioning absorbs)
        self.zero_err = np.deg2rad(np.array([0.7, 1.5, -1.0, 0.8, -1.2]))
        self.tau = 0.035  # s, servo lag behind the goal
        self._apply_fault(self.fault)

    def _apply_fault(self, f: dict):
        for ft in f.get("faults", [f]):
            t = ft["type"]
            if t == "camera_moved":
                self.cam[:3, :3] = Rotation.from_rotvec(ft["rotvec"]).as_matrix() @ self.cam[:3, :3]
                self.cam[:3, 3] += np.array(ft["translation"])
            elif t == "tool_bent":
                self.tip = self.tip + np.array(ft["delta"])
            elif t == "encoder_bias":
                self.zero_err[ft["joint"]] += ft["bias"]
            # tcp_config and latency are applied outside the physical world

    # the true joint angles given the raw goal history
    def q_true(self, t, hist_t, hist_raw):
        tt = np.atleast_1d(t) - self.tau
        raw = np.stack([np.interp(tt, hist_t, hist_raw[:, i]) for i in range(ARM.n)], 1)
        return raw - self.zero_err

    def tag_poses_cam(self, q):
        """(tag_id, size, T_cam_tag) for every tag, arm at q (true angles)."""
        tags = self.cfg["tags"]
        F = ARM.fk(q)
        Ci = np.linalg.inv(self.cam)
        out = [(tags["wrist_id"], tags["wrist_size"], Ci @ F @ self.wrist_tag)]
        Tt = np.eye(4)
        Tt[:3, :3], Tt[:3, 3] = self.tip_R, self.tip
        out.append((tags["tip_id"], tags["tip_size"], Ci @ F @ Tt))
        for tid, p in list(self.table.items()) + [(tags["target_id"], self.target)]:
            T = np.eye(4)
            T[:3, 3] = p
            size = tags["table_size"] if tid != tags["target_id"] else tags["target_size"]
            out.append((tid, size, Ci @ T))
        return out


class SimArm:
    """Same interface as so101.SO101Arm."""

    def __init__(self, world: SimRig, clock: SimClock, cfg: dict, rng=None):
        self.w, self.clock, self.cfg = world, clock, cfg
        self.rng = rng or np.random.default_rng(0)
        self.off = np.deg2rad(np.array(cfg["joint_zero_offsets_deg"], float))
        self.sign = np.array(cfg["joint_sign"], float)
        # arm starts at rest: the raw goal equals where it sits
        self.hist_t = [-10.0, 0.0]
        self.hist_raw = [self.w.zero_err + np.deg2rad(READY_DEG)] * 2

    def connect(self):
        return self

    def raw_now(self, t):
        return self.q_true_at(t) + self.w.zero_err

    def q_true_at(self, t):
        return self.w.q_true(t, np.array(self.hist_t), np.array(self.hist_raw))[0]

    def read(self):
        t = self.clock.now()
        raw = self.raw_now(t)
        raw = np.round(raw / TICK) * TICK  # encoder ticks
        return t, self.sign * raw - self.off

    def write(self, q_model):
        t = self.clock.now()
        raw = self.sign * (np.asarray(q_model) + self.off)
        self.hist_t.append(t)
        self.hist_raw.append(raw)
        return t

    def torque(self, on: bool):
        pass

    def close(self):
        pass


class SimCamera:
    """Renders frames of the simulated desk and runs the real detector on them."""

    def __init__(self, world: SimRig, arm: SimArm, cfg: dict, fps: float = 15.0, rng=None,
                 save_dir: str | None = None):
        self.w, self.arm, self.cfg = world, arm, cfg
        self.dt = 1.0 / fps
        self.next_t = 0.0
        self.rng = rng or np.random.default_rng(1)
        self.detector = make_detector()
        self.frames = []
        self.save_dir = save_dir
        self.intr = world.intr  # what the rig "calibrated" (dry run: exact)
        self.t0 = 0.0

    def start(self, t0: float = 0.0):
        self.frames = []
        self.t0 = t0

    def poll(self, t: float, still: bool):
        """t is relative to the start of the motion, like the joint stream."""
        t_abs = self.arm.clock.now()
        if t_abs < self.next_t:
            return
        self.next_t = t_abs + self.dt
        if not still:  # like the real camera thread, only detect when the arm is still
            return
        img = self.render(t_abs)
        obs = observe(img, self.intr, self.cfg["tags"], self.detector)
        self.frames.append((t, obs))
        if self.save_dir:
            import cv2
            cv2.imwrite(f"{self.save_dir}/frame_{len(self.frames):04d}.jpg", img)

    def render(self, t: float):
        q = self.arm.q_true_at(t)
        return render.render(self.intr, self.w.tag_poses_cam(q), rng=self.rng)

    def grab(self):
        """One frame right now (for the touch test)."""
        t = self.arm.clock.now()
        return t, observe(self.render(t), self.intr, self.cfg["tags"], self.detector)

    def stop(self):
        return self.frames

    def close(self):
        pass


def random_fault(kind: str, rng: np.random.Generator) -> dict:
    """Faults sized like the ones you'd cause on the desk."""
    if kind == "none":
        return {"type": "none"}
    if kind == "camera_moved":  # a bump: ~1 cm and ~1.5 deg
        d = rng.normal(size=3)
        r = rng.normal(size=3)
        return {"type": kind, "translation": (d / np.linalg.norm(d) * rng.uniform(0.008, 0.015)).tolist(),
                "rotvec": (r / np.linalg.norm(r) * np.deg2rad(rng.uniform(1.0, 2.0))).tolist()}
    if kind == "tool_bent":  # pointer bent ~6-10 mm sideways
        d = rng.normal(size=3) * np.array([1.0, 1.0, 0.3])
        return {"type": kind, "delta": (d / np.linalg.norm(d) * rng.uniform(0.006, 0.010)).tolist()}
    if kind == "tcp_config":  # someone typed a wrong tool offset
        d = np.zeros(3)
        d[int(rng.integers(0, 3))] = rng.choice([-1, 1]) * rng.uniform(0.005, 0.010)
        return {"type": kind, "delta": d.tolist()}
    if kind == "encoder_bias":  # calibration file edit, 30-60 ticks
        j = int(rng.integers(1, 4))
        ticks = int(rng.integers(30, 61)) * int(rng.choice([-1, 1]))
        return {"type": kind, "joint": j, "bias": ticks * TICK, "ticks": ticks}
    if kind == "latency":  # sleep in the control loop
        return {"type": kind, "extra_s": float(rng.choice([0.04, 0.06, 0.08]))}
    raise ValueError(kind)
