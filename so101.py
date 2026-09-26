"""Thin reader/writer for a LeRobot SO-101 follower arm.

Uses the `lerobot` package (imported only when you connect). Joint angles
in and out of this class are in the Kintrace model convention: radians,
URDF directions, with the rig's joint zero offsets applied:

    q_model = sign * radians(lerobot_degrees) - radians(joint_zero_offsets_deg)

LeRobot reports body joints in degrees (use_degrees=True), where 0 is the
middle of each joint's calibrated range. The gripper is held at a fixed
value (0-100) because the pointer is clipped to the fixed jaw.

Not tested on a real arm yet. Things to check the first time:
  * the import path (lerobot moved the SO-101 classes between versions)
  * that joint directions match the URDF (see `kintrace rig commission --search-signs`)
  * read/write timing at the control rate you ask for
"""
from __future__ import annotations

import time

import numpy as np

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"


def _import_lerobot():
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # lerobot, Sept 2026 main
        return SO101Follower, SO101FollowerConfig
    except ImportError:
        pass
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig  # older lerobot
        return SO101Follower, SO101FollowerConfig
    except ImportError as e:
        raise RuntimeError(
            "The real arm needs the `lerobot` package with Feetech support:\n"
            "  pip install 'lerobot[feetech]'\n"
            "Then calibrate the arm once with LeRobot (lerobot-calibrate) before using Kintrace.\n"
            "To try everything without hardware, add --dry-run."
        ) from e


class SO101Arm:
    def __init__(self, cfg: dict, max_step_deg: float = 4.0):
        self.cfg = cfg
        self.off = np.deg2rad(np.array(cfg["joint_zero_offsets_deg"], float))
        self.sign = np.array(cfg["joint_sign"], float)
        self.gripper = float(cfg.get("gripper_hold", 10.0))
        self.max_step = np.deg2rad(max_step_deg)  # per write, a speed limit of last resort
        self.robot = None
        self._last = None

    def connect(self):
        Follower, Config = _import_lerobot()
        try:
            # keep holding position when Kintrace exits, so the arm doesn't drop
            conf = Config(port=self.cfg["port"], id=self.cfg["robot_id"], use_degrees=True,
                          disable_torque_on_disconnect=False)
        except TypeError:  # older lerobot without that option
            conf = Config(port=self.cfg["port"], id=self.cfg["robot_id"], use_degrees=True)
        self.robot = Follower(conf)
        if not self.robot.calibration:
            raise RuntimeError(
                f"no LeRobot calibration file for arm id '{self.cfg['robot_id']}'. "
                "Run lerobot-calibrate with that id first.")
        # If the servos disagree with the calibration file (for example after
        # you edited it on purpose), LeRobot asks whether to write the file to
        # the servos. Press ENTER to use the file.
        self.robot.connect(calibrate=True)
        return self

    def read(self):
        """(timestamp, q_model) from the servos' present positions."""
        obs = self.robot.bus.sync_read("Present_Position")
        t = time.monotonic()
        raw = np.deg2rad(np.array([obs[m] for m in MOTORS], float))
        q = self.sign * raw - self.off
        if self._last is None:
            self._last = q
        return t, q

    def write(self, q_model):
        """Send joint goals (q_model, rad). Returns the send time."""
        from ..arms import SO101_LIMITS

        q = np.clip(np.asarray(q_model, float), SO101_LIMITS[:, 0], SO101_LIMITS[:, 1])
        if self._last is not None:
            q = self._last + np.clip(q - self._last, -self.max_step, self.max_step)
        self._last = q
        raw_deg = np.rad2deg(self.sign * (q + self.off))
        goal = {m: float(v) for m, v in zip(MOTORS, raw_deg)}
        goal[GRIPPER] = self.gripper
        self.robot.bus.sync_write("Goal_Position", goal)
        return time.monotonic()

    def torque(self, on: bool):
        if on:
            self.robot.bus.enable_torque()
        else:
            self.robot.bus.disable_torque()

    def close(self):
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None
