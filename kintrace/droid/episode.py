"""Read one raw DROID episode.

A raw episode folder (gs://gresearch/robotics/droid_raw/1.0.1/<lab>/<success|failure>/<date>/<name>/) holds

  trajectory.h5             joint stream, timestamps, per-step camera extrinsics
  metadata_<name>.json      camera serials, configured extrinsics, episode info
  recordings/MP4/<serial>.mp4   fixed and wrist cameras (stereo side by side)
  recordings/SVO/<serial>.svo   ZED native, with depth. Not read here.

h5 key names below are the ones in DROID 1.0.1. Anything missing is
reported rather than guessed, since the point of this adapter is honesty
about what the data really contains.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .extrinsics import configured_from_metadata, pose6_to_T

JOINTS = "observation/robot_state/joint_positions"
JOINTS_CMD = "action/joint_position"
GRIPPER = "observation/robot_state/gripper_position"
T_ROBOT = ("observation/timestamp/robot_state/read_start",
           "observation/timestamp/robot_state/robot_timestamp_seconds")
CAM_EXT = "observation/camera_extrinsics"
T_CAM_PREFIX = "observation/timestamp/cameras"
DEFAULT_HZ = 15.0


@dataclass
class Episode:
    path: str
    meta: dict
    t: np.ndarray            # (N,) s, robot state timestamps
    q: np.ndarray            # (N, 7) measured joints, rad
    q_cmd: np.ndarray        # (N, 7) commanded joints (action), rad
    gripper: np.ndarray      # (N,) 0 open .. 1 closed
    configured: dict         # {serial: 4x4} extrinsics the cell ran with
    per_step_ext: dict = field(default_factory=dict)  # {serial: (N, 4, 4)} if logged
    t_cam: dict = field(default_factory=dict)         # {serial: (N,)} capture times if logged
    serials: dict = field(default_factory=dict)       # {"ext1": serial, "ext2": ..., "wrist": ...}
    missing: list = field(default_factory=list)

    @property
    def key(self) -> str:
        """Key into the corrected extrinsics: the episode's path under the bucket root."""
        return self.meta.get("_key") or os.path.basename(self.path.rstrip("/"))

    def video(self, serial: str) -> str | None:
        p = os.path.join(self.path, "recordings", "MP4", f"{serial}.mp4")
        return p if os.path.exists(p) else None

    def frames(self, serial: str, every: int = 1, left_only: bool = True):
        """Yield (index, frame) from the camera's MP4. ZED MP4s are left|right side by side."""
        import cv2  # local import, only needed for the detect level

        p = self.video(serial)
        if p is None:
            raise FileNotFoundError(f"no MP4 for camera {serial} in {self.path}")
        cap = cv2.VideoCapture(p)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                if left_only and frame.shape[1] >= 2 * frame.shape[0]:
                    frame = frame[:, : frame.shape[1] // 2]
                yield i, frame
            i += 1
        cap.release()


def _first(h5, keys):
    for k in keys:
        if k in h5:
            return np.asarray(h5[k])
    return None


def load(path: str, key: str | None = None) -> Episode:
    import h5py

    metas = sorted(glob.glob(os.path.join(path, "metadata_*.json")))
    meta = {}
    if metas:
        with open(metas[0], encoding="utf-8") as f:
            meta = json.load(f)
    if key:
        meta["_key"] = key
    missing = []
    with h5py.File(os.path.join(path, "trajectory.h5"), "r") as h5:
        q = _first(h5, [JOINTS])
        if q is None:
            raise KeyError(f"{JOINTS} not in trajectory.h5")
        q = q.astype(float)
        n = len(q)
        q_cmd = _first(h5, [JOINTS_CMD])
        if q_cmd is None:
            missing.append(JOINTS_CMD)
            q_cmd = q.copy()
        grip = _first(h5, [GRIPPER])
        if grip is None:
            missing.append(GRIPPER)
            grip = np.zeros(n)
        t = _first(h5, list(T_ROBOT))
        if t is None:
            missing.append(T_ROBOT[0])
            t = np.arange(n) / DEFAULT_HZ
        else:
            t = t.astype(float)
            if t.max() > 1e9 * 1000:   # nanoseconds
                t = t / 1e9
            t = t - t[0]
        per_step, t_cam = {}, {}
        if CAM_EXT in h5:
            for name in h5[CAM_EXT]:
                arr = np.asarray(h5[CAM_EXT][name]).astype(float)
                if arr.ndim == 2 and arr.shape[1] == 6:
                    serial = name[:-5] if name.endswith(("_left", "_right")) else name
                    if serial not in per_step:
                        per_step[serial] = np.stack([pose6_to_T(p) for p in arr])
        if T_CAM_PREFIX in h5:
            for name in h5[T_CAM_PREFIX]:
                if name.endswith("_estimated_capture"):
                    serial = name[: -len("_estimated_capture")]
                    tc = np.asarray(h5[T_CAM_PREFIX][name]).astype(float)
                    if tc.max() > 1e9 * 1000:
                        tc = tc / 1e9
                    t_cam[serial] = tc
    configured = configured_from_metadata(meta)
    if not configured and per_step:
        # metadata missing: fall back to the first logged per-step pose
        configured = {s: T[0] for s, T in per_step.items()}
        missing.append("metadata extrinsics (using per-step h5 pose)")
    serials = {k: str(meta[f"{k}_cam_serial"]) for k in ("ext1", "ext2", "wrist") if f"{k}_cam_serial" in meta}
    return Episode(path, meta, t, q, q_cmd.astype(float), np.asarray(grip, dtype=float).ravel()[:n],
                   configured, per_step, t_cam, serials, missing)
