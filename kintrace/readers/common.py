"""Shared pieces for the readers.

A reader turns a recording in someone else's format into a Stream: joint
angles over time, commanded angles if the recording has them, and camera
frames. assemble() then turns a Stream into a kintrace Log by locating the
wrist in each frame with whatever `locate` function fits the arm: the
printed AprilTag from the desk rig, or a learned detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from .. import arms
from ..logio import Log


@dataclass
class Stream:
    t: np.ndarray                 # (N,) s, joint stream timestamps, starting at 0
    q_meas: np.ndarray            # (N, J) rad
    q_cmd: np.ndarray | None      # (N, J) rad, or None when the recording has no commands
    frames: Callable[[], Iterator[tuple[float, np.ndarray]]] | None  # yields (t, image)
    source: str
    meta: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)   # what the recording did not have

    def describe(self) -> str:
        n = len(self.t)
        dur = float(self.t[-1] - self.t[0]) if n > 1 else 0.0
        hz = (n - 1) / dur if dur > 0 else 0.0
        lines = [f"{self.source}: {n} joint samples over {dur:.1f} s ({hz:.0f} Hz), {self.q_meas.shape[1]} joints"]
        lines.append("commanded joints: " + ("yes" if self.q_cmd is not None else "no (latency check skipped)"))
        lines.append("camera frames: " + ("yes" if self.frames is not None else "no (camera checks skipped)"))
        for m in self.missing:
            lines.append(f"missing: {m}")
        return "\n".join(lines)


def to_rad(q: np.ndarray, units: str) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if units == "rad":
        return q
    if units == "deg":
        return np.deg2rad(q)
    raise ValueError(f"units must be 'rad' or 'deg', got {units!r}")


def apply_zero(q: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Apply the rig's joint signs and zero offsets, if the config has them."""
    if not cfg:
        return q
    sign = np.array(cfg.get("joint_sign", [1.0] * q.shape[1]), float)
    off = np.deg2rad(np.array(cfg.get("joint_zero_offsets_deg", [0.0] * q.shape[1]), float))
    return q * sign - off


def locate_tag(intr, tag_id: int, size_m: float):
    """A locate function that finds one printed AprilTag and returns its
    four corners in the camera frame, like the desk rig."""
    from ..hw import markers

    det = markers.make_detector()

    def locate(image):
        found = markers.detect(image, det)
        if tag_id not in found:
            return None
        R, t, err, amb = markers.tag_pose(found[tag_id], size_m, intr)
        if err > 2.0 or amb > 0.6:
            return None
        return markers.tag_corners_cam(R, t, size_m)

    return locate


def assemble(stream: Stream, config: dict, locate=None, every: int = 1, max_frames: int | None = None) -> Log:
    """Build a Log. config needs arm, camera_extrinsic, markers, tcp_offset.

    locate(image) -> (K, 3) points in the camera frame or None. K must match
    len(config["markers"]). With no locate or no frames, the Log has no
    visible frames and the camera checks are skipped.
    """
    arm = arms.for_config(config)
    if stream.q_meas.shape[1] != arm.n:
        raise ValueError(f"recording has {stream.q_meas.shape[1]} joints, arm '{arm.name}' has {arm.n}")
    K = len(config["markers"])
    t_cam, pts, vis = [], [], []
    if stream.frames is not None and locate is not None:
        for i, (t, img) in enumerate(stream.frames()):
            if i % every:
                continue
            if max_frames and len(t_cam) >= max_frames:
                break
            p = locate(img)
            t_cam.append(t)
            vis.append(p is not None)
            pts.append(np.asarray(p, float).reshape(K, 3) if p is not None else np.zeros((K, 3)))
    if not t_cam:
        t_cam, pts, vis = [stream.t[0]], [np.zeros((K, 3))], [False]
    q_cmd = stream.q_cmd if stream.q_cmd is not None else stream.q_meas.copy()
    cfg = dict(config)
    cfg.setdefault("source", stream.source)
    cfg["has_commands"] = stream.q_cmd is not None
    return Log(
        t_joint=stream.t, q_cmd=q_cmd, q_meas=stream.q_meas,
        t_cam=np.array(t_cam, float), markers_cam=np.array(pts, float), visible=np.array(vis, bool),
        pick_t=np.zeros(0), pick_ok=np.zeros(0, bool), config=cfg,
    )
