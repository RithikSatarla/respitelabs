"""Configured vs corrected camera extrinsics, and the labels built from them.

DROID stores a camera pose as 6 floats [x, y, z, roll, pitch, yaw] in metres
and radians, camera frame expressed in the robot base frame (cam2base).
Euler order is scipy "xyz", the convention in the DROID collection code.
Kintrace's config["camera_extrinsic"] is the same thing as a 4x4.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def pose6_to_T(p) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 16:
        return p.reshape(4, 4)
    if p.size != 6:
        raise ValueError(f"expected 6 (xyz rpy) or 16 floats, got {p.size}")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", p[3:]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def T_to_pose6(T) -> np.ndarray:
    T = np.asarray(T)
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz")]


def pose_error(T_belief, T_truth) -> dict:
    """How far the configured camera pose is from the corrected one."""
    d = np.linalg.inv(T_belief) @ T_truth
    t = np.linalg.norm(d[:3, 3])
    r = np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())
    return {"translation_mm": float(t * 1000), "rotation_deg": float(np.rad2deg(r))}


# --------------------------------------------------------------------------
# the corrected set from KarlP/droid
# --------------------------------------------------------------------------
_SIDES = ("_left", "_right")
META_FIELDS = ("source", "quality_metric", "metric_type", "relative_path")


def _strip_side(serial: str) -> str:
    for s in _SIDES:
        if serial.endswith(s):
            return serial[: -len(s)]
    return serial


@dataclass
class Corrected:
    """cam2base_extrinsics.json, indexed so lookups work by episode path or id."""
    by_key: dict  # key -> {serial: 4x4}
    id_to_path: dict
    meta_by_key: dict = None  # key -> {source, quality_metric, metric_type, relative_path} as present

    @staticmethod
    def load(root: str) -> "Corrected":
        with open(os.path.join(root, "cam2base_extrinsics.json"), encoding="utf-8") as f:
            raw = json.load(f)
        id_to_path = {}
        p = os.path.join(root, "episode_id_to_path.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                id_to_path = json.load(f)
        by_key, meta_by_key = {}, {}
        for key, cams in raw.items():
            if not isinstance(cams, dict):
                continue
            entry = {}
            # GT entries carry relative_path, Pred entries do not
            meta = {k: cams[k] for k in META_FIELDS if k in cams}
            # the left ZED sensor is the one the DROID extrinsics are defined for;
            # keep it when both sides are listed
            for serial, pose in sorted(cams.items(), key=lambda kv: not str(kv[0]).endswith("_left")):
                base = _strip_side(str(serial))
                if base in entry:
                    continue
                try:
                    entry[base] = pose6_to_T(pose)
                except (ValueError, TypeError):
                    continue
            if entry:
                by_key[key] = entry
                by_key[key.rstrip("/")] = entry
                meta_by_key[key] = meta_by_key[key.rstrip("/")] = meta
        return Corrected(by_key, id_to_path, meta_by_key)

    def __len__(self):
        return len({id(v) for v in self.by_key.values()})

    def _find(self, episode_key: str) -> str | None:
        for k in (episode_key, self.id_to_path.get(episode_key, ""), episode_key.rstrip("/")):
            if k and k in self.by_key:
                return k
        # tolerate a path prefix difference (bucket root vs relative)
        tail = episode_key.rstrip("/").split("/")[-1]
        for k in self.by_key:
            if k.rstrip("/").endswith(tail):
                return k
        return None

    def lookup(self, episode_key: str) -> dict | None:
        """episode_key may be the episode id, its path, or a tail of the path."""
        k = self._find(episode_key)
        return self.by_key[k] if k is not None else None

    def meta(self, episode_key: str) -> dict | None:
        """The entry's non-pose fields (source, quality_metric, metric_type, relative_path)."""
        k = self._find(episode_key)
        return (self.meta_by_key or {}).get(k) if k is not None else None

    def keys(self):
        seen, out = set(), []
        for k, v in self.by_key.items():
            if id(v) not in seen:
                seen.add(id(v))
                out.append(k)
        return out


# --------------------------------------------------------------------------
# configured poses, from an episode's metadata json
# --------------------------------------------------------------------------
def configured_from_metadata(meta: dict) -> dict:
    """{serial: 4x4} for the fixed cameras the episode was configured with."""
    out = {}
    for name in ("ext1", "ext2", "wrist"):
        serial = meta.get(f"{name}_cam_serial")
        pose = meta.get(f"{name}_cam_extrinsics")
        if serial is None or pose is None:
            continue
        try:
            out[str(serial)] = pose6_to_T(pose)
        except (ValueError, TypeError):
            continue
    return out


def labels_for(configured: dict, corrected: dict) -> dict:
    """Per camera: configured, corrected and the error between them."""
    out = {}
    for serial, T_belief in configured.items():
        T_truth = corrected.get(serial)
        if T_truth is None:
            continue
        out[serial] = {"belief": T_belief, "truth": T_truth, **pose_error(T_belief, T_truth)}
    return out
