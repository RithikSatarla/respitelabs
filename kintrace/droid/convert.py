"""Turn a DROID episode into a kintrace Log.

The Log needs one thing DROID does not ship: a point on the gripper located
in the fixed camera's frame, per frame (markers_cam). On the desk rig that
is a printed fiducial on the wrist. DROID arms wear none, so the point has
to come from the image. detect_gripper() is where that goes.

Everything else maps directly:
  q_meas   observation/robot_state/joint_positions   (7 joints, Panda)
  q_cmd    action/joint_position
  camera_extrinsic (the belief)   the extrinsic the episode was configured with
  markers / tcp_offset            the Franka Hand fingertip centre in the flange frame
"""
from __future__ import annotations

import numpy as np

from .. import arms
from ..logio import Log
from .episode import Episode


def detect_gripper(frame: np.ndarray, intrinsics: np.ndarray | None = None,
                   depth: np.ndarray | None = None) -> np.ndarray | None:
    """Locate the Franka Hand fingertip centre in the camera frame for one image.

    Returns (3,) metres in the camera frame, or None when not visible.
    Not implemented yet. Two honest routes:
      1. keypoint model for the Franka Hand (train on a few hundred frames
         labelled by projecting FK through the *corrected* extrinsics), then
         back-project with the ZED depth or the stereo pair;
      2. segment the hand (SAM or a colour/shape prior on the white hand +
         black fingers), take the centroid, and read depth from the SVO.
    Either way the output is one point per frame and the rest of the check
    does not change.
    """
    raise NotImplementedError(
        "detect_gripper() has no detector yet. Run with --synthetic to test the "
        "pipeline, or implement one of the routes in its docstring.")


def _config(ep: Episode, serial: str, T_belief: np.ndarray) -> dict:
    return {
        "arm": "panda",
        "camera_extrinsic": np.asarray(T_belief).tolist(),
        "camera_serial": serial,
        "markers": [arms.PANDA_HAND_TIP.tolist()],
        "tcp_offset": arms.PANDA_HAND_TIP.tolist(),
        "droid_episode": ep.key,
        "droid_lab": ep.meta.get("lab", ""),
        "source": "droid_raw_1.0.1",
    }


def synthetic_detections(ep: Episode, T_truth: np.ndarray, noise_mm: float = 3.0,
                         visible_frac: float = 0.8, seed: int = 0):
    """Where the fingertip *would* be seen if the camera sat at T_truth.

    Stands in for detect_gripper() so the rest of the pipeline can be tested.
    A result produced this way is a pipeline test, not a real-image result,
    and gets labelled "synthetic" in every report.
    """
    rng = np.random.default_rng(seed)
    F = arms.PANDA.fk(ep.q)
    tip_base = np.einsum("nij,j->ni", F[:, :3, :3], arms.PANDA_HAND_TIP) + F[:, :3, 3]
    Tinv = np.linalg.inv(T_truth)
    tip_cam = (Tinv[:3, :3] @ tip_base.T).T + Tinv[:3, 3]
    tip_cam += rng.normal(0, noise_mm / 1000, tip_cam.shape)
    visible = rng.random(len(tip_cam)) < visible_frac
    visible &= tip_cam[:, 2] > 0.2  # in front of the camera
    return tip_cam[:, None, :], visible


def to_log(ep: Episode, serial: str, T_belief: np.ndarray, markers_cam: np.ndarray,
           visible: np.ndarray, t_cam: np.ndarray | None = None, truth: dict | None = None) -> Log:
    t_cam = ep.t if t_cam is None else t_cam
    n = min(len(ep.t), len(markers_cam), len(visible), len(t_cam))
    return Log(
        t_joint=ep.t,
        q_cmd=ep.q_cmd,
        q_meas=ep.q,
        t_cam=np.asarray(t_cam[:n], dtype=float),
        markers_cam=np.asarray(markers_cam[:n], dtype=float),
        visible=np.asarray(visible[:n], dtype=bool),
        pick_t=np.zeros(0),
        pick_ok=np.zeros(0, dtype=bool),
        config=_config(ep, serial, T_belief),
        truth=truth or {},
    )


def episode_to_log(ep: Episode, serial: str, T_belief: np.ndarray, T_truth: np.ndarray | None = None,
                   synthetic: bool = False, every: int = 1, intrinsics=None, seed: int = 0) -> Log:
    """One camera of one episode as a Log, with the configured extrinsic as the belief."""
    truth = {}
    if T_truth is not None:
        truth = {"camera_extrinsic_corrected": np.asarray(T_truth).tolist(), "source": "KarlP/droid cam2base_extrinsics.json"}
    if synthetic:
        if T_truth is None:
            raise ValueError("synthetic detections need the corrected extrinsic")
        pts, vis = synthetic_detections(ep, T_truth, seed=seed)
        truth["detections"] = "synthetic"
        return to_log(ep, serial, T_belief, pts, vis, truth=truth)
    pts, vis, t_cam = [], [], []
    tc = ep.t_cam.get(serial)
    for i, frame in ep.frames(serial, every=every):
        p = detect_gripper(frame, intrinsics)
        pts.append(p if p is not None else np.zeros(3))
        vis.append(p is not None)
        t_cam.append(tc[i] if tc is not None and i < len(tc) else (ep.t[i] if i < len(ep.t) else ep.t[-1]))
    truth["detections"] = "detect_gripper"
    return to_log(ep, serial, T_belief, np.asarray(pts)[:, None, :], np.asarray(vis), np.asarray(t_cam), truth)
