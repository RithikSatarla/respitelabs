"""Turn a DROID episode into a kintrace Log.

The Log needs one thing DROID does not ship: a point on the gripper located
in the fixed camera's frame, per frame (markers_cam). On the desk rig that
is a printed fiducial on the wrist. DROID arms wear none, so the point has
to come from the image. detect_gripper() is where that goes.

Everything else maps directly:
  q_meas   observation/robot_state/joint_positions   (7 joints, Panda)
  q_cmd    action/joint_position
  camera_extrinsic (the belief)   the extrinsic the episode was configured with
  markers / tcp_offset            the Robotiq 2F-85 fingertip centre in the flange frame
"""
from __future__ import annotations

import numpy as np

from .. import arms
from ..logio import Log
from .episode import Episode


DEFAULT_MODEL = "data/droid/detector_final.pt"


def detect_gripper(frame: np.ndarray, intrinsics: np.ndarray | None = None,
                   depth: np.ndarray | None = None, model_path: str = DEFAULT_MODEL,
                   min_visible: float = 0.5) -> np.ndarray | None:
    """Locate the Robotiq 2F-85 fingertip centre in one full-res BGR frame.

    Returns (2,) (u, v) in full-res pixels, or None when the model says the
    tip is not in view. 2D only: DROID MP4s are one view per camera with no
    depth, so the camera check runs in the image (check2d.py). intrinsics and
    depth are accepted for the 3D route later and are not used here.
    """
    from . import detector

    d = detector.detect(frame, model_path, min_visible=min_visible)
    return None if d is None else np.array([d.u, d.v])


def detect_gripper_keypoints(frame: np.ndarray, model_path: str = DEFAULT_MODEL, min_visible: float = 0.5) -> dict:
    """All keypoints the detector has (flange, hand centre, fingertip centre), each
    {"uv": (u, v) full-res pixels, "conf": heatmap peak, "visible_p": ...} or None."""
    from . import detector

    return {k: (None if d is None else dict(uv=np.array([d.u, d.v]), conf=d.conf, visible_p=d.visible_p))
            for k, d in detector.detect_keypoints(frame, model_path, min_visible=min_visible).items()}


def _config(ep: Episode, serial: str, T_belief: np.ndarray) -> dict:
    return {
        "arm": "panda",
        "camera_extrinsic": np.asarray(T_belief).tolist(),
        "camera_serial": serial,
        "markers": [arms.ROBOTIQ_2F85_TIP.tolist()],
        "tcp_offset": arms.ROBOTIQ_2F85_TIP.tolist(),
        "droid_episode": ep.key,
        "droid_lab": ep.meta.get("lab", ""),
        "source": "droid_raw_1.0.1",
        # DROID's command-to-motion lag runs past the 0.2 s default (it pinned there on 48/50 episodes)
        "delay_search_s": 1.5,
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
    tip_base = np.einsum("nij,j->ni", F[:, :3, :3], arms.ROBOTIQ_2F85_TIP) + F[:, :3, 3]
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
    # diagnose() wants a 3D point per frame. DROID MP4s give one view and no depth,
    # so real frames go through the 2D check instead (check2d.py, `detect` without --synthetic).
    raise NotImplementedError("no 3D gripper point on DROID real frames (one view, no depth); "
                              "the camera check runs in 2D via kintrace/droid/check2d.py")
