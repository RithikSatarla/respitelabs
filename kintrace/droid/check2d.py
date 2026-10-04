"""Monocular camera check: is the camera where the calibration says?

DROID MP4s give one view per camera and no depth, so the full diagnose()
path (which wants a 3D point per frame) does not apply yet. This check works
in the image instead:

  1. Project the FK fingertip through the believed extrinsic and measure how
     far it lands from the detected fingertip (residual under belief, px).
  2. Fit the camera pose from the whole trajectory: 3D fingertips in the base
     frame from FK, 2D detections, cv2.solvePnPRansac then solvePnPRefineLM.
  3. Call "moved" when the fit explains the detections clearly better than
     the belief (by more than a noise margin) AND the fitted pose is more than
     10 mm or 1 deg from the belief.

Poses are cam2base 4x4, the DROID convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import arms
from .extrinsics import pose_error

MOVED_MM = 10.0
MOVED_DEG = 1.0
NOISE_PX = 5.0  # the fit must beat the belief's median residual by more than this


def fk_tips(q: np.ndarray, tip=arms.ROBOTIQ_2F85_TIP) -> np.ndarray:
    """(N, 7) joints -> (N, 3) fingertip in the robot base frame."""
    F = arms.PANDA.fk(q)
    return np.einsum("nij,j->ni", F[:, :3, :3], tip) + F[:, :3, 3]


def project(T_cam2base: np.ndarray, K: np.ndarray, pts_base: np.ndarray) -> np.ndarray:
    Tinv = np.linalg.inv(T_cam2base)
    pc = pts_base @ Tinv[:3, :3].T + Tinv[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3]


@dataclass
class Check2D:
    frames: int
    residual_belief_px: float
    residual_fit_px: float
    fit_vs_belief_mm: float
    fit_vs_belief_deg: float
    moved: bool
    T_fit: np.ndarray | None
    inliers: int


def camera_check_2d(q: np.ndarray, uv: np.ndarray, K: np.ndarray, T_belief: np.ndarray,
                    tip=arms.ROBOTIQ_2F85_TIP, noise_px: float = NOISE_PX) -> Check2D | None:
    """q (N, 7) joints at each detection's time, uv (N, 2) detections in full-res px. None if too few."""
    import cv2

    P = fk_tips(np.asarray(q, float), tip)
    uv = np.asarray(uv, float)
    if len(uv) < 12:
        return None
    r_belief = np.linalg.norm(project(T_belief, K, P) - uv, axis=1)

    # start the fit from the belief so RANSAC searches near it
    Tinv = np.linalg.inv(T_belief)
    rvec0, _ = cv2.Rodrigues(Tinv[:3, :3])
    tvec0 = Tinv[:3, 3].reshape(3, 1)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        P.astype(np.float64), uv.astype(np.float64), K, None, rvec0.copy(), tvec0.copy(),
        useExtrinsicGuess=True, reprojectionError=max(3 * noise_px, 15.0), iterationsCount=500,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 8:
        return Check2D(len(uv), float(np.median(r_belief)), float("nan"), float("nan"), float("nan"), False, None, 0)
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(P[inl].astype(np.float64), uv[inl].astype(np.float64), K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T_base2cam = np.eye(4)
    T_base2cam[:3, :3], T_base2cam[:3, 3] = R, tvec.ravel()
    T_fit = np.linalg.inv(T_base2cam)

    r_fit = np.linalg.norm(project(T_fit, K, P) - uv, axis=1)
    e = pose_error(T_belief, T_fit)
    better = np.median(r_belief) - np.median(r_fit) > noise_px
    far = e["translation_mm"] > MOVED_MM or e["rotation_deg"] > MOVED_DEG
    return Check2D(len(uv), float(np.median(r_belief)), float(np.median(r_fit)),
                   e["translation_mm"], e["rotation_deg"], bool(better and far), T_fit, int(len(inl)))
