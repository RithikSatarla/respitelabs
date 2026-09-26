"""Kinematics for a UR5e-class 6-joint arm.

Standard DH parameters (published by Universal Robots). Everything is
vectorised over a batch of joint vectors so diagnosis can run over whole
logs at once.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# UR5e standard DH: a, d, alpha
DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"]


def fk(q: np.ndarray) -> np.ndarray:
    """Flange pose(s) in the robot base/world frame.

    q: (6,) or (N, 6) joint angles in radians. Returns (4, 4) or (N, 4, 4).
    """
    q = np.asarray(q, dtype=float)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    n = q.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(6):
        ct, st = np.cos(q[:, i]), np.sin(q[:, i])
        ca, sa = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
        A = np.zeros((n, 4, 4))
        A[:, 0, 0] = ct
        A[:, 0, 1] = -st * ca
        A[:, 0, 2] = st * sa
        A[:, 0, 3] = DH_A[i] * ct
        A[:, 1, 0] = st
        A[:, 1, 1] = ct * ca
        A[:, 1, 2] = -ct * sa
        A[:, 1, 3] = DH_A[i] * st
        A[:, 2, 1] = sa
        A[:, 2, 2] = ca
        A[:, 2, 3] = DH_D[i]
        A[:, 3, 3] = 1.0
        T = T @ A
    return T[0] if single else T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply pose(s) T (N,4,4) to points pts (K,3) -> (N,K,3)."""
    T = np.atleast_3d(T) if T.ndim == 2 else T
    if T.ndim == 2:
        T = T[None]
    return np.einsum("nij,kj->nki", T[:, :3, :3], pts) + T[:, None, :3, 3]


def pose(rotvec, trans) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = trans
    return T


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera pose (camera z axis looks at target)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def tool_target(position, yaw, tilt_x=0.0, tilt_y=0.0) -> np.ndarray:
    """Tool pose pointing down with a yaw and a small tilt."""
    R = (
        Rotation.from_euler("z", yaw)
        * Rotation.from_euler("y", tilt_y)
        * Rotation.from_euler("x", np.pi + tilt_x)
    )
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T[:3, 3] = position
    return T


def ik(target_tool: np.ndarray, tcp: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Numerical IK: joint angles putting the tool frame (flange + tcp
    translation) at target_tool."""

    def err(q):
        F = fk(q)
        tip = F[:3, :3] @ tcp + F[:3, 3]
        rot = Rotation.from_matrix(target_tool[:3, :3].T @ F[:3, :3]).as_rotvec()
        return np.concatenate([(tip - target_tool[:3, 3]) * 10.0, rot])

    seed = np.asarray(seed, dtype=float)
    rng = np.random.default_rng(0)
    seeds = [seed, HOME_Q] + [seed + rng.normal(scale=0.4, size=6) for _ in range(12)]
    best = None
    for s0 in seeds:
        sol = least_squares(err, s0, method="lm", xtol=1e-12, ftol=1e-12)
        if np.linalg.norm(sol.fun) < 1e-5:
            # prefer the solution closest to the seed (no wrist flips)
            x = seed + (sol.x - seed + np.pi) % (2 * np.pi) - np.pi
            if best is None or np.linalg.norm(x - seed) < np.linalg.norm(best - seed):
                best = x
            if np.linalg.norm(x - seed) < 3.0:
                break
    if best is None:
        raise RuntimeError("IK did not converge")
    return best


HOME_Q = np.array([0.0, -1.9, 1.9, -1.57, -1.57, 0.0])
