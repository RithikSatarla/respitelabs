"""Load a robot's kinematic chain from a URDF into the same Arm interface as arms.py.

Standard library only (xml.etree). A chain is the path of joints from a base link to a
tip link. Revolute, continuous and prismatic joints move; fixed joints only add their
origin. URDF origins use fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll).

    from kintrace import urdf
    arm = urdf.load("panda.urdf", tip="panda_link8")
    T = arm.fk(q)            # (4, 4) or (N, 4, 4), tip pose in the base link frame

Only the body (kinematics) loads from a URDF. Cameras, markers and tool offsets still
come from the log's config.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .arms import Arm

MOVING = ("revolute", "continuous", "prismatic")


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray        # 4x4, parent link frame -> joint frame at q = 0
    axis: np.ndarray          # unit vector in the joint frame
    lower: float
    upper: float


def _vec(s, default):
    return np.array([float(x) for x in s.split()]) if s else np.array(default, dtype=float)


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = Rotation.from_euler("xyz", _vec(el.get("rpy"), [0, 0, 0])).as_matrix()
        T[:3, 3] = _vec(el.get("xyz"), [0, 0, 0])
    return T


def parse(path: str) -> tuple:
    """(robot name, {child link: Joint}) for every top-level joint (transmission blocks are ignored)."""
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        jt = j.get("type")
        lo = float(lim.get("lower", -np.pi)) if lim is not None else -np.pi
        hi = float(lim.get("upper", np.pi)) if lim is not None else np.pi
        if jt == "continuous":
            lo, hi = -np.pi, np.pi
        ax = j.find("axis")
        axis = _vec(ax.get("xyz") if ax is not None else None, [1, 0, 0])
        joints[j.find("child").get("link")] = Joint(
            j.get("name"), jt, j.find("parent").get("link"), j.find("child").get("link"),
            _origin(j.find("origin")), axis / (np.linalg.norm(axis) or 1.0), lo, hi)
    return root.get("name", "robot"), joints


def chain(joints: dict, base: str | None, tip: str | None) -> list:
    """Joints from base to tip. Defaults: the root link, and the leaf reached through the most moving joints."""
    links = set(joints) | {j.parent for j in joints.values()}
    root = next(link for link in links if link not in joints)
    base = base or root

    def path_to(link):
        out = []
        while link != base:
            if link not in joints:
                raise ValueError(f"link '{link}' is not below base '{base}'")
            out.append(joints[link])
            link = joints[link].parent
        return out[::-1]

    if tip is None:
        leaves = [link for link in links if link not in {j.parent for j in joints.values()}]
        scored = []
        for leaf in leaves:
            try:
                p = path_to(leaf)
            except ValueError:
                continue
            scored.append((sum(j.type in MOVING for j in p), len(p), leaf))
        tip = max(scored)[2]
    return path_to(tip)


def _motion(j: Joint, q: np.ndarray) -> np.ndarray:
    n = len(q)
    M = np.tile(np.eye(4), (n, 1, 1))
    if j.type == "prismatic":
        M[:, :3, 3] = q[:, None] * j.axis
    else:
        M[:, :3, :3] = Rotation.from_rotvec(q[:, None] * j.axis).as_matrix()
    return M


def fk_fn(joint_chain: list):
    moving = [j for j in joint_chain if j.type in MOVING]

    def fk(q):
        q = np.asarray(q, dtype=float)
        single = q.ndim == 1
        q = np.atleast_2d(q)
        if q.shape[1] != len(moving):
            raise ValueError(f"expected {len(moving)} joint values, got {q.shape[1]}")
        T = np.tile(np.eye(4), (len(q), 1, 1))
        k = 0
        for j in joint_chain:
            T = T @ j.origin
            if j.type in MOVING:
                T = T @ _motion(j, q[:, k])
                k += 1
        return T[0] if single else T

    return fk


@dataclass(frozen=True)
class URDFArm(Arm):
    limits: np.ndarray = None   # (n, 2) lower, upper
    base: str = ""
    tip: str = ""
    source: str = ""


def load(path: str, base: str | None = None, tip: str | None = None, name: str | None = None) -> URDFArm:
    robot, joints = parse(path)
    ch = chain(joints, base, tip)
    moving = [j for j in ch if j.type in MOVING]
    if not moving:
        raise ValueError(f"no moving joints between {base or 'root'} and {tip or 'tip'} in {path}")
    limits = np.array([[j.lower, j.upper] for j in moving])
    home = np.clip(np.zeros(len(moving)), limits[:, 0], limits[:, 1])
    fk = fk_fn(ch)
    reach = fk(home)[:3, 3]
    return URDFArm(name=name or robot, joint_names=tuple(j.name for j in moving), fk_fn=fk, home_q=home,
                   work_pts=np.array([reach * [1, 1, 0] + [0, 0, 0.05]]), limits=limits,
                   base=ch[0].parent, tip=ch[-1].child, source=path)


def random_q(arm: URDFArm, n: int, rng) -> np.ndarray:
    lo, hi = arm.limits[:, 0], arm.limits[:, 1]
    lo, hi = np.maximum(lo, -np.pi), np.minimum(hi, np.pi)
    return rng.uniform(lo, hi, size=(n, len(lo)))
