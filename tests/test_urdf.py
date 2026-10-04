"""URDF loader: Panda and UR5e URDFs give the same forward kinematics as the hand-written models."""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from kintrace import arms, robot, urdf

HERE = os.path.join(os.path.dirname(__file__), "urdf")


def _diff(A, B):
    pos = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1)
    rel = np.einsum("nji,njk->nik", A[:, :3, :3], B[:, :3, :3])
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    return pos.max(), ang.max()


def test_panda_urdf_matches_arms_panda():
    arm = urdf.load(os.path.join(HERE, "panda.urdf"), tip="panda_link8")
    assert arm.n == 7
    q = urdf.random_q(arm, 1000, np.random.default_rng(0))
    pos, ang = _diff(arm.fk(q), arms.panda_fk(q))
    assert pos < 1e-6 and ang < 1e-6   # metres, degrees


def test_ur5e_urdf_matches_dh_model():
    # base_link_inertia is the DH base (base_link is it turned 180 deg about z); tool0 is the DH flange
    arm = urdf.load(os.path.join(HERE, "ur5e.urdf"), base="base_link_inertia", tip="tool0")
    assert arm.n == 6
    q = np.random.default_rng(1).uniform(-np.pi, np.pi, size=(1000, 6))
    pos, ang = _diff(arm.fk(q), robot.fk(q))
    assert pos < 1e-6 and ang < 1e-6


def test_config_urdf_key_selects_the_body():
    path = os.path.join(HERE, "panda.urdf")
    a = arms.for_config({"arm": "ur5e", "urdf": path, "urdf_tip": "panda_link8"})
    assert a.n == 7
    assert arms.for_config({"arm": "ur5e"}) is arms.UR5E
