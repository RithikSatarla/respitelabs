"""Pluggable arm models: the UR5e path must be exactly what it was."""
import numpy as np

from kintrace import arms, robot, sim


def test_ur5e_is_robot_py():
    q = np.random.default_rng(0).uniform(-3, 3, size=(50, 6))
    assert np.array_equal(arms.UR5E.fk(q), robot.fk(q))
    assert arms.UR5E.n == 6


def test_old_logs_default_to_ur5e():
    assert arms.for_config(sim.base_config()) is arms.UR5E
    assert arms.for_config({}) is arms.UR5E
    assert arms.for_config({"arm": "so101"}) is arms.SO101


def test_so101_chain_shape_and_reach():
    q = np.zeros((3, 5))
    F = arms.SO101.fk(q)
    assert F.shape == (3, 4, 4)
    R = F[0, :3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    # gripper_link at the URDF zero pose: arm stretched forward, about 0.3 m out
    p = F[0, :3, 3]
    assert 0.2 < np.linalg.norm(p[:2]) < 0.4 and 0.1 < p[2] < 0.35
    # shoulder pan turns the arm about the vertical axis only
    F2 = arms.SO101.fk(np.array([0.5, 0, 0, 0, 0]))
    assert np.isclose(F2[2, 3], F[0, 2, 3])
