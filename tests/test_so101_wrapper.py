"""The LeRobot wrapper, against a fake `lerobot` package (no arm needed).

This checks our side of the interface: unit and sign conversion, zero
offsets, the speed cap, and the error when lerobot is missing. It does not
prove the real lerobot API behaves the same way. See HARDWARE.md.
"""
import sys
import types

import numpy as np
import pytest

from kintrace.hw import so101


class _FakeBus:
    def __init__(self):
        self.pos = {m: 0.0 for m in so101.MOTORS + [so101.GRIPPER]}
        self.writes = []

    def sync_read(self, reg):
        assert reg == "Present_Position"
        return dict(self.pos)

    def sync_write(self, reg, goal):
        assert reg == "Goal_Position"
        self.writes.append(dict(goal))
        self.pos.update(goal)


class _FakeFollower:
    def __init__(self, conf):
        self.conf = conf
        self.calibration = {"shoulder_pan": object()}
        self.bus = _FakeBus()
        self.connected = False

    def connect(self, calibrate=True):
        self.connected = True

    def disconnect(self):
        self.connected = False


class _FakeConfig:
    def __init__(self, port, id, use_degrees, disable_torque_on_disconnect=True):
        self.port, self.id, self.use_degrees = port, id, use_degrees
        self.disable_torque_on_disconnect = disable_torque_on_disconnect


@pytest.fixture
def fake_lerobot(monkeypatch):
    mods = {name: types.ModuleType(name) for name in ("lerobot", "lerobot.robots", "lerobot.robots.so_follower")}
    mods["lerobot.robots.so_follower"].SO101Follower = _FakeFollower
    mods["lerobot.robots.so_follower"].SO101FollowerConfig = _FakeConfig
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return mods


CFG = dict(port="/dev/null", robot_id="t", joint_zero_offsets_deg=[1.0, -2.0, 0.5, 0.0, 3.0],
           joint_sign=[1, -1, 1, 1, -1], gripper_hold=10.0)


def test_roundtrip_units_signs_offsets(fake_lerobot):
    arm = so101.SO101Arm(CFG, max_step_deg=90).connect()
    assert arm.robot.conf.use_degrees and not arm.robot.conf.disable_torque_on_disconnect
    q = np.deg2rad([10.0, -20.0, 30.0, 15.0, -5.0])
    arm.write(q)
    sent = arm.robot.bus.writes[-1]
    assert sent[so101.GRIPPER] == 10.0
    # lerobot degrees = sign * (q + offset)
    expect = np.array(CFG["joint_sign"]) * (np.rad2deg(q) + CFG["joint_zero_offsets_deg"])
    assert np.allclose([sent[m] for m in so101.MOTORS], expect)
    t, back = arm.read()
    assert np.allclose(back, q)
    arm.close()
    assert arm.robot is None


def test_speed_cap(fake_lerobot):
    arm = so101.SO101Arm(CFG, max_step_deg=4.0).connect()
    _, q0 = arm.read()
    arm.write(np.deg2rad([60.0, 0, 0, 0, 0]))
    _, q1 = arm.read()
    step = np.rad2deg(np.abs(q1 - q0))
    assert step.max() <= 4.0 + 1e-6 and step[0] > 3.9  # moved, but only 4 deg


def test_missing_lerobot_says_how_to_install(monkeypatch):
    for k in ("lerobot", "lerobot.robots", "lerobot.robots.so_follower", "lerobot.robots.so101_follower"):
        monkeypatch.setitem(sys.modules, k, None)
    with pytest.raises(RuntimeError, match="lerobot"):
        so101.SO101Arm(CFG).connect()
