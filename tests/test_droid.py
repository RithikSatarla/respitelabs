"""Panda kinematics and the DROID adapter, on a synthetic fixture (no network)."""
import json
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from kintrace import arms
from kintrace.droid.extrinsics import Corrected, pose6_to_T, pose_error, labels_for, configured_from_metadata


def test_panda_home_matches_published_pose():
    T = arms.panda_fk(arms.PANDA_HOME_Q)
    assert np.allclose(T[:3, 3], [0.307, 0.0, 0.59], atol=2e-3)
    tip = T[:3, :3] @ arms.PANDA_HAND_TIP + T[:3, 3]
    assert abs(tip[2] - 0.487) < 2e-3
    assert arms.for_config({"arm": "panda"}).n == 7


def test_pose6_roundtrip_and_error():
    p = [0.5, -0.2, 0.7, 0.1, -0.4, 2.0]
    T = pose6_to_T(p)
    assert np.allclose(R.from_matrix(T[:3, :3]).as_euler("xyz"), p[3:])
    T2 = T.copy()
    T2[:3, 3] += [0.01, 0.0, 0.0]
    e = pose_error(T, T2)
    assert abs(e["translation_mm"] - 10) < 1e-6 and e["rotation_deg"] < 1e-6


def test_corrected_lookup_by_id_path_and_tail(tmp_path):
    corr = {"LAB/success/d/ep1": {"111_left": [0, 0, 0, 0, 0, 0], "111_right": [1, 0, 0, 0, 0, 0]}}
    (tmp_path / "cam2base_extrinsics.json").write_text(json.dumps(corr))
    (tmp_path / "episode_id_to_path.json").write_text(json.dumps({"id1": "LAB/success/d/ep1"}))
    c = Corrected.load(str(tmp_path))
    for key in ("id1", "LAB/success/d/ep1", "LAB/success/d/ep1/", "bucket/LAB/success/d/ep1"):
        got = c.lookup(key)
        assert got is not None and "111" in got
    meta = {"ext1_cam_serial": 111, "ext1_cam_extrinsics": [0.02, 0, 0, 0, 0, 0]}
    lbl = labels_for(configured_from_metadata(meta), c.lookup("id1"))
    assert abs(lbl["111"]["translation_mm"] - 20) < 1e-6


def test_synthetic_pipeline_catches_moved_camera():
    h5py = pytest.importorskip("h5py")
    from kintrace.droid import episode as ep_mod
    from kintrace.droid.convert import episode_to_log
    from kintrace.droid.bench import _baseline_for
    from kintrace.diagnose import diagnose
    import tempfile

    rng = np.random.default_rng(0)
    n, t = 300, np.arange(300) / 15
    q = arms.PANDA_HOME_Q + 0.5 * np.sin(np.outer(t, rng.uniform(0.2, 0.8, 7)))
    T = np.eye(4)
    T[:3, 3] = [0.9, -0.7, 0.6]
    T[:3, :3] = R.from_euler("xyz", [np.pi / 2 + 0.5, 0, np.pi * 0.75]).as_matrix()
    with tempfile.TemporaryDirectory() as d:
        with h5py.File(os.path.join(d, "trajectory.h5"), "w") as h:
            h["observation/robot_state/joint_positions"] = q
            h["action/joint_position"] = np.roll(q, -1, 0)
            h["observation/timestamp/robot_state/read_start"] = (t + 1.7e9) * 1e9
        meta = {"ext1_cam_serial": "111", "ext1_cam_extrinsics": list(np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_euler("xyz")])}
        with open(os.path.join(d, "metadata_x.json"), "w") as f:
            json.dump(meta, f)
        ep = ep_mod.load(d, key="x")
        assert ep.q.shape == (n, 7) and "111" in ep.configured
        truth = T.copy()
        truth[:3, 3] += [0.03, 0.0, 0.0]
        for T_truth, expect in ((truth, True), (T, False)):
            log = episode_to_log(ep, "111", T, T_truth, synthetic=True)
            assert log.config["arm"] == "panda" and log.markers_cam.shape[1:] == (1, 3)
            dg = diagnose(log, _baseline_for(log, 4.0))
            assert any(f.fault == "camera_moved" for f in dg.findings) == expect


def test_injected_latency_is_measured_on_joint_streams():
    from kintrace.droid.bench import _shift_measured
    from kintrace.diagnose import estimate_delay
    from kintrace.logio import Log

    n, t = 600, np.arange(600) / 15
    q = arms.PANDA_HOME_Q + 0.4 * np.sin(np.outer(t, np.linspace(0.3, 0.9, 7)))
    log = Log(t, q, q, t, np.zeros((n, 1, 3)), np.zeros(n, bool), np.zeros(0), np.zeros(0, bool), {})
    tau0 = estimate_delay(log)
    tau1 = estimate_delay(_shift_measured(log, 0.03))
    assert abs((tau1 - tau0) - 0.03) < 0.004
