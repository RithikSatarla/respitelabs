"""Readers for other people's recordings, on small files built here."""
import json
import os

import numpy as np
import pytest

from kintrace import arms, readers
from kintrace.diagnose import diagnose, self_baseline

N, HZ = 240, 30.0


def _joints():
    t = np.arange(N) / HZ
    q = 0.6 * np.sin(np.outer(t, np.linspace(0.3, 0.9, 5)))
    return t, q


def _config():
    return dict(arm="so101", camera_extrinsic=np.eye(4).tolist(), markers=[[0, 0, 0.05]], tcp_offset=[0, 0, 0.1])


# --------------------------------------------------------------------------
def test_lerobot_v2(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    t, q = _joints()
    deg = np.rad2deg(q)
    state = np.c_[deg, np.full(N, 10.0)]           # 5 joints + gripper, degrees
    action = np.c_[np.roll(deg, -1, 0), np.full(N, 10.0)]
    df = pd.DataFrame({"observation.state": list(state), "action": list(action), "timestamp": t,
                       "frame_index": np.arange(N), "episode_index": np.zeros(N, int)})
    df.to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")
    info = dict(fps=HZ, robot_type="so101_follower", chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                features={"observation.state": {"shape": [6]}, "action": {"shape": [6]},
                          "observation.images.front": {"dtype": "video"}})
    (root / "meta" / "info.json").write_text(json.dumps(info))
    s = readers.lerobot.read(str(root), episode=0, units="deg")
    assert s.q_meas.shape == (N, 5) and s.q_cmd is not None
    assert np.allclose(s.q_meas, q)
    assert any("video" in m for m in s.missing)   # no mp4 written, reader says so
    log = readers.assemble(s, _config())
    assert log.config["arm"] == "so101" and not log.visible.any()
    d = diagnose(log, self_baseline(log))
    assert d.primary == "none"
    assert any("skipped" in c.detail for c in d.checks)


# --------------------------------------------------------------------------
def test_ros2_bag(tmp_path):
    rosbags = pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    JointState = ts.types["sensor_msgs/msg/JointState"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    t, q = _joints()
    names = ["j1", "j2", "j3", "j4", "j5"]
    bag = tmp_path / "bag"
    with Writer(bag, version=8) as w:
        cj = w.add_connection("/joint_states", JointState.__msgtype__, typestore=ts)
        cc = w.add_connection("/joint_cmd", JointState.__msgtype__, typestore=ts)
        for i in range(N):
            ns = int(t[i] * 1e9) + 10**12
            for conn, qq in ((cj, q[i]), (cc, q[min(i + 1, N - 1)])):
                m = JointState(Header(Time(ns // 10**9, ns % 10**9), "base"), names, qq, np.zeros(5), np.zeros(5))
                w.write(conn, ns, ts.serialize_cdr(m, JointState.__msgtype__))
    assert "/joint_states" in readers.rosbag.topics(str(bag))
    s = readers.rosbag.read(str(bag), cmd_topic="/joint_cmd")
    assert s.q_meas.shape == (N, 5) and np.allclose(s.q_meas, q)
    assert s.q_cmd is not None and np.allclose(s.q_cmd[:-1], q[1:], atol=1e-6)
    assert abs(s.t[-1] - t[-1]) < 1e-6


# --------------------------------------------------------------------------
def test_ur_rtde_csv(tmp_path):
    pytest.importorskip("pandas")
    t = np.arange(N) / 125.0
    q = 0.5 * np.sin(np.outer(t, np.linspace(0.3, 0.9, 6)))
    cols = ["timestamp"] + [f"target_q_{i}" for i in range(6)] + [f"actual_q_{i}" for i in range(6)]
    rows = np.c_[t, np.roll(q, -1, 0), q]
    p = tmp_path / "rec.csv"
    p.write_text(" ".join(cols) + "\n" + "\n".join(" ".join(f"{v:.6f}" for v in r) for r in rows))
    s = readers.ur_rtde.read(str(p))
    assert s.q_meas.shape == (N, 6) and s.q_cmd is not None
    assert readers.guess_format(str(p)) == "ur_rtde"
    cfg = dict(camera_extrinsic=np.eye(4).tolist(), markers=[[0, 0, 0.05]], tcp_offset=[0, 0, 0.18])
    log = readers.assemble(s, cfg)
    assert arms.for_config(log.config).name == "ur5e"


# --------------------------------------------------------------------------
def test_locate_tag_on_rendered_image():
    cv2 = pytest.importorskip("cv2")
    from kintrace.hw import markers, render

    intr = markers.Intrinsics.guess(640, 480)
    size = 0.04
    T = np.eye(4)
    # tilted, since a head-on square tag has an ambiguous pose and locate() drops it on purpose
    T[:3, :3] = cv2.Rodrigues(np.array([0.6, 0.2, 0.0]))[0] @ np.diag([1.0, -1.0, -1.0])  # face the camera, then tilt
    T[:3, 3] = [0.02, -0.01, 0.25]
    img = render.render(intr, [(0, size, T)])
    locate = readers.locate_tag(intr, 0, size)
    pts = locate(img)
    assert pts is not None and pts.shape == (4, 3)
    assert np.linalg.norm(pts.mean(0) - T[:3, 3]) < 0.005
