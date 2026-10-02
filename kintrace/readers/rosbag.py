"""ROS 1 bags, ROS 2 bags (sqlite3 or MCAP storage) and plain MCAP files.

Uses the `rosbags` package (pure Python, no ROS install needed):
  pip install rosbags

Topics are found by type, so you rarely have to name them:
  joints     sensor_msgs/msg/JointState       measured angles (position)
  commands   trajectory_msgs/msg/JointTrajectory, or a second JointState topic
             you name with cmd_topic (many controllers publish the command as
             a JointState too)
  camera     sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import Stream

JOINT_TYPES = ("sensor_msgs/msg/JointState",)
TRAJ_TYPES = ("trajectory_msgs/msg/JointTrajectory",)
IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")


def _reader(path: str):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError("ROS bag reading needs the rosbags package: pip install rosbags") from e
    return AnyReader([Path(path)])


def topics(path: str) -> dict:
    """topic -> message type, for picking the right ones."""
    with _reader(path) as r:
        return {c.topic: c.msgtype for c in r.connections}


def _stamp(msg, fallback_ns: int) -> float:
    h = getattr(msg, "header", None)
    if h is not None and getattr(h, "stamp", None) is not None:
        s = h.stamp
        sec = getattr(s, "sec", None)
        nsec = getattr(s, "nanosec", getattr(s, "nsec", 0))
        if sec is not None and (sec or nsec):
            return float(sec) + float(nsec) * 1e-9
    return fallback_ns * 1e-9


def _decode_image(msg, msgtype: str):
    import cv2

    if msgtype.endswith("CompressedImage"):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    enc = msg.encoding
    buf = np.frombuffer(msg.data, np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc in ("mono8", "8UC1"):
        return buf.reshape(msg.height, msg.width).copy()
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR)
    raise ValueError(f"image encoding {enc} not handled")


def read(path: str, joint_topic: str | None = None, cmd_topic: str | None = None,
         image_topic: str | None = None, joint_names: list | None = None) -> Stream:
    missing = []
    with _reader(path) as r:
        by_type = {}
        for c in r.connections:
            by_type.setdefault(c.msgtype, []).append(c.topic)
        jt = joint_topic or next((t for ty in JOINT_TYPES for t in by_type.get(ty, [])), None)
        if jt is None:
            raise ValueError(f"no JointState topic in {path}. Topics: {topics(path)}")
        ct = cmd_topic or next((t for ty in TRAJ_TYPES for t in by_type.get(ty, [])), None)
        it = image_topic or next((t for ty in IMAGE_TYPES for t in by_type.get(ty, [])), None)
        conns = {c.topic: c for c in r.connections}

        t_j, q_j, names = [], [], None
        for conn, ts, raw in r.messages(connections=[conns[jt]]):
            m = r.deserialize(raw, conn.msgtype)
            if names is None:
                names = list(m.name)
                order = [names.index(n) for n in joint_names] if joint_names else list(range(len(names)))
            pos = np.asarray(m.position, float)
            if len(pos) < len(order):
                continue
            t_j.append(_stamp(m, ts))
            q_j.append(pos[order])
        t_j, q_j = np.array(t_j), np.array(q_j)
        if len(t_j) == 0:
            raise ValueError(f"JointState topic {jt} has no usable messages")

        q_c = None
        if ct is not None:
            t_c, qq = [], []
            for conn, ts, raw in r.messages(connections=[conns[ct]]):
                m = r.deserialize(raw, conn.msgtype)
                if conn.msgtype in TRAJ_TYPES:
                    t0 = _stamp(m, ts)
                    cn = list(m.joint_names)
                    o = [cn.index(n) for n in (joint_names or names)] if cn else order
                    for p in m.points:
                        d = p.time_from_start
                        t_c.append(t0 + float(d.sec) + float(getattr(d, "nanosec", getattr(d, "nsec", 0))) * 1e-9)
                        qq.append(np.asarray(p.positions, float)[o])
                else:
                    cn = list(m.name)
                    o = [cn.index(n) for n in (joint_names or names)] if cn else order
                    t_c.append(_stamp(m, ts))
                    qq.append(np.asarray(m.position, float)[o])
            if t_c:
                t_c, qq = np.array(t_c), np.array(qq)
                k = np.argsort(t_c)
                q_c = np.stack([np.interp(t_j, t_c[k], qq[k, i]) for i in range(qq.shape[1])], 1)
        if q_c is None:
            missing.append("commanded joints (no JointTrajectory topic; pass cmd_topic if the command is a JointState)")

        frames = None
        if it is not None:
            msgtype = conns[it].msgtype

            def frames(path=path, it=it, msgtype=msgtype, t0=t_j[0]):
                with _reader(path) as rr:
                    cc = {c.topic: c for c in rr.connections}[it]
                    for conn, ts, raw in rr.messages(connections=[cc]):
                        m = rr.deserialize(raw, conn.msgtype)
                        yield _stamp(m, ts) - t0, _decode_image(m, msgtype)
        else:
            missing.append("camera images (no Image/CompressedImage topic)")

    t0 = t_j[0]
    return Stream(t_j - t0, q_j, q_c, frames, f"rosbag:{Path(path).name}",
                  meta=dict(joint_topic=jt, cmd_topic=ct, image_topic=it, joint_names=names), missing=missing)
