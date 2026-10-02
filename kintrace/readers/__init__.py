"""Readers: other people's recordings in, kintrace Logs out.

  lerobot    LeRobot datasets (SO-100/SO-101 and most public HF datasets)
  rosbag     ROS 1 bags, ROS 2 bags, MCAP (via the rosbags package)
  ur_rtde    Universal Robots RTDE recordings (text table), plus a video

Each reader returns a Stream (joints, commands, frames). assemble() turns a
Stream into a Log with a locate function that finds the wrist in each frame.

  from kintrace import readers
  s = readers.lerobot.read("path/to/dataset", episode=0, units="deg")
  print(s.describe())
  log = readers.assemble(s, config, locate=readers.locate_tag(intr, tag_id=0, size_m=0.04))

Or from the shell:  kintrace import <path> --format lerobot --config rig.json -o log.npz
"""
from . import lerobot, rosbag, ur_rtde
from .common import Stream, assemble, locate_tag

FORMATS = {"lerobot": lerobot.read, "rosbag": rosbag.read, "mcap": rosbag.read, "ur_rtde": ur_rtde.read}


def guess_format(path: str) -> str | None:
    import os

    p = path.rstrip("/")
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta", "info.json")):
        return "lerobot"
    if p.endswith((".bag", ".mcap", ".db3")) or (os.path.isdir(p) and os.path.exists(os.path.join(p, "metadata.yaml"))):
        return "rosbag"
    if p.endswith((".csv", ".txt")):
        return "ur_rtde"
    return None


__all__ = ["Stream", "assemble", "locate_tag", "lerobot", "rosbag", "ur_rtde", "FORMATS", "guess_format"]
