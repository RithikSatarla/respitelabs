"""LeRobot datasets (v2.x and v3.0 layouts).

The format every SO-100/SO-101 arm records in, and what hundreds of public
datasets on Hugging Face use. A dataset root looks like

  meta/info.json          fps, features (observation.state, action, observation.images.<cam>)
  data/chunk-000/episode_000000.parquet      v2: one file per episode
  data/chunk-000/file-000.parquet            v3: many episodes per file
  videos/chunk-000/observation.images.<cam>/episode_000000.mp4   v2
  videos/observation.images.<cam>/chunk-000/file-000.mp4         v3 (episode time range in meta)

Joint angles: observation.state is the measured joints, action the commanded
ones. SO-101 datasets store degrees (LeRobot's use_degrees) or raw motor
ticks depending on how they were recorded; pass units="deg" or "rad" and
check the first few values make sense. The gripper is the last column and
is dropped for the 5-joint arm model.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .common import Stream, apply_zero, to_rad


def _info(root: str) -> dict:
    with open(os.path.join(root, "meta", "info.json"), encoding="utf-8") as f:
        return json.load(f)


def _episode_rows(root: str, info: dict, episode: int):
    import pandas as pd

    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk = episode // int(info.get("chunks_size", 1000))
    try:
        p = os.path.join(root, data_path.format(episode_chunk=chunk, episode_index=episode))
    except KeyError:  # v3 templates use {chunk_index}/{file_index}: the episode is found below
        p = None
    if p and os.path.exists(p):
        return pd.read_parquet(p)
    # v3: episodes live in shared files; find the one that holds this episode
    for f in sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True)):
        df = pd.read_parquet(f, columns=["episode_index"])
        if (df["episode_index"] == episode).any():
            df = pd.read_parquet(f)
            return df[df["episode_index"] == episode].reset_index(drop=True)
    raise FileNotFoundError(f"episode {episode} not found under {root}/data")


def _video_path(root: str, info: dict, episode: int, cam_key: str):
    vp = info.get("video_path")
    chunk = episode // int(info.get("chunks_size", 1000))
    cands = []
    if vp:
        try:
            cands.append(os.path.join(root, vp.format(episode_chunk=chunk, episode_index=episode, video_key=cam_key)))
        except KeyError:  # v3 template: handled below from the episode metadata
            pass
    cands.append(os.path.join(root, "videos", f"chunk-{chunk:03d}", cam_key, f"episode_{episode:06d}.mp4"))
    for c in cands:
        if os.path.exists(c):
            return c, None
    # v3: one file per chunk and a time range per episode
    files = sorted(glob.glob(os.path.join(root, "videos", cam_key, "**", "*.mp4"), recursive=True))
    ep_meta = _episode_meta(root, episode)
    if files and ep_meta:
        key = f"videos/{cam_key}"
        start = ep_meta.get(f"{key}/from_timestamp")
        end = ep_meta.get(f"{key}/to_timestamp")
        idx = int(ep_meta.get(f"{key}/chunk_index", 0))
        fi = int(ep_meta.get(f"{key}/file_index", 0))
        for f in files:
            if f.endswith(f"chunk-{idx:03d}{os.sep}file-{fi:03d}.mp4") or len(files) == 1:
                return f, (start, end)
    return None, None


def _episode_meta(root: str, episode: int) -> dict | None:
    p = os.path.join(root, "meta", "episodes.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("episode_index") == episode:
                    return d
    for f in sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True)):
        import pandas as pd

        df = pd.read_parquet(f)
        row = df[df["episode_index"] == episode]
        if len(row):
            return {k: (v.item() if hasattr(v, "item") and getattr(v, "size", 1) == 1 else v) for k, v in row.iloc[0].items()}
    return None


def cameras(root: str) -> list:
    info = _info(root)
    return [k for k, v in info.get("features", {}).items() if k.startswith("observation.images.")]


def read(root: str, episode: int = 0, camera: str | None = None, units: str = "deg",
         n_joints: int | None = None, rig_cfg: dict | None = None) -> Stream:
    info = _info(root)
    df = _episode_rows(root, info, episode)
    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy()) if "action" in df else None
    t = df["timestamp"].to_numpy(dtype=float)
    t = t - t[0]
    missing = []
    nj = n_joints or (state.shape[1] - 1 if state.shape[1] in (6,) else state.shape[1])
    q = apply_zero(to_rad(state[:, :nj], units), rig_cfg)
    qc = apply_zero(to_rad(action[:, :nj], units), rig_cfg) if action is not None else None
    if qc is None:
        missing.append("action (commanded joints)")
    cams = cameras(root)
    cam_key = camera or (cams[0] if cams else None)
    frames = None
    if cam_key:
        path, trange = _video_path(root, info, episode, cam_key)
        if path is None:
            missing.append(f"video for {cam_key}")
        else:
            fps = float(info.get("fps", 30))

            def frames(path=path, trange=trange, fps=fps):
                import cv2

                cap = cv2.VideoCapture(path)
                if trange and trange[0] is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(trange[0]) * 1000)
                i = 0
                while True:
                    ok, img = cap.read()
                    if not ok:
                        break
                    tt = i / fps
                    if trange and trange[1] is not None and trange[0] is not None and tt > float(trange[1]) - float(trange[0]):
                        break
                    yield tt, img
                    i += 1
                cap.release()
    else:
        missing.append("no camera feature in meta/info.json")
    return Stream(t, q, qc, frames, f"lerobot:{os.path.basename(root.rstrip('/'))}#{episode}",
                  meta=dict(fps=info.get("fps"), camera=cam_key, robot=info.get("robot_type"), units=units),
                  missing=missing)
