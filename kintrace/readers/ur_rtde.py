"""Universal Robots RTDE recordings.

The `rtde` example recorder (record.py from ur_rtde / the RTDE Python
client) writes a text table with a header like

  timestamp target_q_0 ... target_q_5 actual_q_0 ... actual_q_5 ...

space or comma separated. target_q is the commanded joints, actual_q the
measured ones, both in radians, at 125 or 500 Hz. No camera in the file;
pass the camera video separately with video= to get frames.
"""
from __future__ import annotations

import os

import numpy as np

from .common import Stream


def read(path: str, video: str | None = None, video_offset_s: float = 0.0, fps: float | None = None) -> Stream:
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.strip(): c for c in df.columns}
    act = [cols[f"actual_q_{i}"] for i in range(6) if f"actual_q_{i}" in cols]
    tgt = [cols[f"target_q_{i}"] for i in range(6) if f"target_q_{i}" in cols]
    if len(act) != 6:
        raise ValueError(f"expected actual_q_0..5 columns, found {list(df.columns)[:12]}")
    t = df[cols["timestamp"]].to_numpy(float) if "timestamp" in cols else np.arange(len(df)) / 125.0
    t = t - t[0]
    q = df[act].to_numpy(float)
    qc = df[tgt].to_numpy(float) if len(tgt) == 6 else None
    missing = [] if qc is not None else ["target_q (commanded joints)"]
    frames = None
    if video:
        def frames(video=video, off=video_offset_s, fps=fps):
            import cv2

            cap = cv2.VideoCapture(video)
            f = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
            i = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield i / f + off, img
                i += 1
            cap.release()
    else:
        missing.append("camera video (pass video=)")
    return Stream(t, q, qc, frames, f"ur_rtde:{os.path.basename(path)}", meta=dict(video=video), missing=missing)
