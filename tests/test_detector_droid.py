"""detect() on raw full-res DROID frames gives the same numbers as the evaluation script.

Needs the local DROID data, an evaluation run (data/droid/eval_*.json with its _preds.npz)
and torch, so it skips anywhere those are missing (CI included).
"""
import glob
import json
import os

import numpy as np
import pytest

DATA = "data/droid"


def _latest_eval():
    runs = sorted(glob.glob(os.path.join(DATA, "eval_*.json")), key=os.path.getmtime)
    return runs[-1] if runs else None


@pytest.mark.skipif(_latest_eval() is None, reason="no local DROID evaluation run")
def test_detect_matches_eval_on_saved_frames():
    pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from kintrace.droid import detector

    ev = json.load(open(_latest_eval(), encoding="utf-8"))
    if not os.path.exists(ev["checkpoint"]) or not os.path.exists(ev["preds_cache"]):
        pytest.skip("checkpoint or prediction cache missing")
    z = np.load(ev["preds_cache"])
    manifest = {m["key"]: m for m in json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))}

    key, serial = str(z["key"][0]), str(z["serial"][0])  # key is the DROID episode id, as in the manifest
    rows = np.flatnonzero((z["key"] == key) & (z["serial"] == serial))
    m = manifest[key]
    video = os.path.join(m["dir"], "recordings", "MP4", f"{serial}.mp4")
    picks = rows[[0, len(rows) // 2, len(rows) - 1]]

    cap = cv2.VideoCapture(video)
    frames, i = {}, 0
    want = {int(z["frame"][r]) for r in picks}
    while len(frames) < len(want):
        ok, f = cap.read()
        assert ok, "video ended before the picked frames"
        if i in want:
            frames[i] = f
        i += 1
    cap.release()

    names = ev.get("keypoints", ["fingertip_centre"])
    for r in picks:
        dets = detector.detect_keypoints(frames[int(z["frame"][r])], ev["checkpoint"], min_visible=0.0, min_peak=0.0)
        uv = z["uv_full"][r].reshape(len(names), 2)
        peak = np.atleast_1d(z["peak"][r])
        for j, name in enumerate(names):
            d = dets[name]
            assert d is not None
            assert abs(d.u - uv[j][0]) < 0.05 and abs(d.v - uv[j][1]) < 0.05
            assert abs(d.conf - peak[j]) < 1e-3
