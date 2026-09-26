"""The desk rig end to end with --dry-run (simulated SO-101 and webcam).

The dry run renders AprilTag images of a simulated desk and runs the real
detector, solvePnP, commissioning, check, fix and touch test on them. Only
the arm and the camera are fake.

Each check motion takes about 20 s of CPU, so the default run covers
commission, capture and one fault. Set KINTRACE_SLOW=1 to run all five faults.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

from kintrace.cli import main  # noqa: E402
from kintrace.logio import Log  # noqa: E402

SLOW = os.environ.get("KINTRACE_SLOW") == "1"


@pytest.fixture(scope="module")
def rigdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("rig")
    main(["rig", "commission", "--dry-run", "-c", str(d / "rig.json")])
    return d


def _cfg(d):
    with open(d / "rig.json") as f:
        return json.load(f)


def test_commission_writes_known_good(rigdir):
    cfg = _cfg(rigdir)
    assert cfg["commissioned"] and cfg["dry_run"]
    kg = cfg["known_good"]
    assert 0.02 < kg["tau"] < 0.06  # simulated servo lag is 35 ms
    assert len(cfg["fixed_markers"]) == 4
    assert len(cfg["check_poses_deg"]) == 8
    assert "target_world" in cfg["touch"]


def test_capture_writes_a_kintrace_log(rigdir):
    out = rigdir / "cap.npz"
    main(["rig", "capture", "--dry-run", "-c", str(rigdir / "rig.json"), "-o", str(out)])
    log = Log.load(str(out))
    assert log.config["arm"] == "so101"
    assert log.q_cmd.shape[1] == 5 and log.q_meas.shape == log.q_cmd.shape
    assert log.markers_cam.shape[1:] == (4, 3) and len(log.t_cam) >= 20
    assert log.fixed_cam is not None and log.fixed_cam.shape[1:] == (4, 3)
    assert len(log.probe["tip_cam"]) >= 20
    assert len(log.pick_ok) == 0
    # offline analysis of a healthy capture finds nothing
    from kintrace.hw import check, rig

    r = check.analyze(log, rig.load(str(rigdir / "rig.json")))
    assert r["findings"] == [], [f.label for f in r["findings"]]


def _check(rigdir, fault, seed=0):
    rec = rigdir / f"rec_{fault}_{seed}"
    main(["rig", "check", "--dry-run", "-c", str(rigdir / "rig.json"), "--fault", fault,
          "--seed", str(seed), "--record", str(rec)])
    with open(str(rec) + ".json") as f:
        return json.load(f)


def _causes(r):
    return {f["fault"] for f in r["before"]["findings"]}


def test_camera_bump_found_fixed_and_go(rigdir):
    r = _check(rigdir, "camera_moved")
    assert _causes(r) == {"camera_moved"}
    f = r["before"]["findings"][0]
    injected = np.linalg.norm(r["dry_run_fault"]["translation"]) * 1000
    measured = float(f["size"].split(" mm")[0])
    assert abs(measured - injected) < 2.0
    assert r["status"] == "GO"
    assert r["touch"]["ok"]


@pytest.mark.skipif(not SLOW, reason="set KINTRACE_SLOW=1")
@pytest.mark.parametrize("fault,expect,status", [
    ("none", set(), "GO"),
    ("tool_bent", {"tool_bent"}, "GO"),
    ("tcp_config", {"tcp_config"}, "GO"),
    ("encoder_bias", {"encoder_bias"}, "GO"),
    ("latency", {"latency"}, "NO-GO"),
])
def test_each_fault(rigdir, fault, expect, status):
    r = _check(rigdir, fault)
    assert _causes(r) == expect
    assert r["status"] == status
