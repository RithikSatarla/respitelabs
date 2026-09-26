"""kintrace watch: slow drift is caught before picks fail, healthy runs stay quiet."""
import numpy as np
import pytest

from kintrace import sim, watch
from kintrace.diagnose import learn_baseline


@pytest.fixture(scope="module")
def baseline():
    return learn_baseline(sim.simulate({"type": "none"}, seed=1, with_probe=False))


def _first_miss(log):
    miss = np.where(~log.pick_ok)[0]
    return float(log.pick_t[miss[0]]) if len(miss) else None


def test_existing_faults_unchanged():
    # the drift code is additive: a classic fault log still has no drift in it
    log = sim.simulate(sim.random_fault("camera_moved", np.random.default_rng(0)), seed=0, n_picks=3)
    assert "drift" not in log.truth
    assert sim.FAULTS == ["none", "camera_moved", "tool_bent", "tcp_config", "encoder_bias", "latency"]


def test_healthy_no_warning(baseline):
    log = sim.simulate_drift({"type": "none"}, seed=2, n_picks=40)
    r = watch.watch(log, baseline)
    assert r["warning"] is None
    assert min(w["health"] for w in r["windows"]) > 90


def test_camera_sag_warns_early(baseline):
    d = sim.random_drift("camera_sag", np.random.default_rng(0))
    log = sim.simulate_drift(d, seed=0, n_picks=45)
    r = watch.watch(log, baseline)
    wn = r["warning"]
    assert wn is not None and wn["cause"] == "camera_sag"
    fm = _first_miss(log)
    assert fm is not None and wn["t"] < fm
    assert wn["eta_s"] is not None and wn["eta_s"] > 0
    # health falls as the camera sags
    assert r["windows"][0]["health"] > r["windows"][-1]["health"]


def test_joint_creep_names_the_joint(baseline):
    d = sim.random_drift("joint_creep", np.random.default_rng(1002))
    log = sim.simulate_drift(d, seed=1002, n_picks=45)
    wn = watch.watch(log, baseline)["warning"]
    assert wn is not None and wn["cause"] == "joint_creep"
    assert wn["size"]["joint"] == d["joint"] + 1
    assert np.sign(wn["size"]["bias_deg"]) == np.sign(d["rate_rad_s"])


def test_warning_record_shape(baseline):
    d = sim.random_drift("camera_sag", np.random.default_rng(3))
    r = watch.watch(sim.simulate_drift(d, seed=3, n_picks=35), baseline)
    rec = watch.warning_record(r, "x")
    assert rec["kind"] == "watch_warning" and rec["cause"] == "camera_sag" and rec["at"]
