"""Ledger: ingest records and warnings, per-cell report."""
import json

from kintrace import certify, ledger


def _rec(status, faults, start="2026-09-01T08:00:00", onset=10.0):
    return {"start_time": start, "onset_s": onset, "status": status,
            "findings": [{"fault": f, "label": f, "size": "1 mm", "fix": ""} for f in faults]}


def _warn(cause, at):
    return {"kind": "watch_warning", "cause": cause, "text": "drift", "start_time": at, "t": 0.0}


def test_report(tmp_path):
    db = str(tmp_path / "l.jsonl")
    es = [
        ledger.entry_from_record(_rec("GO", ["camera_moved"]), "c1", "r1"),
        ledger.entry_from_record(_rec("NO-GO", ["latency"], start="2026-09-03T08:00:00"), "c1", "r1"),
        ledger.entry_from_record(_warn("camera_sag", "2026-09-05T08:00:00"), "c1", "r1"),
        ledger.entry_from_record(_rec("GO", [], start="2026-09-10T08:00:00"), "c2", "r2"),
    ]
    ledger.append(db, es)
    r = ledger.report(ledger.load(db), hours_per_go=4, hours_per_warning=2)
    c1, c2 = r["cells"]["c1"], r["cells"]["c2"]
    assert c1["incidents"] == 2 and c1["warnings"] == 1 and c1["go"] == 1 and c1["no_go"] == 1
    assert c1["causes"] == {"camera_moved": 1, "latency": 1, "camera_sag": 1}
    assert abs(c1["mean_days_between_changes"] - 2.0) < 1e-3
    # one GO fix (4 h) + one warning (2 h); the healthy GO in c2 fixed nothing
    assert c1["downtime_avoided_h"] == 6 and c2["downtime_avoided_h"] == 0
    assert r["ranked"][0] == "c1"
    txt = ledger.text_report(r)
    assert "c1" in txt and "Downtime avoided" in txt and "SIMULATED" not in txt


def test_signed_record_checked_on_ingest(tmp_path):
    priv, _ = certify.init_keys(str(tmp_path / "k"))
    s = certify.sign(_rec("GO", ["tool_bent"]), certify.load_private(priv))
    assert ledger.entry_from_record(s, "c", "r")["signature_ok"] is True
    s["status"] = "NO-GO"
    assert ledger.entry_from_record(s, "c", "r")["signature_ok"] is False


def test_simulated_flag_shows(tmp_path):
    e = ledger.entry_from_record(_rec("GO", ["tcp_config"]), "c", "r", simulated=True)
    assert "SIMULATED" in ledger.text_report(ledger.report([e]))


def test_cli_add_and_report(tmp_path, capsys):
    from kintrace.cli import main

    db = str(tmp_path / "l.jsonl")
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_rec("GO", ["encoder_bias"])))
    main(["ledger", "add", str(p), "--cell", "cell-9", "--robot", "arm-1", "--db", db])
    main(["ledger", "report", "--db", db])
    out = capsys.readouterr().out
    assert "cell-9" in out and "joint zero off" in out
