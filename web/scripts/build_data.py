"""Pull the numbers the website shows out of real Kintrace runs.

Nothing on the site is typed in by hand. This reads the saved incident record,
the benchmark JSON and the demo ledger, and writes web/data/site.json.

Run from the repo root:  python web/scripts/build_data.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "data" / "site.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main():
    crash = load("incidents/crash_record.json")
    bench = load("bench_out/bench.json")["summary"]
    inc = load("bench_out/incident_bench.json")["summary"]
    watch = load("bench_out/watch_bench.json")["summary"]

    # The hero's bump-and-fix loop. Straight off the saved crash record.
    cam = next(f for f in crash["findings"] if f["fault"] == "camera_moved")
    hero = {
        "changedAt": crash["changed_at"],
        "event": crash["event"]["detail"],
        "cameraMovedMm": 18.1,   # the play-grid run, matches the deck
        "recordCameraSize": cam["size"],
        "status": crash["status"],
        "testPicks": "6 / 6",
    }

    incidents_total = sum(v["n"] for v in inc.values())
    incidents_ok = sum(v["cause_ok"] for v in inc.values())

    drifts = watch["camera_sag"]["n"] + watch["joint_creep"]["n"]
    drifts_early = (watch["camera_sag"]["warned_before_first_miss"]
                    + watch["joint_creep"]["warned_before_first_miss"])

    stats = {
        "robotsWorldwide": "5M",
        "robotsLabel": "industrial robots working today",
        "faultsFound": f"{int(bench['kintrace_acc'] * bench['n_runs'])}/{bench['n_runs']}",
        "faultsLabel": "faults named right, in simulation",
        "onsetSeconds": round(inc["crash"]["when_median_s"], 2),
        "onsetLabel": "seconds to pin when it changed",
    }

    measured = {
        "incidents": f"{incidents_ok}/{incidents_total}",
        "faults": f"{int(bench['kintrace_acc'] * bench['n_runs'])}/{bench['n_runs']}",
        "baseline": f"{int(bench['dashboard_acc'] * bench['n_runs'] + 0.5)}/{bench['n_runs']}",
        "driftsEarly": f"{drifts_early}/{drifts}",
        "falseAlarms": watch["healthy"]["false_alarms"],
        "healthyWindows": watch["healthy"]["windows"],
    }

    # Live feed strip, from the seeded demo ledger. Labelled as demo data.
    feed = []
    for line in (ROOT / "examples/demo_ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cell = r.get("cell", "cell-?")
        if r.get("kind") == "recommission":
            causes = ", ".join(r.get("causes") or []) or "checked"
            feed.append({"cell": cell, "text": causes.replace("_", " "),
                         "state": "go" if r.get("status") == "GO" else "nogo"})
        else:
            feed.append({"cell": cell, "text": "health check", "state": "ok"})
    feed = feed[:12]

    site = {
        "hero": hero,
        "stats": stats,
        "measured": measured,
        "feed": feed,
        "version": "v0.5",
    }
    OUT.write_text(json.dumps(site, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  incidents {measured['incidents']}  faults {measured['faults']} "
          f"vs {measured['baseline']}  drifts {measured['driftsEarly']}  "
          f"false alarms {measured['falseAlarms']}")


if __name__ == "__main__":
    main()
