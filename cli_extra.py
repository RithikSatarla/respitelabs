"""CLI for watch, ledger, keys, certify and verify.

  kintrace watch LOG -b baseline.json            health score + early warning from a production log
  kintrace watch --sim camera_sag --seed 3       same, on a simulated slow drift
  kintrace watch --bench -n 10                   lead-time benchmark -> bench_out/watch_bench.json
  kintrace ledger add RECORD.json --cell c1 --robot r1
  kintrace ledger report
  kintrace ledger demo                           seed a SIMULATED ledger (6 cells, 4 weeks)
  kintrace keys init
  kintrace certify RECORD.json                   sign a record (ed25519)
  kintrace verify RECORD.json --pub keys/public.pem
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

DEFAULT_KEYDIR = "kintrace_keys"
DEFAULT_DB = "kintrace_ledger.jsonl"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# watch
# --------------------------------------------------------------------------
def _cmd_watch(a):
    from . import sim, watch
    from .diagnose import learn_baseline
    from .logio import Log

    if a.bench:
        print(f"Watch benchmark: {a.n} runs each of camera sag, joint creep and healthy "
              f"({a.picks} picks, about {a.picks * sim.PICK_S / 60:.0f} min of simulated log per run)")
        out = watch.run_bench(a.n, a.seed, n_picks=a.picks)
        s = out["summary"]
        for k in ("camera_sag", "joint_creep"):
            v = s[k]
            print(f"\n{k}: speed {min(v['drift_speed_pick_mm_per_min']):.1f}-{max(v['drift_speed_pick_mm_per_min']):.1f} "
                  "mm/min at the pick area (sped up)")
            print(f"  warned before the first missed pick : {v['warned_before_first_miss']}/{v['n']}")
            print(f"  right cause                          : {v['cause_right']}/{v['n']}")
            print(f"  median lead time                     : {v['median_lead_s']:.0f} s (min {v['min_lead_s']:.0f} s), "
                  f"{v['median_lead_share']*100:.0f}% of the time from drift start to first miss")
            print(f"  median error when the warning fires  : {v['median_warn_mm']:.1f} mm "
                  f"(vs {v['median_error_at_first_miss_mm']:.1f} mm at the first miss)")
            print(f"  projected time to first miss, error  : median {v['median_eta_error_s']:+.0f} s "
                  f"(abs {v['median_abs_eta_error_s']:.0f} s)")
        h = s["healthy"]
        print(f"\nhealthy: {h['false_alarms']}/{h['n']} runs with a false alarm ({h['windows']} windows scored)")
        print("\n" + s["settings"]["note"])
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nwrote {a.out}")
        return

    if a.sim:
        d = sim.random_drift(a.sim, np.random.default_rng(a.seed))
        log = sim.simulate_drift(d, seed=a.seed, n_picks=a.picks)
        b = learn_baseline(sim.simulate({"type": "none"}, seed=1, with_probe=False))
        name = f"simulated {a.sim} (seed {a.seed})"
        if a.sim != "none":
            print(f"Simulated drift, sped up: error at the pick area grows {d['pick_mm_per_min']:.1f} mm/min "
                  f"({d['deg_per_min']:.2f} deg/min) from t+{d['start_s']:.0f}s. Real drift is far slower.\n")
    else:
        if not a.log or not a.baseline:
            raise SystemExit("give a log and -b baseline.json, or --sim KIND, or --bench")
        log, b, name = Log.load(a.log), _load_json(a.baseline), os.path.basename(a.log)
    r = watch.watch(log, b, window_s=a.window, warn_at=a.warn_at / 100, tol_xy_mm=a.tol_mm)
    print(watch.text_report(r, name, every=a.every))
    if a.sim and len(log.pick_ok):
        miss = np.where(~log.pick_ok)[0]
        if len(miss):
            from .recommission import _clock
            print(f"\n(simulation: first missed pick at {_clock(r['start_time'], log.pick_t[miss[0]])})")
    if a.json:
        rec = watch.warning_record(r, name) or dict(kind="watch_ok", source=name)
        rec["windows"] = r["windows"]
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, default=float)
        print(f"json : {a.json}")


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------
def _cmd_ledger(a):
    from . import ledger

    if a.cmd_ledger == "add":
        es = []
        for p in a.records:
            rec = _load_json(p)
            rec.pop("windows", None)
            es.append(ledger.entry_from_record(rec, a.cell, a.robot, at=a.at, source=os.path.basename(p)))
        ledger.append(a.db, es)
        for e in es:
            sig = "" if not e["signed"] else ("  signature ok" if e["signature_ok"] else "  SIGNATURE FAILS")
            print(f"added {e['kind']} for {e['cell']}/{e['robot']} at {e['at']}: {', '.join(e['causes']) or 'nothing changed'}{sig}")
    elif a.cmd_ledger == "report":
        es = ledger.load(a.db)
        if not es:
            raise SystemExit(f"{a.db} is empty or missing. Try `kintrace ledger demo --db {a.db}`.")
        r = ledger.report(es, a.hours_per_fix, a.hours_per_warning)
        print(ledger.text_report(r))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2)
    elif a.cmd_ledger == "demo":
        from .certify import load_private

        if os.path.exists(a.db) and not a.append:
            raise SystemExit(f"{a.db} exists. Delete it or pass --append.")
        key = load_private(a.key) if a.key else None
        print("Seeding a SIMULATED ledger: Kintrace runs on simulated incidents and drifts in 6 cells over 4 weeks.")
        es = ledger.seed_demo(a.db, weeks=a.weeks, seed=a.seed, key=key)
        print(f"wrote {len(es)} simulated entries to {a.db}\n")
        print(ledger.text_report(ledger.report(ledger.load(a.db))))


# --------------------------------------------------------------------------
# keys, certify, verify
# --------------------------------------------------------------------------
def _cmd_keys(a):
    from .certify import init_keys, key_id, load_public

    try:
        priv, pub = init_keys(a.dir, force=a.force)
    except FileExistsError as e:
        raise SystemExit(str(e))
    print(f"wrote {priv} (keep private) and {pub} (share with whoever checks your records)")
    print(f"key id {key_id(load_public(pub))}")


def _cmd_certify(a):
    from .certify import load_private, sign

    rec = _load_json(a.record)
    signed = sign(rec, load_private(a.key))
    out = a.out or a.record
    with open(out, "w", encoding="utf-8") as f:
        json.dump(signed, f, indent=2)
    st = signed.get("status", "")
    print(f"signed {out}{' (' + st + ')' if st else ''} with key {signed['signature']['key_id']} at {signed['signed_at']}")


def _cmd_verify(a):
    from .certify import load_public, verify

    trusted = load_public(a.pub) if a.pub else None
    bad = 0
    for p in a.records:
        ok, why = verify(_load_json(p), trusted)
        print(f"{'VALID  ' if ok else 'INVALID'} {p}: {why}")
        bad += not ok
    if trusted is None and not bad:
        print("(checked against the key inside the record. Pass --pub to check it is your key.)")
    if bad:
        sys.exit(1)


def add_commands(sub):
    from . import sim

    s = sub.add_parser("watch", help="physical health score and early warning from a production log (no check motion)")
    s.add_argument("log", nargs="?")
    s.add_argument("-b", "--baseline")
    s.add_argument("--sim", choices=["none"] + sim.DRIFTS, help="run on a simulated slow drift instead")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--picks", type=int, default=60, help="simulated picks (about 3.9 s each)")
    s.add_argument("--window", type=float, default=10.0, help="seconds per health score")
    s.add_argument("--warn-at", type=float, default=15.0, help="warn when this %% of the grasp tolerance is used")
    s.add_argument("--tol-mm", type=float, default=8.0, help="gripper's sideways tolerance (mm)")
    s.add_argument("--every", type=int, default=3, help="print every Nth window")
    s.add_argument("--json", help="write the warning (and window scores) as JSON, for `ledger add`")
    s.add_argument("--bench", action="store_true", help="lead-time benchmark over simulated drifts")
    s.add_argument("-n", type=int, default=10, help="--bench: runs per drift type")
    s.add_argument("-o", "--out", default="bench_out/watch_bench.json", help="--bench output")
    s.set_defaults(fn=_cmd_watch)

    s = sub.add_parser("ledger", help="per-cell history of changes, records and warnings")
    ls = s.add_subparsers(dest="cmd_ledger", required=True)
    x = ls.add_parser("add", help="add recommission records or watch warnings")
    x.add_argument("records", nargs="+")
    x.add_argument("--cell", required=True)
    x.add_argument("--robot", required=True)
    x.add_argument("--at", help="ISO time of the change (default: from the record)")
    x = ls.add_parser("report", help="per-cell summary")
    x.add_argument("--json")
    x.add_argument("--hours-per-fix", type=float, default=4.0, help="assumed technician wait saved per GO fix")
    x.add_argument("--hours-per-warning", type=float, default=2.0, help="assumed unplanned stop saved per early warning")
    x = ls.add_parser("demo", help="seed a SIMULATED ledger: 6 cells, a few weeks")
    x.add_argument("--weeks", type=int, default=4)
    x.add_argument("--seed", type=int, default=7)
    x.add_argument("--key", help="sign each record with this private key")
    x.add_argument("--append", action="store_true")
    for x in ls.choices.values():
        x.add_argument("--db", default=DEFAULT_DB, help=f"ledger file (default {DEFAULT_DB})")
    s.set_defaults(fn=_cmd_ledger)

    s = sub.add_parser("keys", help="signing keys for certificates")
    ks = s.add_subparsers(dest="cmd_keys", required=True)
    x = ks.add_parser("init", help="make an ed25519 key pair")
    x.add_argument("--dir", default=DEFAULT_KEYDIR)
    x.add_argument("--force", action="store_true")
    s.set_defaults(fn=_cmd_keys)

    s = sub.add_parser("certify", help="sign a recommission record (ed25519)")
    s.add_argument("record")
    s.add_argument("--key", default=os.path.join(DEFAULT_KEYDIR, "private.pem"))
    s.add_argument("-o", "--out", help="write the signed record here (default: overwrite)")
    s.set_defaults(fn=_cmd_certify)

    s = sub.add_parser("verify", help="check a signed record")
    s.add_argument("records", nargs="+")
    s.add_argument("--pub", help="trusted public key; the record must be signed by it")
    s.set_defaults(fn=_cmd_verify)
