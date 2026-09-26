"""kintrace ledger: a history of every change, per cell.

A plain JSONL file (one entry per line) under --db. It takes recommission
records (from `kintrace incident`, signed or not) and watch warnings (from
`kintrace watch --json`), tagged by cell and robot. `report` shows, per cell,
what changed, how often and why, and an estimate of downtime avoided.

Downtime avoided is an estimate from a stated assumption, not a measurement:
  * a recommission that found a change, fixed it and ended GO saves the wait for a technician
    (default 4 h per incident)
  * a watch warning lets the fix happen at a planned stop instead of after
    picks fail (default 2 h per warning)
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timedelta

HOURS_PER_GO = 4.0
HOURS_PER_WARNING = 2.0

CAUSE_LABEL = {
    "camera_moved": "camera moved",
    "tool_bent": "tool bent",
    "tcp_config": "tool offset edited",
    "encoder_bias": "joint zero off",
    "latency": "command latency",
    "camera_sag": "camera sag (slow)",
    "joint_creep": "joint creep (slow)",
    "unexplained": "unexplained",
}


def load(db: str) -> list:
    if not os.path.exists(db):
        return []
    with open(db) as f:
        return [json.loads(line) for line in f if line.strip()]


def append(db: str, entries: list) -> None:
    d = os.path.dirname(db)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(db, "a") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")


def _at(rec: dict) -> str | None:
    st = rec.get("start_time")
    t = rec.get("onset_s", rec.get("t"))
    if st and t is not None:
        return (datetime.fromisoformat(st) + timedelta(seconds=float(t))).isoformat(timespec="seconds")
    return st


def entry_from_record(rec: dict, cell: str, robot: str, at: str | None = None, source: str = "",
                      simulated: bool | None = None) -> dict:
    """Turn a recommission record or a watch warning into a ledger entry."""
    from .certify import verify

    signed = "signature" in rec
    sig_ok = verify(rec)[0] if signed else None
    if rec.get("kind") == "watch_warning":
        kind = "watch_warning"
        causes = [rec["cause"]]
        status = None
        detail = rec.get("text", "")
    elif "status" in rec and "findings" in rec:
        kind = "recommission"
        causes = [f["fault"] for f in rec["findings"]]
        status = rec["status"]
        detail = "; ".join(f"{f['label']}: {f['size']}" for f in rec["findings"]) or "nothing changed"
    else:
        raise ValueError("not a recommission record or a watch warning")
    e = dict(kind=kind, cell=cell, robot=robot, at=at or _at(rec), causes=causes, status=status,
             detail=detail, signed=signed, signature_ok=sig_ok, source=source,
             simulated=bool(simulated if simulated is not None else rec.get("simulated", "truth" in rec)))
    e["id"] = hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()[:12]
    return e


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def report(entries: list, hours_per_go: float = HOURS_PER_GO, hours_per_warning: float = HOURS_PER_WARNING) -> dict:
    cells = {}
    for c in sorted({e["cell"] for e in entries}):
        ee = sorted((e for e in entries if e["cell"] == c), key=lambda e: e["at"] or "")
        times = [datetime.fromisoformat(e["at"]) for e in ee if e["at"]]
        gaps = [(b - a).total_seconds() / 86400 for a, b in zip(times, times[1:])]
        span_days = (times[-1] - times[0]).total_seconds() / 86400 if len(times) > 1 else 0.0
        rec = [e for e in ee if e["kind"] == "recommission"]
        warn = [e for e in ee if e["kind"] == "watch_warning"]
        causes = Counter(x for e in ee for x in e["causes"] if x != "none")
        n_go = sum(e["status"] == "GO" for e in rec)
        n_fixed = sum(e["status"] == "GO" and bool(e["causes"]) for e in rec)
        cells[c] = dict(
            robots=sorted({e["robot"] for e in ee}),
            incidents=len(rec),
            warnings=len(warn),
            go=n_go,
            no_go=sum(e["status"] == "NO-GO" for e in rec),
            causes=dict(causes.most_common()),
            mean_days_between_changes=sum(gaps) / len(gaps) if gaps else None,
            first=ee[0]["at"], last=ee[-1]["at"], span_days=span_days,
            downtime_avoided_h=n_fixed * hours_per_go + len(warn) * hours_per_warning,
            unsigned=sum(1 for e in rec if not e["signed"]),
            bad_signatures=sum(1 for e in ee if e["signature_ok"] is False),
        )
    all_times = sorted(e["at"] for e in entries if e["at"])
    weeks = 1.0
    if len(all_times) > 1:
        weeks = max((datetime.fromisoformat(all_times[-1]) - datetime.fromisoformat(all_times[0])).days / 7, 1.0)
    for v in cells.values():
        v["changes_per_week"] = (v["incidents"] + v["warnings"]) / weeks
    ranked = sorted(cells, key=lambda c: (-cells[c]["changes_per_week"], -cells[c]["warnings"], c))
    return dict(cells=cells, ranked=ranked, weeks=weeks, n_entries=len(entries),
                simulated=any(e.get("simulated") for e in entries),
                causes=dict(Counter(x for e in entries for x in e["causes"] if x != "none").most_common()),
                downtime_avoided_h=sum(v["downtime_avoided_h"] for v in cells.values()),
                assumption=dict(hours_per_go=hours_per_go, hours_per_warning=hours_per_warning))


def text_report(r: dict) -> str:
    out = ["KINTRACE LEDGER", "=" * 64]
    if r["simulated"]:
        out.append("SIMULATED DATA: these entries come from the simulator, not from real cells.")
    out.append(f"{r['n_entries']} entries, {len(r['cells'])} cells, over {r['weeks']:.1f} weeks")
    out.append("")
    out.append(f"  {'cell':<10}{'robot':<10}{'incidents':>10}{'warnings':>10}{'GO/NO-GO':>10}"
               f"{'days apart':>12}{'per week':>10}  top causes")
    for c in r["ranked"]:
        v = r["cells"][c]
        mtb = "-" if v["mean_days_between_changes"] is None else f"{v['mean_days_between_changes']:.1f}"
        top = ", ".join(f"{CAUSE_LABEL.get(k, k)} x{n}" for k, n in list(v["causes"].items())[:3]) or "-"
        out.append(f"  {c:<10}{','.join(v['robots']):<10}{v['incidents']:>10}{v['warnings']:>10}"
                   f"{str(v['go']) + '/' + str(v['no_go']):>10}{mtb:>12}{v['changes_per_week']:>10.1f}  {top}")
    out.append("")
    if r["ranked"]:
        top = r["ranked"][:2]
        out.append("Most changes: " + "; ".join(
            f"{c} ({r['cells'][c]['changes_per_week']:.1f} changes/week, mostly "
            f"{CAUSE_LABEL.get(next(iter(r['cells'][c]['causes']), '-'), '-')})" for c in top))
    drifty = sorted((c for c in r["cells"] if r["cells"][c]["warnings"]), key=lambda c: -r["cells"][c]["warnings"])
    out.append("Most slow drift (watch warnings): " + (", ".join(
        f"{c} ({r['cells'][c]['warnings']})" for c in drifty[:3]) or "none"))
    out.append("All causes: " + ", ".join(f"{CAUSE_LABEL.get(k, k)} {n}" for k, n in r["causes"].items()))
    a = r["assumption"]
    out.append(f"Downtime avoided (estimate): {r['downtime_avoided_h']:.0f} h. Assumes each fix that ended GO "
               f"saves {a['hours_per_go']:.0f} h waiting for a technician and each early warning saves "
               f"{a['hours_per_warning']:.0f} h of unplanned stop.")
    bad = sum(v["bad_signatures"] for v in r["cells"].values())
    uns = sum(v["unsigned"] for v in r["cells"].values())
    out.append(f"Records: {uns} unsigned, {bad} with a signature that fails.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# demo ledger: real Kintrace runs on simulated incidents
# --------------------------------------------------------------------------
# how each simulated cell tends to fail (weights over incident kinds)
DEMO_CELLS = {
    "cell-1": ("ur5e-01", {"crash": 1, "tcp_edit": 1}),
    "cell-2": ("ur5e-02", {"camera_sag": 3, "crash": 1}),
    "cell-3": ("ur5e-03", {"joint_creep": 2, "joint_service": 1}),
    "cell-4": ("ur5e-04", {"software_update": 1, "tcp_edit": 1}),
    "cell-5": ("ur5e-05", {"crash": 2, "camera_sag": 1, "joint_creep": 1}),
    "cell-6": ("ur5e-06", {"healthy": 1}),
}
DEMO_COUNTS = {"cell-1": 3, "cell-2": 5, "cell-3": 4, "cell-4": 2, "cell-5": 5, "cell-6": 1}


def seed_demo(db: str, weeks: int = 4, seed: int = 7, key=None, progress=print) -> list:
    """Run Kintrace on simulated incidents in 6 cells over a few weeks and
    write the results to the ledger. Every entry is marked simulated."""
    import numpy as np

    from . import sim, watch
    from .bench_incident import SCEN
    from .certify import sign
    from .diagnose import learn_baseline
    from .recommission import recommission_sim, to_dict

    rng = np.random.default_rng(seed)
    b = learn_baseline(sim.simulate({"type": "none"}, seed=1, with_probe=False))
    t0 = datetime(2026, 8, 31, 6, 0)
    entries = []
    for c, (robot, mix) in DEMO_CELLS.items():
        kinds = list(mix)
        p = np.array([mix[k] for k in kinds], float)
        days = np.sort(rng.uniform(0, weeks * 7, DEMO_COUNTS[c]))
        for day in days:
            kind = str(rng.choice(kinds, p=p / p.sum()))
            s = int(rng.integers(1e6))
            at = t0 + timedelta(days=float(day))
            if kind in ("camera_sag", "joint_creep"):
                d = sim.random_drift(kind, np.random.default_rng(s))
                r = watch.watch(sim.simulate_drift(d, seed=s, n_picks=40), b)
                rec = watch.warning_record(r, f"{c} watch")
                if rec is None:
                    continue
                rec["simulated"] = True
            else:
                fs = [sim.random_fault(k, np.random.default_rng(s)) for k in SCEN[kind]]
                fault = {"type": "none"} if not fs else fs[0] if len(fs) == 1 else {"type": "multi", "faults": fs}
                log = sim.simulate(fault, seed=s, onset_pick=5 if fs else 0, n_picks=10, with_probe=False)
                rec = to_dict(recommission_sim(log, b, sim._World(fault), seed=s))
                for k in ("series", "picks"):
                    rec.pop(k, None)
                rec.update(scenario=kind, simulated=True)
            if key is not None:
                rec = sign(rec, key)
            entries.append(entry_from_record(rec, c, robot, at=at.isoformat(timespec="seconds"),
                                             source=f"simulated {kind}", simulated=True))
        progress(f"  {c} done ({len([e for e in entries if e['cell'] == c])} entries)")
    append(db, entries)
    return entries
