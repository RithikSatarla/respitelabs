"""Benchmark for the check-fix-verify flow on random incidents."""
from __future__ import annotations

import json
import time

import numpy as np

from . import sim
from .diagnose import learn_baseline
from .recommission import recommission_sim

SCEN = {
    "crash": ["tool_bent", "camera_moved"],
    "tcp_edit": ["tcp_config"],
    "joint_service": ["encoder_bias"],
    "software_update": ["latency"],
    "healthy": [],
}


def run(n: int = 10, seed: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    b = learn_baseline(sim.simulate({"type": "none"}, seed=1, with_probe=False))
    rows = []
    t0 = time.time()
    for name, kinds in SCEN.items():
        for i in range(n):
            faults = [sim.random_fault(k, rng) for k in kinds]
            fault = {"type": "none"} if not faults else faults[0] if len(faults) == 1 else {"type": "multi", "faults": faults}
            onset = int(rng.integers(4, 9)) if faults else 0
            log = sim.simulate(fault, seed=int(rng.integers(1e9)), onset_pick=onset, with_probe=False)
            r = recommission_sim(log, b, sim._World(fault), seed=int(rng.integers(1e6)))
            found = {f.fault for f in r["before"]["findings"]}
            want = set(kinds)
            when_err = None
            if faults and r["when"]["onset"] is not None:
                when_err = abs(r["when"]["onset"] - log.truth["onset_s"])
            expect = "NO-GO" if "latency" in kinds else "GO"
            rows.append(dict(scenario=name, cause_ok=found == want, status=r["status"], status_ok=r["status"] == expect,
                             when_err=when_err, test_picks=r["test_picks"]))
        print(f"  {name:<16} done ({time.time()-t0:.0f}s)")
    summ = {}
    for name in SCEN:
        rr = [x for x in rows if x["scenario"] == name]
        we = [x["when_err"] for x in rr if x["when_err"] is not None]
        summ[name] = dict(n=len(rr), cause_ok=sum(x["cause_ok"] for x in rr), status_ok=sum(x["status_ok"] for x in rr),
                          when_median_s=float(np.median(we)) if we else None, when_max_s=float(np.max(we)) if we else None)
    return dict(summary=summ, rows=rows)


if __name__ == "__main__":
    out = run()
    print(json.dumps(out["summary"], indent=2))
    json.dump(out, open("bench_out/incident_bench.json", "w", encoding="utf-8"), indent=2, default=str)
