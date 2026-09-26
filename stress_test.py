"""Stress test: smaller faults and a noisier camera."""
import numpy as np
from kintrace import sim
from kintrace.diagnose import learn_baseline, diagnose

def run(miss_mm, noise_x, n=6, seed=0):
    old = sim.CAM_NOISE.copy()
    sim.CAM_NOISE[:] = old * noise_x
    rng = np.random.default_rng(seed)
    b = learn_baseline(sim.simulate({"type": "none"}, seed=999, with_probe=False))
    ok = tot = 0
    wrong = []
    for c in sim.FAULTS:
        for _ in range(n):
            f = sim.random_fault(c, rng, miss_mm)
            d = diagnose(sim.simulate(f, seed=int(rng.integers(1e9))), b, run_probe=True)
            tot += 1
            if d.primary == c: ok += 1
            else: wrong.append((c, d.primary))
    sim.CAM_NOISE[:] = old
    return ok / tot, wrong

for miss, nx in [(20, 1), (8, 1), (5, 1), (20, 3), (8, 3)]:
    acc, wrong = run(miss, nx)
    print(f"miss {miss:>2} mm, camera noise x{nx}: {acc*100:5.1f}%  wrong: {wrong}")
