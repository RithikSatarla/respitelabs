"""Score a detector checkpoint on the held-out DROID GT cameras.

Uses the trainer's own split (detector.split_by_serial, seed 0, 8 serials over all
FK-labelled cameras), the trainer's preprocessing (detector.predict), and only
held-out serials. Frames where the FK fingertip is outside the image are not
scored and are counted as skipped.

Every held-out prediction is cached (uv, visibility probability, heatmap peak and
entropy) so the confidence and drift studies can run on the CPU afterwards.

Run: python -m kintrace.droid.eval_detector --ckpt data/droid/<name>.pt --out data/droid/eval_<tag>.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from . import detector

# printed by the Phase 4 training run (data/droid/runs/phase4_20261003_104703.log)
PHASE4_TEST_SERIALS = ["20103212", "20252535", "21582473", "22246076", "23897859", "24400334", "25947356", "29838012"]


predict_conf = detector.predict_conf  # one inference path for evaluation and detect()


def held_out(data: str, seed: int = 0) -> tuple:
    cams = detector.load_cameras(data)
    _, te, test_serials = detector.split_by_serial(cams, seed=seed)
    return te, test_serials


def _predict_all(model, te, heat, pause_s):
    rows = dict(key=[], serial=[], frame=[], visible=[], uv=[], uv_full=[], label_full=[], p=[], peak=[], ent=[], scale=[])
    for c in te:
        uv, p, peak, ent = predict_conf(model, detector.frames_of(c), heat, pause_s=pause_s)
        s = c["full_w"] / detector.SMALL_W
        n = len(uv)
        rows["key"] += [c["key"]] * n
        rows["serial"] += [c["serial"]] * n
        rows["frame"] += list(range(n))
        rows["visible"].append(c["visible"])
        rows["uv"].append(uv)
        rows["uv_full"].append(uv * s)
        rows["label_full"].append(c["uv_full"])
        rows["p"].append(p)
        rows["peak"].append(peak)
        rows["ent"].append(ent)
        rows["scale"].append(np.full(n, s))
    return rows


def run(ckpt: str, out: str, data: str = "data/droid", cache: str | None = None, pause_s: float = 0.0) -> dict:
    """GPU under the heat rules; above HOT_C it switches to the CPU and starts the evaluation again."""
    import torch

    te, test_serials = held_out(data)
    if sorted(test_serials) != sorted(PHASE4_TEST_SERIALS):
        raise SystemExit(f"held-out serials {test_serials} differ from the Phase 4 split {PHASE4_TEST_SERIALS}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    use_resnet = ck.get("use_resnet", any(k.startswith("layer1.") for k in ck["state"]))
    model = detector.build_model(ck.get("ch", 32), use_resnet=use_resnet)
    model.load_state_dict(ck["state"])
    heat = detector.Heat()
    ran_on = str(detector.device())
    try:
        heat.settle("before evaluation (idle)")
        rows = _predict_all(model.to(detector.device()), te, heat, pause_s)
        heat.read("after evaluation")
    except detector.Overheat as e:
        note = f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {e} while evaluating {ckpt} on the GPU; switched to the CPU."
        detector.write_progress(open("PROGRESS.md", encoding="utf-8").read() + f"\n## Heat stop during evaluation\n\n{note}\n")
        print(f"HEAT STOP: {note}")
        torch.cuda.empty_cache()
        ran_on = "cpu (after a GPU heat stop)"
        rows = _predict_all(model.to("cpu"), te, None, 0.0)
    arr = {k: (np.concatenate(v) if k not in ("key", "serial", "frame") else np.asarray(v)) for k, v in rows.items()}
    cache = cache or out.replace(".json", "_preds.npz")
    np.savez_compressed(cache, **arr)

    v = arr["visible"].astype(bool)
    err = np.linalg.norm(arr["uv_full"] - arr["label_full"], axis=1)[v]
    ser = arr["serial"][v]
    per_serial = {s: dict(frames=int((ser == s).sum()), median_px=float(np.median(err[ser == s])))
                  for s in sorted(set(ser.tolist()))}
    res = dict(
        checkpoint=ckpt, device=ran_on, pause_s=pause_s, checkpoint_epochs=int(ck.get("epoch", -1)) if "epoch" in ck else None,
        data="DROID GT cameras, held out by camera serial (never seen in training)",
        held_out_serials=sorted(test_serials), cameras=len(te),
        frames_total=int(len(v)), frames_scored=int(v.sum()), frames_skipped_out_of_view=int((~v).sum()),
        px_full=dict(median=float(np.median(err)), mean=float(err.mean()), p90=float(np.percentile(err, 90)),
                     p95=float(np.percentile(err, 95))),
        within_px={str(t): float((err <= t).mean()) for t in (5, 12, 25)},
        per_serial=per_serial, preds_cache=cache, heat_log=heat.log)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"held-out DROID GT cameras: {res['cameras']} cameras, {len(test_serials)} serials, "
          f"{res['frames_scored']} frames scored, {res['frames_skipped_out_of_view']} skipped (tip out of view)")
    e = res["px_full"]
    print(f"  full-res error: median {e['median']:.1f} px  mean {e['mean']:.1f}  p90 {e['p90']:.1f}  p95 {e['p95']:.1f}")
    print("  within 5 / 12 / 25 px: " + " / ".join(f"{100 * res['within_px'][k]:.1f}%" for k in ("5", "12", "25")))
    print("  per serial (median px, frames): " + ", ".join(f"{s} {d['median_px']:.1f} ({d['frames']})" for s, d in per_serial.items()))
    print("  temperatures: " + ", ".join(f"{lbl} {c} C" for _, lbl, c in heat.log))
    print(f"wrote {out} and {cache}")
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.eval_detector")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/droid")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to rest after every batch (gentle GPU load)")
    a = ap.parse_args(argv)
    run(a.ckpt, a.out, a.data, pause_s=a.pause)


if __name__ == "__main__":
    main()
