"""kintrace command line.

  kintrace simulate --fault camera_moved -o bad.npz     make a test log
  kintrace baseline healthy.npz -o baseline.json        learn what normal looks like
  kintrace diagnose bad.npz -b baseline.json            name the cause, size and fix
  kintrace diagnose bad.npz -b baseline.json --probe    also use the 1.2 s probe
  kintrace import <recording> --config rig.json -o log.npz   LeRobot / ROS bag / MCAP / UR RTDE -> log
  kintrace check log.npz                                one log, no baseline needed
  kintrace incident                                      after a crash or change: check, fix, verify, certify
  kintrace bench                                        Kintrace vs a standard dashboard
  kintrace demo                                         run everything into ./demo
  kintrace rig ...                                      the SO-101 desk rig (see HARDWARE.md)
  kintrace watch / ledger / keys / certify / verify     see cli_extra.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import bench, sim
from .diagnose import diagnose, learn_baseline
from .logio import Log
from .report import chart, text_report


def _cmd_simulate(a):
    rng = np.random.default_rng(a.seed)
    fault = sim.random_fault(a.fault, rng, a.miss_mm)
    log = sim.simulate(fault, seed=a.seed)
    log.save(a.out)
    print(f"wrote {a.out}  ({a.fault}, {int((~log.pick_ok).sum())}/{len(log.pick_ok)} picks missed)")


def _cmd_baseline(a):
    b = learn_baseline(Log.load(a.log))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(b, f, indent=2)
    print(f"wrote {a.out}  (delay {b['tau']*1000:.1f} ms, camera noise {np.array(b['sigma_cam'])*1000} mm)")


def _cmd_diagnose(a):
    log = Log.load(a.log)
    with open(a.baseline) as f:
        b = json.load(f)
    d = diagnose(log, b, run_probe=a.probe)
    name = os.path.basename(a.log)
    print(text_report(d, name))
    if a.chart:
        chart(d, a.chart, name)
        print(f"\nchart: {a.chart}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(d.to_json())
        print(f"json : {a.json}")


def _cmd_import(a):
    """Someone else's recording -> a kintrace Log."""
    from . import readers

    fmt = a.format or readers.guess_format(a.path)
    if fmt is None:
        sys.exit("could not guess the format, pass --format lerobot|rosbag|mcap|ur_rtde")
    kw = {}
    if fmt == "lerobot":
        kw = dict(episode=a.episode, units=a.units, camera=a.camera)
    elif fmt in ("rosbag", "mcap"):
        kw = dict(joint_topic=a.joint_topic, cmd_topic=a.cmd_topic, image_topic=a.image_topic)
    elif fmt == "ur_rtde":
        kw = dict(video=a.video)
    cfg = None
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
    if fmt == "lerobot" and cfg and "joint_zero_offsets_deg" in cfg:
        kw["rig_cfg"] = cfg
    stream = readers.FORMATS[fmt](a.path, **kw)
    print(stream.describe())
    if cfg is None:
        sys.exit("\nno --config: need arm, camera_extrinsic, markers and tcp_offset to build a Log (a rig.json works)")
    if "camera_extrinsic" not in cfg and "known_good" in cfg:
        cfg = dict(cfg, **{k: cfg["known_good"][k] for k in ("camera_extrinsic", "tcp_offset") if k in cfg["known_good"]})
    if "markers" not in cfg and "tags" in cfg:
        from .hw import rig
        cfg["markers"] = rig.wrist_markers(cfg).tolist()
    locate = None
    if a.intrinsics and stream.frames is not None:
        from .hw.markers import Intrinsics
        tag = cfg.get("tags", {})
        locate = readers.locate_tag(Intrinsics.load(a.intrinsics), int(a.tag_id if a.tag_id is not None else tag.get("wrist_id", 0)),
                                    float(a.tag_size or tag.get("wrist_size", 0.04)))
    log = readers.assemble(stream, cfg, locate=locate, every=a.every)
    log.save(a.out)
    n = int(log.visible.sum())
    print(f"wrote {a.out}  (wrist located in {n}/{len(log.visible)} frames" + ("" if locate else ", no --intrinsics so no camera detections") + ")")


def _cmd_check(a):
    """Single-session check: no earlier healthy run needed."""
    from .diagnose import self_baseline

    log = Log.load(a.log)
    d = diagnose(log, self_baseline(log, a.sigma_mm))
    name = os.path.basename(a.log)
    print(text_report(d, name))
    print("\n(single-session mode: settings and timing are taken as given, the question is whether the sensors agree with them)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(d.to_json())
        print(f"json : {a.json}")


def _cmd_bench(a):
    print(f"Running benchmark ({a.n} runs per fault type)...")
    out = bench.run(a.n, a.seed)
    s = out["summary"]
    print(f"\nCorrect cause, {s['n_runs']} runs:")
    print(f"  standard dashboard     {s['dashboard_acc']*100:5.1f}%")
    print(f"  Kintrace, passive only {s['kintrace_passive_acc']*100:5.1f}%  (bent tool flagged for probe)")
    print(f"  Kintrace, with probe   {s['kintrace_acc']*100:5.1f}%")
    print(f"  two faults at once, both found: {s['two_fault_both_found']*100:.0f}%")
    print("\nPer fault (dashboard / Kintrace):")
    for c, v in s["per_class"].items():
        print(f"  {c:<14} {v['dashboard']*100:5.0f}% / {v['kintrace']*100:5.0f}%")
    print("\nMedian size error:")
    units = {"camera_moved": "mm", "encoder_bias": "deg", "latency": "ms", "tcp_config": "mm", "tool_bent": "mm"}
    for c, v in s["size_err_median"].items():
        print(f"  {c:<14} {v:.3f} {units[c]}")
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    bench.chart(s, os.path.join(a.outdir, "bench.png"))
    print(f"\nwrote {a.outdir}/bench.json and bench.png")


SCENARIOS = {
    "crash": ("A collision bends the gripper finger and knocks the camera", ["tool_bent", "camera_moved"]),
    "tcp_edit": ("A technician reloads an old installation file with the wrong tool offset", ["tcp_config"]),
    "joint_service": ("A joint is serviced and its zero comes back off", ["encoder_bias"]),
    "software_update": ("A cell PC update slows the command path", ["latency"]),
}


def _cmd_incident(a):
    from .recommission import recommission_sim, text_record, to_dict

    os.makedirs(a.outdir, exist_ok=True)
    healthy = sim.simulate({"type": "none"}, seed=1, with_probe=False)
    b = learn_baseline(healthy)
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    for i, name in enumerate(names):
        story, kinds = SCENARIOS[name]
        rng = np.random.default_rng(a.seed + i)
        faults = [sim.random_fault(k, rng) for k in kinds]
        fault = faults[0] if len(faults) == 1 else {"type": "multi", "faults": faults}
        log = sim.simulate(fault, seed=a.seed + 20 + i, onset_pick=6, with_probe=False)
        log.save(f"{a.outdir}/{name}_production.npz")
        r = recommission_sim(log, b, sim._World(fault), seed=a.seed + i)
        txt = text_record(r, cell="Cell 3")
        with open(f"{a.outdir}/{name}_record.txt", "w", encoding="utf-8") as f:
            f.write(txt)
        d = to_dict(r)
        d.update(scenario=name, story=story, truth=fault, truth_onset_s=log.truth["onset_s"])
        with open(f"{a.outdir}/{name}_record.json", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        print(f"\n### {name}: {story}\n{txt}")


def _cmd_demo(a):
    d = a.outdir
    os.makedirs(d, exist_ok=True)
    healthy = sim.simulate({"type": "none"}, seed=1)
    healthy.save(f"{d}/healthy.npz")
    b = learn_baseline(healthy)
    with open(f"{d}/baseline.json", "w", encoding="utf-8") as f:
        json.dump(b, f, indent=2)
    for i, kind in enumerate(sim.FAULTS[1:]):
        rng = np.random.default_rng(100 + i)
        log = sim.simulate(sim.random_fault(kind, rng), seed=200 + i)
        path = f"{d}/{kind}.npz"
        log.save(path)
        diag = diagnose(log, b, run_probe=True)
        with open(f"{d}/{kind}.txt", "w", encoding="utf-8") as f:
            f.write(text_report(diag, f"{kind}.npz"))
        chart(diag, f"{d}/{kind}.png", f"{kind}.npz")
        print(f"{kind:<14} -> {diag.findings[0].label}: {diag.findings[0].size}")
    print(f"\nlogs, reports and charts in {d}/")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kintrace", description="Tell which physical thing changed on a robot, from its own logs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a simulated log with one fault")
    s.add_argument("--fault", choices=sim.FAULTS, default="camera_moved")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--miss-mm", type=float, default=20.0)
    s.add_argument("-o", "--out", default="log.npz")
    s.set_defaults(fn=_cmd_simulate)

    s = sub.add_parser("baseline", help="learn normal behavior from a healthy log")
    s.add_argument("log")
    s.add_argument("-o", "--out", default="baseline.json")
    s.set_defaults(fn=_cmd_baseline)

    s = sub.add_parser("diagnose", help="diagnose a log")
    s.add_argument("log")
    s.add_argument("-b", "--baseline", required=True)
    s.add_argument("--probe", action="store_true", help="use the 1.2 s probe if sensors all agree")
    s.add_argument("--chart", help="write a PNG chart")
    s.add_argument("--json", help="write the result as JSON")
    s.set_defaults(fn=_cmd_diagnose)

    s = sub.add_parser("import", help="turn a LeRobot dataset, ROS bag/MCAP or UR RTDE recording into a log")
    s.add_argument("path")
    s.add_argument("--format", choices=["lerobot", "rosbag", "mcap", "ur_rtde"])
    s.add_argument("--config", help="rig.json or any json with arm, camera_extrinsic, markers, tcp_offset")
    s.add_argument("--intrinsics", help="camera intrinsics json (from kintrace rig calibrate)")
    s.add_argument("--tag-id", type=int, help="wrist AprilTag id (default from config)")
    s.add_argument("--tag-size", type=float, help="wrist tag size in metres (default from config)")
    s.add_argument("--every", type=int, default=1, help="use every Nth frame")
    s.add_argument("--episode", type=int, default=0, help="lerobot: episode index")
    s.add_argument("--units", choices=["deg", "rad"], default="deg", help="lerobot: joint units in the dataset")
    s.add_argument("--camera", help="lerobot: observation.images.<name> to use")
    s.add_argument("--joint-topic")
    s.add_argument("--cmd-topic")
    s.add_argument("--image-topic")
    s.add_argument("--video", help="ur_rtde: camera video recorded alongside")
    s.add_argument("-o", "--out", default="log.npz")
    s.set_defaults(fn=_cmd_import)

    s = sub.add_parser("check", help="check one log with no earlier baseline")
    s.add_argument("log")
    s.add_argument("--sigma-mm", type=float, default=4.0, help="expected wrist detection noise")
    s.add_argument("--json")
    s.set_defaults(fn=_cmd_check)

    s = sub.add_parser("bench", help="Kintrace vs a standard monitoring dashboard")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--outdir", default="bench_out")
    s.set_defaults(fn=_cmd_bench)

    s = sub.add_parser("incident", help="after a crash or change: check, fix, verify, certify (simulated cell)")
    s.add_argument("--scenario", choices=["all"] + list(SCENARIOS), default="all")
    s.add_argument("--seed", type=int, default=4)
    s.add_argument("--outdir", default="incidents")
    s.set_defaults(fn=_cmd_incident)

    s = sub.add_parser("demo", help="generate example logs, reports and charts")
    s.add_argument("--outdir", default="demo")
    s.set_defaults(fn=_cmd_demo)

    from .hw.cli_rig import add_parser as _add_rig  # SO-101 desk rig, see HARDWARE.md
    _add_rig(sub)

    from .cli_extra import add_commands  # watch, ledger, keys, certify, verify
    add_commands(sub)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
