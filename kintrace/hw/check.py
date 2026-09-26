"""Commission, check, fix, verify and certify the desk rig.

  commission  run the check motion on a healthy rig, learn the camera pose,
              tag mounting and joint zeros, then run it again and save that
              run as the known-good state (rig.json "known_good")
  check       run the check motion, name what changed, write the fix to
              rig.json, run it again to verify, touch the target tag, GO / NO-GO

The analysis is the same code as the simulated cell (recommission.analyze_check).
Only the arm model (SO-101) and the source of the numbers (printed tags
seen by a webcam) are different.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np

from ..diagnose import learn_baseline
from ..recommission import analyze_check
from . import capture, rig


# --------------------------------------------------------------------------
# hardware or simulator
# --------------------------------------------------------------------------
class Parts:
    """The arm, camera and clock for one session, real or simulated."""

    def __init__(self, arm, camera, clock, world=None, fault=None):
        self.arm, self.camera, self.clock = arm, camera, clock
        self.world, self.fault = world, fault or {"type": "none"}

    def close(self):
        for p in (self.camera, self.arm):
            try:
                p.close()
            except Exception:
                pass


def open_parts(cfg: dict, dry_run: bool, seed: int = 0, fault: dict | None = None,
               save_frames: str | None = None) -> Parts:
    if save_frames:
        os.makedirs(save_frames, exist_ok=True)
    if dry_run:
        from . import rigsim

        # the desk is the one this rig.json was commissioned on; seed only
        # changes the fault and the image noise
        desk = int(cfg.get("dry_run_desk", seed))
        world = rigsim.SimRig(cfg, seed=desk, fault=fault)
        clock = rigsim.SimClock()
        arm = rigsim.SimArm(world, clock, cfg)
        cam = rigsim.SimCamera(world, arm, cfg, rng=np.random.default_rng(seed + 1), save_dir=save_frames)
        return Parts(arm, cam, clock, world, fault)
    from .so101 import SO101Arm

    clock = capture.RealClock()
    cam = capture.WebCamera(cfg, clock, save_dir=save_frames)
    arm = SO101Arm(cfg).connect()
    return Parts(arm, cam, clock)


def _set_zero_offsets(parts: Parts, cfg: dict):
    parts.arm.off = np.deg2rad(np.array(cfg["joint_zero_offsets_deg"], float))
    parts.arm.sign = np.array(cfg["joint_sign"], float)


# --------------------------------------------------------------------------
# commissioning
# --------------------------------------------------------------------------
def commission(cfg: dict, parts: Parts, search_signs: bool = False, echo=print) -> tuple[dict, dict]:
    """Returns (new cfg with known_good, report)."""
    echo("1/2  learning the rig geometry")
    log, stats = capture.capture(cfg, parts.arm, parts.camera, parts.clock, echo=echo)
    _require(stats, echo)
    new, rep = rig.commission(log, cfg, search_signs=search_signs, echo=echo)
    echo(f"     wrist tag fit {rep['wrist_rms_mm']:.2f} mm rms, tip fit {rep['tip_rms_mm']:.2f} mm rms")
    if rep["wrist_rms_mm"] > 3.0:
        echo("     warning: wrist tag fit is poor. Check the camera calibration and that the tag is flat.")
    _set_zero_offsets(parts, new)
    # freeze the check poses so every later check repeats this exact motion
    new["check_poses_deg"] = np.rad2deg(rig.check_poses(new)).round(3).tolist()

    echo("2/2  recording the known-good state")
    log2, stats2 = capture.capture(new, parts.arm, parts.camera, parts.clock, echo=echo)
    _require(stats2, echo)
    b = learn_baseline(log2)
    tgt = _target_world(log2, parts, new)
    new["known_good"] = dict(
        tau=b["tau"], sigma_cam=b["sigma_cam"], tcp_offset=new["tcp_offset"],
        camera_extrinsic=new["camera_extrinsic"], recorded=datetime.now().isoformat(timespec="seconds"))
    if tgt is not None:
        new["touch"]["target_world"] = tgt.round(5).tolist()
    rep.update(tau_ms=b["tau"] * 1000, sigma_mm=(np.array(b["sigma_cam"]) * 1000).tolist(), stats=stats2)
    return new, rep


def _require(stats, echo):
    probs = capture.check_stats(stats)
    if probs:
        raise RuntimeError("capture is not usable: " + "; ".join(probs)
                           + ". Check lighting, focus and that the tags face the camera (see HARDWARE.md).")


def _target_world(log, parts, cfg):
    """Where the target tag sits (world), seen while the arm is out of the way."""
    t, obs = parts.camera.grab()
    if obs.target is None:
        return None
    C = np.array(cfg["camera_extrinsic"])
    return C[:3, :3] @ obs.target + C[:3, 3]


# --------------------------------------------------------------------------
# touch test: the rig's version of test picks
# --------------------------------------------------------------------------
def touch_test(cfg: dict, parts: Parts, echo=print) -> dict | None:
    """Find the target tag with the camera, hover the pointer tip hover_mm
    above it, and measure the miss in the camera image. Like a pick, it uses
    the camera calibration, the joint zeros and the tool offset all at once.
    Returns None if the target tag isn't set up."""
    tags = cfg["tags"]
    if tags.get("target_id") is None:
        return None
    # Measure the target first, while the arm is away from it. Once the
    # pointer hovers over it, the tip flag covers part of the target tag.
    seen = []
    for _ in range(5):
        _, obs = parts.camera.grab()
        if obs.target is not None:
            seen.append(obs.target)
    if not seen:
        echo("touch test skipped: target tag not seen")
        return None
    target_cam = np.median(seen, axis=0)
    C = np.array(cfg["camera_extrinsic"])
    hover = cfg["touch"]["hover_mm"] / 1000.0
    goal = C[:3, :3] @ target_cam + C[:3, 3] + [0.0, 0.0, hover]
    eye = C[:3, 3]
    n_f = rig.pose6(cfg["wrist_tag_pose"])[:3, 2]  # tip flag faces the same way as the wrist tag
    d = eye - goal
    d /= np.linalg.norm(d)
    _, q0 = parts.arm.read()
    q_touch = rig.ik_point_dir(np.array(cfg["tcp_offset"]), goal, n_f, d, seed=q0, w_dir=0.3)
    t, q, still = capture.plan_motion(q0, [q_touch], cfg)
    _, _, _, frames = capture.run_motion(parts.arm, parts.camera, parts.clock, t, q, still, echo=echo)
    tips = [o.tip for _, o in frames if o.tip is not None]
    if len(tips) < 3:
        echo(f"touch test: tip flag seen in only {len(tips)} frames at the hover pose")
        return dict(ok=False, miss_mm=None, frames=len(tips))
    diff = np.median(tips, axis=0) - target_cam  # camera frame
    miss = C[:3, :3] @ diff - [0.0, 0.0, hover]  # world frame, minus the planned hover
    miss_mm = float(np.linalg.norm(miss) * 1000)
    ok = miss_mm < cfg["tolerances"].get("touch_mm", 4.0)
    return dict(ok=bool(ok), miss_mm=miss_mm, miss_xyz_mm=(miss * 1000).round(1).tolist(), frames=len(tips))


# --------------------------------------------------------------------------
# check, fix, verify, certify
# --------------------------------------------------------------------------
def _check_dict(log) -> dict:
    """The frames where both the wrist tag and the tip flag were seen, in the
    shape recommission.analyze_check wants."""
    p = log.probe
    return dict(t_joint=log.t_joint, q_cmd=log.q_cmd, q_meas=log.q_meas, t_cam=p["t_cam"],
                markers_cam=p["markers_cam"], tip_cam=p["tip_cam"], fixed_cam=p.get("fixed_cam"))


def analyze(log, cfg: dict) -> dict:
    kg = cfg.get("known_good")
    if not kg:
        raise RuntimeError("rig.json has no known_good state. Run `kintrace rig commission` on a healthy rig first.")
    good = dict(tcp_offset=kg["tcp_offset"], camera_extrinsic=kg["camera_extrinsic"], tau=kg["tau"])
    tol = cfg["tolerances"]
    sigma = np.maximum(np.array(kg["sigma_cam"], float),
                       np.array(tol.get("tag_noise_floor_mm", [0.0, 0.0, 0.0]), float) / 1000.0)
    return analyze_check(_check_dict(log), log.config, good, sigma,
                         tip_tol=tol["tip_mm"] / 1000.0, tau_tol=tol["tau_ms"] / 1000.0)


def apply_fix(cfg: dict, fix: dict) -> dict:
    new = copy.deepcopy(cfg)
    if "camera_extrinsic" in fix:
        new["camera_extrinsic"] = np.asarray(fix["camera_extrinsic"]).tolist()
    if "tcp_offset" in fix:
        new["tcp_offset"] = np.asarray(fix["tcp_offset"]).round(6).tolist()
    if "joint_zero_offsets_deg" in fix:
        # the reading was off by +b, so add b to the offset Kintrace subtracts
        z = np.array(new["joint_zero_offsets_deg"], float) + np.array(fix["joint_zero_offsets_deg"], float)
        new["joint_zero_offsets_deg"] = z.round(4).tolist()
    return new


def run_check(cfg: dict, parts: Parts, fix: bool = True, add_latency_s: float = 0.0,
              touch: bool = True, log_path: str | None = None, echo=print) -> dict:
    """The whole flow. Returns a record dict; the new cfg is in record['cfg_after']."""
    echo("check motion")
    log, stats = capture.capture(cfg, parts.arm, parts.camera, parts.clock,
                                 add_latency_s=add_latency_s, truth=parts.fault, echo=echo)
    _require(stats, echo)
    if log_path:
        log.save(log_path)
    before = analyze(log, cfg)
    after = None
    cfg_after = cfg
    if before["fix"] and fix:
        cfg_after = apply_fix(cfg, before["fix"])
        _set_zero_offsets(parts, cfg_after)
        # the fixed values are now the ones to verify against
        cfg_v = copy.deepcopy(cfg_after)
        cfg_v["known_good"] = dict(cfg["known_good"], tcp_offset=cfg_after["tcp_offset"],
                                   camera_extrinsic=cfg_after["camera_extrinsic"])
        echo("fix applied, re-running the check motion")
        log2, stats2 = capture.capture(cfg_after, parts.arm, parts.camera, parts.clock,
                                       add_latency_s=add_latency_s, truth=parts.fault, echo=echo)
        _require(stats2, echo)
        after = analyze(log2, cfg_v)
    touch_r = None
    if touch:
        echo("touch test")
        touch_r = touch_test(cfg_after, parts, echo=echo)

    verified = after if after is not None else before
    blockers = list(before["blockers"]) + ([] if after is None else after["blockers"])
    all_ok = all(c.agrees for c in verified["checks"])
    if after is not None and not all_ok:
        blockers.append("a check still fails after the fix")
    if before["fix"] and not fix:
        blockers.append("fix found but not applied (--no-fix)")
    if touch_r is not None and not touch_r["ok"]:
        blockers.append("touch test missed")
    status = "GO" if not blockers and all_ok else "NO-GO"
    return dict(when=datetime.now().isoformat(timespec="seconds"), before=before, after=after,
                touch=touch_r, status=status, blockers=blockers, cfg_after=cfg_after,
                fault=parts.fault, stats=stats)


def text_record(r: dict, name: str = "SO-101 desk rig") -> str:
    out = [f"KINTRACE  check record, {name}", "=" * 64, f"Run at       : {r['when']}", ""]
    out.append("What changed (check motion)")
    b = r["before"]
    if not b["findings"]:
        out.append("  nothing: rig matches its known-good state")
    for f in b["findings"]:
        out.append(f"  - {f.label}: {f.size}")
    if r["after"] is not None:
        fx = b["fix"]
        out += ["", "Fix written to rig.json"]
        if "camera_extrinsic" in fx:
            out.append("  - camera calibration updated")
        if "tcp_offset" in fx:
            out.append("  - tool offset set to [" + ", ".join(f"{v*1000:.1f}" for v in fx["tcp_offset"]) + "] mm")
        if "joint_zero_offsets_deg" in fx:
            for j, v in enumerate(fx["joint_zero_offsets_deg"]):
                if v:
                    out.append(f"  - J{j+1} ({rig.ARM.joint_names[j]}) zero offset {v:+.2f} deg")
    shown = r["after"] if r["after"] is not None else b
    out += ["", "Verification (check motion re-run)" if r["after"] is not None else "Checks"]
    for c in shown["checks"]:
        out.append(f"  [{'ok  ' if c.agrees else 'FAIL'}] {c.name:<28} {c.detail}")
    t = r["touch"]
    if t is not None:
        if t["miss_mm"] is None:
            out.append("  [FAIL] Touch test                   tip or target not seen")
        else:
            out.append(f"  [{'ok  ' if t['ok'] else 'FAIL'}] Touch test                   "
                       f"pointer {t['miss_mm']:.1f} mm from the target spot")
    out += ["", f"STATUS: {r['status']}" + ("" if not r["blockers"] else "  (" + "; ".join(r["blockers"]) + ")")]
    return "\n".join(out)


def record_json(r: dict) -> str:
    def part(a):
        if a is None:
            return None
        return dict(checks=[asdict(c) for c in a["checks"]], findings=[asdict(f) for f in a["findings"]],
                    fix=a["fix"], blockers=a["blockers"], tau_ms=a["tau"] * 1000,
                    tip_measured_mm=(np.array(a["tip_measured"]) * 1000).round(2).tolist())

    d = dict(when=r["when"], status=r["status"], blockers=r["blockers"], before=part(r["before"]),
             after=part(r["after"]), touch=r["touch"], capture=r["stats"], dry_run_fault=r["fault"])
    return json.dumps(d, indent=2, default=float)
