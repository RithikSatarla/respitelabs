"""Run the check motion on the rig and write a Kintrace log.

The arm visits a few poses where the wrist plate and the pointer tip face
the camera. At each pose it stops for dwell_s; the camera only keeps frames
taken after the arm has been still for settle_s, so camera timing and motion
blur don't matter. The joint stream (commanded and measured) is recorded the
whole time, which is what the timing check needs.

The output is a normal kintrace Log (.npz):
  q_cmd, q_meas, t_joint       the whole motion, 5 joints, radians
  t_cam, markers_cam, visible  frames where the wrist tag was seen (4 corners)
  fixed_cam                    the 4 table tags, same frames
  pick_t, pick_ok              empty (no picks on the desk rig)
  probe_*                      the frames where the tip flag was seen too, in
                               the shape the recommission check expects
  config                       the rig settings in use (tool offset, camera
                               calibration, joint zero offsets, tag layout)
"""
from __future__ import annotations

import collections
import os
import threading
import time
from datetime import datetime

import numpy as np

from ..logio import Log
from ..sim import _build_trajectory
from . import rig
from .markers import Intrinsics, make_detector, observe


class RealClock:
    def __init__(self):
        self.t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self.t0

    def wait_until(self, t: float) -> float:
        d = t - self.now()
        if d > 0:
            time.sleep(d)
        return self.now()


class WebCamera:
    """USB webcam read in a background thread. Tags are detected only while
    the capture loop says the arm is still, to keep the CPU free for the
    control loop."""

    def __init__(self, cfg: dict, clock: RealClock, save_dir: str | None = None):
        import cv2

        c = cfg["camera"]
        self.cfg, self.clock, self.save_dir = cfg, clock, save_dir
        path = c.get("intrinsics", "camera.json")
        if not os.path.exists(path):
            raise RuntimeError(f"camera intrinsics file '{path}' not found. Run `kintrace rig calibrate-camera` first.")
        self.intr = Intrinsics.load(path)
        self.cap = cv2.VideoCapture(c.get("index", 0))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        self.cap.set(cv2.CAP_PROP_FPS, c.get("fps", 30))
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # focus must stay where it was calibrated
        if c.get("focus") is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, c["focus"])
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {c.get('index', 0)}")
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (self.intr.width, self.intr.height):
            raise RuntimeError(f"camera gives {w}x{h} but was calibrated at {self.intr.width}x{self.intr.height}")
        self.latency = float(c.get("frame_latency_s", 0.0))
        self.still = False
        self.frames = []
        self._run = False
        self._thread = None
        self._last = None
        self.t0 = 0.0

    def _loop(self):
        import cv2

        det = make_detector()
        n = 0
        while self._run:
            ok, img = self.cap.read()
            t = self.clock.now() - self.latency - self.t0  # same time base as the joint stream
            if not ok:
                continue
            self._last = (t, img)
            if not self.still:
                continue
            obs = observe(img, self.intr, self.cfg["tags"], det)
            if self.still:  # still after the detection too
                self.frames.append((t, obs))
                if self.save_dir:
                    n += 1
                    cv2.imwrite(os.path.join(self.save_dir, f"frame_{n:04d}.jpg"), img)

    def start(self, t0: float = 0.0):
        self.frames = []
        self.t0 = t0
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def poll(self, t: float, still: bool):
        self.still = still

    def grab(self):
        t, img = self._last if self._last else (None, None)
        while img is None:
            time.sleep(0.02)
            t, img = self._last if self._last else (None, None)
        return t, observe(img, self.intr, self.cfg["tags"])

    def stop(self):
        self.still = False
        self._run = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.frames

    def close(self):
        self.cap.release()


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------
def order_poses(q_start, poses):
    """Visit the poses nearest-first (largest single joint move), so the arm
    doesn't swing its wrist back and forth between pose families."""
    left = [np.asarray(p, float) for p in poses]
    out, q = [], np.asarray(q_start, float)
    while left:
        i = int(np.argmin([np.abs(p - q).max() for p in left]))
        q = left.pop(i)
        out.append(q)
    return out


def plan_motion(q_start, poses, cfg: dict):
    """Joint trajectory through the check poses (nearest first) and back to
    the start. Returns t, q_cmd and a bool 'still' per sample (arm settled at
    a pose)."""
    speed = np.deg2rad(cfg["speed_deg_s"])
    dwell, settle = cfg["dwell_s"], cfg["settle_s"]
    dt = 1.0 / cfg["control_hz"]
    via = rig.ARM.home_q.copy()
    q_start = np.asarray(q_start, float)
    seq = [(q_start, False)]
    for p, is_pose in [(p, True) for p in order_poses(q_start, poses)] + [(q_start, False)]:
        if rig.path_min_height(seq[-1][0], p, cfg) < cfg.get("min_tip_height_m", 0.04):
            seq.append((via, False))  # go through a pose held high, clear of the table
        seq.append((p, is_pose))
    waypoints = [(q_start, 0.0, 0.2)]
    for (a, _), (b, is_pose) in zip(seq[:-1], seq[1:]):
        move = max(1.0, 1.6 * np.abs(b - a).max() / speed)  # min-jerk peak speed is ~1.9x average
        waypoints.append((b, move, dwell if is_pose else 0.1))
    t, q = _build_trajectory(waypoints, dt)
    moving = np.r_[True, np.abs(np.diff(q, axis=0)).max(1) > 1e-9]
    still = np.zeros(len(t), bool)
    last_move = -1e9
    for i in range(len(t)):
        if moving[i]:
            last_move = t[i]
        still[i] = t[i] - last_move >= settle
    return t, q, still


def run_motion(arm, camera, clock, t_plan, q_plan, still, add_latency_s: float = 0.0,
               echo=print):
    """Stream the plan to the arm, record commanded/measured joints and let
    the camera collect frames at the still points.

    add_latency_s holds every command back before it's sent: the "sleep in
    the control loop" fault, done without slowing the loop itself.
    """
    queue = collections.deque()
    t_j, q_c, q_m = [], [], []
    t0 = clock.now()
    camera.start(t0)
    try:
        for k in range(len(t_plan)):
            t = clock.wait_until(t0 + t_plan[k])
            queue.append((t, q_plan[k]))
            while queue and t - queue[0][0] >= add_latency_s - 1e-9:
                arm.write(queue.popleft()[1])
            _, qm = arm.read()
            t_j.append(t - t0)
            q_c.append(q_plan[k])
            q_m.append(qm)
            camera.poll(t - t0, bool(still[k]))
    except KeyboardInterrupt:
        echo("\nstopped: holding the current position")
        _, qm = arm.read()
        arm.write(qm)
        raise
    finally:
        frames = camera.stop()
    return np.array(t_j), np.array(q_c), np.array(q_m), frames


# --------------------------------------------------------------------------
# frames -> log
# --------------------------------------------------------------------------
def build_log(cfg: dict, t_joint, q_cmd, q_meas, frames, truth=None) -> tuple[Log, dict]:
    tags = cfg["tags"]
    table_ids = list(tags["table_ids"])
    n_frames = len(frames)
    wrist = [(t, o) for t, o in frames if o.wrist is not None]
    # Table tags don't move during a capture, so every frame gets each tag's
    # median over the whole capture. That also fills a tag the arm hid, and
    # ignores a frame where the arm half covered a tag and bent its corners.
    fixed_med = {}
    for tid in table_ids:
        seen = [o.table[tid] for _, o in frames if tid in o.table]
        if len(seen) >= 3:
            fixed_med[tid] = np.median(seen, axis=0)
    have_table = len(fixed_med) == len(table_ids)

    def fixed_of(o):
        return np.array([fixed_med[tid] for tid in table_ids])

    t_cam = np.array([t for t, _ in wrist])
    markers_cam = np.array([o.wrist for _, o in wrist]).reshape(-1, 4, 3)
    fixed_cam = np.array([fixed_of(o) for _, o in wrist]).reshape(-1, len(table_ids), 3) if have_table else None
    both = [(t, o) for t, o in wrist if o.tip is not None]
    probe = dict(
        t_joint=t_joint, q_cmd=q_cmd, q_meas=q_meas,
        t_cam=np.array([t for t, _ in both]),
        markers_cam=np.array([o.wrist for _, o in both]).reshape(-1, 4, 3),
        tip_cam=np.array([o.tip for _, o in both]).reshape(-1, 3),
    )
    if have_table:
        probe["fixed_cam"] = np.array([fixed_of(o) for _, o in both]).reshape(-1, len(table_ids), 3)
    conf = rig.controller_config(cfg)
    conf["start_time"] = datetime.now().isoformat(timespec="seconds")
    log = Log(
        t_joint=t_joint, q_cmd=q_cmd, q_meas=q_meas,
        t_cam=t_cam, markers_cam=markers_cam, visible=np.ones(len(t_cam), bool),
        pick_t=np.zeros(0), pick_ok=np.zeros(0, bool),
        config=conf, probe=probe, truth=truth or {}, fixed_cam=fixed_cam,
    )
    stats = dict(frames=n_frames, wrist=len(wrist), tip=len(both), table_tags=len(fixed_med),
                 table_needed=len(table_ids))
    return log, stats


def check_stats(stats: dict) -> list[str]:
    """Problems with a capture that would make the diagnosis meaningless."""
    out = []
    if stats["wrist"] < 20:
        out.append(f"wrist tag seen in only {stats['wrist']} still frames (want 20+)")
    if stats["tip"] < 20:
        out.append(f"tip flag seen in only {stats['tip']} still frames (want 20+)")
    if stats["table_tags"] < stats["table_needed"]:
        out.append(f"only {stats['table_tags']} of {stats['table_needed']} table tags seen")
    return out


def capture(cfg: dict, arm, camera, clock, poses=None, add_latency_s=0.0, truth=None, echo=print):
    """Run the check motion once. Returns (Log, stats)."""
    poses = rig.check_poses(cfg) if poses is None else poses
    _, q0 = arm.read()
    t, q, still = plan_motion(q0, poses, cfg)
    echo(f"check motion: {len(poses)} poses, {t[-1]:.0f} s")
    tj, qc, qm, frames = run_motion(arm, camera, clock, t, q, still, add_latency_s, echo)
    return build_log(cfg, tj, qc, qm, frames, truth)
