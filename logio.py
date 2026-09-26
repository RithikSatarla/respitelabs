"""Log format.

A Kintrace log is a single .npz file with these arrays. Anything that can
export these fields (a ROS 2 bag, an MCAP file, a controller dump) can be
diagnosed. The ROS 2 / MCAP adapter is the next piece to build.

  t_joint      (N,)      seconds, joint stream timestamps
  q_cmd        (N, 6)    commanded joint angles (rad)
  q_meas       (N, 6)    measured joint angles from the encoders (rad)
  t_cam        (M,)      seconds, camera frame timestamps
  markers_cam  (M, K, 3) fiducial marker points on the wrist, in the camera frame (m)
  visible      (M,)      bool, marker detected in that frame
  fixed_cam    (M, F, 3) optional: fixed markers on the table, in the camera frame
  pick_t       (P,)      time of each grasp
  pick_ok      (P,)      bool, grasp succeeded
  config       json str  controller + cell settings the robot is running with
                         (tcp offset, camera extrinsic, marker layout, conveyor speed)
  probe_*      optional  same fields, recorded during a probe motion where
                         the fingertip is visible (tip_cam: (Q, 3))
  truth        json str  simulation only, ground truth. Kintrace never reads it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Log:
    t_joint: np.ndarray
    q_cmd: np.ndarray
    q_meas: np.ndarray
    t_cam: np.ndarray
    markers_cam: np.ndarray
    visible: np.ndarray
    pick_t: np.ndarray
    pick_ok: np.ndarray
    config: dict
    probe: dict | None = None
    truth: dict = field(default_factory=dict)
    fixed_cam: np.ndarray | None = None

    def save(self, path: str) -> None:
        arrays = dict(
            t_joint=self.t_joint,
            q_cmd=self.q_cmd,
            q_meas=self.q_meas,
            t_cam=self.t_cam,
            markers_cam=self.markers_cam,
            visible=self.visible,
            pick_t=self.pick_t,
            pick_ok=self.pick_ok,
            config=np.array(json.dumps(self.config)),
            truth=np.array(json.dumps(self.truth)),
        )
        if self.fixed_cam is not None:
            arrays["fixed_cam"] = self.fixed_cam
        if self.probe is not None:
            for k, v in self.probe.items():
                arrays[f"probe_{k}"] = v
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path: str) -> "Log":
        z = np.load(path, allow_pickle=False)
        probe = {k[6:]: z[k] for k in z.files if k.startswith("probe_")} or None
        return Log(
            t_joint=z["t_joint"],
            q_cmd=z["q_cmd"],
            q_meas=z["q_meas"],
            t_cam=z["t_cam"],
            markers_cam=z["markers_cam"],
            visible=z["visible"].astype(bool),
            pick_t=z["pick_t"],
            pick_ok=z["pick_ok"].astype(bool),
            config=json.loads(str(z["config"])),
            probe=probe,
            truth=json.loads(str(z["truth"])) if "truth" in z.files else {},
            fixed_cam=z["fixed_cam"] if "fixed_cam" in z.files else None,
        )
