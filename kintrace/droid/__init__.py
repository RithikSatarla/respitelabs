"""DROID adapter: real Franka arm data for the check.

DROID (droid-dataset.github.io) is 76k teleop episodes on Franka Panda arms
with two fixed ZED stereo cameras and a wrist camera. Each episode stores
the camera-to-base extrinsics the cell was *configured* with. In 2025 the
DROID team re-solved those extrinsics for ~36k episodes and published the
corrected values (huggingface.co/KarlP/droid), because the stored
calibration could not be trusted for those episodes.

That gives us the thing the check layer is built to catch, on real robots,
with ground truth:

  belief  = extrinsic the cell was running with (episode metadata)
  truth   = corrected extrinsic (KarlP/droid cam2base_extrinsics.json)
  label   = how far the camera really was from where the stack thought

Two levels of result, both on real data:

  labels   no vision needed. How often, and by how much, were DROID cameras
           off from their configured pose. Run: python -m kintrace.droid labels
  detect   feed joint positions + fixed-camera frames through diagnose() with
           the configured extrinsic as the belief, and score the camera-moved
           call against the label. Needs a point on the gripper located in
           the camera frame per frame. DROID has no wrist fiducial, so
           convert.detect_gripper() is the one function to fill in (a Franka
           hand keypoint model, or a mask + stereo depth from the ZED pair).
           Until it is, --synthetic runs the same pipeline with detections
           made from the *corrected* extrinsic plus noise, which tests
           everything except the detector itself.

Modules
  download    pull the extrinsics JSONs and N raw episodes (needs internet)
  extrinsics  parse configured and corrected camera poses, compute labels
  episode     read one raw episode (joints, timestamps, serials, frames)
  convert     build a kintrace Log from an episode
  bench       the two result levels, and the CLI (python -m kintrace.droid)
"""
