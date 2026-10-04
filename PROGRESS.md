# Progress

## Gripper detector on real DROID frames (Steps 1 to 7, done 2026-10-03 14:37)

**81 px → 19.9 px median on held-out cameras.**

The 81 px is the Phase 3 small CNN on its own held-out set (16 cameras). The 19.9 px is
the ResNet-18 at epoch 50 on the Phase 4 held-out set (120 cameras). Both are cameras the
model never saw in training, but they are not the same cameras.

This is DROID, a real Franka dataset. The split is by camera serial: 8 serials held out.
The 317 GT episodes come from 26 physical cameras, not 317. Raw output: `RESULTS_DROID.txt`.

**It missed the 12 px target.** From a by-eye sample of 24 big misses (over 100 px), one
per camera:

- real misses, gripper in plain view and the detector elsewhere: 14 of 24. Several land
  a link or two up the arm.
- label problems: 7 of 24. The FK label says "in view" but the gripper is hidden at the
  frame edge or behind the arm (5), or the gripper is visible somewhere else (2).
- unclear: 3 of 24.

A visibility check on the labels would remove about a third of the big misses in this
sample. The rest needs a better detector.

| Held-out DROID GT cameras, full res | Epoch 45 | Epoch 50 (final) |
|---|---|---|
| Median | 19.7 px | 19.9 px |
| Mean | 67.5 px | 70.4 px |
| p90 / p95 | 219.4 / 324.6 px | 232.4 / 347.4 px |
| Within 5 / 12 / 25 px | 8.5 / 30.9 / 59.4% | 8.3 / 30.2 / 58.9% |

30,949 frames scored, 1,098 skipped (fingertip out of view). Per serial at epoch 50:
7.3 px (21582473) to 40.1 px (29838012). Training stopped at epoch 50 of 60 on heat; loss
was flat (3.679 to 3.676), so epoch 50 is the model: `data/droid/detector_resnet18_pretrained_frozen.pt`.

- Step 3: `detect()` matches the evaluation on raw full-res frames (test passes).
  `convert.detect_gripper()` calls it.
- Step 4: kept if visibility >= 0.5 and heatmap peak >= 0.0657: 55.7% of in-view frames,
  kept median 13.9 px, p90 76.6 px. No threshold reaches 12 px. The sharpest peaks are not
  the most accurate (top 9% by peak: 34.0 px). 250 of 1,098 out-of-view frames still pass.
- Step 5 (injected calibration error on held-out GT cameras, per camera, median residual
  over confident frames): healthy median 15.4 px; 5 cm gives 40.3 px, 5 deg 46.7 px,
  2 cm 20.7 px. At 1 false alarm in 100, nothing is caught: the alarm level is 330.8 px
  because 8 of 120 healthy cameras already sit above 100 px. Per-frame tail, healthy,
  confident frames: median 14.1 px, p90 89.1, p99 580.4; 9.5% over 100 px.
- Step 6: figures in `data/droid/figures/`.

## Heat so far today

- Training stopped on heat twice: 86 C at 13:29:59 (epoch 46) and 86 C at 14:09:07
  (epoch 51). Both spikes came within 30 s of restarting after a pause. Before 13:20,
  training held 65 to 70 C for over an hour.
- The epoch-50 evaluation at a gentle load (50 ms rest per batch of 16): 57 to 66 C.
- Logs: `data/droid/runs/phase4_20261003_104703.log`, `data/droid/runs/resume45b.log`.

## Big session (started 2026-10-03 14:37)

Status every 30 minutes below.

### 14:45 status
- Steps 3 to 7 done (above). Starting Phase 0.
- Baseline before code changes: `pytest tests` 42 passed, 5 skipped. `kintrace incident` runs.
- Blocker found: `python -m kintrace bench -n 4` crashes with "IK did not converge" in the
  UR5e simulator. It crashes the same way on 8ec04f5 (before this week's work) and on
  a2a3f25 (GitHub now), so it predates these changes. Not fixed: a fix would change sim
  outputs. For the identical-output check I use the crashing command (must still crash the
  same way) plus `bench -n 3 --seed 7`, which completes.

### 14:50 status: Phase 0 done
- Baseline (epoch 50, test, 120 cameras): median 19.9 px, p90 232.4, within 5/12/25 px
  8.3/30.2/58.9%, per serial 7.3 to 40.1 px. (Unchanged from above.)
- Validation split: 3 serials (25455306, 25916956, 28451778), 25 cameras, 10,146 frames,
  picked at random (seed 1) among non-test serials with at most 20 cameras. Train: 15
  serials, 172 cameras. Test: the same 8 serials, 120 cameras. `data/droid/split.json`.
- Epoch 50 on validation: 2.1 px median. Not a fair reference: epoch 50 trained on these
  serials. The fair reference is E0 (same recipe, trained without the validation serials).
- Label visibility rule (written before use, in `kintrace/droid/experiments.py`): FK tip in
  the image, at least 30 px from the border, and not behind the arm (no arm point within
  50 px in the image and 5 cm nearer the camera). It drops 2,015 of 88,875 in-view frames
  (1,306 border, 709 behind the arm). Training data only; test gets it as a second number.
- What the epoch-50 trainer does:
  - input 320x180 (frames shrunk with INTER_AREA), BGR to RGB, ImageNet normalisation
  - ResNet-18 stem and layer1 to layer3 (layer4 not used), FPN-style head back to stride 4:
    80x45 heatmap, one cell = 4 px at 320x180 = 16 px at full res
  - target: Gaussian heatmap, sigma 1.5 cells; loss = heatmap cross-entropy + 0.05 x L1 on
    the soft-argmax point + BCE on a visibility head (trained on "FK tip in image")
  - augmentation: zoom-crop 85 to 100%, shift +-12% horizontal and +-5% vertical (label
    moves with it), brightness x0.7 to 1.3 and offset. No colour jitter, blur or erasing.
  - frozen: stem and layer1 (BatchNorm stats too). One learning rate for everything,
    AdamW 2e-3, one-cycle (10% warm-up, cosine), batch 16, float16.
  - Note: the best held-out frames are at 0.1 px, so output resolution is not what limits
    the median; the misses are.
- Ladder (validation only, every 3rd training frame, 6 epochs each, gentle GPU): E0 the
  epoch-50 recipe, E1 stride-2 head with sigma 2 cells, E2 stronger augmentation (colour
  gain, blur, random erasing), E3 unfreeze everything with backbone LR x0.1, E4 the label
  rule (drops hidden-tip frames from the heatmap loss and trains the visibility head on the
  rule). Rule to keep a change: val median down 3% or more, or median within 3% and p90
  down 10% or more.

### 15:00 status
- Phase 1 ladder running (gentle GPU). E0 (epoch-50 recipe, no validation serials):
  val median 19.4 px, p90 188.2, within 12 px 28.5%. E1 (stride-2 head) training.
  GPU so far 55 to 68 C, no heat events.
- More distinct cameras: DROID's GT set has only 31 camera serials; we had 26 plus the
  excluded one, so only 4 new physical cameras exist. Added 18 episodes from them
  (23748752, 24013089, 26638268, 29513368), 14 labelled cameras, 4,778 frames. Train only,
  used in the final run. Labelled cameras: 331 from 30 physical cameras.
- Phase 3 started on the CPU while the GPU trains: `kintrace/urdf.py` (standard library
  XML). Panda URDF (bullet3) vs arms.panda_fk over 1,000 random poses: max 6.9e-9 mm.
  UR5e URDF (ros-industrial, flattened) base_link_inertia to tool0 vs the DH model: max
  2.1e-7 mm. No fitting either way. `--urdf` on import and check; a config "urdf" key
  selects the body everywhere. tests/test_urdf.py passes.

### 15:17 status
- Ladder (validation only; E0 is the epoch-50 recipe without the validation serials):
  - E0 baseline: median 19.4 px, p90 188.2, within 12 px 28.5%. Kept as the reference.
  - E1 stride-2 head, sigma 2 cells: median 20.4, p90 232.4. Not kept (resolution is not the limit).
  - E2 stronger augmentation (colour gain, blur, random erasing): median 19.9, p90 133.9
    (down 29%), within 12 px 25.7%. Kept.
  - E3 unfreeze everything (backbone LR x0.1) training now; E4 label rule after.
- GPU 55 to 69 C in gentle mode, rest stayed at 50 ms per batch, no heat events.
- Phase 3, CPU, alongside:
  - Cross-robot timing and joint-offset check on real public LeRobot data, 10 episodes each,
    reference = the robot's other episodes. Healthy alarms / injected caught:
    ALOHA 0/10, latency 30/30, offset 20/20. Koch 0/10 latency, offset 2/10 healthy alarms,
    latency 30/30, offset 19/20. SO-100 0/10, 30/30, 20/20. Unitree G1 arms 0/10, 30/30,
    20/20. LeKiwi arm 0/10, 30/30, 20/20. LeKiwi wheels 3/10 healthy alarms, latency 23/30.
  - URDFs load: Unitree G1 (pelvis to wrist, 10 joints), SO-100, SO-101 (matches arms.py to 2e-13 mm).
  - `kintrace preflight` works on LeRobot: SO-100 healthy GO (exit 0), +20 ms injected NO-GO
    (exit 1, measured +22.5 ms), no reference: exit 2.
  - Fixed: the LeRobot reader failed on every v3 dataset (path template keys, array metadata).
    `kintrace import` + `check` now run on SO-100 v3 with its URDF: no fault on healthy data.

### 15:45 status
- Ladder finished (validation, 6 epochs on every 3rd frame):
  E0 19.4 / p90 188.2 (reference), E1 stride-2 20.4 / 232.4 (not kept), E2 stronger
  augmentation 19.9 / 133.9 (kept), E3 + unfreeze 19.1 / 172.4 (kept, best), E4 + label
  rule 21.5 / 126.5 (not kept). Every run lands between 19 and 22 px on new cameras;
  augmentation and the label rule shrink the tail, nothing moves the median much.
- Finalists: E3 (ladder winner) and E2 (lowest p90 of the kept runs, frozen early layers).
  Training now on all 19 training serials (incl. the 4 new cameras), every 2nd frame,
  12 epochs, gentle GPU, 64 to 65 C. E3 done about 16:15, E2 about 16:45.
- No heat events this session.

### 16:12 status
- Finalist E3 (all 19 training serials, every 2nd frame, 12 epochs): validation median
  19.3 px, p90 164.4, within 5/12/25 px 4.9/26.4/64.5%. Visible-only frames: 19.0 px.
  More data and epochs did not move the median (ladder run: 19.1 px).
- Validation error split (visible-only frames, 25 cameras): each camera has a constant
  offset between detections and FK labels, median 11.8 px (2.3 to 239.5); the scatter
  around that offset is 13.4 px. So about half the median error is a fixed per-camera
  shift, which is what a slightly wrong calibration or tip point gives, not detector noise.
- Finalist E2 training (done about 16:40). GPU up to 69 C, no heat events.

### 16:25 status: where the per-camera offset comes from (validation only, CPU)
1. Intrinsics are right: K comes per episode and camera from KarlP intrinsics.json (24
   distinct fx over 25 cameras, 522.8 to 545.4), calibrated at 1280x720 and the MP4s are
   1280x720 (scale 1), order fx, cx, fy, cy checked (cx about 640, cy about 360). No fix needed.
2. Not a keypoint-definition offset: in the gripper's frame the per-camera offsets point every
   way (direction consistency 0.21, 1 = all the same way), median (2.3, -1.1, -3.1) mm. One
   shared tip correction changes nothing (18.9 -> 18.8 px; fitted on half the cameras, scored
   on the other half: 24.7 -> 24.8 px). The same physical camera has different offsets in
   different episodes (25455306: 2 mm in one, 45 mm in another).
3. Not shown to be calibration error: a 6-DoF extrinsic refit on the first half of each
   camera's frames makes the second half worse (median 16.7 -> 22.6 px, offset 10.8 ->
   14.7 px). The refits are large (median 95 mm, 8.9 deg): from detections this noisy over one
   episode's motion, a camera pose is poorly constrained. So the source of the offset is not
   determined. It also means the fitted-pose part of the 2D check will be unreliable.

### 16:45 status: Phase 1 done
- More offset checks (validation, CPU):
  - Timing: per episode, the video-to-joints shift (within +-200 ms) that fits best has a
    median of +30 ms (p10 -19, p90 +95; one at the -200 ms limit). Shifting barely helps:
    median 17.4 -> 17.0 px, constant offset 11.8 -> 11.8 px. A small timing mismatch exists
    in DROID's video, but it is not the offset.
  - Gripper pose: weak. Within episodes, error vs opening Spearman 0.18, vs tool angle to
    the camera 0.05; across episodes 0.07 and -0.25. Median error 18.8 px open, 15.0 px closed.
    Not enough to justify more keypoints as the fix for the offset.
- Finalists, validation: E2 18.3 px (p90 173.3, within 12 px 30.7%), E3 19.3 px (p90 164.4).
  Final model = E2, chosen on validation. `data/droid/detector_final.pt` and `.json`.
- Test, scored once per finalist (8 held-out serials, 120 cameras, original labels):
  - E2 (final): median 21.8 px, p90 220.4, within 5/12/25 px 6.8/27.2/55.4%; visible-only 21.1 px.
  - E3: median 21.5 px, p90 194.8, within 5/12/25 px 5.8/25.4/55.8%; visible-only 21.0 px.
  - Reference: epoch 50 is 19.9 px (p90 232.4); Phase 3 small CNN 81.0 px (other held-out set).
  - Phase 1 did not beat epoch 50 on the test median. It cut the tail (p90 220 and 195 vs 232).
    Epoch 50 also trained on the 3 validation serials, all frames, 50 epochs.
- Heat: one reading at 77 C during the E2 finalist (rest stays doubled only above 78 C). No stops.
- Phase 2 caching the final detector's outputs now (validation, test, Pred cameras).

## Final summary (17:05)

**Gripper detector: 81 px -> 19.9 px median on held-out real cameras.** (DROID, split by
camera serial; the 81 px is the Phase 3 CNN on a different held-out set.) The 12 px target
was missed. Today's tuning did not beat 19.9 px on test.

What improved
- Tail of the detector error: test p90 232 px (epoch 50) -> 220 px (final) and 195 px (E3).
- The drift check against a camera's own baseline works in the typical case: a healthy
  camera's residual moves -0.8 px (the fixed offset cancels); an injected 5 cm move adds
  +22 px, 5 deg +30 px, 2 cm +5 px.
- Other robots: timing check on public LeRobot data for ALOHA, Koch, SO-100, Unitree G1 arms
  and the LeKiwi arm: 0 healthy alarms, injected latency 30/30 each. Joint-offset check (new):
  20/20 on four sets, Koch 19/20 with 2/10 healthy alarms.
- URDF loader: Panda, UR5e and SO-101 FK match the hand-written models to < 1e-6 mm.
- `kintrace preflight` with exit codes 0 / 1 / 2; works on LeRobot (SO-100: GO, +20 ms NO-GO).
- Fixed: LeRobot v3 datasets did not load at all; `python -m kintrace` dropped exit codes;
  test URDFs were under an ignored `data/` folder.

What did not work
- Detector median: every configuration lands at 19 to 22 px on new cameras. Finer output,
  stronger augmentation, unfreezing and a hidden-gripper label rule moved the tail, not the median.
- Half the median error is a fixed per-camera/episode offset (11.8 px). Not the intrinsics,
  not the keypoint, not timing; a one-keypoint camera refit is too weak to show calibration error.
- Drift at 1 false alarm in 100: not caught at any size (0.5 to 5 cm, 0.5 to 5 deg). A few
  cameras where the detector fails set the alarm at about 200 px.
- Real drift (Pred cameras): 6/45 flagged, 0/63 false alarms on healthy test cameras. The
  translation-only fix is hundreds of mm from DROID's re-solved pose: weakly constrained.
- Preflight on a DROID camera: GO healthy (5.7 px) but also GO with 2 cm (14.1 px) and 5 cm
  (37.0 px) injected, because the alarm is 204.8 px.
- Encoder drift on DROID (2D): 0/63 flagged at 10 and 20 mm.
- LeKiwi wheels: 3/10 healthy alarms, 23/30 injected latency.

Heat events this session (gentle mode, 50 ms rest per batch, batch 16)
- No heat stops. No reading above 77 C (one reading, during the E2 finalist). Most runs 55
  to 69 C. GPU jobs this session: ladder (5 runs), 2 finalists, 2 test scorings, Phase 2 cache,
  epoch-50 validation scoring, the consistency test.

Tests, bench, incident (before vs after)
- `pytest tests`: 42 passed, 5 skipped -> 46 passed, 5 skipped (new: URDF x3, LeRobot v3).
- `kintrace bench -n 4`: crashes before and after with "IK did not converge" (pre-existing,
  also on 8ec04f5 and a2a3f25); only traceback line numbers changed. Flagged as a separate task.
- `kintrace bench -n 3 --seed 7`: bench.json identical before and after (ignoring run time).
- `kintrace incident`: all four records and the console output identical.

Files changed (not committed)
- New: kintrace/urdf.py, kintrace/preflight.py, kintrace/crossrobot.py,
  kintrace/droid/{experiments,phase2,offset_study,check2d,eval_detector,study}.py,
  tests/test_urdf.py, tests/test_detector_droid.py, tests/urdf/ (Panda and UR5e URDFs, sources),
  incidents/droid_IPRL+7790ec0a+2023-06-30-17h-22m-36s_21582473/ (signed record).
- Changed: README.md, RESULTS_DROID.txt, kintrace/{__main__,arms,cli}.py,
  kintrace/readers/lerobot.py, kintrace/droid/{__main__,bench,convert,detector,labels_fk,train_detector_cmd}.py,
  tests/test_readers.py.
- Local only (ignored): data/ (models, caches, figures, runs), kintrace_keys/ (the signing key).

What I would do next
1. Fix the tail at the source: find why the detector fails for whole stretches of some
   episodes (the cameras that set the 200 px alarm). A per-camera alarm level learned from
   that camera's own healthy history would already separate 2 cm on cameras like 20252535.
2. Labels: the fixed offset changes per episode on the same camera. Check DROID's per-episode
   calibration quality metric against it, and try two-fingertip plus flange keypoints.
3. A 3D point per frame (stereo from the SVO files) so the full diagnose() path can run.
4. Fix the bench -n 4 IK crash (separate task offered).

## Drift alarm fix (started 2026-10-03 17:10). Rules written before any new test result

Data: the cached final-detector outputs (every frame of every episode; no new detections).
A frame is used if the visibility head says >= 0.5 ("confident"). Residual per frame = pixel
distance between the detection and the FK fingertip projected through the believed pose.

1. Reliability gate (per camera, from its healthy baseline only, no drift labels).
   Reliable if the baseline's median residual <= X px and at least Y of its frames are
   confident. X and Y are picked on validation from X in {15, 20, 30, 40, 60} and Y in
   {0.3, 0.5, 0.7}: the pair that keeps the most validation cameras while the 99th
   percentile of healthy |change in median residual| between the two halves of an episode,
   over the cameras kept, is at most 20 px. Ties: the stricter pair. Unreliable cameras
   answer "can't tell (detector unreliable on this view)", exit code 2, and are left out of
   every threshold.
2. Per-camera alarm. Baseline frames give the camera's own spread: sigma = 1.4826 x MAD of
   its residuals. With n frames, the standard error of a median is 1.2533 x sigma /
   sqrt(n / 5) (blocks of 5 frames, because neighbouring frames are correlated). The
   decision statistic is z = (median residual now - median residual at baseline) /
   sqrt(SE_baseline^2 + SE_now^2), both SEs from the baseline sigma. Alarm if z > k.
   k is the 99th percentile of z over healthy validation decisions (1 false alarm in 100),
   picked separately for within-session and between-session decisions and for each decision
   window n in {5, 10, 20, 50, 100, all}.
   - Healthy validation decisions, within a session: baseline = the first part of an
     episode, test = the rest, split at 1/3, 1/2 and 2/3 and in both directions; for small n,
     every non-overlapping window of n frames after the baseline.
   - Healthy validation decisions, between sessions: every ordered pair of episodes on the
     same validation camera.
3. Test is scored once with these rules: injected drift within a session (bump in the
   second half), injected drift between sessions (earlier episode on the same camera as the
   baseline), frames to decide, and the number of "can't tell".
4. Real drift: a Pred episode is tested against a healthy GT episode on the same camera
   (earlier when one exists). Only Pred cameras on validation or test serials (not seen in
   training). A Pred camera with no GT episode on its camera answers "can't tell (no healthy
   baseline)". Healthy GT pairs on test cameras give the false alarms.

### 17:30 validation calibration of the rules above (no test results seen yet)
- Gate picked on validation: reliable if baseline median residual <= 15 px and >= 70% of
  frames confident. Keeps 8 of 25 validation cameras (their half-to-half p99 change 15.4 px).
- k at 1 in 100 comes out very large: within a session 35 to 80, between sessions 57 to 301,
  growing with the number of frames. So the standard-error model is wrong for this data: the
  residual drifts slowly within an episode (error depends on the arm's pose), so it does not
  average down with more frames, and the MAD understates it.
- Rule 2 is scored on test exactly as written. Added now, before any test result, a second
  rule (rule 2b), also set on validation only: alarm if the camera's median residual rises
  by more than d px over its baseline, d = the 99th percentile of healthy rises over gated
  validation decisions, per decision window and separately within and between sessions.
  Same gate, same decisions.

### 17:40 drift alarm fix: result (test scored once)
Measured on held-out DROID cameras. Rules and thresholds from validation only (above).

| | Old fleet threshold | Per-camera, rule 2 (z) | Per-camera, rule 2b (px) |
|---|---|---|---|
| Within session: decided / can't tell | 67 / none | 39 / 81 | 39 / 81 |
| Within, all frames: healthy false alarms | 1.5% | 1/39 | 0/39 |
| Within, all frames: caught 2 cm / 5 cm / 5 deg | 1.5% / 1.5% / 1.5% | 4% / 5% / 8% | 0% / 1% / 2% |
| Between sessions, all frames: false alarms | 0% | 2/340 | 2/340 |
| Between sessions, all frames: caught 5 cm | 0% | 1% | 0% |
| Real drift (Pred): decided / moved / can't tell | 45 / 6 / none | 10 / 1 / 18 | 10 / 1 / 18 |

- The per-camera alarm does not fix it. The gate leaves a third of the test cameras decidable.
  On those, healthy sessions still jump: between-session noise floor p50 8.4 px, p90 36 px,
  p99 172 px (short windows: p90 268 px). More frames do not help: the swings are slow and
  systematic, not frame noise.
- One camera shows why: healthy GT sessions of 20252535 range from 8.9 px to 483.4 px.
  Either the detector fails on whole sessions, or DROID's kept calibration is wrong for some
  sessions. Telling those apart is the next question.
- Preflight on TRI 20252535 against its 2023-11-01 session: healthy +1.5 px (z 1.8), 2 cm
  +6.3 px (z 7.7), 5 cm +25.7 px (z 31.2). All GO: the alarm is at z > 300.9. 2 cm is not
  caught; 5 cm is not caught either. What is caught: nothing at 1 false alarm in 100.
- The signed record (camera 21582473) keeps its decision (still called moved); not changed.
- Files: new kintrace/droid/alarm.py; kintrace/preflight.py (--baseline, "can't tell" exit 2).
  pytest: 46 passed, 5 skipped. No GPU used.

## Arms session (started 2026-10-03 17:30). Rules written before any new test result

Same heat rules (gentle GPU, 50 ms rest per batch, batch 16, temperature every 30 s, double
the rest above 78 C, stop above 85 C, CPU only after 2 heat stops) and validation rules
(test serials never used to train, choose or set thresholds; test scored once per finalist).

Step 1, multi-keypoint detector
- Keypoints, from FK through DROID's kept (GT) pose: flange (0 mm), hand centre (on the tool
  axis, 110 mm), left and right fingertips (170 mm along the tool axis, +-half the opening
  across it). Opening = 85 mm x (1 - gripper state), the Robotiq 2F-85 stroke. The axis the
  fingers open along is picked by eye on a contact sheet before any training.
- Model: the final single-keypoint recipe (E2: stronger augmentation, stem + layer1 frozen),
  one heatmap and one visibility output per keypoint. Trained on the 19 training serials,
  every 2nd frame, 12 epochs, gentle GPU. One finalist unless time allows two.
- Scored on validation per keypoint (median, p90); test once.
- 6-DoF refit with all keypoints: on validation, fit on the first half of each episode, score
  the second half. Only called "calibration error in DROID's kept poses" if the refit removes
  most of the per-episode offset on the held-out half.
- 17:40 Finger-axis check (data/droid/figures/finger_axis_check.png, finger_axis_zoom.png):
  on open-gripper frames from 8 GT cameras, no single opening direction (0, 45, 90, 135 deg in
  the flange plane) clearly lands on both fingertips; several fit equally on some cameras and
  the fingers are seen edge-on on others. Not reliable enough to label from. Decision: use
  three keypoints on the tool axis (flange 0 mm, hand centre 110 mm, fingertip centre 170 mm);
  left/right fingertips left out. Three points per frame, swept through 3D by the arm's motion,
  are what the 6-DoF refit needs.

### 17:40 status
- Step 1: three-keypoint labels built for 331 cameras (flange in view 88.9% of frames, hand
  centre 94.1%, tip 95.2%; the tip labels equal the old single-keypoint labels exactly).
  Training now: 12 epochs, gentle GPU, 63 to 64 C, done about 18:05.
- Step 6 (wrist camera) is blocked as specified: DROID's corrected set has no wrist-camera
  calibration (ground truth does not exist), and the detector is trained on fixed-camera
  views, not the wrist view. Will run injected-only later only if time remains.
- 18:10 Step 1, validation: flange 15.9 px (p90 79.3), hand centre 16.7 (127.7), fingertip
  18.6 (215.7); per-episode offset 12.6 / 11.3 / 12.1 px (single keypoint was 11.8): not
  shrunk. 6-DoF refit with all three keypoints, first half -> second half: median 18.4 ->
  15.2 px, offset 12.9 -> 9.9 px (23% removed, shrank on 19 of 25 cameras), refit size
  median 44 mm / 4.5 deg. Not "most" of the offset: not called calibration error.
  Final model choice (validation): three-keypoint model (fingertip within 2% of the
  single-keypoint 18.3 px, and only it constrains the refit). Test scored once next.
- 18:12 Step 2 rule (before test): the drift-alarm-fix rules unchanged (gate, rule 2 z, rule 2b
  px, within and between sessions, real drift vs a GT session), re-tuned on validation with the
  three-keypoint detector. A frame counts if at least 2 of the 3 keypoints are confident; its
  residual is the mean distance over its confident keypoints.
- 18:15 With three keypoints no gate in the grid passes the rule (p99 of healthy half-to-half
  change <= 20 px): cameras with a low baseline still jump 29 to 36 px between halves. Grid
  widened on validation only to X in {8, 10, 12, 15, 20, 30, 40, 60} (test still unseen). The
  single-keypoint calibration is not affected (it picked X = 15 from the old grid; rerun to confirm).
- 18:18 Step 2 result on validation: with three keypoints no gate passes even with X down to
  8 px (the strict gates keep fewer than 3 cameras). The three-keypoint drift alarm cannot be
  calibrated under the rule, so it is not scored on test. Single-keypoint calibration unchanged.

Step 3 and 7 rules (written before any Pred/fix numbers):
- Fix = 6-DoF camera pose fitted from the three keypoints over the episode's confident frames
  (frames with >= 2 confident keypoints): solvePnPRansac starting from the believed pose, then
  solvePnPRefineLM ("ours"). Standard method for comparison: plain cv2.solvePnP (iterative, no
  starting pose, no RANSAC) on the same points.
- Run on every cached Pred camera; reported separately for serials not seen in training.
  Accuracy = fitted pose vs DROID's corrected pose (mm, deg), next to configured vs corrected.
- Re-check: fit on the first half, residual on the second half under the fix. GO if that held-out
  median residual is at or under the p90 of healthy validation cameras' median residual
  (three-keypoint), else NO-GO.
- Error bars: 200 bootstrap fits over blocks of 10 frames; 90% intervals on the fix's
  translation and rotation. Coverage checked on injected bumps of known size on test GT
  cameras (0.5 to 5 cm, 0.5 to 5 deg): how often the true size falls inside the 90% interval.
- 18:22 Step 3 (fix vs DROID, all 58 Pred cameras; 28 on untrained serials): configured pose
  153 mm / 15.0 deg from DROID's corrected pose; our three-keypoint fix 79 mm / 7.2 deg
  (untrained: 79 mm / 7.1 deg); plain PnP 729 mm / 68 deg. Ours closer than configured on 35
  of 58 (17 of 28); at least as close as plain PnP on 43 of 58. Held-out re-check GO 47 of 58
  (GO line 41.5 px from validation).
- 18:23 Step 7, bootstrap intervals on injected bumps (test GT, 240 cases): coverage of the 90%
  interval only 30% (translation) and 25% (rotation); size error median 36 mm / 3.0 deg. The
  bootstrap sees frame noise, not the per-episode offset. Rule added now: widen each interval
  by +- the 90th percentile of size error on validation injected bumps; test coverage checked
  once with the widened intervals.
- 18:28 Step 7 widened intervals (validation-set +-123.6 mm, +-10.4 deg): test coverage 92%
  (translation) and 93% (rotation). Honest but wide.
- 18:30 Step 4 blinded test (kintrace/droid/blinded.py; 40 sealed cases, sha256 05461f2b...):
  can't tell 24; of the decided, caught 1 of 14 bumps (2-5 cm: 1 of 9; 1-2 cm: 0 of 4; under
  1 cm: 0 of 1), false alarms 0 of 2 healthy decided. Size error median 34.5 mm / 3.1 deg;
  widened 90% intervals held the truth 90% / 85%.

Step 5 rules (before results): per camera serial (validation and test serials only), sessions
in date order. Kintrace's pose per session = the three-keypoint fix started from that session's
configured pose. Kintrace flags a session if its fitted pose moved from the previous session's
by more than the 99th percentile of that change on validation session pairs where DROID's own
pose did not change (< 10 mm and < 1 deg). DROID says moved if its corrected pose changed by
>= 10 mm or >= 1 deg. Agreement, misses and false flags over consecutive session pairs on test.

### 18:35 status
- Step 5 (test serials, 142 sessions on 8 cameras, date order): Kintrace's per-session fitted
  camera pose tracks DROID's corrected pose with a median error of 59 mm / 5.2 deg (p90 279 mm /
  23 deg) while the cameras themselves move 100 to 600 mm between sessions
  (figures/final/sessions_top3.png). DROID's pose changes between 102 of 134 consecutive sessions
  (>= 10 mm or 1 deg); Kintrace flags 127. Of the 32 pairs DROID calls unchanged, Kintrace
  false-flags 26: the flag threshold (56 mm / 4.1 deg) came from only 4 validation pairs, too few.
  So session tracking follows DROID's pose but does not yet separate real moves from fit noise.
- Bug fixed: DROID ids are LAB+hash+date, so sorting by id is not date order. sessions.py and
  alarm.py now sort by the date part. The drift-alarm and blinded tests ran with the old sort
  (it only changes which same-camera session counts as "earlier"); not re-scored.
- Heat: no stops; all GPU work this session 51 to 65 C.
- Next: Step 8 (HTML report), Step 9 (second dataset check), write-up.
- 18:45 Step 8: `kintrace report <episode> --camera SERIAL` writes one self-contained HTML page
  (kintrace/droid/report_html.py): frame strip with expected (calibration), detected and
  after-fix keypoints, what moved with the calibrated 90% range, before/after on held-out
  frames, DROID's pose when there is one. Two pages in data/droid/reports/:
  - moved: IRIS 29838012 (test serial, picked because its calibration is 304 mm off DROID's,
    a large real displacement; not the median case): NO-GO, fix ready; 307 mm / 21.3 deg;
    held-out error 79 -> 15 px.
  - healthy: TRI 20252535 2023-11-02: GO; 22 mm / 1.9 deg fitted (inside the noise); 25 -> 23 px.
  Checked on desktop and a 375 px phone viewport: fits, no sideways scroll.
- 18:50 Step 9: RH20T provides per-camera calibration to the robot base (its site says every
  global camera is calibrated), on Flexiv, UR5, Franka and KUKA configs. But it ships only as
  whole-config archives on Google Drive / Baidu (smallest 4.4 GB at 320x180, 15 GB for the UR5
  cfg4), and the Hugging Face mirrors checked (robot-lev/rh20t_cfg4_task0064, rh20t_cfg3,
  anqil/rh20t_camera_samples) drop the calibration. cfg4 is a UR5 with a Robotiq 2F-85 (same
  gripper as DROID). A small subset with calibration is not available, so stopped there.

## Arms session summary (17:30 to 18:30)

Headlines (DROID, real Franka recordings; not run on a live arm):
- Detector (held-out cameras): fingertip 20.9 px median; hand centre 17.9 px; flange 18.7 px
  (single keypoint: 19.9 px epoch 50, 21.8 px after tuning). 12 px still missed.
- Smallest drift caught at 1 false alarm in 100: none, with fleet-wide or per-camera alarms,
  single or three keypoints. The signal is there on a typical camera (5 cm about +22 to 26 px),
  but healthy sessions of the same camera range from 9 to 483 px.
- Fix accuracy vs DROID (measured on held-out cameras): running calibration 153 mm / 15 deg from
  DROID's corrected pose, Kintrace's three-keypoint fix 79 mm / 7.2 deg; plain PnP 729 mm / 68 deg.
- Blinded test: 40 sealed cases, 24 can't tell, 1 of 14 bumps caught, 0 false alarms; size error
  34 mm / 3.1 deg; widened 90% ranges held the truth 90% / 85%.
- Error bars: raw bootstrap covered the truth only 30% / 25%; widened by the validation size
  error they cover 92% / 93% on test (+-124 mm, +-10 deg: honest but wide).
- Sessions: per-session pose tracks DROID's at 59 mm / 5.2 deg median over 142 sessions; flags
  26 of 32 unchanged sessions, so not yet a move detector.

What did not work or was not run
- Left/right fingertip keypoints (finger axis not identifiable from the contact sheet).
- Three-keypoint drift alarm: no gate passes the validation rule.
- Calibration error in DROID's kept poses: not shown (a three-point refit removes 23% of the
  per-episode offset on held-out frames, not most of it).
- Wrist camera: no ground truth and no wrist-view detector. RH20T: calibration not available
  in a small download. bench -n 4 crash: still there (pre-existing; separate task offered).

Also fixed: pip install -e . failed (flat layout; now packages = kintrace*); new `detector` extra;
kintrace.crossrobot no longer needs pandas at import; preflight works with the three-keypoint model.

Heat: no stops; GPU 51 to 65 C (gentle mode) for training, scoring and caching.
Tests: pytest 46 passed, 5 skipped. bench -n 3 --seed 7 and kintrace incident identical to the
14:37 baseline; bench -n 4 crashes as before.

Files (not committed): new kintrace/droid/{alarm,multikp,fixes,blinded,sessions,report_html}.py,
examples/droid_preflight_report.ipynb, incidents/droid_kp3_IRIS+7dfa2da3+2023-05-03-09h-24m-38s_29838012/;
changed README.md, RESULTS_DROID.txt, PROGRESS.md, pyproject.toml, kintrace/cli.py, kintrace/preflight.py,
kintrace/crossrobot.py, kintrace/droid/{detector,convert,experiments}.py, tests/test_detector_droid.py.
Local only: data/droid/ (models, caches, figures, reports, blinded answers), kintrace_keys/.

Next
1. Find what makes healthy sessions of one camera jump (detector failing on whole sessions vs
   DROID's kept calibration). That jump is what blocks every alarm.
2. More points on the hand (identify the finger axis from the URDF/CAD of the Robotiq mount),
   which should tighten the fix and its 90% ranges.
3. A real-arm test on the SO-101, where the ground truth is known.
