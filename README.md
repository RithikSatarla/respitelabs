# Respite Labs

**The check layer for physical AI. Starting with robot arms.**

When a robot's body is right, its sensors agree. When something physical
changes, they disagree in a pattern that says what moved.

Kintrace is the check: a short routine a robot runs (a 6.3 s check motion in
simulation) to confirm its body is what its software thinks it is. Is the camera where the calibration says, is the tool
the length the controller thinks, are the joint zeros still zero, do commands
land on time. When one of those changes, Kintrace says what moved, by how much,
and writes the fix.

The checks are geometry and statistics, so the same log always gives the same
answer and every answer can be checked. One small vision model finds the
gripper in camera images; that is the only learned part.

Website: [respitelabs.net](https://www.respitelabs.net). Contact: contact@respitelabs.net

## Where it stands

| | Status |
|---|---|
| Simulated UR5e cell | 120/120 faults named and sized. See [Results](#results-simulation). |
| Real Franka data (DROID) | Held-out cameras: gripper detector 81 px -> 19.9 px median, measured on different held-out camera sets (12 px target missed). Camera fix on real moved cameras: 153 mm / 15 deg -> 79 mm / 7 deg from DROID's own corrected pose. Drift alarm: nothing reliably caught at 1 false alarm in 100. Blinded test: 1 of 14 bumps caught, 0 false alarms, 24 of 40 "can't tell"; size error 34 mm / 3 deg with honest 90% ranges. Injected delays caught 145/150. Tested on recordings, not on a live arm. See [DROID](#real-robot-data-droid-in-progress). |
| Other robots (public LeRobot data) | Timing and joint-offset checks on ALOHA, Koch, SO-100, Unitree G1 arms and LeKiwi: injected latency 30/30 on every arm set, 0 healthy alarms. See [Also runs on other robots](#also-runs-on-other-robots-timing-and-joint-offsets). |
| Real arm (SO-101 desk rig) | Next: SO-101 desk rig. Nothing measured on a real arm yet. |
| Someone else's robot | Send one recording and we'll run it: `kintrace import` + `kintrace check`, or `kintrace preflight`. |

Every number in this file says which row it comes from.

## Try it (2 minutes)

Windows: double-click `setup.bat`. Mac or Linux: `bash setup.sh`. Or by hand:

```bash
pip install -r requirements.txt
python -m kintrace incident                    # 4 incidents: find, check, fix, certify
python -m kintrace watch --sim camera_sag      # slow drift, warned before picks fail
python -m kintrace bench -n 20                 # Kintrace vs a standard dashboard
```

## Bring your own recording

One session from your arm is enough: joint positions over time, the fixed
camera's video, and the camera pose you're using. No earlier healthy run needed.

```bash
pip install rosbags pandas pyarrow h5py        # readers
kintrace import my_dataset/ --config rig.json --intrinsics cam.json -o log.npz   # LeRobot dataset
kintrace import session.mcap --config rig.json --intrinsics cam.json -o log.npz  # ROS 2 bag / MCAP / ROS 1 bag
kintrace import rtde.csv --video cam.mp4 --config cell.json -o log.npz          # UR RTDE recording
kintrace check log.npz                                                          # one log, no baseline
```

`import` reads the joints and commands from the recording and finds the
wrist in each frame. Today that needs a printed AprilTag on the wrist (two
minutes of tape; `kintrace rig tags` prints one). A markerless detector exists
only for DROID's Franka arms with a Robotiq gripper (see the DROID section);
other grippers still need the tag. `check` runs in
single-session mode: the settings the recording was made with count as
"last known good", and the question is whether the sensors still agree with
them. What a recording is missing (no commanded joints, no camera) is
reported, and that check is skipped rather than guessed.

## After a crash or a change: `kintrace incident`

1. **Find** when it changed, from the logs, and match it to the logged event
   (protective stop, settings edit, service, software update).
2. **Check** what moved with a 6.3 s motion: 8 poses facing the camera. A
   marker fixed to the table tells a moved camera apart from a base joint that's off.
3. **Fix**: write the new camera calibration, tool offset or joint zeros.
   Late commands can't be fixed with a number, so those go to a person.
4. **Certify**: run the check again, then 6 test picks, then GO or NO-GO.

Real output for the crash scenario:

```
Changed at   : 13:40:23  (lines up with protective stop at 13:40:23)
What changed (6 s check motion)
  - Camera moved: 21.2 mm, 0.11 deg (parts appear 23 mm off in the pick area)
  - Gripper / tool bent: fingertip moved 24.4 mm (x -2.7, y +24.0, z +3.8)
Fix written to controller
  - camera calibration updated
  - tool offset set to [-2.7, 24.0, 183.8] mm
Verification: all checks ok, test picks 6/6
STATUS: GO
```

Incident benchmark (`python -m kintrace.bench_incident`): 50 of 50 incidents
got the right cause and the right GO / NO-GO. That's crash, tool edit, joint
service, software update and healthy, 10 each. The crash onset was found
within 0.01 s (median).

## Always on: `kintrace watch`

Reads the logs the robot already writes, with no check motion, and gives the
robot a physical health score every 10 s. It flags slow drift (a sagging
camera mount, a creeping joint) before picks start missing.

`python -m kintrace watch --bench`, 10 runs each of camera sag, joint creep and healthy:

| | camera sag | joint creep |
|---|---|---|
| warned before the first missed pick | 9/10 | 10/10 |
| cause named right | 10/10 | 10/10 |
| error when it warned (median) | 2.5 mm | 2.5 mm |
| error at the first miss (median) | 5.9 mm | 7.2 mm |

0 false alarms in 230 healthy windows. The drift is sped up so a run takes
minutes. We have not measured lead time at real-world drift rates.

## Cell ledger: `kintrace ledger`

Every record and warning goes into a per-cell history. `ledger report` shows
which cells change most, why, and how long between changes.
`ledger demo` seeds 6 simulated cells over a few weeks so you can see it.

## Signed records: `kintrace certify` / `verify`

```bash
python -m kintrace keys init --dir keys
python -m kintrace certify incidents/crash_record.json --key keys/private.pem -o crash.signed.json
python -m kintrace verify crash.signed.json
```

Records are signed with ed25519. Change any field after signing and `verify`
says INVALID. A fleet can hand the record to its customer as proof the cell
was checked.

## Desk rig: `kintrace rig`

An SO-101 arm, a webcam and printed AprilTags, about $300 to $500 in parts.
See [HARDWARE.md](HARDWARE.md). `kintrace rig ... --dry-run` runs the whole
path on the simulator. It has not run on the real arm yet.

## Diagnose one log

```bash
python -m kintrace simulate --fault none -o healthy.npz
python -m kintrace baseline healthy.npz -o baseline.json     # learn "normal" once
python -m kintrace simulate --fault encoder_bias --seed 4 -o bad.npz
python -m kintrace diagnose bad.npz -b baseline.json --chart bad.png
```

## How it works

When nothing is wrong, a robot's sensors agree with each other. Each fault breaks
a different agreement, in a different shape:

| Check | What it compares | What breaks it |
|---|---|---|
| Command timing | commanded joints vs measured joints | commands arriving late (network, CPU, driver) |
| Controller settings | running tool offset / camera calibration vs last known-good | someone edited the tool offset |
| Camera vs joint encoders | where the camera sees the wrist plate vs where the encoders + arm model say it is | camera bumped, encoder zero drifted |
| Pick success | success rate vs normal | anything that causes misses |

Camera moved and encoder drift both break the same check, but differently:
a moved camera shifts everything by one fixed amount, while a drifted encoder
causes an error that changes with the arm's pose. Kintrace fits every
explanation and keeps the only one that brings the error down to sensor noise.

A bent fingertip breaks nothing the robot can normally see. When picks fail but
every sensor agrees, Kintrace asks for a **1.2 second probe**: the arm shows
its fingertip to the camera while the wrist sweeps, and Kintrace measures the
tool directly.

## Results (simulation)

120 runs, every fault sized to cause the same ~20 mm miss. The baseline is a nearest-signature classifier on the numbers a typical robot dashboard tracks (camera tracking error, joint tracking error, pick success).

| | Baseline dashboard (ours) | Kintrace |
|---|---|---|
| Correct cause | 69% | 100% (83% without the probe) |
| Encoder drift | 20% | 100% |
| Tool offset edited | 40% | 100% |
| Bent tool | 60% | 100% |
| Size error | none given | 0.2 mm / 0.003 deg / 0.01 ms (median) |
| Two faults at once | n/a | both found, 100% |
| False alarms on healthy robots | 0% | 0% |

Stress test: still 100% with faults down to 8 mm and a camera 3x noisier.
At 5 mm a bent tool doesn't make picks fail, so Kintrace reports the robot as fine.

**Limits:** this is a simulation we built, and we built the baseline too. Real sensors have biases,
dropouts and unmodeled effects. The two sections below are how we get past that.

## Real robot data: DROID (in progress)

[DROID](https://droid-dataset.github.io/) is 76k episodes on Franka Panda arms
with fixed stereo cameras. Each episode records the camera extrinsics the cell
was configured with. In 2025 the DROID team re-solved the extrinsics for ~36k
episodes ([KarlP/droid](https://huggingface.co/KarlP/droid)) because
the stored calibration could not be trusted for those episodes. That gives real
cases where the running calibration was replaced, with DROID's corrected pose
as the reference. It is DROID's estimate, not a measured ground truth.

GT = DROID kept the original calibration. Pred = DROID re-solved it. Only Pred
entries can show a moved camera, and a Pred difference is camera movement or a
calibration that was never right; the data cannot tell which.

`kintrace/droid/` is the adapter. Franka Panda kinematics are in `arms.py`.

```
python -m kintrace.droid download --out data/droid --episodes 50   # extrinsics JSONs + 50 raw episodes
python -m kintrace.droid survey   --data data/droid                 # GT vs Pred across the whole corrected set
python -m kintrace.droid labels   --data data/droid                 # how far configured cameras were from the truth
python -m kintrace.droid faults   --data data/droid                 # inject the other faults into real logs
python -m kintrace.droid detect   --data data/droid --synthetic     # pipeline test with synthetic detections
```

What each fault can and cannot get from DROID:

| Fault | On DROID | Needs |
|---|---|---|
| Camera moved | Real cases (Pred) and injected bumps. The gripper detector plus a 2D check; results below. | runs today |
| Commands late | Injected: measured joint stream shifted against the commanded one. Real motion, real timing noise. | runs today |
| Encoder zero drifted | Injected: a bias added to one joint's measured angle. In 2D it looks like a moved camera: 0 of 63 flagged. | a 3D point per frame |
| Tool offset edited | A settings comparison against the known-good config; not run on frames. | nothing |
| Bent tool | Not possible. Nothing in DROID records the tool. Waits for the desk arm. | the SO-101 |

Injected faults on real logs are how most fault-detection work gets evaluated, and
they are labelled "injected" everywhere here. They are not the same as a camera
someone actually bumped, which is why the camera row matters most.

DROID arms carry no wrist marker, so the camera checks need a markerless gripper
detector (`kintrace/droid/detector.py`, `detect_gripper()` in `convert.py`). Before
it existed, `--synthetic` ran the pipeline with detections made from the corrected
pose plus 3 mm noise: on a 12-episode fixture, 10/10 moved cameras caught. That is a
pipeline test, not a result; the real-frame results are below.

### What has run on real DROID data (50 episodes, Oct 2026)

Raw output: [RESULTS_DROID.txt](RESULTS_DROID.txt).

- **DROID corrected extrinsics, all entries.** 36084 entries: 30790 GT (85.3%),
  5294 Pred (14.7%).
- **Our DROID sample.** 50 episodes, one fixed camera each: 43 GT, 7 Pred.
- **Commands late (injected delays on real Franka joint streams).** Caught
  145/150: 48/50 at 10 ms, 48/50 at 20 ms, 49/50 at 50 ms. Size error median
  4.40 ms.

### Gripper detector on real DROID frames (Oct 2026)

**Gripper detector: 81 px -> 19.9 px median on held-out real cameras.**

DROID is a real Franka dataset. Labels are free: on episodes where DROID kept the
calibration (GT), the fingertip from forward kinematics is projected into the image.
Train, validation and test are split by camera serial. The first 317 labelled GT
episodes come from 26 physical cameras, not 317; 4 more cameras were added for
training later.

| Held-out test cameras (8 serials, 120 cameras) | Median | p90 | Within 12 px |
|---|---|---|---|
| Small CNN (Phase 3, its own held-out set) | 81 px | 302 px | n/a |
| ResNet-18, epoch 50 | 19.9 px | 232 px | 30.2% |
| After tuning on a validation split (chosen on validation) | 21.8 px | 220 px | 27.2% |

The 81 px and the other rows were measured on different held-out cameras. The
tuning round (finer output, stronger augmentation, unfreezing, a label rule for a
hidden gripper) shrank the tail but did not beat 19.9 px on test.

It missed the 12 px target. On validation about half the median error is a fixed
offset per camera and episode (median 11.8 px), the rest is scatter (13.4 px). The
offset is not the intrinsics (checked), not the keypoint definition (no shared
direction in the gripper frame), not timing (a +30 ms median shift between video and
joints exists but removes nothing) and not a recoverable calibration error (a camera
refit from one keypoint is too weakly constrained to say). In a by-eye sample of 24
big misses, 14 were real detector misses and 7 were label problems (gripper hidden
or elsewhere).

### Camera drift on DROID (injected and real)

The check compares a camera with its own baseline, so a fixed offset cancels. On
held-out GT cameras, with a known calibration error injected:

| Injected | 0.5 cm | 1 cm | 2 cm | 5 cm | 0.5 deg | 1 deg | 2 deg | 5 deg |
|---|---|---|---|---|---|---|---|---|
| Rise in the camera's median residual, within a session | -0.6 px | 0.8 px | 5.1 px | 22.0 px | -0.2 px | 1.7 px | 6.7 px | 29.8 px |

Healthy session-to-session spread on the same camera: median 7.8 px, p90 45.4 px,
p99 177.8 px. At 1 false alarm in 100 the alarm level is about 200 px, set by a few
cameras where the detector fails for part of an episode, so no drift size is caught
at that rate yet. Real drift on cameras DROID re-solved (Pred): 6 of 45 flagged, with
0 false alarms on 63 healthy test cameras. Raw output and both filter settings are in
RESULTS_DROID.txt.

A per-camera alarm (`kintrace/droid/alarm.py`: a reliability gate, then each camera
against its own healthy baseline, all set on validation) did not fix this. Measured on
held-out DROID cameras, test scored once: the gate leaves 39 of 120 cameras decidable
within a session (81 answer "can't tell"); at 1 false alarm in 100 it catches at most 8%
of 5 deg and 5% of 5 cm bumps. Between sessions on the same camera the healthy noise floor
is median 8.4 px but p90 36 px and p99 172 px: healthy sessions of one camera range from
9 px to 483 px. On real DROID re-solved cameras it decided 10, called 1 moved, and answered
"can't tell" on 18. The open question is whether those healthy jumps are the detector or
DROID's kept calibrations.

### Fixing real moved cameras (DROID, three keypoints)

The detector now finds three points on the tool axis: flange, hand centre and
fingertip (held-out cameras: 18.7, 17.9 and 20.9 px median). With three points a
6-DoF camera pose can be fitted from one episode. On every camera DROID re-solved
(58; 28 never seen in training), measured against DROID's own corrected pose:

| | Distance to DROID's corrected pose (median) |
|---|---|
| The calibration the robot was running | 153 mm / 15.0 deg |
| Kintrace's fix | 79 mm / 7.2 deg (untrained cameras: 79 mm / 7.1 deg) |
| Plain PnP on the same points, no starting pose | 729 mm / 68 deg |

The fix is closer than the running calibration on 35 of 58 cameras and passes a
held-out re-check on 47. The best case, IRIS camera 29838012, went from 304 mm to
18 mm; that is a best case, the median is 153 mm to 79 mm over 58 cameras. It roughly halves the error but is not a calibration
tool yet: one episode pins the camera to about a decimetre. Every size comes with a
90% range calibrated on held-out cameras (true size inside it 92% / 93% of the time).

Blinded test (`python -m kintrace.droid.blinded`, sealed answers): 40 cases, 24
"can't tell", 1 of 14 bumps caught, 0 false alarms; size error 34 mm / 3.1 deg, and
the 90% ranges held the truth 90% / 85% of the time.

Over 142 sessions on 8 held-out cameras, Kintrace's per-session camera pose follows
DROID's corrected pose (median 59 mm / 5.2 deg apart) while the cameras move 100 to
600 mm between sessions. It does not yet tell a real move from fit noise: it flags
26 of the 32 sessions where DROID's pose did not change.

**What we learned from DROID.** DROID's cameras move 100 to 600 mm between
sessions, and healthy sessions of one camera range from 9 px to 483 px of error.
So DROID cannot validate cm-level drift detection: a 2 cm bump has not been shown
to be caught on real data. What it does show is that the fix moves a badly off
calibration about halfway to DROID's own corrected pose. The next step is a
controlled test on a real arm, with known bumps.

`kintrace report <episode> --camera SERIAL` writes one HTML page per camera: where
the calibration says the hand is, where the camera sees it, the fix, and before vs
after.

The camera result comes from the 2D check in `kintrace/droid/phase2.py` (detector,
then residual against the forward-kinematics fingertip), not from the full
`diagnose()` path, which needs a 3D point per frame that one DROID view does not give.

- **Encoder zero drifted (injected, 10 and 20 mm at the fingertip).** Flagged 0 of 63.
  In 2D a joint that reads wrong looks like a camera that moved; the check cannot
  separate them.
- **Tool offset edited.** A settings comparison; it does not depend on the frames.

## Also runs on other robots (timing and joint offsets)

The camera checks need a calibrated fixed camera, which public sets on other
robots mostly don't have. What needs only commanded and measured joints runs on
any recording that logs both. Real public LeRobot recordings, 10 episodes each; the reference is the
same robot's other episodes. Latency faults are injected (the measured stream
shifted by 20, 50 and 100 ms); joint offsets are injected (2 and 5 deg on one
joint). The joint-offset check is new and simple: each joint's median of measured
minus commanded.

| Robot (dataset) | Joints | Delay | Healthy latency alarms | Injected latency caught | Healthy offset alarms | Injected offset caught |
|---|---|---|---|---|---|---|
| ALOHA, bimanual (lerobot/aloha_static_coffee) | 12 | 84.8 ms | 0/10 | 30/30 | 0/10 | 20/20 |
| Koch (lerobot/koch_pick_place_5_lego) | 5 | 88.4 ms | 0/10 | 30/30 | 2/10 | 19/20 |
| SO-100 (lerobot/svla_so100_pickplace) | 5 | 115.2 ms | 0/10 | 30/30 | 0/10 | 20/20 |
| Unitree G1 humanoid, both arms (unitreerobotics/G1_Dex1_Stack_Block) | 14 | 85.9 ms | 0/10 | 30/30 | 0/10 | 20/20 |
| LeKiwi mobile manipulator, arm (QianGroup/lekiwi_pick_sponge) | 5 | 50.4 ms | 0/10 | 30/30 | 0/10 | 20/20 |
| LeKiwi, wheeled base (same set) | 3 wheels | 56.2 ms | 3/10 | 23/30 | not run | not run |

Not tested on these robots: anything that needs a camera (camera moved, encoder
drift seen by the camera, bent tool). The wheels are the weakest row.

`kintrace/urdf.py` loads a robot body from a URDF. Checked against the hand-written
models over 1,000 random poses: Franka Panda max 6.9e-9 mm, UR5e max 2.1e-7 mm,
SO-101 max 2e-13 mm. A Unitree G1 URDF and an SO-100 URDF load and run forward
kinematics; nothing more is claimed for them.

`kintrace preflight <recording>` runs every check that applies, prints one line
per check (ok, problem, or skipped and why), then GO or NO-GO. Exit code 0 is GO,
1 is NO-GO, 2 is not enough data, so a lab can put it in front of a
data-collection or shift-start script.

## Where this goes (roadmap, not built yet)

| Robot type | Sensors that should agree | What drift looks like | Status |
|---|---|---|---|
| Arms | joint encoders, commands, fixed and wrist cameras | camera bumped, tool bent, joint zero off, commands late | now |
| Mobile robots | wheel odometry, IMU, lidar or camera | wheel slip, wheel radius off, sensor mount moved | not built (base timing only, above) |
| Humanoids | joint encoders, IMU, foot contact, head cameras | joint zero off, camera knocked, foot sensor bias | not built (arm timing only, above) |
| Drones | IMU, GPS, motor commands, camera | IMU misaligned, motor weakening, camera mount loose | not built |

## How we're different

- Observability tools such as Ember and Foxglove collect and show robot data, so
  engineers can investigate.
- Robot makers' tools such as FANUC ZDT predict part wear on their own brand.
- Calibration tools such as RoboDK or Dynalog fix the robot with measuring
  hardware.
- Kintrace works out the physical cause from the robot's own sensors, sizes it,
  writes the fix and signs off, with no fixtures.

## Real arm results

None yet. Next: an SO-101 desk rig; `kintrace rig` is the capture and check path
for it (`HARDWARE.md`). Nothing has been measured on a real arm. The plan is the same faults as the
simulation, caused on purpose, on that arm, and this section gets the
numbers as they come in, good or bad.

## Log format

See `kintrace/logio.py`. Anything that can export commanded joints, measured
joints, camera detections of a wrist marker, pick outcomes and the controller's
tool/camera settings can be diagnosed. `kintrace/readers/` does this for
LeRobot datasets, ROS 1/2 bags, MCAP and UR RTDE recordings.

## Files

- `kintrace/robot.py`, `kintrace/arms.py` arm kinematics (UR5e-class, SO-101, Franka Panda)
- `kintrace/readers/` LeRobot, ROS bag / MCAP, UR RTDE -> kintrace log
- `kintrace/droid/` DROID adapter: download, labels, gripper detector (training, eval,
  `detect()`), per-camera alarm, 6-DoF fix with error bars, blinded test, HTML report
- `kintrace/urdf.py` robot bodies from URDF; `kintrace/preflight.py` GO / NO-GO with exit codes
- `kintrace/crossrobot.py` timing and joint-offset check on public LeRobot recordings
- `kintrace/sim.py` simulated pick cell with fault injection
- `kintrace/diagnose.py` the checks and the explanation fitting
- `kintrace/recommission.py` find, check, fix, certify
- `kintrace/watch.py` health score and early warning
- `kintrace/ledger.py` per-cell history
- `kintrace/certify.py` ed25519 signing
- `kintrace/hw/` SO-101 desk rig: markers, capture, check
- `kintrace/bench.py`, `bench_incident.py` benchmarks
- `incidents/` the four example incidents, logs and records
- `bench_out/` benchmark results
- `docs/` the GitHub Pages site and the interactive demo
- `pitch/` the deck and its sources
- `tests/` run with `python -m pytest tests` (also run on every push, see Actions)
