# Respite Labs

**The check layer for robot arms.** Kintrace is the check: a few seconds a
robot spends confirming its body is what its software thinks it is.

Monitoring tools watch a robot's software. Kintrace watches its body: is the
camera where the calibration says, is the tool the length the controller
thinks, are the joint zeros still zero, do commands land on time. When one of
those changes, Kintrace says what moved, by how much, and writes the fix.

No AI model inside. It's geometry and statistics, so the same log always gives
the same answer, and every answer can be checked.

Website: [respitelabs.net](https://www.respitelabs.net). Contact: contact@respitelabs.net

## Where it stands

| | Status |
|---|---|
| Simulated UR5e cell | 120/120 faults named and sized. See [Results](#results-simulation). |
| Real Franka data (DROID) | Ran on 50 DROID episodes: injected delays caught 145/150. Camera, encoder and tool checks wait on a gripper detector. See [DROID](#real-robot-data-droid-in-progress). |
| Real arm (SO-101 desk rig) | Hardware on order. Nothing measured yet. |
| Someone else's robot | Not yet. Send one recording and we'll run it: `kintrace import` + `kintrace check`. |

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
minutes of tape; `kintrace rig tags` prints one). A markerless detector for
the Franka hand is the next piece, see the DROID section. `check` runs in
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
cameras had moved during collection and nobody had re-calibrated. That is the
fault the check exists for, on real robots, with ground truth: the configured
pose is the belief, the corrected pose is the truth.

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
| Camera moved | Real. Ground truth from the corrected extrinsics. | gripper detector |
| Commands late | Injected: measured joint stream shifted against the commanded one. Real motion, real timing noise. | nothing, runs today |
| Encoder zero drifted | Injected: a bias added to one joint's measured angle, sized by how far it moves the fingertip. Real motion, real images. | gripper detector |
| Tool offset edited | Injected: config change. The check is a settings diff, so the data only has to stay quiet. | gripper detector |
| Bent tool | Not possible. Nothing in DROID records the tool. Waits for the desk arm. | the SO-101 |

Injected faults on real logs are how most fault-detection work gets evaluated, and
they are labelled "injected" everywhere here. They are not the same as a camera
someone actually bumped, which is why the camera row matters most.

Two levels of result:

1. **Labels (no vision).** How often, and by how much, DROID cameras sat away
   from their configured pose. Real data, nothing of ours in the loop. Numbers
   go here once we have run it on a few hundred episodes.
2. **Detection.** Joints plus fixed-camera frames through `diagnose()` with the
   configured extrinsic as the belief, scored against the label. DROID arms
   have no wrist fiducial, so this needs one point on the gripper per frame:
   `kintrace/droid/convert.py: detect_gripper()` is the function to fill
   (Franka Hand keypoint model, or a mask plus ZED depth). It is not written
   yet. `--synthetic` runs the same pipeline with detections made from the
   corrected pose plus 3 mm noise, which tests everything except the detector.
   On a 12-episode synthetic fixture: 10/10 moved cameras caught, 1 false
   alarm on a camera 7.9 mm off (just under the 10 mm threshold, 21 frames).
   Injected faults on the same fixture: latency 72/72 at 10 to 50 ms, tool
   offset 33/33, encoder drift 17/22 at 10 and 20 mm of tip motion and 1/11
   at 5 mm (under the 3 mm detector noise assumed). Pipeline tests, not results.

### What has run on real DROID data (50 episodes, Oct 2026)

Raw output: [RESULTS_DROID.txt](RESULTS_DROID.txt).

- **DROID corrected extrinsics, all entries.** 36084 entries: 30790 GT (85.3%),
  5294 Pred (14.7%).
- **Our DROID sample.** 50 episodes, one fixed camera each: 43 GT, 7 Pred.
- **Commands late (injected delays on real Franka joint streams).** Caught
  145/150: 48/50 at 10 ms, 48/50 at 20 ms, 49/50 at 50 ms. Size error median
  4.40 ms.
- **Encoder zero drifted.** Not run. Waits on the gripper detector.
- **Tool offset edited.** Not run. Waits on the gripper detector.
- **Camera moved.** Not run on real frames. Waits on the gripper detector.

## Real arm results

None yet. An SO-101 desk arm is on order and `kintrace rig` is the capture and
check path for it (`HARDWARE.md`). The plan is the same faults as the
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
- `kintrace/droid/` DROID adapter: download, labels, injected faults, detection run
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
