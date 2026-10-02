#Respitelabs

Physical observability for robot arms.

Monitoring tools watch a robot's software. Kintrace watches its body: is the
camera where the calibration says, is the tool the length the controller
thinks, are the joint zeros still zero, do commands land on time. When one of
those changes, Kintrace says what moved, by how much, and writes it back.

No AI model inside. It's geometry and statistics, so the same log always gives
the same answer, and every answer can be checked.

Everything below runs on a simulated UR5e-class pick cell. The desk rig in
[HARDWARE.md](HARDWARE.md) is how it gets onto a real arm.

## Try it (2 minutes)

Windows: double-click `setup.bat`. Mac or Linux: `bash setup.sh`. Or by hand:

```bash
pip install -r requirements.txt
python -m kintrace incident                    # 4 incidents: find, check, fix, certify
python -m kintrace watch --sim camera_sag      # slow drift, warned before picks fail
python -m kintrace bench -n 20                 # Kintrace vs a standard dashboard
```

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
dropouts and unmodeled effects. The next step is the same five faults, caused
on purpose, on a real arm at CMU.

## Log format

See `kintrace/logio.py`. Anything that can export commanded joints, measured
joints, camera detections of a wrist marker, pick outcomes and the controller's
tool/camera settings can be diagnosed. Next: a ROS 2 bag / MCAP adapter.

## Files

- `kintrace/robot.py`, `kintrace/arms.py` arm kinematics (UR5e-class, SO-101)
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
- `docs/` the website (GitHub Pages) and the interactive demo
- `tests/` run with `python -m pytest tests`
