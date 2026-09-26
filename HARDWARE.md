# Kintrace desk rig (SO-101)

A cheap physical rig to show Kintrace on real hardware: an SO-101 hobby arm,
a USB webcam and printed AprilTags. You cause each of the five faults by hand,
and Kintrace finds it, sizes it, writes the fix and gives GO or NO-GO.

**Status, honestly:** the software is written and tested against a simulated
desk. The simulated desk renders real AprilTag images and runs the real
detector, solvePnP, commissioning, check, fix and touch test on them. None of
it has run on a physical SO-101 yet. See "Tested vs not tested" at the end.

The rig uses the same analysis code as the simulated UR5e cell
(`recommission.analyze_check`). Only the arm model (`kintrace/arms.py`) and the
source of the numbers (tags seen by a webcam) change.

## 1. What to buy

Prices are approximate, in USD, as of September 2026. Check before ordering.

| Part | Price | Notes |
|---|---|---|
| SO-101 follower arm, parts | ~$122 | 6x Feetech STS3215 (7.4 V, 1/345 gear), Waveshare bus servo board, 5 V supply, USB-C cable, table clamps. BOM in github.com/TheRobotStudio/SO-ARM100 |
| 3D printed arm parts | $15-25 filament, or $60-100 from a print service | STLs in the same repo |
| or: SO-101 kit instead of the two lines above | ~$250-500 | Seeed, WowRobo, PartaBot and others sell parts kits and pre-built arms. Pre-built saves about 4 hours |
| Logitech C920 (or any 1080p USB webcam with manual focus) | ~$60 | |
| Clamp arm with a 1/4-20 camera screw | ~$20 | It needs to hold still, and move when you nudge it |
| Pointer tips, 2 or 3 | ~$5 | 3D printed clip plus a 3 mm steel rod or a long bolt. Make one straight and one bent about 8 mm at the end |
| Tag prints | ~$10 | Laser printer, matte paper, glued to foam board or 2 mm acrylic. Keep them flat |
| Digital calipers | ~$20 | To measure the printed tag sizes |
| Soft stop (foam block) and a sturdy table | $0-10 | |
| **Total** | **~$250-300 DIY, ~$350-600 with a kit** | Plus a laptop with Python 3.10+ |

You only need the follower arm. Kintrace commands it directly, so no leader arm.

## 2. Build (about 2 weeks part time)

**Days 1-5: the arm.** Print and assemble the follower as in the LeRobot SO-101
guide (huggingface.co/docs/lerobot/so101). Then:

```bash
pip install 'lerobot[feetech]'
lerobot-find-port
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kintrace_arm
```

Remember the id. Kintrace uses the calibration file LeRobot saves for it.

**Days 6-7: pointer and tags.**

```bash
pip install -e '.[rig]'          # adds OpenCV with the aruco module
python -m kintrace rig init --port /dev/ttyACM0 --robot-id kintrace_arm
python -m kintrace rig tags --outdir tags
```

`rig tags` writes the tags and a ChArUco calibration board. Print at 100%.
Measure each black square with calipers and put the real sizes (meters) in
the `tags` block of `rig.json`. A 1% size error is a 1% depth error.

| Tag | ID | Size | Where |
|---|---|---|---|
| Wrist plate | 0 | 40 mm | Flat on the side of the gripper body that faces the camera. Glue it hard. Exact spot doesn't matter, commissioning measures it |
| Tip flag | 1 | 20 mm | A small flat flag at the very end of the pointer, facing the same way as the wrist tag. Kintrace treats the flag's center as the tool tip |
| Table | 10, 11, 12, 13 | 50 mm | Taped flat in a rough rectangle around the work area, about 25 cm apart, all in view, none under the arm's path |
| Target | 20 | 50 mm | Loose on the table in easy reach, in view. Used by the touch test |

Clip the pointer to the fixed jaw of the gripper. Kintrace holds the gripper
at a fixed value (`gripper_hold` in rig.json) and never opens it.

**Day 8: camera.** Put the webcam on the clamp arm about 40 cm to one side of
the arm and 40 cm up, looking down at the work area at about 45 degrees. Put
a rough guess of where it is in `rig.json` under `camera_guess` (meters, arm
base frame: x forward along the stretched-out arm, y to its left, z up).
Within 5 cm is fine.

Turn autofocus off and leave focus fixed. Kintrace sets `camera.focus` from
rig.json through OpenCV. If your OS ignores that, on Linux use
`v4l2-ctl -c focus_automatic_continuous=0 -c focus_absolute=0` (older kernels:
`focus_auto`). Refocusing changes the intrinsics.

**Day 9: camera calibration.**

```bash
python -m kintrace rig snap --outdir calib_images -n 25      # move the board between shots
python -m kintrace rig calibrate-camera --images calib_images --square-mm 30 --marker-mm 22
```

Use the measured square and marker sizes. Tilt the board 20 to 45 degrees
in different directions, near and far, and fill the corners of the image.
You want under 0.5 px reprojection error. A plain checkerboard works too:
`--chessboard 9x6 --square-mm 25` (inner corners, whole board in every photo).
This writes `camera.json`.

**Day 10: commission.** Start with the arm in a safe raised pose, clear of the
table, with the pointer on. Keep a hand near the power switch.

```bash
python -m kintrace rig commission --search-signs
```

The arm runs the check motion twice (about a minute each). The first run fits
the camera pose, the wrist plate mounting, the tip position and the joint
zeros. `--search-signs` also tries flipping joint directions, in case a
servo turns the other way from the URDF. The second run is saved as the
known-good state. Look at the printed fit: wrist tag under 1 mm rms is good,
over 3 mm means something is off (bad camera calibration, tag not flat,
wrong tag size).

**Days 11-14: practice the faults and the demo** (sections 4 and 5).

## 3. Everyday use

```bash
python -m kintrace rig check --record rec/today      # check, fix, verify, touch test, GO / NO-GO
python -m kintrace rig check --no-fix                # report only
python -m kintrace rig capture -o log.npz            # just record the check motion
python -m kintrace rig check --log log.npz           # analyze a saved capture, move nothing
```

`rig.json` plays the robot controller. The tool offset, camera calibration
and joint zero offsets live there, and Kintrace writes its fixes there. Each
save keeps the previous file as `rig.json.bak`.

Everything also runs with no hardware:

```bash
python -m kintrace rig commission --dry-run -c rig_dry.json
python -m kintrace rig check --dry-run -c rig_dry.json --fault camera_moved --seed 1
```

The dry run simulates the arm, the servo lag and 12-bit encoders, and renders
webcam frames of the tags. It is a best case: no motion blur, glare or
leftover lens distortion.

Tolerances are in `rig.json` under `tolerances`: tip 3 mm, command delay 8 ms,
touch miss 4 mm, and a floor on tag noise of 0.5 / 0.5 / 1.5 mm (x, y, depth).
These are guesses for a webcam and hobby servos. Tune them after the first
week of real runs.

## 4. Causing each fault safely

Rules for all of them: cause the fault with the arm still. Keep the work area
clear of hands while the arm moves. Keep the foam block under the work area.
Always keep a clean copy of `rig.json` and the LeRobot calibration file
(`cp` them to a `clean/` folder after commissioning).

| Fault | How to cause it | Size to use | What Kintrace should say | Fix it writes |
|---|---|---|---|---|
| Camera moved | Loosen the clamp, nudge the camera, tighten it | 1 cm and a degree or two. A firm tap on the clamp arm also works | Camera moved, N mm, N deg | New camera calibration in rig.json |
| Tool bent | Swap in the pre-bent pointer, or bend the rod by hand | 5-10 mm at the tip | Tool bent, fingertip moved N mm (x, y, z) | Tool offset set to the measured tip |
| Tool offset edited | Edit `tcp_offset` in rig.json by hand. It's in meters: change one number by 0.008 | 5-10 mm | Tool offset changed in controller, N mm | Tool offset restored to known-good |
| Servo zero shifted | Edit `homing_offset` for `elbow_flex` or `wrist_flex` in the LeRobot calibration file (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/kintrace_arm.json`, folder `so101_follower` in older LeRobot) by 30 to 60. On the next run LeRobot sees the servo and file disagree and asks: press ENTER to write the file to the servo | 30-60 ticks = 2.6-5.3 deg | Joint encoder zero drifted, J3 or J4 off by N deg | Joint zero offset in rig.json |
| Command latency | Add `--add-latency-ms 80` to the check. It holds every command back 80 ms, like a `sleep` in the control loop | 40-80 ms | Commands arriving late, +N ms. NO-GO, because no number fixes latency | None. It escalates |

Notes:
- The servo zero fault moves the arm about 1 to 2 cm from where it thinks it
  is. The planned check poses keep the tip about 6 cm or more above the table,
  and any move that would take the tip or wrist under 4 cm
  (`min_tip_height_m`) goes through a raised pose instead. That margin is
  from the URDF model, which puts the table at the bottom of the base. Watch
  the first slow run by eye. Don't use more than 60 ticks.
- `tc netem` only delays network traffic. The arm is on USB serial, so netem
  does nothing unless you run the control loop over a network (for example
  serial over TCP with ser2net, or a ROS 2 bridge). That path is not built or
  tested. Use `--add-latency-ms`.
- Put things back after each demo: restore `rig.json` and the LeRobot
  calibration file from `clean/`, straighten the camera to its tape marks,
  put the straight pointer back. Then run `rig check` once and expect GO.
  If it doesn't come back GO, re-run `rig commission`.

## 5. The 5-minute investor demo

Setup before people arrive: commissioned rig, GO on a check within the last
hour, straight pointer on, `clean/` copies ready, tape marks on the table
under the camera clamp. Font size up in the terminal.

| Time | Do | Say |
|---|---|---|
| 0:00 | Point at the arm, camera and tags | "This is a $300 arm and a $60 webcam. The tags are paper. Kintrace treats this like a robot cell in a factory." |
| 0:30 | `python -m kintrace rig check --no-fix` | "First a healthy check. The arm shows its wrist and tool to the camera from 8 angles." |
| 1:40 | Point at GO | "Every check agrees with known-good. GO." |
| 2:00 | Tap the camera clamp so the camera shifts about a centimeter | "Someone bumps the camera. This happens all the time in real cells. The robot has no idea. Its picks would now land a centimeter off." |
| 2:15 | `python -m kintrace rig check --record rec/demo` | "Now Kintrace runs the same check." |
| 3:20 | Point at "Camera moved: N mm, N deg" | "It says the camera moved, by how much, and that nothing else changed. The table tags tell a moved camera apart from a joint that slipped." |
| 3:30 | Arm re-runs the check, then touches the target | "It wrote the new camera calibration and is checking its own fix. Then it points at a target it finds with the camera, like a pick." |
| 4:40 | Point at GO and the record file | "GO, with a record of what changed and the fix. No one re-taught anything." |
| 5:00 | Stop | |

If there's time for a second fault, the servo zero edit is the most
convincing because the camera can't see anything wrong with the arm.

If a check fails live, say so plainly, show the record and move on. Don't
retry more than once in front of people.

## 6. Safety

- The SO-101 is light and its servos are weak, but the pointer is a rod.
  Wear glasses near it and keep faces out of the work area.
- The check motion is limited to 40 deg/s and each command step is capped
  at 4 deg. Faster (`speed_deg_s`) is untested.
- Ctrl-C stops the motion and holds the current position.
- When Kintrace exits, the arm keeps holding its last position (Kintrace asks
  LeRobot not to turn the torque off). Cut the power to make it limp. With an
  older LeRobot that lacks that option, the torque goes off on exit and the
  arm drops. The check ends back where it started, so start from a raised
  pose that is safe either way.
- Never cause a fault with the arm moving.

## 7. Tested vs not tested on hardware

**Tested (in software only):**
- SO-101 kinematics taken from the official URDF (`so101_new_calib.urdf`,
  TheRobotStudio/SO-ARM100). The UR5e path gives exactly the same numbers as
  before this change (`bench -n 4 --seed 1` and `incident`, compared line by line).
- Tag detection and solvePnP on rendered tags at a known pose: center within
  about 0.2-1.1 mm at 35-60 cm, corners within 1.2 mm (`tests/test_markers.py`).
- Checkerboard calibration on rendered photos: focal length within 2%.
- Full dry run: commission, capture, check, fix, re-check, touch test,
  GO / NO-GO, for all five faults (`tests/test_rig_dry_run.py`,
  `KINTRACE_SLOW=1` for all faults). In the dry runs a 9 to 14 mm camera
  bump is sized within about 0.5 mm and 0.1 deg, and the fix passes.

**Not tested at all on a real arm:**
- The `lerobot` connection in `kintrace/hw/so101.py`: import path, read and
  write timing at 50 Hz, the calibration prompt, torque behavior.
- Whether the URDF zero and joint directions match a real LeRobot
  calibration. Commissioning fits the zeros and `--search-signs` handles
  flipped joints, but that has only run in simulation.
- Real webcam behavior: motion blur, rolling shutter, auto exposure,
  glare on tags, lens distortion after calibration, USB frame latency.
- Real servo behavior: backlash, sag under gravity, load-dependent error,
  the servo's own position smoothing. The dry run models only a fixed 35 ms
  lag and 12-bit encoder steps. Gravity sag could look like a joint zero
  error on some poses.
- Whether the tolerances (tip 3 mm, 8 ms, touch 4 mm) avoid false alarms
  on a real desk.
- The time each check takes (about 64 s of motion planned, about 2.5 min
  for a check with a fix) and whether it can safely go faster.
- The latency fault via `tc netem`. Only `--add-latency-ms` is built.
- Camera calibration with `rig snap` on a real C920.

## 8. Files

- `kintrace/arms.py` SO-101 and UR5e arm models
- `kintrace/hw/markers.py` AprilTag detection, tag pose, camera intrinsics, calibration
- `kintrace/hw/so101.py` thin wrapper over LeRobot's SO-101 follower
- `kintrace/hw/capture.py` the check motion, webcam thread, log writer
- `kintrace/hw/rig.py` rig.json, tag geometry, check poses, commissioning fit
- `kintrace/hw/check.py` check, fix, verify, touch test, record
- `kintrace/hw/rigsim.py`, `render.py` the dry-run desk
- `kintrace/hw/cli_rig.py` the `kintrace rig` commands
