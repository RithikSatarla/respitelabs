# Kintrace: notes for Claude Code

## What this is
Kintrace is physical observability for robot arms. It checks that a robot is
physically what its software thinks it is (camera, tool, joint zeros, command
timing). After a crash or a change it finds when it changed, what moved and by
how much, writes the fix, re-checks with test picks, and signs a GO / NO-GO
record. `kintrace watch` gives an always-on health score that warns before picks fail.

Line: "Ember watches the code. Kintrace watches the body."
Competitors: see COMPETITORS.md.

## State today
- Everything runs on a simulated UR5e-class pick cell (`kintrace/sim.py`).
- Measured in sim: 50/50 incidents right, 120/120 faults vs 83/120 for a dashboard baseline, 19/20 slow drifts flagged before the first miss.
- SO-101 desk rig code exists in `kintrace/hw/` and runs in `--dry-run`. It has NOT run on a real arm.
- `kintrace/readers/` reads LeRobot datasets, ROS 1/2 bags, MCAP and UR RTDE into logs; `kintrace check` runs one log with no baseline (single-session mode).
- `kintrace/droid/` turns DROID (real Franka data, corrected camera extrinsics as ground truth) into labels and injected-fault runs. `detect_gripper()` is a stub: no markerless detector yet.
- Company name is Respite Labs (respitelabs.net). Kintrace stays the product / CLI name.
- Website in `docs/` (main site, `docs/play/` prototype, `docs/demo/`).
- Run tests: `python -m pytest tests`.

## Rules
- Never claim a result we haven't measured. Sim numbers say "in simulation".
- Keep UR5e sim outputs identical when changing code. Re-run `python -m kintrace bench -n 4` and `python -m kintrace incident --outdir /tmp/check` before and after.
- Writing (README, site, docs, commits): short, plain, human. No em dashes. No buzzwords.
- Don't commit keys, `*.pem` or ledger files.
- Commit messages: plain, present tense, no AI attribution lines.

## The plan, in order
Do one step at a time. Finish it, show the result, then wait for me before the next.

1. **Repo live.** git init, first commit, push to my GitHub. Replace the placeholder GitHub link in `docs/index.html` and `docs/play/index.html` with the real repo URL. Add a GitHub Actions workflow that runs the tests on push.
2. **Website live.** Turn on GitHub Pages from `/docs` on `main`. Check it on desktop and phone. Add a simple contact form (Tally or Formspree, free tier) next to the email. Add privacy-friendly page analytics if free. Tell me the live URL.
3. **Real-log reader.** Done: `kintrace/readers/` (LeRobot, ROS bag / MCAP, UR RTDE), `kintrace import`, `kintrace check`. Next: a markerless wrist detector so recordings without an AprilTag work (`kintrace/droid/convert.py: detect_gripper`).
4. **Desk rig.** Walk me through HARDWARE.md on the real SO-101: camera calibration, commissioning, a healthy baseline, then each fault on purpose. Fix whatever breaks on real hardware. Save every run's record in `rig_runs/`. Measure honestly and write the results in README under "Real arm".
5. **Demo video.** Help me script and cut a 60 second video: healthy, bump the camera, run Kintrace, fix, GO. Put it on the site's hero and in `docs/play/`.
6. **Pitch update.** Update the numbers on the website and in `pitch/one-pager.md` with the real-arm results. Keep sim and real numbers clearly separate.
