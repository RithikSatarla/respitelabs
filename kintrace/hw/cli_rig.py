"""`kintrace rig ...`: the SO-101 desk rig from the command line.

  kintrace rig init                     write a starting rig.json
  kintrace rig tags                     printable AprilTags and the calibration board
  kintrace rig snap                     photos of the board with the webcam
  kintrace rig calibrate-camera         camera intrinsics from those photos
  kintrace rig commission               learn the rig and its known-good state
  kintrace rig capture -o log.npz       run the check motion, write a Kintrace log
  kintrace rig check                    check, fix, verify, touch test, GO / NO-GO

Add --dry-run to commission, capture or check to use the simulated rig
(no arm, no camera). See HARDWARE.md.

OpenCV and lerobot are imported only inside the commands, so the rest of
the Kintrace CLI works without them.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..logio import Log

RIG_FAULTS = ["none", "camera_moved", "tool_bent", "tcp_config", "encoder_bias", "latency"]


def _rig_load(a, need_known_good=False):
    from . import rig

    if not os.path.exists(a.config):
        if a.cmd_rig == "commission" or getattr(a, "dry_run", False) and a.cmd_rig == "capture":
            cfg = rig.default_config()
        else:
            raise SystemExit(f"{a.config} not found. Run `kintrace rig init` (or `kintrace rig commission --dry-run`).")
    else:
        cfg = rig.load(a.config)
    dry = getattr(a, "dry_run", False)
    if cfg.get("dry_run") and not dry:
        raise SystemExit(f"{a.config} was made by a dry run. Use --dry-run, or commission the real rig into another file.")
    if dry and os.path.exists(a.config) and not cfg.get("dry_run"):
        raise SystemExit(f"{a.config} belongs to the real rig. Use another file for dry runs, e.g. -c rig_dry.json")
    if need_known_good and not cfg.get("known_good"):
        raise SystemExit(f"{a.config} has no known-good state yet. Run `kintrace rig commission` first.")
    cfg["dry_run"] = bool(dry)
    return cfg


def _rig_fault(a, cfg):
    """Dry-run fault: returns (fault dict, extra latency s). tcp_config is
    applied to cfg here, like someone editing rig.json."""
    if not a.dry_run:
        return {"type": "none"}, (a.add_latency_ms or 0.0) / 1000.0
    from . import rigsim

    f = rigsim.random_fault(a.fault, np.random.default_rng(a.seed))
    lat = (a.add_latency_ms or 0.0) / 1000.0
    if f["type"] == "tcp_config":
        cfg["tcp_offset"] = (np.array(cfg["tcp_offset"]) + f["delta"]).tolist()
    if f["type"] == "latency":
        lat += f["extra_s"]
    return f, lat


def _cmd_rig_init(a):
    from . import rig

    if os.path.exists(a.config) and not a.force:
        raise SystemExit(f"{a.config} exists (use --force to overwrite)")
    cfg = rig.default_config()
    cfg["port"] = a.port
    cfg["robot_id"] = a.robot_id
    rig.save(cfg, a.config)
    print(f"wrote {a.config}. Edit camera_guess (rough camera position, m, in the arm base frame) before commissioning.")


def _cmd_rig_tags(a):
    import cv2

    from . import markers, rig

    os.makedirs(a.outdir, exist_ok=True)
    t = rig.default_config()["tags"]
    sizes = {t["wrist_id"]: ("wrist", t["wrist_size"]), t["tip_id"]: ("tip", t["tip_size"]),
             t["target_id"]: ("target", t["target_size"])}
    sizes.update({i: (f"table{k}", t["table_size"]) for k, i in enumerate(t["table_ids"])})
    for tid, (name, size) in sizes.items():
        img = markers.marker_image(tid, 800)
        img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
        path = os.path.join(a.outdir, f"tag_{tid:02d}_{name}_{size*1000:.0f}mm.png")
        cv2.imwrite(path, img)
    cv2.imwrite(os.path.join(a.outdir, "charuco_7x5_30mm.png"), markers.board_image())
    print(f"wrote {len(sizes)} tags and the calibration board to {a.outdir}/")
    print("Print at 100% scale. The size in each file name is the black square's edge; measure it and put the real size in rig.json.")


def _cmd_rig_snap(a):
    import time

    import cv2

    from . import rig

    cfg = rig.load(a.config) if os.path.exists(a.config) else rig.default_config()
    c = cfg["camera"]
    cap = cv2.VideoCapture(c.get("index", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if c.get("focus") is not None:
        cap.set(cv2.CAP_PROP_FOCUS, c["focus"])
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {c.get('index', 0)}")
    os.makedirs(a.outdir, exist_ok=True)
    print(f"taking {a.n} photos, one every {a.every:.1f} s. Move the board between shots.")
    for i in range(a.n):
        t_end = time.time() + a.every
        while time.time() < t_end:
            cap.grab()
        ok, img = cap.read()
        if ok:
            cv2.imwrite(os.path.join(a.outdir, f"calib_{i:03d}.png"), img)
            print(f"  {i+1}/{a.n}")
    cap.release()


def _cmd_rig_calibrate_camera(a):
    import glob

    import cv2

    from . import markers

    files = sorted(glob.glob(os.path.join(a.images, "*.png")) + glob.glob(os.path.join(a.images, "*.jpg")))
    if not files:
        raise SystemExit(f"no .png or .jpg files in {a.images}")
    images = [cv2.imread(f) for f in files]
    if a.chessboard:
        cols, rows = (int(v) for v in a.chessboard.lower().split("x"))
        intr, n = markers.calibrate_chessboard(images, (cols, rows), a.square_mm / 1000.0)
    else:
        board = markers.charuco_board(square=a.square_mm / 1000.0, marker=a.marker_mm / 1000.0)
        intr, n = markers.calibrate(images, board)
    intr.save(a.out)
    print(f"wrote {a.out}: {n}/{len(files)} photos used, reprojection error {intr.rms_px:.2f} px")
    if intr.rms_px > 1.0:
        print("That's high (want under 0.5 px). Retake with the board flat, well lit and in focus.")


def _cmd_rig_commission(a):
    from . import check, rig

    cfg = _rig_load(a)
    parts = check.open_parts(cfg, a.dry_run, seed=a.seed, save_frames=a.save_frames)
    try:
        new, rep = check.commission(cfg, parts, search_signs=a.search_signs)
    finally:
        parts.close()
    if a.dry_run:
        new["dry_run_desk"] = int(a.seed)  # later dry runs use this same simulated desk
    rig.save(new, a.config)
    print(f"\nwrote {a.config} (old one kept as {a.config}.bak)")
    print(f"  normal command delay {rep['tau_ms']:.0f} ms, tag noise {np.round(rep['sigma_mm'], 2).tolist()} mm (x, y, depth)")
    print(f"  joint zeros added (deg): {np.round(rep['zeros_added_deg'], 2).tolist()}")
    if a.dry_run:
        print("  dry run: these numbers come from the simulated rig")


def _cmd_rig_capture(a):
    from . import capture, check

    cfg = _rig_load(a)
    fault, lat = _rig_fault(a, cfg)
    parts = check.open_parts(cfg, a.dry_run, seed=a.seed, fault=fault, save_frames=a.save_frames)
    try:
        log, stats = capture.capture(cfg, parts.arm, parts.camera, parts.clock, add_latency_s=lat, truth=fault)
    finally:
        parts.close()
    log.save(a.out)
    print(f"wrote {a.out}: {stats['frames']} still frames, wrist tag in {stats['wrist']}, "
          f"tip flag in {stats['tip']}, {stats['table_tags']}/{stats['table_needed']} table tags")
    for p in capture.check_stats(stats):
        print("  warning:", p)


def _cmd_rig_check(a):
    from . import check, rig

    cfg = _rig_load(a, need_known_good=True)
    if a.log:
        # offline: analyze a capture made earlier, change nothing
        r = check.analyze(Log.load(a.log), cfg)
        for f in r["findings"]:
            print(f"  - {f.label}: {f.size}")
        for c in r["checks"]:
            print(f"  [{'ok  ' if c.agrees else 'FAIL'}] {c.name:<28} {c.detail}")
        if not r["findings"]:
            print("  nothing changed")
        return
    fault, lat = _rig_fault(a, cfg)
    parts = check.open_parts(cfg, a.dry_run, seed=a.seed, fault=fault, save_frames=a.save_frames)
    try:
        r = check.run_check(cfg, parts, fix=not a.no_fix, add_latency_s=lat, touch=not a.no_touch,
                            log_path=a.out_log)
    finally:
        parts.close()
    txt = check.text_record(r)
    print("\n" + txt)
    if a.dry_run and fault["type"] != "none":
        print(f"\n(dry run, injected: {json.dumps(fault)})")
    if a.record:
        with open(a.record + ".txt", "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        with open(a.record + ".json", "w", encoding="utf-8") as f:
            f.write(check.record_json(r))
        print(f"record: {a.record}.txt / .json")
    if r["cfg_after"] is not cfg and not a.dry_run:
        rig.save(r["cfg_after"], a.config)
        print(f"fix saved to {a.config} (old one kept as {a.config}.bak)")
    elif r["cfg_after"] is not cfg:
        print("dry run: fix not saved (the simulated desk resets every run)")


def add_parser(sub):
    """Add `kintrace rig ...` to the main CLI's subparsers."""
    r = sub.add_parser("rig", help="SO-101 desk rig: commission, capture, check (see HARDWARE.md)")
    rs = r.add_subparsers(dest="cmd_rig", required=True)

    def common(s, dry=True):
        s.add_argument("-c", "--config", default="rig.json", help="rig settings file (default rig.json)")
        if dry:
            s.add_argument("--dry-run", action="store_true", help="simulated arm and camera, no hardware")
            s.add_argument("--seed", type=int, default=0, help="dry run: which fault and image noise (commission: which simulated desk)")
            s.add_argument("--save-frames", metavar="DIR", help="save the camera frames used")

    s = rs.add_parser("init", help="write a starting rig.json")
    common(s, dry=False)
    s.add_argument("--port", default="/dev/ttyACM0")
    s.add_argument("--robot-id", default="kintrace_arm", help="the id you gave lerobot-calibrate")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=_cmd_rig_init)

    s = rs.add_parser("tags", help="write printable AprilTags and the calibration board")
    s.add_argument("--outdir", default="tags")
    s.set_defaults(fn=_cmd_rig_tags)

    s = rs.add_parser("snap", help="take photos of the calibration board with the webcam")
    common(s, dry=False)
    s.add_argument("--outdir", default="calib_images")
    s.add_argument("-n", type=int, default=25)
    s.add_argument("--every", type=float, default=2.0, help="seconds between photos")
    s.set_defaults(fn=_cmd_rig_snap)

    s = rs.add_parser("calibrate-camera", help="camera intrinsics from board photos")
    s.add_argument("--images", default="calib_images")
    s.add_argument("--square-mm", type=float, default=30.0, help="measured checker square size")
    s.add_argument("--marker-mm", type=float, default=22.0, help="measured marker size inside a square")
    s.add_argument("--chessboard", metavar="COLSxROWS",
                   help="use a plain checkerboard instead of the ChArUco board, e.g. 9x6 inner corners")
    s.add_argument("-o", "--out", default="camera.json")
    s.set_defaults(fn=_cmd_rig_calibrate_camera)

    s = rs.add_parser("commission", help="learn the rig geometry and its known-good state (healthy rig)")
    common(s)
    s.add_argument("--search-signs", action="store_true", help="also try flipping joint directions (first setup)")
    s.set_defaults(fn=_cmd_rig_commission)

    for name, fn, hlp in [("capture", _cmd_rig_capture, "run the check motion and write a log"),
                          ("check", _cmd_rig_check, "check, fix, verify, touch test, GO / NO-GO")]:
        s = rs.add_parser(name, help=hlp)
        common(s)
        s.add_argument("--fault", choices=RIG_FAULTS, default="none", help="dry run only: fault to inject")
        s.add_argument("--add-latency-ms", type=float, default=0.0,
                       help="hold every command back this long (the latency fault, on real hardware too)")
        if name == "capture":
            s.add_argument("-o", "--out", default="rig_log.npz")
        else:
            s.add_argument("--log", help="analyze this capture instead of moving the arm")
            s.add_argument("--no-fix", action="store_true", help="report only, don't write the fix")
            s.add_argument("--no-touch", action="store_true", help="skip the touch test")
            s.add_argument("--out-log", help="also save the first capture here")
            s.add_argument("--record", help="write the record to RECORD.txt and RECORD.json")
        s.set_defaults(fn=fn)
