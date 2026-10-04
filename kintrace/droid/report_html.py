"""kintrace report: one self-contained HTML page for one DROID camera in one episode.

    python -m kintrace report <episode folder> --camera SERIAL [--out report.html]

The page shows, for a strip of frames, where the keypoints should be (from the joints and the
camera calibration), where the detector sees them, and where they land after the fix; then what
moved with error bars, the fix, and the residual before vs after on held-out frames. If DROID
re-solved this camera, its pose is shown as a reference. Images are embedded, so the file can be
emailed as is. The detector runs on the CPU.
"""
from __future__ import annotations

import base64
import glob
import html
import io
import json
import os

import numpy as np

from .. import arms
from . import experiments as X
from . import fixes, labels_fk, multikp
from .extrinsics import Corrected, pose_error

DATA = "data/droid"
NAMES = ("flange", "hand centre", "fingertip")
STRIP_DEBUG: list = []   # last strip's frames, for checking the page by eye


def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _b64_jpg(img) -> str:
    import cv2
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


def _project(T, K, q, off):
    from .check2d import fk_tips, project
    return project(T, K, fk_tips(q, tip=np.array([0, 0, off])))


def build(ep_dir: str, serial: str, out: str, model_path: str = "data/droid/detector_final.pt") -> dict:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import episode as ep_mod

    ep = ep_mod.load(ep_dir)
    meta_file = sorted(glob.glob(os.path.join(ep_dir, "metadata_*.json")))[0]
    key = os.path.basename(meta_file)[9:-5]
    intr = json.load(open(os.path.join(DATA, "intrinsics.json"), encoding="utf-8"))
    video = ep.video(serial)
    small, (w, h) = labels_fk.read_small(video)
    K = labels_fk.intrinsics_for(intr, key, serial, (w, h))
    n = min(len(small), len(ep.t))
    model, _ = X.load_model(model_path)
    model = model.to("cpu")
    uv, p, _ = multikp.predict(model, small[:n], None, "report")
    uv = uv * (w / 320)
    tc = ep.t_cam.get(serial)
    t = tc[:n] if tc is not None and len(tc) >= n else ep.t[:n]
    q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(7)], 1)
    T_bel = ep.configured[serial]
    c = dict(q=q, K=K, uv3=uv, p3=p, T_belief=T_bel)
    k = fixes.frames_ok(c)
    if len(k) < 30:
        raise SystemExit(f"can't tell: only {len(k)} frames with at least 2 confident keypoints")
    widen = json.load(open(fixes.WIDEN))
    T_fix = fixes.fit(c, k, T_bel)
    hlf = len(k) // 2
    T_half = fixes.fit(c, k[:hlf], T_bel)
    r_before = fixes.residual(c, T_bel, k[hlf:])
    r_after = fixes.residual(c, T_half, k[hlf:])
    go_px = fixes.healthy_level()
    status = "GO" if np.median(r_after) <= go_px and np.median(r_before) <= go_px else (
        "NO-GO, fix ready" if np.median(r_after) <= go_px else "NO-GO")
    moved = pose_error(T_bel, T_fix)
    ci = fixes.bootstrap(c, k, T_bel, np.random.default_rng(0), n=100) or dict(translation_mm=(np.nan, np.nan), rotation_deg=(np.nan, np.nan))
    lo_t, hi_t = max(0.0, ci["translation_mm"][0] - widen["translation_mm"]), ci["translation_mm"][1] + widen["translation_mm"]
    lo_r, hi_r = max(0.0, ci["rotation_deg"][0] - widen["rotation_deg"]), ci["rotation_deg"][1] + widen["rotation_deg"]
    droid = (Corrected.load(DATA).lookup(key) or {}).get(serial)
    meta = Corrected.load(DATA).meta(key)

    # frame strip: 4 frames spread over the episode
    picks = k[np.linspace(0, len(k) - 1, 4).astype(int)]
    cap = cv2.VideoCapture(video)
    frames, want, i = {}, set(int(x) for x in picks), 0
    while len(frames) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f
        i += 1
    cap.release()
    strip = []
    for fi in picks:
        img = frames[int(fi)].copy()
        for j, off in enumerate((0.0, 0.11, 0.17)):
            e = _project(T_bel, K, q[fi:fi + 1], off)[0]
            fx = _project(T_fix, K, q[fi:fi + 1], off)[0]
            cv2.circle(img, (int(e[0]), int(e[1])), 11, (60, 60, 230), 2)          # expected, calibration (red)
            cv2.circle(img, (int(fx[0]), int(fx[1])), 7, (230, 140, 40), 2)        # expected, after the fix (blue)
            if p[fi, j] >= 0.5:
                d = uv[fi, j]
                cv2.drawMarker(img, (int(d[0]), int(d[1])), (70, 200, 70), cv2.MARKER_TILTED_CROSS, 18, 3)  # detected (green)
        # crop around the hand the detector sees; widen it (up to the full frame) so the calibration's
        # expected point is in view too
        exp_c = _project(T_bel, K, q[fi:fi + 1], 0.11)[0]
        det_c = uv[fi, 1] if p[fi, 1] >= 0.5 else exp_c
        pts = np.array([exp_c, det_c])
        half_w = float(np.clip(max(320, (np.ptp(pts[:, 0]) / 2 + 120)), 320, w / 2))
        half_h = half_w * 9 / 16
        cx, cy = pts.mean(0)
        x0 = int(np.clip(cx - half_w, 0, w - 2 * half_w))
        y0 = int(np.clip(cy - half_h, 0, h - 2 * half_h))
        crop = img[y0:y0 + int(2 * half_h), x0:x0 + int(2 * half_w)]
        strip.append(_b64_jpg(cv2.resize(crop, (480, 270))))
        STRIP_DEBUG.append(cv2.resize(crop, (480, 270)))

    fig, ax = plt.subplots(figsize=(7, 2.6))
    tt = t[k] - t[k][0]
    ax.plot(tt, fixes.residual(c, T_bel, k), lw=1, color="#d9534f", label="with the current calibration")
    ax.plot(tt, fixes.residual(c, T_fix, k), lw=1, color="#2a7ae2", label="with the fix")
    ax.axhline(go_px, color="k", ls="--", lw=0.8, label=f"healthy level ({go_px:.0f} px)")
    ax.axvline(tt[hlf], color="#999", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("time in episode (s)")
    ax.set_ylabel("keypoint error (px)")
    ax.legend(fontsize=7, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    chart = _b64_png(fig)
    plt.close(fig)

    droid_rows = ""
    if droid is not None:
        b, a = pose_error(T_bel, droid), pose_error(T_fix, droid)
        droid_rows = (f"<tr><td>DROID's re-solved pose ({html.escape(str(meta.get('source', '')))})</td>"
                      f"<td>calibration was {b['translation_mm']:.0f} mm / {b['rotation_deg']:.1f}&deg; from it</td>"
                      f"<td>the fix is {a['translation_mm']:.0f} mm / {a['rotation_deg']:.1f}&deg; from it</td></tr>")
    colour = {"GO": "#1e8e3e", "NO-GO": "#c5221f"}.get(status.split(",")[0], "#c5221f")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kintrace camera check, {html.escape(serial)}</title>
<style>
:root {{ --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --bg:#ffffff; --card:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e6edf3; --muted:#9198a1; --line:#3d444d; --bg:#0d1117; --card:#151b23; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:28px 0 8px; }}
.muted {{ color:var(--muted); }}
.status {{ display:inline-block; padding:4px 12px; border-radius:999px; color:#fff; background:{colour}; font-weight:600; margin:12px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
.grid img {{ width:100%; border-radius:6px; border:1px solid var(--line); display:block; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.card b {{ display:block; font-size:20px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }} td {{ border-top:1px solid var(--line); padding:8px 6px; vertical-align:top; }}
.key span {{ margin-right:14px; white-space:nowrap; }} img.chart {{ width:100%; max-width:760px; }}
</style></head><body><main>
<h1>Camera check: {html.escape(serial)}</h1>
<div class="muted">DROID episode {html.escape(ep.meta.get('_key') or os.path.basename(ep_dir))}, {html.escape(ep.meta.get('lab', ''))} Franka Panda.
Real recording; the check ran on the recorded joints and video.</div>
<div class="status">{html.escape(status)}</div>

<h2>What the camera sees</h2>
<div class="key muted"><span style="color:#e64646">&#9711; where the calibration says the hand is</span>
<span style="color:#46c846">&#10005; where the detector sees it</span><span style="color:#2a8ce6">&#9711; after the fix</span></div>
<div class="grid">{''.join(f'<img alt="frame" src="{s}">' for s in strip)}</div>

<h2>What moved</h2>
<div class="cards">
<div class="card"><span class="muted">camera position</span><b>{moved['translation_mm']:.0f} mm</b>
<span class="muted">90% range {lo_t:.0f} to {hi_t:.0f} mm</span></div>
<div class="card"><span class="muted">camera angle</span><b>{moved['rotation_deg']:.1f}&deg;</b>
<span class="muted">90% range {lo_r:.1f} to {hi_r:.1f}&deg;</span></div>
<div class="card"><span class="muted">error before / after the fix (held-out frames)</span>
<b>{np.median(r_before):.0f} px &rarr; {np.median(r_after):.0f} px</b><span class="muted">healthy level {go_px:.0f} px</span></div>
</div>

<h2>Before and after</h2>
<img class="chart" alt="error over time" src="{chart}">
<div class="muted">The fix is fitted on the first half of the episode (left of the grey line) and checked on the second half.</div>

<h2>The fix</h2>
<table>
<tr><td>Fix</td><td colspan="2">replace the camera calibration with the fitted pose (6-DoF, from three keypoints on the hand
over {len(k)} frames)</td></tr>
{droid_rows}
<tr><td>How sure</td><td colspan="2">90% ranges are calibrated on held-out cameras (true size inside the range 92% / 93% of the
time on test). They are wide: one episode pins the camera to about a decimetre.</td></tr>
</table>
<p class="muted">Kintrace by Respite Labs. Detector: {html.escape(os.path.basename(model_path))}. Numbers are from this
recording; nothing here was run on a live arm.</p>
</main></body></html>"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w", encoding="utf-8").write(page)
    return dict(out=out, status=status, moved_mm=moved["translation_mm"], moved_deg=moved["rotation_deg"],
                before_px=float(np.median(r_before)), after_px=float(np.median(r_after)))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="kintrace report")
    ap.add_argument("episode")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", default="kintrace_report.html")
    a = ap.parse_args(argv)
    print(build(a.episode, a.camera, a.out))
    return 0
