"""Multi-keypoint gripper detector for DROID: flange, hand centre and fingertip centre on the tool axis.

Labels come from FK through DROID's kept (GT) camera pose, like the single-keypoint labels. The model
is the final single-keypoint recipe (experiments.build) with one heatmap and one visibility output
per keypoint. Training follows the same heat rules (experiments.Gentle).

Run: python -m kintrace.droid.multikp labels | train | score --on val|test | refit
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import detector as D
from . import experiments as X

KP_NAMES = ("flange", "hand_centre", "fingertip_centre")
KP_OFFSETS = (0.0, 0.11, 0.17)      # metres along the tool axis from the flange
DATA = X.DATA
KP_DIR = os.path.join(DATA, "kp_labels")
CKPT = os.path.join(DATA, "exp", "F_multikp.pt")


# --------------------------------------------------------------------------
def build_labels():
    from . import episode as ep_mod
    from . import labels_fk

    os.makedirs(KP_DIR, exist_ok=True)
    man = {m["key"]: m for m in json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))}
    n_cam = 0
    for c in D.load_cameras(DATA):
        out = os.path.join(KP_DIR, f"{c['key']}_{c['serial']}.npz")
        if os.path.exists(out):
            n_cam += 1
            continue
        z = np.load(c["file"])
        m = man[c["key"]]
        ep = ep_mod.load(m["dir"], key=m["path"])
        T = ep.configured[c["serial"]]
        w, h = (int(v) for v in z["full_size"])
        uvs, viss = [], []
        for off in KP_OFFSETS:
            pc, _ = labels_fk.tip_in_camera(ep, c["serial"], T, c["n"], tip=np.array([0.0, 0.0, off]))
            uv, vis = labels_fk.project(z["K"], pc, (w, h))
            uvs.append(uv)
            viss.append(vis)
        uv_full = np.stack(uvs, 1).astype(np.float32)          # (N, 3, 2)
        np.savez_compressed(out, uv_full=uv_full, uv=(uv_full * (D.SMALL_W / w)).astype(np.float32),
                            visible=np.stack(viss, 1))
        n_cam += 1
    return n_cam


def load_kp(c):
    z = np.load(os.path.join(KP_DIR, f"{c['key']}_{c['serial']}.npz"))
    return z["uv"], z["uv_full"], z["visible"]


# --------------------------------------------------------------------------
def augment(x, uv, vis, torch):
    """X.augment (strong) with (B, n, 2) labels: the same random crop/shift moves every keypoint."""
    F = torch.nn.functional
    b = x.shape[0]
    dev = x.device
    s = torch.empty(b, device=dev).uniform_(0.85, 1.0)
    tx = torch.empty(b, device=dev).uniform_(-0.12, 0.12)
    ty = torch.empty(b, device=dev).uniform_(-0.05, 0.05)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, tx, ty
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                      padding_mode="border", align_corners=False)
    nx = (uv[..., 0] + 0.5) / D.SMALL_W * 2 - 1
    ny = (uv[..., 1] + 0.5) / D.SMALL_H * 2 - 1
    ox, oy = (nx - tx[:, None]) / s[:, None], (ny - ty[:, None]) / s[:, None]
    uv = torch.stack([(ox + 1) / 2 * D.SMALL_W - 0.5, (oy + 1) / 2 * D.SMALL_H - 0.5], -1)
    vis = vis & (ox.abs() < 1) & (oy.abs() < 1)
    x = x * torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3) + torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.3, 0.3)
    # colour gain, blur, random erasing: the strong part of X.augment, with labels unchanged
    x = x * torch.empty(b, 3, 1, 1, device=dev).uniform_(0.8, 1.2)
    k = torch.ones(3, 1, 3, 3, device=dev) / 9
    blur = torch.rand(b, device=dev) < 0.33
    if blur.any():
        x = x.clone()
        x[blur] = F.conv2d(F.pad(x[blur], (1, 1, 1, 1), mode="replicate"), k, groups=3)
    for i in torch.nonzero(torch.rand(b, device=dev) < 0.5).flatten().tolist():
        area = float(torch.empty(1).uniform_(0.05, 0.2)) * D.SMALL_W * D.SMALL_H
        bw = int(min(D.SMALL_W - 1, max(8, (area * float(torch.empty(1).uniform_(0.5, 2.0))) ** 0.5)))
        bh = int(min(D.SMALL_H - 1, max(8, area / bw)))
        x0, y0 = int(torch.randint(0, D.SMALL_W - bw, (1,))), int(torch.randint(0, D.SMALL_H - bh, (1,)))
        x[i, :, y0:y0 + bh, x0:x0 + bw] = torch.randn(3, bh, bw, device=dev) * 0.5
    return x, uv, vis


def train(logfile, every=2, epochs=12):
    import torch

    lad = json.load(open(os.path.join(X.EXP, "finalists.json")))
    cfg = dict(lad["E2_strong_aug"]["cfg"], keypoints=len(KP_OFFSETS), name="F_multikp", every=every, epochs=epochs)
    cams = D.load_cameras(DATA)
    sp = X.load_split()
    tr = X.cams_by(cams, sp["train_final"])
    cache = os.path.join(X.EXP, f"trainfinal_every{every}_fk.npy")   # same cameras and order as the finalists
    X.build_subset_cache(tr, every, cache, None)
    uv = np.concatenate([load_kp(c)[0][np.arange(0, c["n"], every)] for c in tr]).astype(np.float32)
    vis = np.concatenate([load_kp(c)[2][np.arange(0, c["n"], every)] for c in tr])
    assert len(uv) == np.load(cache, mmap_mode="r").shape[0]
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    dev = D.device()
    loader = torch.utils.data.DataLoader(D.FrameSet(cache, uv, vis), batch_size=X.BATCH, shuffle=True, num_workers=2,
                                         persistent_workers=True, drop_last=True, generator=torch.Generator().manual_seed(0))
    model = X.build(cfg).to(dev)
    groups = [dict(params=model.head_params(), lr=cfg["lr"])]
    bb = model.backbone_params()
    if bb:
        groups.append(dict(params=bb, lr=cfg["lr"] * cfg["backbone_lr_mult"]))
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * len(loader),
                                                pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    gentle = X.Gentle(logfile)
    t0 = time.time()
    try:
        for ep in range(epochs):
            model.train()
            tot = n = 0
            for fr, u, v in loader:
                gentle.step(f"multikp epoch {ep + 1}")
                x = D._to_input(fr, dev)
                u, v = u.to(dev), v.to(dev)
                x, u, v = augment(x, u, v, torch)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    hm, vl = model(x)
                b, k = hm.shape[:2]
                pred, p = X.soft_argmax(hm.reshape(b * k, *hm.shape[2:]))
                uf, vf = u.reshape(b * k, 2), v.reshape(b * k)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(vl.float(), v.float())
                if vf.any():
                    tgt = X.gauss_target(uf[vf], p.shape[1], p.shape[2], cfg["sigma_cells"], torch)
                    loss = loss - (tgt * torch.log(p[vf].clamp_min(1e-9))).sum((1, 2)).mean() \
                        + 0.05 * (pred[vf] - uf[vf]).abs().sum(1).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * len(fr)
                n += len(fr)
            X.log(f"  multikp epoch {ep + 1}/{epochs} loss {tot / n:.3f} ({time.time() - t0:.0f} s)", logfile)
            torch.save(dict(state=model.state_dict(), cfg=cfg, epoch=ep + 1), CKPT)
    except D.Overheat as e:
        torch.save(dict(state=model.state_dict(), cfg=cfg, epoch=ep), CKPT)
        k = X.record_heat_stop("F_multikp", str(e))
        X.log(f"HEAT STOP {k}: {e}; checkpoint {CKPT}", logfile)
        raise
    return gentle.readings


# --------------------------------------------------------------------------
def predict(model, frames, gentle, label):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    uvs, ps, peaks = [], [], []
    with torch.no_grad():
        for i in range(0, len(frames), X.BATCH):
            if gentle is not None:
                gentle.step(label)
            x = D._to_input(frames[i:i + X.BATCH], dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                hm, vl = model(x)
            b, k = hm.shape[:2]
            uv, p = X.soft_argmax(hm.reshape(b * k, *hm.shape[2:]))
            uvs.append(uv.reshape(b, k, 2).cpu().numpy())
            ps.append(torch.sigmoid(vl.float()).cpu().numpy())
            peaks.append(p.reshape(b, k, -1).max(-1).values.cpu().numpy())
    return np.concatenate(uvs), np.concatenate(ps), np.concatenate(peaks)


def score(on: str, logfile: str) -> dict:
    """Per-keypoint error on validation or test (original FK labels), plus the per-episode offset; caches predictions."""
    model, _ = X.load_model(CKPT)
    cams = X.cams_by(D.load_cameras(DATA), X.load_split()[on])
    gentle = X.Gentle(logfile)
    errs = {n: [] for n in KP_NAMES}
    offsets = {n: [] for n in KP_NAMES}
    cache = {}
    for c in cams:
        uv, p, peak = predict(model, D.frames_of(c), gentle, f"multikp score {on}")
        _, lab, vis = load_kp(c)
        s = c["full_w"] / D.SMALL_W
        cache[f"{c['key']}|{c['serial']}"] = dict(uv_full=uv * s, p=p, peak=peak)
        for j, n in enumerate(KP_NAMES):
            v = vis[:, j]
            if v.sum() < 5:
                continue
            d = uv[v, j] * s - lab[v, j]
            errs[n].append(np.linalg.norm(d, axis=1))
            if v.sum() >= 30:
                offsets[n].append(float(np.linalg.norm(np.median(d, 0))))
    res = dict(on=on, cameras=len(cams), heat=gentle.readings)
    for n in KP_NAMES:
        e = np.concatenate(errs[n])
        res[n] = dict(frames=int(len(e)), median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                      within_12=float((e <= 12).mean()), episode_offset_median=float(np.median(offsets[n])))
    np.save(os.path.join(X.EXP, f"multikp_{on}_preds.npy"), np.array([cache], dtype=object), allow_pickle=True)
    json.dump(res, open(os.path.join(X.EXP, f"multikp_{on}.json"), "w"), indent=1)
    return res


# --------------------------------------------------------------------------
def refit(on: str = "val") -> dict:
    """6-DoF camera refit from all keypoints: fit on the first half of each episode's frames, score the second half."""
    import cv2

    from .. import arms
    from . import episode as ep_mod
    from .extrinsics import pose_error

    preds = np.load(os.path.join(X.EXP, f"multikp_{on}_preds.npy"), allow_pickle=True)[0]
    man = {m["key"]: m for m in json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))}
    rows = []
    for c in X.cams_by(D.load_cameras(DATA), X.load_split()[on]):
        pr = preds[f"{c['key']}|{c['serial']}"]
        z = np.load(c["file"])
        _, lab, vis = load_kp(c)
        ok = vis.all(1) & (pr["p"] >= 0.5).all(1)
        fr = np.flatnonzero(ok)
        if len(fr) < 30:
            continue
        m = man[c["key"]]
        ep = ep_mod.load(m["dir"], key=m["path"])
        T = ep.configured[c["serial"]]
        q = np.stack([np.interp(z["t"][fr], ep.t, ep.q[:, j]) for j in range(7)], 1)
        F = arms.PANDA.fk(q)
        P = np.stack([np.einsum("nij,j->ni", F[:, :3, :3], np.array([0, 0, o])) + F[:, :3, 3] for o in KP_OFFSETS], 1)
        uv = pr["uv_full"][fr]
        h = len(fr) // 2
        Ti = np.linalg.inv(T)
        rv, _ = cv2.Rodrigues(Ti[:3, :3])
        tv = Ti[:3, 3].reshape(3, 1).copy()
        Pa, ua = P[:h].reshape(-1, 3).astype(np.float64), uv[:h].reshape(-1, 2).astype(np.float64)
        okp, rv, tv, inl = cv2.solvePnPRansac(Pa, ua, z["K"], None, rv.copy(), tv, useExtrinsicGuess=True,
                                              reprojectionError=30.0, iterationsCount=300)
        if not okp or inl is None or len(inl) < 12:
            continue
        inl = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(Pa[inl], ua[inl], z["K"], None, rv, tv)
        R, _ = cv2.Rodrigues(rv)
        Tb2c = np.eye(4)
        Tb2c[:3, :3], Tb2c[:3, 3] = R, tv.ravel()
        Tfit = np.linalg.inv(Tb2c)

        def proj(Tc, pts):
            Tci = np.linalg.inv(Tc)
            pc = pts @ Tci[:3, :3].T + Tci[:3, 3]
            u = pc @ z["K"].T
            return u[..., :2] / u[..., 2:3]

        Pb, ub = P[h:], uv[h:]
        eb, ea = np.linalg.norm(proj(T, Pb) - ub, axis=-1), np.linalg.norm(proj(Tfit, Pb) - ub, axis=-1)
        ob = np.linalg.norm(np.median((proj(T, Pb) - ub).reshape(-1, 2), 0))
        oa = np.linalg.norm(np.median((proj(Tfit, Pb) - ub).reshape(-1, 2), 0))
        pe = pose_error(T, Tfit)
        rows.append(dict(key=c["key"], serial=c["serial"], frames=int(len(fr)), median_before=float(np.median(eb)),
                         median_after=float(np.median(ea)), offset_before=float(ob), offset_after=float(oa),
                         refit_mm=pe["translation_mm"], refit_deg=pe["rotation_deg"]))
    g = lambda k: float(np.median([r[k] for r in rows])) if rows else None  # noqa: E731
    res = dict(on=on, cameras=len(rows), median_before=g("median_before"), median_after=g("median_after"),
               offset_before=g("offset_before"), offset_after=g("offset_after"), refit_mm=g("refit_mm"),
               refit_deg=g("refit_deg"), rows=rows)
    json.dump(res, open(os.path.join(X.EXP, f"multikp_refit_{on}.json"), "w"), indent=1)
    return res


def build_caches(logfile):
    """Phase-2-style caches (cache_<group>_kp3.npy) for val, test and Pred cameras with three-keypoint
    detections: val/test from the scoring run, Pred cameras run through the detector here (gentle GPU)."""
    from . import labels_fk
    from .phase2 import OUT, load_cache

    model = None
    gentle = None
    for group in ("val", "test", "pred"):
        rows = []
        pre = None if group == "pred" else np.load(os.path.join(X.EXP, f"multikp_{group}_preds.npy"), allow_pickle=True)[0]
        for c in load_cache(group):
            c = dict(c)
            n = len(c["p"])
            if pre is not None:
                r = pre[f"{c['key']}|{c['serial']}"]
                uv3, p3 = r["uv_full"][:n], r["p"][:n]
            else:
                if model is None:
                    model, _ = X.load_model(CKPT)
                    gentle = X.Gentle(logfile)
                m = next(m for m in json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8")) if m["key"] == c["key"])
                from . import episode as ep_mod
                ep = ep_mod.load(m["dir"], key=m["path"])
                small, (w, h) = labels_fk.read_small(ep.video(c["serial"]))
                uv, p, _ = predict(model, small[:n], gentle, "kp3 cache pred")
                uv3, p3 = uv * (w / D.SMALL_W), p
            c["uv3"], c["p3"] = uv3, p3
            rows.append(c)
        np.save(os.path.join(OUT, f"cache_{group}_kp3.npy"), np.array(rows, dtype=object), allow_pickle=True)
        X.log(f"kp3 cache {group}: {len(rows)} cameras", logfile)
    if gentle is not None:
        X.log(f"heat readings: {gentle.readings}", logfile)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.multikp")
    ap.add_argument("what", choices=("labels", "train", "score", "refit", "caches"))
    ap.add_argument("--on", default="val", choices=("val", "test"))
    a = ap.parse_args(argv)
    logfile = os.path.join(DATA, "runs", f"multikp_{time.strftime('%Y%m%d')}.log")
    if a.what == "labels":
        print("keypoint labels for", build_labels(), "cameras")
    elif a.what == "train":
        if not X.gpu_allowed():
            raise SystemExit("2 heat stops this session: no GPU jobs")
        train(logfile)
    elif a.what == "score":
        r = score(a.on, logfile)
        print(json.dumps({k: v for k, v in r.items() if k != "heat"}, indent=1))
    elif a.what == "caches":
        build_caches(logfile)
    else:
        r = refit(a.on)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
