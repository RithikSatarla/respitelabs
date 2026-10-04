"""Detector experiments on DROID: validation split, label rule, gentle GPU training, ladder.

Split
    test        the 8 Phase 4 serials (120 GT cameras). Never trained on, never used to choose.
    validation  3 of the remaining serials, picked at random (seed 1) among serials with at most
                20 cameras. Used for every comparison and every threshold.
    train       everything else.

Label visibility rule (decided before use; applied to training data, and to test only as a second,
labelled number). A frame counts as "gripper visible" only if
    1. the FK tip is in front of the camera and inside the image,
    2. it is at least BORDER_PX (full res) from the image border, and
    3. no point on the arm's own links (joint origins and link midpoints, joints 1 to 7) projects
       within OCCLUDE_PX of the tip while sitting at least OCCLUDE_M closer to the camera.

Ladder rule: each experiment adds one change to the best configuration so far. The change is kept
if it lowers the validation median by at least 3%, or keeps the median within 3% and lowers the
validation p90 by at least 10%.

Heat (gentle mode, every GPU job): REST_S after every batch, batch 16. Temperature read every 30 s.
Above 78 C the rest doubles (up to 1.6 s) until a reading is under 72 C. Above 85 C the job saves a
checkpoint and stops. After 2 heat stops in the session no more GPU jobs run.

Run: python -m kintrace.droid.experiments split|rule|ladder|final|test ...
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from .. import arms
from . import detector as D
from .eval_detector import PHASE4_TEST_SERIALS

DATA = "data/droid"
EXP = os.path.join(DATA, "exp")
SPLIT = os.path.join(DATA, "split.json")
RULE = os.path.join(DATA, "fk_labels", "visible_rule.npz")
HEAT_STOPS = os.path.join(EXP, "heat_stops.json")

BORDER_PX = 30.0
OCCLUDE_PX = 50.0
OCCLUDE_M = 0.05

REST_S = 0.05
REST_MAX_S = 1.6
HOT_REST_C = 78
COOL_REST_C = 72
HOT_STOP_C = 85
READ_S = 30
BATCH = 16


def log(msg, path=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------
# heat, gentle mode
# --------------------------------------------------------------------------
class Gentle:
    """Steady load: a rest after every batch, longer when hot, stop above HOT_STOP_C. No long pauses."""

    def __init__(self, logfile=None):
        self.rest = REST_S
        self.last = 0.0
        self.readings = []
        self.logfile = logfile

    def step(self, label: str):
        if time.time() - self.last >= READ_S:
            t = D.gpu_temp_c()
            self.last = time.time()
            self.readings.append((time.strftime("%H:%M:%S"), label, t))
            if t is None:
                raise D.Overheat("GPU temperature could not be read")
            if t > HOT_STOP_C:
                raise D.Overheat(f"GPU at {t} C (limit {HOT_STOP_C} C)")
            if t > HOT_REST_C:
                self.rest = min(self.rest * 2, REST_MAX_S)
            elif t < COOL_REST_C:
                self.rest = REST_S
            log(f"  [heat] {label}: {t} C, rest {1000 * self.rest:.0f} ms/batch", self.logfile)
        time.sleep(self.rest)

    def tick(self, label: str):
        """Same as step(); lets detector.predict_conf pace itself batch by batch."""
        self.step(label)


def heat_stops() -> list:
    return json.load(open(HEAT_STOPS)) if os.path.exists(HEAT_STOPS) else []


def record_heat_stop(job: str, err: str):
    s = heat_stops() + [dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), job=job, reading=err)]
    os.makedirs(EXP, exist_ok=True)
    json.dump(s, open(HEAT_STOPS, "w"), indent=1)
    return len(s)


def gpu_allowed() -> bool:
    return len(heat_stops()) < 2


# --------------------------------------------------------------------------
# split and label rule
# --------------------------------------------------------------------------
def make_split(cams) -> dict:
    test = set(PHASE4_TEST_SERIALS)
    count = {}
    for c in cams:
        if c["serial"] not in test:
            count[c["serial"]] = count.get(c["serial"], 0) + 1
    pool = sorted(s for s, n in count.items() if n <= 20)
    val = sorted(np.random.default_rng(1).choice(pool, size=3, replace=False).tolist())
    split = dict(test=sorted(test), val=val, train=sorted(s for s in count if s not in val),
                 rule="val: 3 serials at random (seed 1) among non-test serials with <= 20 cameras")
    json.dump(split, open(SPLIT, "w"), indent=1)
    return split


def load_split():
    return json.load(open(SPLIT))


def cams_by(cams, serials):
    s = set(serials)
    return [c for c in cams if c["serial"] in s]


def _link_points(q):
    """(N, 7) -> (N, P, 3) points on the Panda's links in the base frame (joint origins and midpoints)."""
    q = np.atleast_2d(q)
    n = len(q)
    theta = np.c_[q, np.zeros(n)]
    T = np.tile(np.eye(4), (n, 1, 1))
    origins = []
    for i in range(8):
        ct, st = np.cos(theta[:, i]), np.sin(theta[:, i])
        ca, sa = np.cos(arms.PANDA_ALPHA[i]), np.sin(arms.PANDA_ALPHA[i])
        A = np.zeros((n, 4, 4))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 3] = ct, -st, arms.PANDA_A[i]
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st * ca, ct * ca, -sa, -sa * arms.PANDA_D[i]
        A[:, 2, 0], A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = st * sa, ct * sa, ca, ca * arms.PANDA_D[i]
        A[:, 3, 3] = 1.0
        T = T @ A
        origins.append(T[:, :3, 3].copy())
    o = np.stack(origins[:7], 1)                       # joints 1..7
    mids = (o[:, 1:] + o[:, :-1]) / 2
    return np.concatenate([o, mids], 1)


def build_rule(data: str = DATA) -> dict:
    """Apply the visibility rule to every FK-labelled camera; save one boolean per frame."""
    from . import episode as ep_mod
    from .check2d import project

    manifest = {m["key"]: m for m in json.load(open(os.path.join(data, "manifest.json"), encoding="utf-8"))}
    out, why = {}, dict(in_view=0, border=0, occluded=0, kept=0)
    for c in D.load_cameras(data):
        z = np.load(c["file"])
        m = manifest[c["key"]]
        ep = ep_mod.load(m["dir"], key=m["path"])
        T = ep.configured[c["serial"]]
        t = z["t"][:c["n"]]
        q = np.stack([np.interp(t, ep.t, ep.q[:, j]) for j in range(7)], 1)
        w, h = (int(v) for v in z["full_size"])
        uv, vis = z["uv_full"], c["visible"]
        tip_depth = z["tip_cam"][:, 2]
        border = (uv[:, 0] < BORDER_PX) | (uv[:, 0] > w - BORDER_PX) | (uv[:, 1] < BORDER_PX) | (uv[:, 1] > h - BORDER_PX)
        pts = _link_points(q)                           # (N, P, 3) base frame
        Tinv = np.linalg.inv(T)
        pc = pts @ Tinv[:3, :3].T + Tinv[:3, 3]        # camera frame
        puv = pc @ z["K"].T
        puv = puv[..., :2] / np.clip(puv[..., 2:3], 1e-6, None)
        near = np.linalg.norm(puv - uv[:, None, :], axis=2) < OCCLUDE_PX
        closer = pc[..., 2] < (tip_depth[:, None] - OCCLUDE_M)
        occluded = (near & closer & (pc[..., 2] > 0)).any(1)
        keep = vis & ~border & ~occluded
        why["in_view"] += int(vis.sum())
        why["border"] += int((vis & border).sum())
        why["occluded"] += int((vis & ~border & occluded).sum())
        why["kept"] += int(keep.sum())
        out[f"{c['key']}|{c['serial']}"] = keep
    np.savez_compressed(RULE, **out)
    return why


def load_rule() -> dict:
    z = np.load(RULE)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build(cfg: dict):
    """ResNet-18 heatmap model. cfg: stride (4 or 2), pretrained, freeze ('stem+layer1' or 'none')."""
    import torch
    import torchvision
    from torch import nn
    F = torch.nn.functional

    class ExpModel(nn.Module):
        def __init__(self):
            super().__init__()
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if cfg.get("pretrained", True) else None
            r = torchvision.models.resnet18(weights=w)
            self.stride = cfg.get("stride", 4)
            self.stem1 = nn.Sequential(r.conv1, r.bn1, r.relu)   # stride 2, 64 ch
            self.pool = r.maxpool
            self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3
            self.r3 = nn.Conv2d(256, 128, 1)
            self.s2 = nn.Conv2d(128, 128, 1)
            self.s1 = nn.Conv2d(64, 128, 1)
            self.up2 = D._block(128, 128, 1)
            self.up1 = D._block(128, 64, 1)
            if self.stride == 2:
                self.s0 = nn.Conv2d(64, 64, 1)
                self.up0 = D._block(64, 32, 1)
                self.head = nn.Conv2d(32, cfg.get("keypoints", 1), 1)
            else:
                self.head = nn.Conv2d(64, cfg.get("keypoints", 1), 1)
            self.vis = nn.Linear(256, cfg.get("keypoints", 1))
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.frozen = [self.stem1, self.layer1] if cfg.get("freeze", "stem+layer1") == "stem+layer1" else []
            for m in self.frozen:
                for p in m.parameters():
                    p.requires_grad_(False)

        def backbone_params(self):
            return [p for m in (self.stem1, self.layer1, self.layer2, self.layer3) for p in m.parameters() if p.requires_grad]

        def head_params(self):
            ids = {id(p) for p in self.backbone_params()}
            return [p for p in self.parameters() if p.requires_grad and id(p) not in ids]

        def train(self, mode=True):
            super().train(mode)
            for m in self.frozen:
                m.eval()
            return self

        def forward(self, x):
            x = ((x * 0.25 + 0.45) - self.mean) / self.std
            l0 = self.stem1(x)
            l1 = self.layer1(self.pool(l0))
            l2 = self.layer2(l1)
            l3 = self.layer3(l2)
            y = self.up2(F.interpolate(self.r3(l3), size=l2.shape[-2:], mode="bilinear", align_corners=False) + self.s2(l2))
            y = self.up1(F.interpolate(y, size=l1.shape[-2:], mode="bilinear", align_corners=False) + self.s1(l1))
            if self.stride == 2:
                y = self.up0(F.interpolate(y, size=l0.shape[-2:], mode="bilinear", align_corners=False) + self.s0(l0))
            hm, vl = self.head(y), self.vis(l3.mean((2, 3)))
            return (hm[:, 0], vl[:, 0]) if hm.shape[1] == 1 else (hm, vl)   # (B, n, h, w), (B, n) for n keypoints

    return ExpModel()


def soft_argmax(logits):
    """Any heatmap size -> (u, v) at 320x180."""
    import torch
    b, h, w = logits.shape
    p = torch.softmax(logits.reshape(b, -1).float(), 1).reshape(b, h, w)
    xs = (torch.arange(w, device=p.device, dtype=p.dtype) + 0.5) * (D.SMALL_W / w) - 0.5
    ys = (torch.arange(h, device=p.device, dtype=p.dtype) + 0.5) * (D.SMALL_H / h) - 0.5
    return torch.stack([(p.sum(1) * xs).sum(1), (p.sum(2) * ys).sum(1)], 1), p


def gauss_target(uv, h, w, sigma_cells, torch):
    cw, ch = D.SMALL_W / w, D.SMALL_H / h
    xs = (torch.arange(w, device=uv.device) + 0.5) * cw - 0.5
    ys = (torch.arange(h, device=uv.device) + 0.5) * ch - 0.5
    gx = torch.exp(-((xs[None] - uv[:, :1]) / (sigma_cells * cw)) ** 2 / 2)
    gy = torch.exp(-((ys[None] - uv[:, 1:]) / (sigma_cells * ch)) ** 2 / 2)
    t = gy[:, :, None] * gx[:, None, :]
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-12)


def augment(x, uv, vis, strong: bool, torch):
    x, uv, vis = D._augment(x, uv, vis, torch)          # zoom-crop, shift, brightness/contrast
    if not strong:
        return x, uv, vis
    F = torch.nn.functional
    b = x.shape[0]
    dev = x.device
    # per-channel colour gain
    x = x * torch.empty(b, 3, 1, 1, device=dev).uniform_(0.8, 1.2)
    # blur on about a third of the batch (3x3 box, applied once or twice)
    k = torch.ones(3, 1, 3, 3, device=dev) / 9
    blur = torch.rand(b, device=dev) < 0.33
    if blur.any():
        xb = F.conv2d(F.pad(x[blur], (1, 1, 1, 1), mode="replicate"), k, groups=3)
        x = x.clone()
        x[blur] = xb
    # random erasing: one box, 5 to 20% of the image, on half the batch (occlusion)
    er = torch.rand(b, device=dev) < 0.5
    for i in torch.nonzero(er).flatten().tolist():
        area = float(torch.empty(1).uniform_(0.05, 0.2)) * D.SMALL_W * D.SMALL_H
        bw = int(min(D.SMALL_W - 1, max(8, (area * float(torch.empty(1).uniform_(0.5, 2.0))) ** 0.5)))
        bh = int(min(D.SMALL_H - 1, max(8, area / bw)))
        x0 = int(torch.randint(0, D.SMALL_W - bw, (1,)))
        y0 = int(torch.randint(0, D.SMALL_H - bh, (1,)))
        x[i, :, y0:y0 + bh, x0:x0 + bw] = torch.randn(3, bh, bw, device=dev) * 0.5
    return x, uv, vis


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def build_subset_cache(cams, every: int, path: str, rule: dict | None):
    """Frames every `every`-th frame of each camera into one .npy, a camera at a time. Returns uv, vis."""
    idx = [(c, np.arange(0, c["n"], every)) for c in cams]
    total = sum(len(i) for _, i in idx)
    uv = np.concatenate([c["uv"][i] for c, i in idx]).astype(np.float32)
    vis = np.concatenate([(rule[f"{c['key']}|{c['serial']}"] if rule else c["visible"])[i] for c, i in idx])
    if not (os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == total):
        part = path + ".part.npy"
        mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(total, D.SMALL_H, D.SMALL_W, 3))
        k = 0
        for c, i in idx:
            f = D.frames_of(c)[i]
            mm[k:k + len(f)] = f
            k += len(f)
        mm.flush()
        del mm
        os.replace(part, path)
    return uv, vis


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------
def train(cfg: dict, train_cams, cache: str, gentle: Gentle, ckpt: str, logfile=None):
    import torch

    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = False
    dev = D.device()
    rule = load_rule() if cfg.get("label_rule") else None
    uv, vis = build_subset_cache(train_cams, cfg.get("every", 3), cache, rule)
    loader = torch.utils.data.DataLoader(D.FrameSet(cache, uv, vis), batch_size=BATCH, shuffle=True, num_workers=2,
                                         persistent_workers=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(cfg.get("seed", 0)))
    model = build(cfg).to(dev)
    lr = cfg.get("lr", 2e-3)
    groups = [dict(params=model.head_params(), lr=lr)]
    bb = model.backbone_params()
    if bb:
        groups.append(dict(params=bb, lr=lr * cfg.get("backbone_lr_mult", 1.0)))
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    epochs = cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * len(loader),
                                                pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    sigma = cfg.get("sigma_cells", 1.5)
    t0 = time.time()
    try:
        for ep in range(epochs):
            model.train()
            tot = n = 0
            for fr, u, v in loader:
                gentle.step(f"{cfg['name']} epoch {ep + 1}")
                x = D._to_input(fr, dev)
                u, v = u.to(dev), v.to(dev)
                x, u, v = augment(x, u, v, cfg.get("strong_aug", False), torch)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    hm, vl = model(x)
                pred, p = soft_argmax(hm)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(vl.float(), v.float())
                if v.any():
                    tgt = gauss_target(u[v], p.shape[1], p.shape[2], sigma, torch)
                    loss = loss - (tgt * torch.log(p[v].clamp_min(1e-9))).sum((1, 2)).mean() \
                        + 0.05 * (pred[v] - u[v]).abs().sum(1).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * len(fr)
                n += len(fr)
            log(f"  {cfg['name']} epoch {ep + 1}/{epochs} loss {tot / n:.3f} ({time.time() - t0:.0f} s)", logfile)
            torch.save(dict(state=model.state_dict(), cfg=cfg, epoch=ep + 1), ckpt)
    except D.Overheat:
        torch.save(dict(state=model.state_dict(), cfg=cfg, epoch=ep), ckpt)
        raise
    return model


def predict(model, frames, gentle: Gentle | None, label: str):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = dict(uv=[], p=[], peak=[])
    with torch.no_grad():
        for i in range(0, len(frames), BATCH):
            if gentle is not None:
                gentle.step(label)
            x = D._to_input(frames[i:i + BATCH], dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                hm, vl = model(x)
            uv, p = soft_argmax(hm)
            out["uv"].append(uv.cpu().numpy())
            out["p"].append(torch.sigmoid(vl.float()).cpu().numpy())
            out["peak"].append(p.reshape(len(p), -1).max(1).values.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def score(model, cams, gentle, label, rule: dict | None = None, keep_preds: str | None = None) -> dict:
    """Errors on in-view frames (original FK labels); plus visible-only frames if a rule is given."""
    errs, errs_rule, ser, rows = [], [], [], {k: [] for k in ("key", "serial", "frame", "visible", "rule", "uv_full", "label_full", "p", "peak")}
    for c in cams:
        pr = predict(model, D.frames_of(c), gentle, label)
        s = c["full_w"] / D.SMALL_W
        e = np.linalg.norm(pr["uv"] * s - c["uv_full"], axis=1)
        v = c["visible"]
        errs.append(e[v])
        ser += [c["serial"]] * int(v.sum())
        r = rule[f"{c['key']}|{c['serial']}"] if rule else None
        if r is not None:
            errs_rule.append(e[r])
        if keep_preds:
            n = len(e)
            rows["key"] += [c["key"]] * n
            rows["serial"] += [c["serial"]] * n
            rows["frame"] += list(range(n))
            rows["visible"].append(v)
            rows["rule"].append(r if r is not None else v)
            rows["uv_full"].append(pr["uv"] * s)
            rows["label_full"].append(c["uv_full"])
            rows["p"].append(pr["p"])
            rows["peak"].append(pr["peak"])
    e = np.concatenate(errs)
    ser = np.asarray(ser)
    res = dict(cameras=len(cams), frames=int(len(e)), median=float(np.median(e)), mean=float(e.mean()),
               p90=float(np.percentile(e, 90)), p95=float(np.percentile(e, 95)),
               within={str(t): float((e <= t).mean()) for t in (5, 12, 25)},
               per_serial={s: float(np.median(e[ser == s])) for s in sorted(set(ser.tolist()))})
    if errs_rule:
        er = np.concatenate(errs_rule)
        res["visible_only"] = dict(frames=int(len(er)), median=float(np.median(er)), p90=float(np.percentile(er, 90)),
                                   within={str(t): float((er <= t).mean()) for t in (5, 12, 25)})
    if keep_preds:
        np.savez_compressed(keep_preds, **{k: (np.concatenate(v) if k not in ("key", "serial", "frame") else np.asarray(v))
                                           for k, v in rows.items()})
    return res


def load_model(path: str):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(dict(ck["cfg"], pretrained=False))
    m.load_state_dict(ck["state"])
    return m.to(D.device()), ck


# --------------------------------------------------------------------------
# ladder
# --------------------------------------------------------------------------
LADDER = [
    ("E0_baseline", {}),
    ("E1_stride2", dict(stride=2, sigma_cells=2.0)),
    ("E2_strong_aug", dict(strong_aug=True)),
    ("E3_unfreeze", dict(freeze="none", backbone_lr_mult=0.1)),
    ("E4_label_rule", dict(label_rule=True)),
]
BASE = dict(stride=4, sigma_cells=1.5, freeze="stem+layer1", backbone_lr_mult=1.0, strong_aug=False,
            label_rule=False, every=3, epochs=6, lr=2e-3, seed=0)


def better(new, best) -> bool:
    if new["median"] <= best["median"] * 0.97:
        return True
    return new["median"] <= best["median"] * 1.03 and new["p90"] <= best["p90"] * 0.90


def ladder(logfile: str):
    cams = D.load_cameras(DATA)
    sp = load_split()
    tr, va = cams_by(cams, sp["train"]), cams_by(cams, sp["val"])
    results_path = os.path.join(EXP, "ladder.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    best_cfg, best = dict(BASE), None
    log(f"ladder: train {len(tr)} cameras ({len(sp['train'])} serials), val {len(va)} cameras ({sp['val']})", logfile)
    for name, change in LADDER:
        if not gpu_allowed():
            log("2 heat stops this session: no more GPU jobs", logfile)
            break
        cfg = dict(best_cfg, **change, name=name)
        if name in results and "val" in results[name]:
            res = results[name]
        else:
            gentle = Gentle(logfile)
            ckpt = os.path.join(EXP, f"{name}.pt")
            cache = os.path.join(EXP, f"train_every{cfg['every']}_{'rule' if cfg['label_rule'] else 'fk'}.npy")
            try:
                model = train(cfg, tr, cache, gentle, ckpt, logfile)
                val = score(model, va, gentle, f"{name} val")
            except D.Overheat as e:
                k = record_heat_stop(name, str(e))
                log(f"HEAT STOP {k}: {e} during {name}; checkpoint {ckpt}", logfile)
                results[name] = dict(cfg=cfg, heat_stop=str(e))
                json.dump(results, open(results_path, "w"), indent=1)
                continue
            res = dict(cfg=cfg, val=val, heat=gentle.readings)
            results[name] = res
            json.dump(results, open(results_path, "w"), indent=1)
            del model
        v = res["val"]
        kept = best is None or better(v, best)
        log(f"{name}: val median {v['median']:.1f} px, p90 {v['p90']:.1f}, within 12 px {100 * v['within']['12']:.1f}% "
            f"-> {'kept' if kept else 'not kept'}", logfile)
        if kept:
            best_cfg, best = {k: res["cfg"][k] for k in BASE}, v
        results[name]["kept"] = kept
        json.dump(results, open(results_path, "w"), indent=1)
    results["_best_cfg"] = best_cfg
    json.dump(results, open(results_path, "w"), indent=1)
    log(f"best config: {best_cfg}", logfile)
    return results


def finalists(logfile: str, names: list, every: int = 2, epochs: int = 12) -> dict:
    """Train the chosen ladder configs longer on all training serials (train_final), score on validation.
    Test is scored separately, once per finalist (score-ckpt --on test)."""
    lad = json.load(open(os.path.join(EXP, "ladder.json")))
    cams = D.load_cameras(DATA)
    sp = load_split()
    tr, va = cams_by(cams, sp.get("train_final", sp["train"])), cams_by(cams, sp["val"])
    out_path = os.path.join(EXP, "finalists.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    log(f"finalists {names}: train {len(tr)} cameras ({len(sp.get('train_final', sp['train']))} serials), "
        f"every {every}th frame, {epochs} epochs", logfile)
    for name in names:
        if not gpu_allowed():
            log("2 heat stops this session: no more GPU jobs", logfile)
            break
        if name in out and "val" in out[name]:
            continue
        base = {k: lad[name]["cfg"][k] for k in BASE}
        cfg = dict(base, every=every, epochs=epochs, name=f"F_{name}")
        gentle = Gentle(logfile)
        ckpt = os.path.join(EXP, f"F_{name}.pt")
        cache = os.path.join(EXP, f"trainfinal_every{every}_{'rule' if cfg['label_rule'] else 'fk'}.npy")
        try:
            model = train(cfg, tr, cache, gentle, ckpt, logfile)
            val = score(model, va, gentle, f"F_{name} val", rule=load_rule(),
                        keep_preds=os.path.join(EXP, f"F_{name}_val_preds.npz"))
        except D.Overheat as e:
            k = record_heat_stop(f"F_{name}", str(e))
            log(f"HEAT STOP {k}: {e} during F_{name}; checkpoint {ckpt}", logfile)
            out[name] = dict(cfg=cfg, heat_stop=str(e))
            json.dump(out, open(out_path, "w"), indent=1)
            continue
        out[name] = dict(cfg=cfg, ckpt=ckpt, val=val, heat=gentle.readings)
        json.dump(out, open(out_path, "w"), indent=1)
        log(f"F_{name}: val median {val['median']:.1f} px, p90 {val['p90']:.1f}, within 12 px {100 * val['within']['12']:.1f}%",
            logfile)
        del model
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kintrace.droid.experiments")
    ap.add_argument("what", choices=("split", "rule", "ladder", "score-ckpt", "final"))
    ap.add_argument("--names", nargs="*", default=None, help="final: ladder entries to train longer")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--on", default="val", choices=("val", "test"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.makedirs(EXP, exist_ok=True)
    logfile = os.path.join(DATA, "runs", f"exp_{time.strftime('%Y%m%d')}.log")
    if a.what == "split":
        sp = make_split(D.load_cameras(DATA))
        cams = D.load_cameras(DATA)
        for k in ("train", "val", "test"):
            cs = cams_by(cams, sp[k])
            print(f"{k:5s}: {len(sp[k]):2d} serials, {len(cs):3d} cameras, {sum(c['n'] for c in cs)} frames  {sp[k] if k != 'train' else ''}")
    elif a.what == "rule":
        why = build_rule()
        print(f"visibility rule over all FK-labelled cameras: {why}")
    elif a.what == "ladder":
        ladder(logfile)
    elif a.what == "final":
        finalists(logfile, a.names)
    elif a.what == "score-ckpt":
        if not gpu_allowed():
            raise SystemExit("2 heat stops this session: no GPU jobs")
        cams = D.load_cameras(DATA)
        sp = load_split()
        cs = cams_by(cams, sp[a.on])
        gentle = Gentle(logfile)
        if a.ckpt.endswith("detector_resnet18_pretrained_frozen.pt"):
            model = D.load(a.ckpt)  # the epoch-50 model: scored through detector.predict_conf
            res = _score_old(model, cs, gentle, f"score {a.on}")
        else:
            model, _ = load_model(a.ckpt)
            res = score(model, cs, gentle, f"score {a.on}", rule=load_rule() if os.path.exists(RULE) else None,
                        keep_preds=a.out.replace(".json", "_preds.npz") if a.out else None)
        res["heat"] = gentle.readings
        print(json.dumps({k: v for k, v in res.items() if k != "heat"}, indent=1))
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


def _score_old(model, cams, gentle, label):
    errs = []
    for c in cams:
        frames = D.frames_of(c)
        uvs = []
        for i in range(0, len(frames), BATCH):
            gentle.step(label)
            uv, _, _, _ = D.predict_conf(model, frames[i:i + BATCH])
            uvs.append(uv)
        e = np.linalg.norm(np.concatenate(uvs) * (c["full_w"] / D.SMALL_W) - c["uv_full"], axis=1)
        errs.append(e[c["visible"]])
    e = np.concatenate(errs)
    return dict(cameras=len(cams), frames=int(len(e)), median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                within={str(t): float((e <= t).mean()) for t in (5, 12, 25)})


if __name__ == "__main__":
    main()
