"""Markerless Robotiq 2F-85 fingertip detector for DROID fixed cameras.

A small heatmap network: four strided conv blocks down to 1/16, a light
decoder back to an 80x45 heatmap, soft-argmax to (u, v) at 320x180, and a
visibility logit from the deepest block. Trained on the FK labels from
labels_fk.py (DROID GT cameras only), on CUDA when available.

Held-out cameras are split by serial: every camera of a held-out serial is
left out of training, so the test is on rooms and viewpoints the model has
not seen.

Heat: this runs on a laptop GPU. Training goes in runs of RUN_EPOCHS epochs
with a temperature check and a pause after every run; inference checks every
CHECK_S seconds. Above WARM_C it waits until the GPU is below RESUME_C.
Above HOT_C (or if the temperature cannot be read) it saves a checkpoint,
writes PROGRESS.md and exits; it does not resume on its own.

Run: python -m kintrace.droid train-detector --data data/droid [--wide]
Writes data/droid/detector.pt (or detector_wide.pt) and *_eval.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

SMALL_W, SMALL_H = 320, 180
HM_W, HM_H = 80, 45
STRIDE = SMALL_W // HM_W  # 4

RUN_EPOCHS = 5      # epochs per run
PAUSE_S = 60        # pause after every run, whatever the temperature
WARM_C = 75         # above this, wait ...
RESUME_C = 70       # ... until below this
WAIT_S = 180
HOT_C = 85          # above this, save and stop for good
CHECK_S = 30        # inference: check the temperature this often
WATCH_S = 30        # training: read this often; pause above WARM_C until below RESUME_C, stop above HOT_C
BATCH = 16


def _torch():
    import torch
    return torch


def device():
    torch = _torch()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    print("CUDA not available: running on CPU", file=sys.stderr)
    return torch.device("cpu")


# --------------------------------------------------------------------------
# heat
# --------------------------------------------------------------------------
class Overheat(RuntimeError):
    pass


def gpu_temp_c() -> int | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:
        return None


class Heat:
    """Temperature log plus the rules above. Every reading goes in .log as (time, label, C)."""

    def __init__(self, log=print):
        self.log = []
        self.say = log
        self.last = 0.0

    def read(self, label: str) -> int:
        t = gpu_temp_c()
        self.log.append((time.strftime("%Y-%m-%d %H:%M:%S"), label, t))
        self.say(f"  [heat] {label}: GPU {t if t is not None else 'unreadable'} C")
        self.last = time.time()
        if t is None:
            raise Overheat("GPU temperature could not be read")
        if t > HOT_C:
            raise Overheat(f"GPU at {t} C (limit {HOT_C} C)")
        return t

    def _wait_below(self, label: str) -> int:
        """Once above WARM_C: wait WAIT_S at a time until a reading is below RESUME_C."""
        while True:
            self.say(f"  [heat] above {WARM_C} C: waiting {WAIT_S} s, need below {RESUME_C} C")
            time.sleep(WAIT_S)
            t = self.read(label + " (waiting)")
            if t < RESUME_C:
                return t

    def settle(self, label: str) -> int:
        t = self.read(label)
        return self._wait_below(label) if t > WARM_C else t

    def after_run(self, label: str):
        t = self.read(label)
        time.sleep(PAUSE_S)
        if t > WARM_C:
            self._wait_below(label)

    def tick(self, label: str):
        """Inference: check at most every CHECK_S seconds."""
        if time.time() - self.last >= CHECK_S:
            self.settle(label)


def write_progress(text: str, path: str = "PROGRESS.md"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def _block(cin, cout, stride):
    from torch import nn
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _model_class():
    """GripperHeatmapModel, built on first use so importing this module does not need torch."""
    global GripperHeatmapModel
    if GripperHeatmapModel is not None:
        return GripperHeatmapModel
    import torch
    from torch import nn
    F = torch.nn.functional

    class _GripperHeatmapModel(nn.Module):
        """Heatmap (80x45) plus a visibility logit, on one of two backbones.

        use_resnet=False: the small CNN (four strided conv blocks, light decoder). Default.
        use_resnet=True:  ResNet-18 stem and layer1..layer3 (stride 4/8/16; layer4 and the
                          classifier are dropped), then a 2-block upsampling head back to
                          stride 4. pretrained=True loads torchvision's ImageNet weights.
        """

        def __init__(self, ch: int = 32, use_resnet: bool = False, pretrained: bool = False):
            super().__init__()
            self.use_resnet = use_resnet
            self._frozen = []
            if use_resnet:
                import torchvision
                w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                r = torchvision.models.resnet18(weights=w)
                self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)   # 80x45, 64 ch
                self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3  # 64 / 128 / 256 ch
                self.r3 = nn.Conv2d(256, 128, 1)
                self.s2 = nn.Conv2d(128, 128, 1)
                self.s1 = nn.Conv2d(64, 128, 1)
                self.up2 = _block(128, 128, 1)   # head block 1: stride 16 -> 8
                self.up1 = _block(128, 64, 1)    # head block 2: stride 8 -> 4
                self.head = nn.Conv2d(64, 1, 1)
                self.vis = nn.Linear(256, 1)
                # inputs arrive as (x - 0.45) / 0.25; ResNet wants ImageNet normalisation
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            else:
                self.b1 = _block(3, ch, 2)            # 160x90
                self.b2 = _block(ch, 2 * ch, 2)       # 80x45
                self.b3 = _block(2 * ch, 4 * ch, 2)   # 40x23
                self.b4 = _block(4 * ch, 4 * ch, 2)   # 20x12
                self.s3 = nn.Conv2d(4 * ch, 4 * ch, 1)
                self.s2 = nn.Conv2d(2 * ch, 4 * ch, 1)
                self.u3 = _block(4 * ch, 4 * ch, 1)
                self.u2 = _block(4 * ch, 2 * ch, 1)
                self.head = nn.Conv2d(2 * ch, 1, 1)
                self.vis = nn.Linear(4 * ch, 1)

        def forward(self, x):
            return self._resnet(x) if self.use_resnet else self._cnn(x)

        def _cnn(self, x):
            x2 = self.b2(self.b1(x))
            x3 = self.b3(x2)
            x4 = self.b4(x3)
            y = self.u3(F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False) + self.s3(x3))
            y = self.u2(F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False) + self.s2(x2))
            return self.head(y)[:, 0], self.vis(x4.mean((2, 3)))[:, 0]

        def _resnet(self, x):
            x = ((x * 0.25 + 0.45) - self.mean) / self.std
            l1 = self.layer1(self.stem(x))
            l2 = self.layer2(l1)
            l3 = self.layer3(l2)
            y = self.up2(F.interpolate(self.r3(l3), size=l2.shape[-2:], mode="bilinear", align_corners=False) + self.s2(l2))
            y = self.up1(F.interpolate(y, size=l1.shape[-2:], mode="bilinear", align_corners=False) + self.s1(l1))
            return self.head(y)[:, 0], self.vis(l3.mean((2, 3)))[:, 0]

        def train(self, mode: bool = True):
            super().train(mode)
            for m in self._frozen:   # frozen layers keep their BatchNorm statistics too
                m.eval()
            return self

    GripperHeatmapModel = _GripperHeatmapModel
    return GripperHeatmapModel


GripperHeatmapModel = None  # set by _model_class() on first use


def _freeze_backbone(model) -> int:
    """Freeze the early layers: ResNet stem + layer1, or the simple CNN's first two conv blocks.
    Returns the number of frozen parameters."""
    mods = [model.stem, model.layer1] if model.use_resnet else [model.b1, model.b2]
    n = 0
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(False)
            n += p.numel()
    model._frozen = mods
    model.train(model.training)
    return n


def build_model(ch: int = 32, use_resnet: bool = False, pretrained: bool = False):
    return _model_class()(ch=ch, use_resnet=use_resnet, pretrained=pretrained)


def soft_argmax(logits):
    """(B, 45, 80) logits -> (B, 2) (u, v) at 320x180, and the softmax map."""
    torch = _torch()
    b, h, w = logits.shape
    p = torch.softmax(logits.reshape(b, -1).float(), 1).reshape(b, h, w)
    xs = (torch.arange(w, device=p.device, dtype=p.dtype) + 0.5) * (SMALL_W / w) - 0.5   # cell size from the map
    ys = (torch.arange(h, device=p.device, dtype=p.dtype) + 0.5) * (SMALL_H / h) - 0.5
    return torch.stack([(p.sum(1) * xs).sum(1), (p.sum(2) * ys).sum(1)], 1), p


def _to_input(frames_bgr_u8, dev):
    torch = _torch()
    x = torch.as_tensor(frames_bgr_u8).to(dev)[..., [2, 1, 0]].permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.45) / 0.25


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_cameras(data: str) -> list:
    """Labels for every FK-labelled camera. Frames stay on disk (frames_of) so hundreds of cameras fit in RAM."""
    with open(os.path.join(data, "fk_labels", "index.json"), encoding="utf-8") as f:
        index = json.load(f)["written"]
    cams = []
    for r in index:
        z = np.load(r["file"])
        vis = z["visible"].astype(bool)
        cams.append(dict(key=r["key"], serial=r["serial"], uv=z["uv"], uv_full=z["uv_full"], visible=vis,
                         n=len(vis), full_w=int(z["full_size"][0]), file=r["file"]))
    return cams


def frames_of(cam: dict) -> np.ndarray:
    """(N, 180, 320, 3) BGR uint8 frames for one camera, read from its npz."""
    return np.load(cam["file"])["frames"]


def build_cache(cams: list, path: str) -> str:
    """Write all cameras' frames into one .npy, a camera at a time (never all in RAM)."""
    total = sum(c["n"] for c in cams)
    if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == total:
        return path
    part = path + ".part.npy"
    mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(total, SMALL_H, SMALL_W, 3))
    i = 0
    for c in cams:
        f = frames_of(c)
        mm[i:i + len(f)] = f
        i += len(f)
    mm.flush()
    del mm
    os.replace(part, path)
    return path


def split_by_serial(cams: list, n_test_serials: int = 8, seed: int = 0) -> tuple:
    serials = sorted({c["serial"] for c in cams})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(serials, size=min(n_test_serials, len(serials)), replace=False).tolist())
    return [c for c in cams if c["serial"] not in test], [c for c in cams if c["serial"] in test], sorted(test)


class FrameSet:
    """Training frames as a memory-mapped .npy, so loader workers share it instead of copying it."""

    def __init__(self, npy: str, uv: np.ndarray, vis: np.ndarray):
        self.npy, self.uv, self.vis, self._a = npy, uv, vis, None

    def __len__(self):
        return len(self.uv)

    def __getitem__(self, i):
        if self._a is None:
            self._a = np.load(self.npy, mmap_mode="r")
        return np.array(self._a[i]), self.uv[i], self.vis[i]


def _augment(x, uv, vis, torch):
    """Brightness/contrast, small zoom-crop, horizontal (and slight vertical) jitter, done on the GPU."""
    F = torch.nn.functional
    b = x.shape[0]
    dev = x.device
    s = torch.empty(b, device=dev).uniform_(0.85, 1.0)          # zoom in: crop 85..100% of the image
    tx = torch.empty(b, device=dev).uniform_(-0.12, 0.12)       # horizontal jitter (normalised units)
    ty = torch.empty(b, device=dev).uniform_(-0.05, 0.05)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, tx, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # labels: input normalised p_in = s * p_out + t  ->  p_out = (p_in - t) / s
    nx = (uv[:, 0] + 0.5) / SMALL_W * 2 - 1
    ny = (uv[:, 1] + 0.5) / SMALL_H * 2 - 1
    ox, oy = (nx - tx) / s, (ny - ty) / s
    uv = torch.stack([(ox + 1) / 2 * SMALL_W - 0.5, (oy + 1) / 2 * SMALL_H - 0.5], 1)
    vis = vis & (ox.abs() < 1) & (oy.abs() < 1)
    a = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    c = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.3, 0.3)
    return x * a + c, uv, vis


def _gauss_target(uv, torch, sigma_cells: float = 1.5):
    dev = uv.device
    xs = (torch.arange(HM_W, device=dev) + 0.5) * STRIDE - 0.5
    ys = (torch.arange(HM_H, device=dev) + 0.5) * (SMALL_H / HM_H) - 0.5
    gx = torch.exp(-((xs[None] - uv[:, :1]) / (sigma_cells * STRIDE)) ** 2 / 2)
    gy = torch.exp(-((ys[None] - uv[:, 1:]) / (sigma_cells * STRIDE)) ** 2 / 2)
    t = gy[:, :, None] * gx[:, None, :]
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-12)


# --------------------------------------------------------------------------
# train / predict / evaluate
# --------------------------------------------------------------------------
def train(cams: list, cache_npy: str, heat: Heat, ckpt: str, ch: int = 32, epochs: int = 40,
          batch: int = BATCH, lr: float = 2e-3, seed: int = 0, log=print, model=None,
          start_epoch: int = 0, opt_state: dict | None = None):
    torch = _torch()
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    dev = device()
    build_cache(cams, cache_npy)
    uv = np.concatenate([c["uv"] for c in cams]).astype(np.float32)
    vis = np.concatenate([c["visible"] for c in cams])
    loader = torch.utils.data.DataLoader(FrameSet(cache_npy, uv, vis), batch_size=batch, shuffle=True,
                                         num_workers=2, persistent_workers=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(seed))
    model = (model if model is not None else build_model(ch)).to(dev)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    if start_epoch:
        # resume: optimiser state from the checkpoint, learning-rate schedule moved on to where it was
        import warnings
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch * len(loader)):
                sched.step()
        log(f"  resuming at epoch {start_epoch + 1}/{epochs}, learning rate {sched.get_last_lr()[0]:.2e}")
    t0 = time.time()
    try:
        for ep in range(start_epoch, epochs):
            _epoch(model, loader, opt, sched, scaler, dev, heat, ep, torch, log)
            log(f"  epoch {ep + 1}/{epochs}  loss {model._loss:.3f}  ({time.time() - t0:.0f} s)")
            if (ep + 1) % RUN_EPOCHS == 0 or ep == epochs - 1:
                torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), epoch=ep + 1, ch=ch), ckpt)
                heat.after_run(f"after epoch {ep + 1}")
    except Overheat:
        torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), epoch=ep, ch=ch), ckpt)
        log(f"  checkpoint saved during epoch {ep + 1}: {ckpt}")
        raise
    return model


def _epoch(model, loader, opt, sched, scaler, dev, heat, ep, torch, log):
    model.train()
    tot = n = 0
    for fr, u, v in loader:
        if time.time() - heat.last >= WATCH_S:
            heat.settle(f"during epoch {ep + 1}")  # above WARM_C: pause here until below RESUME_C
        x = _to_input(fr, dev)
        u, v = u.to(dev), v.to(dev)
        x, u, v = _augment(x, u, v, torch)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            hm, vl = model(x)
        pred, p = soft_argmax(hm)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(vl.float(), v.float())
        if v.any():
            tgt = _gauss_target(u[v], torch)
            ce = -(tgt * torch.log(p[v].clamp_min(1e-9))).sum((1, 2)).mean()
            l1 = (pred[v] - u[v]).abs().sum(1).mean()
            loss = loss + ce + 0.05 * l1
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item() * len(fr)
        n += len(fr)
    model._loss = tot / max(n, 1)


def predict(model, frames_small, heat: Heat | None = None, batch: int = BATCH):
    """(N, 180, 320, 3) BGR uint8 -> (N, 2) (u, v) at 320x180 and (N,) visibility probability."""
    torch = _torch()
    dev = next(model.parameters()).device
    model.eval()
    uvs, ps = [], []
    with torch.no_grad():
        for i in range(0, len(frames_small), batch):
            if heat is not None:
                heat.tick("inference")
            x = _to_input(frames_small[i:i + batch], dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                hm, vl = model(x)
            uv, _ = soft_argmax(hm)
            uvs.append(uv.cpu().numpy())
            ps.append(torch.sigmoid(vl.float()).cpu().numpy())
    return np.concatenate(uvs), np.concatenate(ps)


def evaluate(model, cams: list, heat: Heat | None = None) -> dict:
    err_s, err_f, rows = [], [], []
    vis_ok = vis_n = 0
    for c in cams:
        uv, pv = predict(model, frames_of(c), heat)
        v = c["visible"]
        vis_ok += int(((pv > 0.5) == v).sum())
        vis_n += len(v)
        e = np.linalg.norm(uv[v] - c["uv"][v], axis=1)
        scale = c["full_w"] / SMALL_W
        ef = np.linalg.norm(uv[v] * scale - c["uv_full"][v], axis=1)
        err_s.append(e)
        err_f.append(ef)
        for j, k in enumerate(np.flatnonzero(v)):
            rows.append((float(ef[j]), c["key"], c["serial"], int(k)))
    es, ef = np.concatenate(err_s), np.concatenate(err_f)
    rows.sort(reverse=True)
    return dict(cameras=len(cams), frames=int(len(es)),
                px_320=dict(median=float(np.median(es)), p90=float(np.percentile(es, 90))),
                px_full=dict(median=float(np.median(ef)), p90=float(np.percentile(ef, 90))),
                visibility_accuracy=vis_ok / max(vis_n, 1), worst=rows[:3])


def save(model, path: str, ch: int, extra: dict):
    torch = _torch()
    torch.save(dict(state=model.state_dict(), ch=ch, use_resnet=bool(getattr(model, "use_resnet", False)), **extra), path)


_CACHE: dict = {}


def load(path: str):
    """Model from a checkpoint. The backbone is read from the weights, so mid-run *_ckpt.pt files load too."""
    torch = _torch()
    if path not in _CACHE:
        ck = torch.load(path, map_location=device(), weights_only=False)
        if "cfg" in ck:  # experiment checkpoint (experiments.py)
            from .experiments import load_model
            _CACHE[path] = load_model(path)[0].eval()
            return _CACHE[path]
        use_resnet = ck.get("use_resnet", any(k.startswith("layer1.") for k in ck["state"]))
        m = build_model(ck.get("ch", 32), use_resnet=use_resnet).to(device())
        m.load_state_dict(ck["state"])
        m.eval()
        _CACHE[path] = m
    return _CACHE[path]


def predict_conf(model, frames_small, heat: Heat | None = None, batch: int = BATCH, pause_s: float = 0.0):
    """(N, 180, 320, 3) BGR uint8 -> uv (N, 2) at 320x180, visibility probability, heatmap peak, heatmap entropy.

    The one inference path: the evaluation script and detect() both call this.
    """
    torch = _torch()
    dev = next(model.parameters()).device
    model.eval()
    uvs, ps, peaks, ents = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(frames_small), batch):
            if heat is not None:
                heat.tick("inference")
            x = _to_input(frames_small[i:i + batch], dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                hm, vl = model(x)
            uv, p = soft_argmax(hm)
            flat = p.reshape(len(p), -1)
            uvs.append(uv.cpu().numpy())
            ps.append(torch.sigmoid(vl.float()).cpu().numpy())
            peaks.append(flat.max(1).values.cpu().numpy())
            ents.append((-(flat * torch.log(flat.clamp_min(1e-12))).sum(1)).cpu().numpy())
            if pause_s:
                time.sleep(pause_s)  # gentle, steady load: a short rest after every batch
    return np.concatenate(uvs), np.concatenate(ps), np.concatenate(peaks), np.concatenate(ents)


DEFAULT_MODEL = "data/droid/detector_final.pt"
MIN_VISIBLE = 0.5    # visibility head: below this the gripper is called out of view
MIN_PEAK = 0.0       # heatmap peak: below this the detection is not trusted (set from the confidence study)


@dataclass
class Detection:
    u: float          # full-res pixels
    v: float
    conf: float       # heatmap peak probability (higher = sharper, more certain)
    visible_p: float  # visibility head probability


KEYPOINTS_MULTI = ("flange", "hand_centre", "fingertip_centre")   # tool axis: 0, 110, 170 mm from the flange


def detect_keypoints(frame, model_path: str = DEFAULT_MODEL, min_visible: float = MIN_VISIBLE,
                     min_peak: float | None = None) -> dict:
    """One full-res BGR frame -> {keypoint name: Detection or None}, full-res pixels.

    Same preprocessing as training and evaluation (INTER_AREA resize to 320x180). A keypoint is
    None when its visibility output is under min_visible or its heatmap peak under min_peak.
    Single-keypoint models return only "fingertip_centre".
    """
    import cv2

    small = cv2.resize(frame, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)
    model = load(model_path)
    if min_peak is None:  # threshold picked on validation, stored next to the model
        cfg = os.path.splitext(model_path)[0] + ".json"
        min_peak = json.load(open(cfg)).get("min_peak", MIN_PEAK) if os.path.exists(cfg) else MIN_PEAK
    s = frame.shape[1] / SMALL_W
    if getattr(model, "head", None) is not None and model.head.out_channels > 1:
        from .multikp import predict as predict_multi
        uv, p, peak = predict_multi(model, small[None], None, "detect")
        names = KEYPOINTS_MULTI
        uv, p, peak = uv[0], p[0], peak[0]
    else:
        uv, p, peak, _ = predict_conf(model, small[None])
        names = ("fingertip_centre",)
    out = {}
    for j, name in enumerate(names):
        ok = p[j] >= min_visible and peak[j] >= min_peak
        out[name] = Detection(float(uv[j, 0] * s), float(uv[j, 1] * s), float(peak[j]), float(p[j])) if ok else None
    return out


def detect(frame, model_path: str = DEFAULT_MODEL, min_visible: float = MIN_VISIBLE,
           min_peak: float | None = None) -> Detection | None:
    """One full-res BGR frame -> the Robotiq fingertip centre in full-res pixels, or None (see detect_keypoints)."""
    return detect_keypoints(frame, model_path, min_visible, min_peak)["fingertip_centre"]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m kintrace.droid train-detector")
    ap.add_argument("--data", default="data/droid")
    ap.add_argument("--wide", action="store_true", help="double the channels and train 20 more epochs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch = _torch()
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    heat = Heat()
    cams = load_cameras(a.data)
    tr, te, test_serials = split_by_serial(cams, seed=a.seed)
    print(f"DROID GT cameras: {len(cams)} ({len({c['serial'] for c in cams})} serials). "
          f"Train {len(tr)} cameras / {sum(c['n'] for c in tr)} frames. "
          f"Held out {len(te)} cameras / {sum(c['n'] for c in te)} frames, serials {', '.join(test_serials)}")
    print("excluded from training and evaluation: camera 23960472 (labels visibly off), "
          "cameras with all-zero intrinsics or no MP4 (see fk_labels/index.json)")
    ch, epochs = (64, a.epochs + 20) if a.wide else (32, a.epochs)
    name = "detector_wide" if a.wide else "detector"
    print(f"model: {ch} base channels, {epochs} epochs, batch {BATCH}, float16 autocast, runs of {RUN_EPOCHS} epochs")
    ckpt = os.path.join(a.data, f"{name}_ckpt.pt")
    try:
        heat.settle("before training (idle)")
        model = train(tr, os.path.join(a.data, "fk_labels", f"train_frames_seed{a.seed}.npy"), heat, ckpt,
                      ch=ch, epochs=epochs, seed=a.seed)
        heat.settle("before evaluation")
        ev = evaluate(model, te, heat)
        ev_train = evaluate(model, tr, heat)
    except Overheat as e:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        write_progress(
            "# Progress\n\n"
            f"Stopped {stamp}: {e}.\n\n"
            f"Phase 3, training `{name}` ({ch} channels). Checkpoint: `{ckpt}`.\n"
            "Stopped by the heat rule. Do not resume without checking the laptop.\n\n"
            "Temperature log:\n\n" + "\n".join(f"- {t} {lbl}: {c} C" for t, lbl, c in heat.log) + "\n")
        print(f"STOPPED: {e}. Wrote PROGRESS.md")
        sys.exit(3)
    save(model, os.path.join(a.data, f"{name}.pt"), ch, dict(test_serials=test_serials))
    print(f"held-out ({ev['cameras']} cameras, {ev['frames']} frames with the tip in view):")
    print(f"  pixel error at 320x180   median {ev['px_320']['median']:.2f} px   p90 {ev['px_320']['p90']:.2f} px")
    print(f"  pixel error at full res  median {ev['px_full']['median']:.1f} px   p90 {ev['px_full']['p90']:.1f} px")
    print(f"  visibility accuracy {100 * ev['visibility_accuracy']:.1f}%")
    print(f"training cameras, for reference: median {ev_train['px_320']['median']:.2f} px at 320x180")
    print("  worst held-out frames (full-res px, episode, serial, frame):")
    for w in ev["worst"]:
        print(f"    {w[0]:.1f}  {w[1]}  {w[2]}  f{w[3]}")
    out = os.path.join(a.data, f"{name}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(held_out=ev, train=ev_train, test_serials=test_serials, ch=ch, epochs=epochs,
                       heat_log=heat.log), f, indent=2)
    print("temperature log:")
    for t, lbl, c in heat.log:
        print(f"  {t}  {lbl}: {c} C")
    print(f"wrote {out}")
    return ev


if __name__ == "__main__":
    main()
