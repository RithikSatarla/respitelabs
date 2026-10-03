"""Train the DROID gripper detector with a choice of backbone.

  python -m kintrace.droid train_detector --data data/droid [--resnet] [--pretrained] [--freeze]
                                          [--epochs N] [--max-episodes N]

--resnet      ResNet-18 backbone instead of the small CNN
--pretrained  start ResNet-18 from torchvision's ImageNet weights (downloaded once, ~45 MB)
--freeze      freeze the early layers (ResNet stem + layer1, or the CNN's first two blocks)
--max-episodes N   use only the first N GT cameras (one per GT episode), for quick checks

Training uses detector.train(), so every heat rule in detector.py applies:
runs of 5 epochs, a pause after each, wait above 75 C until below 70 C, and
save + PROGRESS.md + exit above 85 C.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from . import detector


def check_gpu_temp() -> int | None:
    """GPU temperature in C from nvidia-smi, or None if it cannot be read."""
    return detector.gpu_temp_c()


class DROIDKeypoints(detector.FrameSet):
    """FK-labelled DROID frames (320x180) with (u, v) and visibility, read from a memory-mapped cache."""

    @classmethod
    def from_cameras(cls, cams: list, cache_npy: str) -> "DROIDKeypoints":
        if not os.path.exists(cache_npy):
            np.save(cache_npy, np.concatenate([c["frames"] for c in cams]))
        uv = np.concatenate([c["uv"] for c in cams]).astype(np.float32)
        vis = np.concatenate([c["visible"] for c in cams])
        return cls(cache_npy, uv, vis)


def train_detector(cams: list, data: str, heat: detector.Heat, ckpt: str, use_resnet: bool = False,
                   pretrained: bool = False, freeze: bool = False, ch: int = 32, epochs: int = 40, seed: int = 0):
    tag = hashlib.md5("|".join(sorted(c["key"] + c["serial"] for c in cams)).encode()).hexdigest()[:8]
    ds = DROIDKeypoints.from_cameras(cams, os.path.join(data, "fk_labels", f"train_frames_{tag}.npy"))
    model = detector.build_model(ch, use_resnet=use_resnet, pretrained=pretrained)
    if freeze:
        n = detector._freeze_backbone(model)
        print(f"froze {n} of {sum(p.numel() for p in model.parameters())} parameters")
    return detector.train(cams, ds.npy, heat, ckpt, ch=ch, epochs=epochs, seed=seed, model=model)


def main(argv=None):
    import argparse

    import torch

    ap = argparse.ArgumentParser(prog="python -m kintrace.droid train_detector")
    ap.add_argument("--data", default="data/droid")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--resnet", action="store_true", help="ResNet-18 backbone instead of the small CNN")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights for the ResNet-18 backbone")
    ap.add_argument("--freeze", action="store_true", help="freeze the early layers")
    ap.add_argument("--ch", type=int, default=32, help="small CNN width (ignored with --resnet)")
    ap.add_argument("--max-episodes", type=int, default=None, help="use only the first N GT cameras")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.pretrained and not a.resnet:
        ap.error("--pretrained needs --resnet")

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    cams = detector.load_cameras(a.data)
    if a.max_episodes:
        cams = sorted(cams, key=lambda c: c["key"])[:a.max_episodes]
    serials = {c["serial"] for c in cams}
    n_test = 8 if len(serials) >= 16 else max(1, len(serials) // 3)
    tr, te, test_serials = detector.split_by_serial(cams, n_test_serials=n_test, seed=a.seed)
    arch = "resnet18" + ("_pretrained" if a.pretrained else "") if a.resnet else f"cnn{a.ch}"
    name = f"detector_{arch}" + ("_frozen" if a.freeze else "") + (f"_smoke{a.max_episodes}" if a.max_episodes else "")
    print(f"DROID GT cameras: {len(cams)} ({len(serials)} serials). Train {len(tr)} cameras / "
          f"{sum(len(c['frames']) for c in tr)} frames. Held out {len(te)} cameras / "
          f"{sum(len(c['frames']) for c in te)} frames, serials {', '.join(test_serials)}")
    print(f"model: {arch}, freeze={a.freeze}, {a.epochs} epochs, batch {detector.BATCH}, float16 autocast, "
          f"runs of {detector.RUN_EPOCHS} epochs")
    heat = detector.Heat()
    ckpt = os.path.join(a.data, f"{name}_ckpt.pt")
    try:
        heat.settle("before training (idle)")
        model = train_detector(tr, a.data, heat, ckpt, use_resnet=a.resnet, pretrained=a.pretrained,
                               freeze=a.freeze, ch=a.ch, epochs=a.epochs, seed=a.seed)
        heat.settle("before evaluation")
        ev = detector.evaluate(model, te, heat)
    except detector.Overheat as e:
        detector.write_progress(
            "# Progress\n\n"
            f"Stopped {time.strftime('%Y-%m-%d %H:%M:%S')}: {e}.\n\n"
            f"Training `{name}`. Checkpoint: `{ckpt}`.\n"
            "Stopped by the heat rule. Do not resume without checking the laptop.\n\n"
            "Temperature log:\n\n" + "\n".join(f"- {t} {lbl}: {c} C" for t, lbl, c in heat.log) + "\n")
        print(f"STOPPED: {e}. Wrote PROGRESS.md")
        sys.exit(3)
    detector.save(model, os.path.join(a.data, f"{name}.pt"), a.ch, dict(test_serials=test_serials, arch=arch))
    print(f"held-out ({ev['cameras']} cameras, {ev['frames']} frames with the tip in view):")
    print(f"  pixel error at 320x180   median {ev['px_320']['median']:.2f} px   p90 {ev['px_320']['p90']:.2f} px")
    print(f"  pixel error at full res  median {ev['px_full']['median']:.1f} px   p90 {ev['px_full']['p90']:.1f} px")
    print(f"  visibility accuracy {100 * ev['visibility_accuracy']:.1f}%")
    out = os.path.join(a.data, f"{name}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(held_out=ev, test_serials=test_serials, arch=arch, freeze=a.freeze, epochs=a.epochs,
                       heat_log=heat.log), f, indent=2)
    print("temperature log:")
    for t, lbl, c in heat.log:
        print(f"  {t}  {lbl}: {c} C")
    print(f"wrote {out}")
    return ev


if __name__ == "__main__":
    main()
