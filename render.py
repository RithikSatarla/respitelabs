"""Draw AprilTags into a synthetic camera image at a known pose.

Used by the dry run and the tests, so the real detector and solvePnP run on
pixels even without a camera. Pinhole camera, no lens distortion, a little
blur and pixel noise. Real images will be worse (motion blur, glare, rolling
shutter), so treat dry-run accuracy as a best case.
"""
from __future__ import annotations

import numpy as np

from .markers import Intrinsics, cv2, marker_image

_CACHE: dict = {}


def _canvas(tag_id: int, cells_px: int):
    """Tag image with a white quiet zone of one cell. 36h11 is 8 cells wide
    including the black border, so the canvas is 10 cells."""
    key = (tag_id, cells_px)
    if key not in _CACHE:
        m = marker_image(tag_id, 8 * cells_px)
        _CACHE[key] = cv2.copyMakeBorder(m, cells_px, cells_px, cells_px, cells_px,
                                         cv2.BORDER_CONSTANT, value=255)
    return _CACHE[key]


def render(intr: Intrinsics, tags, background: int = 150, blur_sigma: float = 0.7,
           noise: float = 2.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """tags: iterable of (tag_id, size_m, T_cam_tag 4x4). Returns a uint8 gray image.

    size_m is the printed black-square edge. A tag facing away from the
    camera, or behind it, is not drawn.
    """
    rng = rng or np.random.default_rng(0)
    img = np.full((intr.height, intr.width), float(background), np.float32)
    # far tags first, so a nearer tag covers them (the tip flag over the target)
    for tag_id, size, T in sorted(tags, key=lambda x: -np.linalg.norm(x[2][:3, 3])):
        R, t = T[:3, :3], T[:3, 3]
        if t[2] < 0.05 or R[:, 2] @ t >= 0:  # behind the camera or facing away
            continue
        h = size / 2 * 10 / 8  # outer edge of the white quiet zone
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        cam = (R @ obj.T).T + t
        uv = (intr.K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        x0, y0 = np.floor(uv.min(0)).astype(int) - 2
        x1, y1 = np.ceil(uv.max(0)).astype(int) + 2
        if x1 < 0 or y1 < 0 or x0 >= intr.width or y0 >= intr.height:
            continue
        edge = max(np.linalg.norm(uv[1] - uv[0]), np.linalg.norm(uv[3] - uv[0]))
        cell = int(np.clip(np.ceil(edge * 2 / 10), 4, 60))  # about 2x oversampled
        can = _canvas(tag_id, cell).astype(np.float32)
        n = can.shape[0]
        # pixel centers sit at integer coordinates, so the canvas edges are at -0.5
        src = np.float32([[0, 0], [n, 0], [n, n], [0, n]]) - 0.5
        dst = np.float32(uv - [x0, y0])
        H = cv2.getPerspectiveTransform(src, dst)
        w, hgt = x1 - x0, y1 - y0
        patch = cv2.warpPerspective(can, H, (w, hgt), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(can), H, (w, hgt), flags=cv2.INTER_LINEAR, borderValue=0)
        # paste the patch, clipped to the image
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x1, intr.width), min(y1, intr.height)
        px0, py0 = ix0 - x0, iy0 - y0
        sl = img[iy0:iy1, ix0:ix1]
        p = patch[py0:py0 + (iy1 - iy0), px0:px0 + (ix1 - ix0)]
        mk = mask[py0:py0 + (iy1 - iy0), px0:px0 + (ix1 - ix0)]
        sl[:] = sl * (1 - mk) + p * mk
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise > 0:
        img = img + rng.normal(scale=noise, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)
