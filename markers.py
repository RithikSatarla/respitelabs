"""Find printed AprilTags in a camera image and turn them into 3D points.

Rig tags are AprilTag 36h11 (cv2.aruco.DICT_APRILTAG_36h11):

  wrist plate  one tag on the wrist, its 4 corners become the "markers" points
  tip flag     a small tag at the end of the pointer, its center is the tool tip
  table        4 tags taped to the table, their centers are the fixed markers
  target       an optional loose tag for the touch test

A tag's pose comes from cv2.solvePnP (IPPE_SQUARE) with the camera
intrinsics, so every point is in the camera frame, in meters.

Camera calibration: a ChArUco board (`calibrate`, from a different
dictionary, DICT_5X5_100, so it never gets confused with the rig tags) or a
plain printed checkerboard (`calibrate_chessboard`). ChArUco is more
forgiving: it still works when part of the board is out of the frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

try:
    import cv2
    import cv2.aruco  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "kintrace.hw needs OpenCV with the aruco module. Install it with:\n"
        "  pip install opencv-contrib-python-headless"
    ) from e

TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
BOARD_DICT = cv2.aruco.DICT_5X5_100


# --------------------------------------------------------------------------
# camera intrinsics
# --------------------------------------------------------------------------
@dataclass
class Intrinsics:
    K: np.ndarray  # 3x3
    dist: np.ndarray  # OpenCV distortion coefficients
    width: int
    height: int
    rms_px: float | None = None

    @staticmethod
    def load(path: str) -> "Intrinsics":
        with open(path) as f:
            d = json.load(f)
        return Intrinsics(np.array(d["K"], float), np.array(d["dist"], float).ravel(),
                          int(d["width"]), int(d["height"]), d.get("rms_px"))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(K=self.K.tolist(), dist=self.dist.ravel().tolist(), width=self.width,
                           height=self.height, rms_px=self.rms_px), f, indent=2)

    @staticmethod
    def guess(width: int = 1280, height: int = 720, hfov_deg: float = 70.0) -> "Intrinsics":
        """Rough pinhole guess from the field of view. Only for dry runs and
        first tests. Calibrate the real camera before trusting any numbers."""
        f = width / 2 / np.tan(np.deg2rad(hfov_deg) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        return Intrinsics(K, np.zeros(5), width, height)


# --------------------------------------------------------------------------
# tags
# --------------------------------------------------------------------------
def dictionary(kind: int = TAG_DICT):
    return cv2.aruco.getPredefinedDictionary(kind)


def make_detector(kind: int = TAG_DICT):
    p = cv2.aruco.DetectorParameters()
    # On rendered test images the AprilTag edge fit was the only refinement
    # with no inward corner bias (SUBPIX read tags about 0.2 px small, which
    # is 2 mm of depth at 0.5 m). Not yet compared on real webcam images.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary(kind), p)


def marker_image(tag_id: int, px: int = 400, border_bits: int = 1) -> np.ndarray:
    """The printable tag image (black square, no white margin)."""
    return cv2.aruco.generateImageMarker(dictionary(), tag_id, px, borderBits=border_bits)


def tag_object_points(size: float) -> np.ndarray:
    """Tag corners in the tag's own frame (z out of the printed face), in the
    order the detector returns them: top-left, top-right, bottom-right, bottom-left."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def detect(image: np.ndarray, detector=None) -> dict:
    """Tag id -> (4, 2) pixel corners."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = detector or make_detector()
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


def tag_pose(corners_px: np.ndarray, size: float, intr: Intrinsics):
    """Pose of one tag in the camera frame.

    Returns (R, t, reproj_err_px, ambiguity). ambiguity is the ratio of the
    reprojection errors of the two IPPE solutions (best / second best). Close
    to 1 means the tag is small or seen head-on and its tilt is uncertain.
    """
    obj = tag_object_points(size)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, corners_px.reshape(4, 1, 2), intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    errs = np.asarray(errs).ravel()
    order = np.argsort(errs)
    i = int(order[0])
    R, _ = cv2.Rodrigues(rvecs[i])
    amb = float(errs[order[0]] / max(errs[order[1]], 1e-12)) if n > 1 else 0.0
    return R, tvecs[i].ravel(), float(errs[i]), amb


def tag_corners_cam(R: np.ndarray, t: np.ndarray, size: float) -> np.ndarray:
    return (R @ tag_object_points(size).T).T + t


@dataclass
class Observation:
    """Everything the rig needs from one camera frame (camera frame, meters)."""
    wrist: np.ndarray | None  # (4, 3) wrist tag corners
    tip: np.ndarray | None  # (3,) pointer tip (tip tag center)
    table: dict  # id -> (3,) table tag centers
    target: np.ndarray | None  # (3,) loose target tag center
    reproj_px: float = 0.0


def observe(image: np.ndarray, intr: Intrinsics, tags: dict, detector=None,
            max_reproj_px: float = 2.0, max_ambiguity: float = 0.6) -> Observation:
    """Detect the rig tags in one image.

    tags: the "tags" block of rig.json (ids and printed sizes in meters).
    A tag whose pose is ambiguous (small, head-on) is dropped rather than
    risk a flipped pose, except table tags, which are big and flat on the table.
    """
    found = detect(image, detector)
    errs = []

    def pose_of(tag_id, size, strict=True):
        if tag_id is None or tag_id not in found:
            return None
        R, t, err, amb = tag_pose(found[tag_id], size, intr)
        if err > max_reproj_px or (strict and amb > max_ambiguity):
            return None
        errs.append(err)
        return R, t

    wrist = tip = target = None
    p = pose_of(tags.get("wrist_id"), tags.get("wrist_size", 0.04))
    if p is not None:
        wrist = tag_corners_cam(*p, tags.get("wrist_size", 0.04))
    p = pose_of(tags.get("tip_id"), tags.get("tip_size", 0.02))
    if p is not None:
        tip = p[1]
    p = pose_of(tags.get("target_id"), tags.get("target_size", 0.05), strict=False)
    if p is not None:
        target = p[1]
    table = {}
    for tid in tags.get("table_ids", []):
        p = pose_of(tid, tags.get("table_size", 0.05), strict=False)
        if p is not None:
            table[tid] = p[1]
    return Observation(wrist, tip, table, target, float(np.mean(errs)) if errs else 0.0)


# --------------------------------------------------------------------------
# camera calibration (ChArUco board)
# --------------------------------------------------------------------------
def charuco_board(squares_x: int = 7, squares_y: int = 5, square: float = 0.030, marker: float = 0.022):
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square, marker, dictionary(BOARD_DICT))


def board_image(board=None, px_per_m: float = 6000.0, margin_px: int = 40) -> np.ndarray:
    """Printable board image. At the default 6000 px/m, print at 152.4 dpi
    (or just print it and measure one square with calipers, then pass that
    size to calibrate)."""
    board = board or charuco_board()
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    size = (int(sx * sq * px_per_m) + 2 * margin_px, int(sy * sq * px_per_m) + 2 * margin_px)
    return board.generateImage(size, marginSize=margin_px)


def calibrate(images, board=None, min_corners: int = 8) -> tuple[Intrinsics, int]:
    """Camera intrinsics from photos of the ChArUco board.

    Returns (intrinsics, number of views used). Use 15 to 30 photos with the
    board at different angles and distances, filling the corners of the image too.
    """
    board = board or charuco_board()
    det = cv2.aruco.CharucoDetector(board)
    obj_all, img_all = [], []
    shape = None
    for im in images:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        shape = g.shape[::-1]
        cc, cid, _, _ = det.detectBoard(g)
        if cid is None or len(cid) < min_corners:
            continue
        obj, img = board.matchImagePoints(cc, cid)
        obj_all.append(obj)
        img_all.append(img)
    if len(obj_all) < 5:
        raise RuntimeError(f"only {len(obj_all)} usable board views, need at least 5 (aim for 15+)")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_all, img_all, shape, None, None)
    return Intrinsics(K, dist.ravel(), shape[0], shape[1], float(rms)), len(obj_all)


def calibrate_chessboard(images, pattern=(9, 6), square: float = 0.025) -> tuple[Intrinsics, int]:
    """Camera intrinsics from photos of a plain checkerboard.

    pattern: inner corners (columns, rows), e.g. (9, 6) for a 10 x 7 square board.
    square: measured square edge in meters. Every photo must show the whole
    board. Returns (intrinsics, number of views used).
    """
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    obj_all, img_all = [], []
    shape = None
    for im in images:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        shape = g.shape[::-1]
        ok, corners = cv2.findChessboardCorners(g, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not ok:
            continue
        corners = cv2.cornerSubPix(g, corners, (7, 7), (-1, -1), crit)
        obj_all.append(obj)
        img_all.append(corners)
    if len(obj_all) < 5:
        raise RuntimeError(f"board found in only {len(obj_all)} photos, need at least 5 (aim for 15+)")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_all, img_all, shape, None, None)
    return Intrinsics(K, dist.ravel(), shape[0], shape[1], float(rms)), len(obj_all)
