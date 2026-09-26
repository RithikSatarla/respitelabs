"""AprilTag detection and pose on synthetic images with a known answer.

The tag is generated with cv2.aruco, warped into a pinhole camera image at a
known pose (hw/render.py), then found with the same detector and solvePnP
the rig uses. Real webcam images are worse than these (motion blur, glare,
lens distortion left over after calibration), so these are best-case numbers.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

from scipy.spatial.transform import Rotation  # noqa: E402

from kintrace.hw import markers, render  # noqa: E402

INTR = markers.Intrinsics.guess(1280, 720, hfov_deg=70.0)


def _pose(dist, tilt_deg, tilt_axis, spin_deg, offset=(0.0, 0.0)):
    """Tag at `dist` m in front of the camera, facing it, then tilted."""
    R_face = np.diag([1.0, -1.0, -1.0])  # tag z points back at the camera
    R = (Rotation.from_rotvec(np.asarray(tilt_axis, float) / np.linalg.norm(tilt_axis) * np.deg2rad(tilt_deg))
         * Rotation.from_matrix(R_face) * Rotation.from_euler("z", spin_deg, degrees=True)).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [offset[0], offset[1], dist]
    return T


CASES = [
    # (distance m, tilt deg, tilt axis, spin deg, offset, tag size m)
    (0.35, 30, (1, 0, 0), 0, (0.00, 0.00), 0.040),
    (0.45, 40, (0, 1, 0), 25, (0.05, -0.03), 0.040),
    (0.55, 25, (1, 1, 0), 60, (-0.08, 0.04), 0.040),
    (0.60, 45, (1, -1, 0), 110, (0.10, 0.06), 0.050),
    (0.50, 35, (0, 1, 0), 200, (-0.12, -0.05), 0.050),
]


@pytest.mark.parametrize("dist,tilt,axis,spin,off,size", CASES)
def test_tag_pose_recovered(dist, tilt, axis, spin, off, size):
    T = _pose(dist, tilt, axis, spin, off)
    img = render.render(INTR, [(7, size, T)], rng=np.random.default_rng(1))
    found = markers.detect(img)
    assert 7 in found, "tag not detected"
    R, t, err, _ = markers.tag_pose(found[7], size, INTR)
    corners = markers.tag_corners_cam(R, t, size)
    truth = markers.tag_corners_cam(T[:3, :3], T[:3, 3], size)
    corner_err_mm = np.linalg.norm(corners - truth, axis=1).max() * 1000
    center_err_mm = np.linalg.norm(t - T[:3, 3]) * 1000
    assert err < 1.0, f"reprojection {err:.2f} px"
    assert center_err_mm < 2.0, f"center off by {center_err_mm:.2f} mm"
    assert corner_err_mm < 3.0, f"a corner is off by {corner_err_mm:.2f} mm"


def test_observe_rig_layout():
    """The rig's own tag layout: wrist tag corners, tip flag center, table tags."""
    tags = dict(wrist_id=0, wrist_size=0.040, tip_id=1, tip_size=0.020,
                table_ids=[10, 11], table_size=0.050, target_id=20, target_size=0.050)
    Tw = _pose(0.40, 35, (1, 0.3, 0), 15, (0.02, -0.02))
    Tt = _pose(0.42, 30, (1, 0.3, 0), 15, (0.06, 0.03))
    Ta = _pose(0.55, 50, (1, 0, 0), 0, (-0.12, 0.08))
    Tb = _pose(0.55, 50, (1, 0, 0), 0, (0.12, 0.08))
    img = render.render(INTR, [(0, 0.040, Tw), (1, 0.020, Tt), (10, 0.050, Ta), (11, 0.050, Tb)],
                        rng=np.random.default_rng(2))
    o = markers.observe(img, INTR, tags)
    assert o.wrist is not None and o.tip is not None
    wrist_truth = markers.tag_corners_cam(Tw[:3, :3], Tw[:3, 3], 0.040)
    assert np.linalg.norm(o.wrist - wrist_truth, axis=1).max() < 0.003
    assert np.linalg.norm(o.tip - Tt[:3, 3]) < 0.003
    assert set(o.table) == {10, 11}
    for tid, T in ((10, Ta), (11, Tb)):
        assert np.linalg.norm(o.table[tid] - T[:3, 3]) < 0.003
    assert o.target is None


def test_intrinsics_roundtrip(tmp_path):
    p = tmp_path / "camera.json"
    INTR.save(str(p))
    back = markers.Intrinsics.load(str(p))
    assert np.allclose(back.K, INTR.K) and np.allclose(back.dist, INTR.dist)
    assert (back.width, back.height) == (INTR.width, INTR.height)


def _render_chessboard(intr, T, cols=10, rows=7, square=0.025, rng=None):
    """A plain checkerboard (cols x rows squares) at pose T, drawn like a tag."""
    px = 40
    board = np.kron((np.indices((rows, cols)).sum(0) % 2) * 255, np.ones((px, px))).astype(np.float32)
    board = cv2.copyMakeBorder(board, px, px, px, px, cv2.BORDER_CONSTANT, value=255)
    w, h = (cols + 2) * square, (rows + 2) * square
    obj = np.array([[-w / 2, h / 2, 0], [w / 2, h / 2, 0], [w / 2, -h / 2, 0], [-w / 2, -h / 2, 0]])
    cam = (T[:3, :3] @ obj.T).T + T[:3, 3]
    uv = (intr.K @ cam.T).T
    uv = (uv[:, :2] / uv[:, 2:3]).astype(np.float32)
    n_w, n_h = board.shape[1], board.shape[0]
    src = np.float32([[0, 0], [n_w, 0], [n_w, n_h], [0, n_h]]) - 0.5
    H = cv2.getPerspectiveTransform(src, uv)
    img = cv2.warpPerspective(board, H, (intr.width, intr.height), borderValue=150)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    img += (rng or np.random.default_rng(0)).normal(scale=2.0, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_chessboard_calibration_recovers_focal_length():
    rng = np.random.default_rng(3)
    images = []
    for i in range(12):
        axis = rng.normal(size=3)
        axis[2] = 0
        T = _pose(rng.uniform(0.45, 0.7), rng.uniform(15, 40), axis, rng.uniform(-20, 20),
                  (rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05)))
        images.append(_render_chessboard(INTR, T, rng=rng))
    intr, n = markers.calibrate_chessboard(images, pattern=(9, 6), square=0.025)
    assert n >= 10
    assert intr.rms_px < 0.5
    assert abs(intr.K[0, 0] / INTR.K[0, 0] - 1) < 0.02
    assert abs(intr.K[1, 1] / INTR.K[1, 1] - 1) < 0.02
    assert abs(intr.K[0, 2] - INTR.K[0, 2]) < 15 and abs(intr.K[1, 2] - INTR.K[1, 2]) < 15
