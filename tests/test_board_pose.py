import copy
import cv2
import numpy as np
import pytest
from packing_assets.board import create_board
from packing_assets.board_pose import calibrate_frames, write_model


def capture(board):
    K = np.array([[925.0, 0, 507.0], [0, 925.0, 387.0], [0, 0, 1.0]])
    frames = []
    for i in range(18):
        angle = 2 * np.pi * i / 6
        center = np.array(
            [0.55 * np.cos(angle), 0.55 * np.sin(angle), 0.25 + 0.25 * (i // 6)]
        )
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.array([right, down, forward])
        t = -R @ center
        markers = {
            key: cv2.projectPoints(
                np.array(xyz), cv2.Rodrigues(R)[0], t, K, np.array([-0.03, 0, 0, 0, 0])
            )[0]
            .reshape(-1, 2)
            .tolist()
            for key, xyz in board["markers"].items()
        }
        frames.append(
            {"name": f"f{i}.jpg", "width": 1024, "height": 768, "markers": markers}
        )
    return frames, K


def test_metric_board_camera_recovery_and_colmap_pixel_convention(tmp_path):
    board = create_board(tmp_path / "board")
    frames, truthK = capture(board)
    K, k1, views, report = calibrate_frames(frames, board)
    np.testing.assert_allclose(K, truthK, atol=0.001)
    assert abs(k1 + 0.03) < 1e-5
    assert report["held_out_rms_mm"] < 0.001
    model = write_model(tmp_path / "model", (1024, 768), K, k1, views, board)
    assert model.num_reg_images() == 18
    for im in model.images.values():
        camera = model.cameras[im.camera_id]
        for p in im.points2D:
            xyz = model.points3D[p.point3D_id].xyz
            expected = camera.img_from_cam(im.cam_from_world() * xyz)
            np.testing.assert_allclose(expected, p.xy, atol=0.001)
    assert not any((pid - 1) // 4 % 3 == 0 for pid in model.points3D)


def test_bad_heldout_board_observations_are_not_used_to_fit(tmp_path):
    board = create_board(tmp_path / "board")
    frames, _ = capture(board)
    bad = copy.deepcopy(frames)
    for frame in bad:
        for key, points in frame["markers"].items():
            if int(key) % 3 == 0:
                frame["markers"][key] = (np.array(points) + [20, -10]).tolist()
    with pytest.raises(ValueError, match="passed validation"):
        calibrate_frames(bad, board)
