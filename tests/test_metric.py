import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from packing_assets.board import create_board, detect
from packing_assets.metric import similarity, transform_points, triangulate, fit_board
import cv2


def test_board_preview_detects_all_markers(tmp_path):
    board = create_board(tmp_path)
    detected = detect(cv2.imread(str(tmp_path / "board-preview.png")))
    assert set(detected) == set(map(int, board["markers"]))
    assert np.isclose(
        np.linalg.norm(
            np.array(board["markers"]["0"])[1] - np.array(board["markers"]["0"])[0]
        ),
        0.032,
    )


def test_metric_fit_withheld_controls_and_bad_scale(tmp_path):
    board = create_board(tmp_path)
    truth = np.eye(4)
    truth[:3, :3] = (
        0.071 * Rotation.from_euler("xyz", [21, 33, 48], degrees=True).as_matrix()
    )
    truth[:3, 3] = [0.5, -0.2, 0.13]
    points = {
        f"{i}:{j}": transform_points([p], np.linalg.inv(truth))[0]
        for i, corners in board["markers"].items()
        for j, p in enumerate(corners)
    }
    got, report = fit_board(points, board)
    np.testing.assert_allclose(got, truth, atol=1e-9)
    assert report["held_out_rms_mm"] < 1e-6
    # Holdout marker damaged by a 3D reconstruction error must reject the job.
    for j in range(4):
        points[f"0:{j}"] += [0.8, 0.8, 0.8]
    with pytest.raises(ValueError, match="held-out"):
        fit_board(points, board)


def test_triangulation_resists_bad_view_and_rejects_zero_baseline():
    point = np.array([0.2, 0.1, 2.0])
    obs = []
    for i in range(7):
        p = np.column_stack([np.eye(3), [-0.25 * i, 0, 0]])
        projected = p @ np.r_[point, 1]
        xy = projected[:2] / projected[2]
        if i == 4:
            xy += [0.4, -0.3]
        obs.append({"P": p, "xy": xy, "focal": 1000})
    got, stats = triangulate(obs)
    np.testing.assert_allclose(got, point, atol=1e-6)
    assert stats["views"] == 6
    with pytest.raises(ValueError):
        triangulate([obs[0]] * 3)


def test_collinear_fit_rejected():
    with pytest.raises(ValueError, match="collinear"):
        similarity([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [2, 0, 0], [4, 0, 0]])
