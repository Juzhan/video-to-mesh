import numpy as np
from packing_assets.board import create_dense_a4
from packing_assets.metric import fit_board, transform_points
from packing_assets.proxy import proxy_scan_roi


def test_partial_dense_board_recovers_scale(tmp_path):
    board = create_dense_a4(tmp_path)
    assert len(board["markers"]) == 48
    assert board["page_mm"] == [297.0, 210.0]
    # Simulate an object hiding the middle rows: top and bottom controls remain.
    ids = [str(i) for i in list(range(8)) + list(range(40, 48))]
    points = {
        f"{i}:{c}": (np.array(p) - [0.1, 0.2, 0.3]) / 2.7
        for i in ids
        for c, p in enumerate(board["markers"][i])
    }
    t, report = fit_board(points, board)
    assert abs(report["metric_scale"] - 2.7) < 1e-9
    assert report["held_out_rms_mm"] < 1e-6
    for key, p in points.items():
        i, c = key.split(":")
        np.testing.assert_allclose(
            transform_points([p], t)[0], board["markers"][i][int(c)], atol=1e-9
        )


def test_paper_size_does_not_crop_adjacent_object(tmp_path):
    board = create_dense_a4(tmp_path)
    p = np.zeros(2, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    p["x"] = [0.17, 0.61]
    p["y"] = [-0.073, 0.073]
    p["z"] = [0.002, 0.035]
    roi, height = proxy_scan_roi(board, p, 0.001)
    assert (
        roi[0, 0] < 0.17 and roi[1, 0] > 0.61
    )  # Outside A4, still inside the scan region.
    assert np.prod(np.ceil((roi[1] - roi[0]) / 0.001)) < 15_000_000
    old = dict(board, proxy_crop_to_observed=False)
    oldroi, _ = proxy_scan_roi(old, p, 0.001)
    np.testing.assert_allclose(oldroi[:, :2], [[-0.7, -0.7], [0.7, 0.7]])
