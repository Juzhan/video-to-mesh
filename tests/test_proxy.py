import numpy as np
import pytest
from packing_assets.proxy import voxel_mesh


def test_voxel_units_and_ground_contact():
    occupied = np.zeros((12, 14, 16), bool)
    occupied[2:8, 3:11, :10] = True
    mesh, count = voxel_mesh(occupied, [-0.01, -0.02, 0], 0.002)
    assert mesh.is_volume and count == 1
    np.testing.assert_allclose(mesh.extents, [0.012, 0.016, 0.020], atol=1e-8)
    np.testing.assert_allclose(mesh.bounds[0], [-0.006, -0.014, 0], atol=1e-8)


def test_empty_or_truncated_proxy_is_rejected():
    occupied = np.zeros((10, 10, 10), bool)
    with pytest.raises(ValueError, match="empty"):
        voxel_mesh(occupied, [0, 0, 0], 0.001)
    occupied[:5, 2:7, :5] = True
    with pytest.raises(ValueError, match="boundary"):
        voxel_mesh(occupied, [0, 0, 0], 0.001)


def test_proxy_removes_disconnected_noise():
    occupied = np.zeros((20, 20, 20), bool)
    occupied[3:9, 3:9, :8] = True
    occupied[15, 15, 15] = True
    mesh, count = voxel_mesh(occupied, [0, 0, 0], 0.001)
    assert count == 2 and mesh.is_volume
    np.testing.assert_allclose(mesh.extents, [0.006, 0.006, 0.008], atol=1e-8)


def test_out_of_frame_and_behind_camera_are_unknown_not_background():
    from types import SimpleNamespace
    from packing_assets.proxy import silhouette_observations, silhouette_consensus

    camera = SimpleNamespace(img_from_cam=lambda cam: cam[:, :2] + 0.5)
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    cam = np.array(
        [[3, 3, 1], [0, 0, 1], [10, 3, 1], [3, 3, -1], [np.nan, 3, 1]], float
    )
    seen, missed = silhouette_observations(camera, cam, mask)
    np.testing.assert_array_equal(seen, [True, True, False, False, False])
    np.testing.assert_array_equal(missed, [False, True, False, False, False])
    # A wing seen in 12 frames and outside two frames remains supported.
    observed = np.zeros(1, int)
    misses = np.zeros(1, int)
    for x in [3] * 12 + [10] * 2:
        s, m = silhouette_observations(camera, np.array([[x, 3, 1.0]]), mask)
        observed += s
        misses += m
    assert silhouette_consensus(observed, misses, 0.97)[0]
    assert observed[0] == 12 and misses[0] == 0


def test_consensus_requires_enough_actual_views_and_uses_their_denominator():
    from packing_assets.proxy import silhouette_consensus

    result = silhouette_consensus(
        np.array([11, 12, 30, 48]), np.array([0, 0, 1, 1]), 0.97
    )
    np.testing.assert_array_equal(result, [False, True, False, True])
