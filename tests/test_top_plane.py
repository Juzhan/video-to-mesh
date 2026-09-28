import numpy as np
import trimesh
from packing_assets.top_plane import fit_supported_top, cap_hull


def test_supported_sloped_keys_cap_and_preserve_side_dimensions():
    rng = np.random.default_rng(29)
    xy = rng.uniform([-0.21, -0.065], [0.21, 0.065], (4000, 2))
    z = 0.031 + 0.06 * xy[:, 1] + rng.normal(0, 0.00015, len(xy))
    top = np.c_[xy, z]
    side = np.c_[
        rng.uniform([-0.22, -0.073], [0.22, 0.073], (1000, 2)),
        rng.uniform(0.001, 0.017, 1000),
    ]
    points = np.r_[top, side]
    normals = np.r_[
        np.tile([0, -0.06, 1.0], (len(top), 1)), np.tile([1.0, 0, 0], (len(side), 1))
    ]
    plane, report = fit_supported_top(points, normals)
    assert plane is not None
    hull = trimesh.creation.box([0.44, 0.146, 0.04])
    hull.apply_translation([0, 0, 0.02])
    clipped = cap_hull(hull, plane)
    assert clipped.is_volume and clipped.is_convex
    np.testing.assert_allclose(clipped.extents[:2], [0.44, 0.146], atol=1e-7)
    assert (
        np.max(clipped.vertices[:, 2] - clipped.vertices[:, :2] @ plane[:2] - plane[2])
        < 1e-7
    )


def test_upper_protrusion_prevents_plane_cut():
    rng = np.random.default_rng(30)
    xy = rng.uniform([-0.2, -0.06], [0.2, 0.06], (4000, 2))
    points = np.c_[xy, np.full(4000, 0.03)]
    points[:40, 2] = 0.045
    normals = np.tile([0.0, 0, 1.0], (4000, 1))
    plane, report = fit_supported_top(points, normals)
    assert plane is None


def test_capped_convex_asset_remains_convex_after_obj_roundtrip(tmp_path):
    from packing_assets.assets import export_asset, load_mesh

    hull = trimesh.creation.icosphere(subdivisions=2, radius=0.13).convex_hull
    mesh = cap_hull(hull, [0.000533, 0.06108, 0.03054])
    export_asset(mesh, tmp_path, {"scale_source": "test"})
    assert load_mesh(tmp_path / "visual.obj").is_convex
