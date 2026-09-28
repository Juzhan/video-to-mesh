import numpy as np
import open3d as o3d
import trimesh

from packing_assets.planar import fit_supported_box
from packing_assets.quality import ray_scene


def points_cloud(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def test_rotated_noisy_box_fits_without_dimension_input():
    box = trimesh.creation.box([0.19, 0.13, 0.032])
    box.apply_translation([0, 0, 0.016])
    rotation = trimesh.transformations.rotation_matrix(0.43, [0, 0, 1])
    box.apply_transform(rotation)
    box.apply_translation([0.015, -0.023, 0])
    xyz, _ = trimesh.sample.sample_surface(box, 18000, seed=31)
    xyz = xyz[xyz[:, 2] > 0.002]
    xyz += np.random.default_rng(32).normal(0, 0.00025, xyz.shape)
    mesh, report = fit_supported_box(points_cloud(xyz))
    assert mesh is not None and report["surface_inlier_fraction"] > 0.95
    assert abs(mesh.volume / box.volume - 1) < 0.04
    surface, _ = trimesh.sample.sample_surface(mesh, 10000, seed=33)
    distances = (
        ray_scene(box).compute_distance(o3d.core.Tensor(surface.astype("f4"))).numpy()
    )
    assert np.percentile(distances, 95) < 0.001


def test_missing_side_does_not_get_invented_as_a_valid_box():
    box = trimesh.creation.box([0.19, 0.13, 0.032])
    box.apply_translation([0, 0, 0.016])
    xyz, _ = trimesh.sample.sample_surface(box, 18000, seed=31)
    xyz = xyz[(xyz[:, 2] > 0.002) & (xyz[:, 0] > -0.094)]
    mesh, report = fit_supported_box(points_cloud(xyz))
    assert mesh is None and report["applicable"] is False


def test_round_object_is_not_forced_into_box():
    cylinder = trimesh.creation.cylinder(radius=0.04, height=0.15)
    cylinder.apply_translation([0, 0, 0.075])
    xyz, _ = trimesh.sample.sample_surface(cylinder, 10000, seed=34)
    mesh, report = fit_supported_box(points_cloud(xyz[xyz[:, 2] > 0.002]))
    assert mesh is None
