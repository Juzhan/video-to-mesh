"""Geometry-only simplification; no simulation or display application required."""

import numpy as np


def light_visual(mesh, target_faces=8000):
    """Reduce display triangles without rescaling or recentering the source frame."""
    if len(mesh.faces) <= target_faces:
        return mesh.copy(), {
            "source_faces": len(mesh.faces),
            "export_faces": len(mesh.faces),
            "p95_distance_mm": 0.0,
        }
    import open3d as o3d
    import trimesh
    from .quality import ray_scene

    native = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    reference = ray_scene(mesh)
    for budget in [target_faces, target_faces * 2, target_faces * 4]:
        reduced = native.simplify_quadric_decimation(budget)
        candidate = trimesh.Trimesh(
            np.asarray(reduced.vertices), np.asarray(reduced.triangles), process=True
        )
        predicted = ray_scene(candidate)
        points_a, _ = trimesh.sample.sample_surface(mesh, 12000, seed=23)
        points_b, _ = trimesh.sample.sample_surface(candidate, 12000, seed=23)
        distances = np.r_[
            reference.compute_distance(
                o3d.core.Tensor(points_b.astype("f4")), nthreads=2
            ).numpy(),
            predicted.compute_distance(
                o3d.core.Tensor(points_a.astype("f4")), nthreads=2
            ).numpy(),
        ]
        p95 = float(np.percentile(distances, 95))
        if (
            p95 <= 0.001
            and np.max(abs(candidate.bounds - mesh.bounds)) <= 0.001
            and (not mesh.is_volume or candidate.is_volume)
        ):
            return candidate, {
                "source_faces": len(mesh.faces),
                "export_faces": len(candidate.faces),
                "p95_distance_mm": p95 * 1000,
                "max_bounds_change_mm": float(
                    np.max(abs(candidate.bounds - mesh.bounds)) * 1000
                ),
                "method": "quadric decimation; bidirectional sampled distance to full reconstruction; no rescaling",
            }
    raise ValueError(
        "visual simplification would exceed 1 mm review policy; inspect source geometry"
    )
