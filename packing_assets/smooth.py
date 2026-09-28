"""Optional appearance mesh; never replaces collision geometry or inertia."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import trimesh
from .assets import load_mesh, audit_mesh_file
from .quality import ray_scene


def smooth_visual(mesh):
    import open3d as o3d

    native = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    reference = ray_scene(mesh)
    original, _ = trimesh.sample.sample_surface(mesh, 12000, seed=41)
    for iterations in [12, 6, 3, 1]:
        result = native.filter_smooth_taubin(
            number_of_iterations=iterations, lambda_filter=0.5, mu=-0.53
        )
        candidate = trimesh.Trimesh(
            np.asarray(result.vertices), np.asarray(result.triangles), process=False
        )
        if np.max(abs(candidate.bounds - mesh.bounds)) > 0.001:
            continue
        points, _ = trimesh.sample.sample_surface(candidate, 12000, seed=41)
        predicted = ray_scene(candidate)
        distances = np.r_[
            reference.compute_distance(
                o3d.core.Tensor(points.astype("f4")), nthreads=2
            ).numpy(),
            predicted.compute_distance(
                o3d.core.Tensor(original.astype("f4")), nthreads=2
            ).numpy(),
        ]
        p95 = float(np.percentile(distances, 95))
        if p95 <= 0.001 and (not mesh.is_volume or candidate.is_volume):
            return candidate, {
                "method": "Taubin geometry smoothing plus interpolated vertex normals",
                "iterations": iterations,
                "p95_distance_mm": p95 * 1000,
                "max_bounds_change_mm": float(
                    np.max(abs(candidate.bounds - mesh.bounds)) * 1000
                ),
                "faces": len(candidate.faces),
                "collision_unchanged": True,
            }
    return mesh.copy(), {
        "method": "interpolated vertex normals only; geometry smoothing exceeded the deviation limit",
        "iterations": 0,
        "faces": len(mesh.faces),
        "p95_distance_mm": 0.0,
        "max_bounds_change_mm": 0.0,
        "collision_unchanged": True,
    }


def ensure_smooth_variant(package: Path):
    report = json.loads((package / "package.json").read_text())
    ordinary_check = audit_mesh_file(package / "visual.obj", require_volume=True)
    has_urdf = (package / "asset.urdf").is_file()
    if (
        report.get("visual_smoothing")
        and (package / "visual_smooth.obj").is_file()
        and (not has_urdf or (package / "asset_smooth.urdf").is_file())
    ):
        report["exported_mesh_checks"] = {
            "visual": ordinary_check,
            "smooth_visual": audit_mesh_file(
                package / "visual_smooth.obj", require_volume=True
            ),
        }
        (package / "package.json").write_text(json.dumps(report, indent=2))
        return report
    mesh = load_mesh(package / "visual.obj")
    smoothed, quality = smooth_visual(mesh)
    cavity = report.get("cavity", {})
    if cavity.get("detected"):
        from .depth_proxy import cavity_probe

        check = cavity_probe(smoothed)
        if not check["detected"] or check["depth_m"] < cavity["depth_m"] * 0.75:
            smoothed = mesh.copy()
            quality.update(
                method="smooth normals only to preserve cavity",
                iterations=0,
                p95_distance_mm=0.0,
                max_bounds_change_mm=0.0,
            )
        quality["cavity_check"] = cavity_probe(smoothed)
    (package / "visual_smooth.obj").write_text(
        trimesh.exchange.obj.export_obj(
            smoothed, include_normals=True, include_texture=False, digits=16
        )
    )
    glb = smoothed.copy()
    glb.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    glb.export(package / "visual_smooth.glb")
    if has_urdf:
        tree = ET.parse(package / "asset.urdf")
        tree.find("link/visual/geometry/mesh").set("filename", "visual_smooth.obj")
        tree.write(
            package / "asset_smooth.urdf", encoding="utf-8", xml_declaration=True
        )
    report["visual_smoothing"] = quality
    report["exported_mesh_checks"] = {
        "visual": ordinary_check,
        "smooth_visual": audit_mesh_file(
            package / "visual_smooth.obj", require_volume=True
        ),
    }
    (package / "package.json").write_text(json.dumps(report, indent=2))
    return report
