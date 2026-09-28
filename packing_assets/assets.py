import json
import hashlib
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path):
    scene = trimesh.load_scene(path, process=False)
    mesh = (
        scene.to_mesh()
    )  # Bake scene-node transforms before any dimension calculation.
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("no triangle geometry in input")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("nonfinite mesh coordinates")
    return mesh


def audit_mesh_file(path: Path, require_volume=False):
    """Inspect serialized geometry, including caches; never repair or fill it."""
    mesh = load_mesh(path)
    counts = np.bincount(mesh.edges_unique_inverse)
    report = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "faces": len(mesh.faces),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "closed_volume": bool(mesh.is_volume),
        "volume_m3": float(mesh.volume) if mesh.is_volume else None,
        "check": "reloaded exported mesh; edge topology and orientation, not proof of physical accuracy or absence of self intersections",
    }
    if require_volume and not report["closed_volume"]:
        raise ValueError(
            f"exported mesh is not a closed volume: {path} "
            f"({report['boundary_edges']} boundary edges, {report['nonmanifold_edges']} nonmanifold edges)"
        )
    return report


def rotate_up(mesh, up_axis):
    rotations = {
        "x": (-np.pi / 2, [0, 1, 0]),
        "-x": (np.pi / 2, [0, 1, 0]),
        "y": (np.pi / 2, [1, 0, 0]),
        "-y": (-np.pi / 2, [1, 0, 0]),
        "z": (0, [1, 0, 0]),
        "-z": (np.pi, [1, 0, 0]),
    }
    angle, axis = rotations[up_axis]
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))


def scale_height(mesh, height_m):
    if not np.isfinite(height_m) or height_m <= 0 or mesh.extents[2] <= 1e-9:
        raise ValueError("height_m must be positive and mesh must have nonzero height")
    factor = height_m / mesh.extents[2]
    mesh.apply_scale(
        factor
    )  # Deliberately isotropic: do not hide aspect-ratio errors by stretching.
    return float(factor)


def export_asset(
    mesh, out: Path, provenance: dict, expected_extents=None, collision=False
):
    out.mkdir(parents=True, exist_ok=True)
    translation = [
        -mesh.bounds[:, 0].mean(),
        -mesh.bounds[:, 1].mean(),
        -mesh.bounds[0, 2],
    ]
    mesh.apply_translation(translation)
    mesh.units = "m"
    dims = mesh.extents
    reasons = ["object_shape_not_validated_against_real_views"]
    if not mesh.is_watertight:
        reasons.append("open_surface_requires_review_do_not_blindly_fill_cup_openings")
    if not mesh.is_winding_consistent:
        reasons.append("inconsistent_winding")
    components = len(mesh.split(only_watertight=False))
    if components > 1:
        reasons.append("multiple_components_check_against_object")
    dimension_check = None
    if expected_extents is not None:
        expected = np.array(expected_extents, float)
        if (
            expected.shape != (3,)
            or not np.isfinite(expected).all()
            or np.any(expected <= 0)
        ):
            raise ValueError(
                "expected_extents_m must have three positive finite XYZ dimensions"
            )
        # Pilot policy, not a demonstrated reconstruction accuracy claim.
        error = abs(dims - expected)
        limits = np.maximum(0.003, 0.03 * expected)
        dimension_check = {
            "expected_m": expected.tolist(),
            "abs_error_mm": (error * 1000).tolist(),
            "allowed_mm": (limits * 1000).tolist(),
            "passed": bool(np.all(error <= limits)),
            "independent_check": provenance.get("scale_source") != "measured_height",
        }
        if not dimension_check["passed"]:
            reasons.append("dimension_check_failed")
    # OBJ is intentionally untextured for collision/measurement. GLB keeps supported visuals.
    (out / "visual.obj").write_text(
        trimesh.exchange.obj.export_obj(mesh, include_texture=False, digits=16)
    )
    # glTF convention is Y up; bake a Z-up -> Y-up rotation only for the GLB viewer asset.
    glb = mesh.copy()
    glb.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    glb.export(out / "visual.glb")
    report = {
        "units": "m",
        "obj_up_axis": "z",
        "glb_up_axis": "y",
        "origin": "bottom_center",
        "extents_m_xyz_z_up": dims.tolist(),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": components,
        "volume_m3": float(mesh.volume) if mesh.is_volume else None,
        "status": "needs_review",
        "review_reasons": reasons,
        "dimensions": dimension_check,
        "provenance": provenance,
        "centering_translation_m": translation,
        "mass_kg": None,
        "inertia": None,
        "collision": {"generated": False},
    }
    if collision:
        if not mesh.is_volume:
            report["collision"] = {
                "generated": False,
                "reason": "requires closed consistently oriented volume; no automatic hole filling",
            }
        else:
            import coacd

            parts = coacd.run_coacd(
                coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.05, seed=17
            )
            folder = out / "collision"
            folder.mkdir(exist_ok=True)
            for index, (vertices, faces) in enumerate(parts):
                trimesh.Trimesh(vertices, faces, process=False).export(
                    folder / f"part_{index:03d}.obj"
                )
            report["collision"] = {
                "generated": True,
                "parts": len(parts),
                "up_axis": "z",
                "units": "m",
                "threshold_normalized": 0.05,
                "seed": 17,
                "validated": False,
                "note": "Check lost holes and collision clearance before simulation.",
            }
    (out / "asset.json").write_text(json.dumps(report, indent=2))
    return report
