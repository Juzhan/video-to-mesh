"""Convex sectors of a measured radial material profile; the opening stays empty."""

import hashlib, json
import numpy as np
import trimesh
from shapely.geometry import Polygon
from .assets import load_mesh
from .collision import union_overlap


def convex_profile_pieces(profile):
    polygon = Polygon(profile)
    if not polygon.is_valid:
        raise ValueError("invalid material profile")
    vertices, triangles = trimesh.creation.triangulate_polygon(polygon, engine="earcut")
    pieces = [Polygon(vertices[face]) for face in triangles]
    while True:
        merged = False
        for i in range(len(pieces)):
            for j in range(i):
                if pieces[i].intersection(pieces[j]).length < 1e-9:
                    continue
                combined = pieces[i].union(pieces[j])
                if (
                    combined.geom_type == "Polygon"
                    and combined.convex_hull.area - combined.area < 1e-12
                ):
                    pieces[j] = combined
                    pieces.pop(i)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return pieces


def build_radial_collision(mesh_path, out, provenance, threshold_m=0.003):
    if out.exists() and any(out.iterdir()):
        raise ValueError("collision output must be empty")
    source = load_mesh(mesh_path)
    pieces = convex_profile_pieces(provenance["radial_profile_m"])
    axis = provenance["axis_xy_m"]
    extra_parts = []
    if provenance.get("appendage_mesh"):
        import coacd
        from .collision import collision_input

        extra = load_mesh(mesh_path.parent / provenance["appendage_mesh"])
        simplified, _ = collision_input(extra, threshold_m)
        coacd.set_log_level("warn")
        raw = coacd.run_coacd(
            coacd.Mesh(simplified.vertices, simplified.faces),
            threshold=0.001,
            real_metric=True,
            max_convex_hull=48,
            preprocess_mode="off",
            decimate=True,
            max_ch_vertex=64,
            seed=17,
        )
        extra_parts = [trimesh.Trimesh(v, f, process=True).convex_hull for v, f in raw]
    for sectors in [32, 48, 64]:
        parts = []
        for i in range(sectors):
            angles = (i + np.array([0, 0.5, 1])) * 2 * np.pi / sectors
            for polygon in pieces:
                rz = np.array(polygon.exterior.coords)[:-1]
                vertices = np.stack(
                    [
                        np.c_[
                            axis[0] + rz[:, 0] * np.cos(a),
                            axis[1] + rz[:, 0] * np.sin(a),
                            rz[:, 1],
                        ]
                        for a in angles
                    ]
                ).reshape(-1, 3)
                parts.append(trimesh.convex.convex_hull(vertices))
        parts += extra_parts
        score = union_overlap(parts, source)
        if score["iou"] >= 0.95 and score["missing_fraction_of_reference"] <= 0.02:
            break
    if score["iou"] < 0.95 or score["missing_fraction_of_reference"] > 0.02:
        raise ValueError("radial collision occupancy check failed")
    out.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for i, part in enumerate(parts):
        (out / f"part_{i:03d}.obj").write_text(
            trimesh.exchange.obj.export_obj(part, include_texture=False, digits=16)
        )
        display = part.copy()
        display.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )
        scene.add_geometry(display)
    scene.export(out / "compound.glb")
    report = {
        "units": "m",
        "obj_up_axis": "z",
        "source_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "pipeline_version": 3,
        "backend": "convex angular sectors of measured radial material profile",
        "threshold_m": threshold_m,
        "parts": len(parts),
        "total_faces": sum(len(p.faces) for p in parts),
        "profile_pieces": len(pieces),
        "angular_sectors": sectors,
        "appendage_parts": len(extra_parts),
        "source_volume_comparison": score,
        "status": "needs_review",
        "limitation": "Preserves the estimated opening; original interior measurements may still be inaccurate.",
    }
    (out / "collision.json").write_text(json.dumps(report, indent=2))
    return report
