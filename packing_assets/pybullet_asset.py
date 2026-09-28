"""Portable single-link URDF with multiple convex collision shapes."""

import hashlib, json, shutil
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from .assets import load_mesh
from .collision import build_collision


# Compatibility import for existing callers.
from .mesh_export import light_visual


def export_pybullet(
    asset: Path, out: Path, mass_kg: float, threshold_m=0.003, smooth=True
):
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("supply a positive object mass_kg")
    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    if out.exists() and any(out.iterdir()):
        raise ValueError("PyBullet package output must be empty")
    meta = json.loads((asset / "asset.json").read_text())
    if meta.get("status") != "needs_review":
        raise ValueError("source geometry is rejected or failed")
    if meta["units"] != "m" or meta["obj_up_axis"] != "z":
        raise ValueError("requires metre Z-up asset")
    mesh_path = asset / "visual.obj"
    mesh = load_mesh(mesh_path)
    if not mesh.is_volume:
        raise ValueError("select a closed outer-occupancy model")
    out.mkdir(parents=True, exist_ok=True)
    prior = asset / "collision/collision.json"
    fingerprint = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    cached = json.loads(prior.read_text()) if prior.exists() else {}
    if (
        cached.get("source_sha256") == fingerprint
        and cached.get("threshold_m") == threshold_m
        and cached.get("pipeline_version") == 3
    ):
        collision = cached
        shutil.copytree(prior.parent, out / "collision")
    elif meta.get("provenance", {}).get("radial_profile_m"):
        from .radial_collision import build_radial_collision

        collision = build_radial_collision(
            mesh_path, out / "collision", meta["provenance"], threshold_m
        )
    elif meta.get("cavity", {}).get("detected"):
        collision = build_collision(
            mesh_path, out / "collision", threshold_m, max_hulls=128
        )
    else:
        collision = build_collision(mesh_path, out / "collision", threshold_m)
    parts = sorted((out / "collision").glob("part_*.obj"))
    if len(parts) != collision["parts"]:
        raise ValueError("collision part count mismatch")
    cavity = meta.get("cavity", {})
    cavity_collision = None
    if meta.get("cavity_required") and not cavity.get("detected"):
        raise ValueError("required cavity is missing")
    if cavity.get("detected"):
        import open3d as o3d
        from .quality import ray_scene

        columns = cavity.get(
            "probe_columns",
            [
                {
                    "xy_m": cavity["probe_m"][:2],
                    "floor_z_m": cavity["floor_z_m"],
                    "rim_z_m": float(mesh.bounds[1, 2]),
                }
            ],
        )
        rays = np.array(
            [[*c["xy_m"], c["rim_z_m"] + 0.01, 0, 0, -1] for c in columns], dtype="f4"
        )
        depth = np.full(len(rays), np.inf)
        for path in parts:
            part = load_mesh(path)
            active = np.flatnonzero(
                (rays[:, :2] >= part.bounds[0, :2] - 1e-7).all(1)
                & (rays[:, :2] <= part.bounds[1, :2] + 1e-7).all(1)
            )
            if len(active):
                hit = (
                    ray_scene(part)
                    .cast_rays(o3d.core.Tensor(rays[active]), nthreads=2)["t_hit"]
                    .numpy()
                )
                depth[active] = np.minimum(depth[active], hit)
        intrusion = [
            float(c["rim_z_m"] + 0.01 - d - c["floor_z_m"])
            for c, d in zip(columns, depth)
        ]
        if not np.isfinite(depth).all() or max(intrusion) > 0.005:
            raise ValueError(
                "collision blocks bridge the required cavity or its floor is missing"
            )
        cavity_collision = {
            "passed": True,
            "columns": len(columns),
            "max_floor_intrusion_m": max(intrusion),
        }
    visual_mesh, visual_report = light_visual(mesh)
    visual_mesh.export(out / "visual.obj")
    # This is an explicitly labelled geometric approximation, not measured mass distribution.
    center = mesh.center_mass
    inertia = mesh.moment_inertia * (mass_kg / mesh.volume)
    if not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("invalid approximate inertia")
    robot = ET.Element("robot", name="scanned_item")
    link = ET.SubElement(robot, "link", name="body")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass_kg))
    ET.SubElement(
        inertial,
        "inertia",
        **{
            k: str(inertia[i, j])
            for k, i, j in [
                ("ixx", 0, 0),
                ("ixy", 0, 1),
                ("ixz", 0, 2),
                ("iyy", 1, 1),
                ("iyz", 1, 2),
                ("izz", 2, 2),
            ]
        },
    )
    visual = ET.SubElement(link, "visual")
    geo = ET.SubElement(visual, "geometry")
    ET.SubElement(geo, "mesh", filename="visual.obj", scale="1 1 1")
    mat = ET.SubElement(visual, "material", name="scan")
    ET.SubElement(mat, "color", rgba=".2 .5 .7 1")
    for i, path in enumerate(parts):
        node = ET.SubElement(link, "collision", name=f"convex_{i:03d}")
        geo = ET.SubElement(node, "geometry")
        ET.SubElement(geo, "mesh", filename=f"collision/{path.name}", scale="1 1 1")
    ET.indent(robot)
    ET.ElementTree(robot).write(
        out / "asset.urdf", encoding="utf-8", xml_declaration=True
    )
    report = {
        "units": "m",
        "up_axis": "z",
        "urdf": "asset.urdf",
        "mass_kg": mass_kg,
        "mass_source": "caller supplied",
        "cavity": cavity,
        "cavity_required": meta.get("cavity_required", False),
        "cavity_collision_check": cavity_collision,
        "inertia_source": "uniform-density reconstructed-solid approximation; replace for accurate dynamics",
        "estimated_center_of_mass_m": center.tolist(),
        "collision_parts": len(parts),
        "collision_union_check": collision["source_volume_comparison"],
        "visual_simplification": visual_report,
        "source_geometry_status": meta["status"],
        "status": "needs_review",
        "friction": "not estimated; configure in PyBullet",
    }
    (out / "package.json").write_text(json.dumps(report, indent=2))
    from .smooth import ensure_smooth_variant

    return ensure_smooth_variant(out) if smooth else report
