"""Carve observed free space from the silhouette bound using metric RGB stereo depth."""

import json
from pathlib import Path
import numpy as np
import trimesh
from .assets import load_mesh, export_asset
from .quality import ray_scene, check_views, surface_release_gate
from .proxy import voxel_mesh
from .progress import event


def cavity_probe(mesh, pitch=0.003):
    """Find top-accessible depressions, without assuming the cavity is at the bbox centre."""
    import open3d as o3d

    lo, hi = mesh.bounds
    xs = np.arange(lo[0] + pitch, hi[0], pitch)
    ys = np.arange(lo[1] + pitch, hi[1], pitch)
    xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    rays = np.c_[
        xy, np.full(len(xy), hi[2] + 0.01), np.zeros((len(xy), 2)), np.full(len(xy), -1)
    ].astype("f4")
    hit = ray_scene(mesh).cast_rays(o3d.core.Tensor(rays), nthreads=2)["t_hit"].numpy()
    depth = hit - 0.01
    good = (
        np.isfinite(depth)
        & (depth > max(0.012, (hi[2] - lo[2]) * 0.2))
        & (depth < (hi[2] - lo[2]) * 0.98)
    )
    # Deep hits alone can be outside sloped outer walls; require surrounding rim hits.
    shape = (len(xs), len(ys))
    shallow = np.isfinite(depth) & (depth < max(0.005, (hi[2] - lo[2]) * 0.10))
    from scipy.ndimage import binary_fill_holes, binary_erosion, label

    interior = (
        binary_fill_holes(shallow.reshape(shape))
        & ~shallow.reshape(shape)
        & good.reshape(shape)
    )
    labels, n = label(interior)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if n and counts.max() * pitch * pitch > 0.0001:
        selected = np.flatnonzero(labels.ravel() == counts.argmax())
        center = selected[
            np.argmin(np.linalg.norm(xy[selected] - xy[selected].mean(0), axis=1))
        ]
        safe = np.flatnonzero(
            binary_erosion(labels == counts.argmax(), iterations=2).ravel()
        )
        safe = safe if len(safe) else selected
        mean = xy[safe].mean(0)
        span = np.ptp(xy[safe], axis=0) * 0.25
        targets = [
            mean,
            mean + [span[0], 0],
            mean - [span[0], 0],
            mean + [0, span[1]],
            mean - [0, span[1]],
        ]
        probes = []
        for target in targets:
            k = safe[np.argmin(np.linalg.norm(xy[safe] - target, axis=1))]
            probes.append(
                {
                    "xy_m": xy[k].tolist(),
                    "floor_z_m": float(hi[2] - depth[k]),
                    "rim_z_m": float(hi[2]),
                }
            )
        floor_z = float(hi[2] - depth[center])
        probe_z = float((hi[2] + floor_z) / 2)
        return {
            "detected": True,
            "opening_area_m2": float(len(selected) * pitch * pitch),
            "depth_m": float(depth[center]),
            "probe_m": [float(xy[center, 0]), float(xy[center, 1]), probe_z],
            "floor_z_m": floor_z,
            "probe_columns": probes,
            "method": "downward rays through a region enclosed by measured rim geometry",
        }
    return {
        "detected": False,
        "method": "no sufficiently large top opening enclosed by rim found",
    }


def build_depth_proxy(
    work: Path,
    bound_asset: Path,
    out: Path,
    pitch=0.001,
    min_views=3,
    clearance=0.002,
    require_cavity=False,
    depth_work=None,
):
    import pycolmap
    import open3d as o3d

    depth_work = depth_work or work
    meta = json.loads((bound_asset / "asset.json").read_text())
    mesh = load_mesh(bound_asset / "visual.obj")
    mesh.apply_translation(-np.array(meta["centering_translation_m"]))
    metric = json.loads((work / "metric.json").read_text())
    transform = np.array(metric["transform_to_m_z_up"])
    inv = np.linalg.inv(transform)
    scale = np.cbrt(np.linalg.det(transform[:3, :3]))
    rec = pycolmap.Reconstruction(str(depth_work / "dense/sparse"))
    lo = mesh.bounds[0] - 3 * pitch
    hi = mesh.bounds[1] + 3 * pitch
    counts = np.ceil((hi - lo) / pitch).astype(int)
    if np.prod(counts) > 15000000:
        raise ValueError("depth proxy grid too large")
    axes = [lo[i] + (np.arange(counts[i]) + 0.5) * pitch for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype("f4")
    scene = ray_scene(mesh)
    inside = np.zeros(len(grid), bool)
    for start in range(0, len(grid), 200000):
        inside[start : start + 200000] = (
            scene.compute_occupancy(
                o3d.core.Tensor(grid[start : start + 200000]), nthreads=2, nsamples=3
            ).numpy()
            > 0.5
        )
    indices = np.flatnonzero(inside)
    points = grid[indices]
    del grid
    free = np.zeros(len(points), np.uint16)
    used = 0
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        camera = rec.cameras[image.camera_id]
        path = depth_work / "dense/stereo/depth_maps" / (image.name + ".geometric.bin")
        if not path.is_file():
            continue
        dm = pycolmap.DepthMap()
        dm.read(str(path))
        depth = dm.to_array()
        matrix = np.vstack([image.cam_from_world().matrix(), [0, 0, 0, 1]]) @ inv
        for start in range(0, len(points), 150000):
            cam = points[start : start + 150000] @ matrix[:3, :3].T + matrix[:3, 3]
            xy = camera.img_from_cam(cam)
            xy *= np.array(
                [depth.shape[1] / camera.width, depth.shape[0] / camera.height]
            )
            ij = np.rint(np.nan_to_num(xy - 0.5, nan=-1, posinf=-1, neginf=-1)).astype(
                int
            )
            valid = (
                (cam[:, 2] > 0)
                & (ij[:, 0] >= 0)
                & (ij[:, 0] < depth.shape[1])
                & (ij[:, 1] >= 0)
                & (ij[:, 1] < depth.shape[0])
            )
            seen = np.flatnonzero(valid)
            d = depth[ij[seen, 1], ij[seen, 0]]
            carved = (d > 0) & np.isfinite(d) & ((d - cam[seen, 2]) * scale > clearance)
            free[start + seen[carved]] += 1
        used += 1
        event("advance", used, rec.num_reg_images(), "视角")
    if used < 12:
        raise ValueError("not enough geometric depth views for cavity reconstruction")
    # Preserve a small uncertain exterior skin around thin handles/side walls.
    # Upward-facing top caps remain carveable, so this never deliberately seals a cup.
    protected = np.zeros(len(points), bool)
    for start in range(0, len(points), 150000):
        sample = points[start : start + 150000]
        closest = scene.compute_closest_points(o3d.core.Tensor(sample), nthreads=2)
        distance = np.linalg.norm(closest["points"].numpy() - sample, axis=1)
        upward = (closest["primitive_normals"].numpy()[:, 2] > 0.4) & (
            sample[:, 2] > mesh.bounds[1, 2] - 0.012
        )
        protected[start : start + 150000] = (distance < 0.002) & ~upward
    occupancy = np.zeros(tuple(counts), bool)
    kept = (free < min_views) | protected
    occupancy.ravel()[indices[kept]] = True
    carved_mesh, _ = voxel_mesh(occupancy, lo, pitch, smoothing=True)
    views = check_views(carved_mesh, work)
    gate = surface_release_gate(carved_mesh, views)
    report = export_asset(
        carved_mesh,
        out,
        {
            "backend": "RGB_stereo_free_space_carving",
            "calibration": metric,
            "scale_source": "triangulated_board",
            "depth_views": used,
            "min_free_views": min_views,
            "depth_clearance_m": clearance,
            "voxel_pitch_m": pitch,
            "removed_fraction": float((~kept).mean()),
            "uncertain_exterior_skin_m": 0.002,
            "limitation": "Only observed free space is recovered; unseen interior still requires additional views.",
        },
    )
    result_mesh = load_mesh(out / "visual.obj")
    cavity = cavity_probe(result_mesh)
    if require_cavity and not cavity["detected"]:
        gate["status"] = "rejected"
        gate["reasons"].append("required_open_cavity_not_observed")
    report.update(
        status=gate["status"],
        quality=gate,
        cavity=cavity,
        cavity_required=require_cavity,
    )
    report["review_reasons"].append(
        "visible_cavity_depth_is_approximate_and_unseen_surfaces_unverified"
    )
    (out / "asset.json").write_text(json.dumps(report, indent=2))
    (out / "quality.json").write_text(
        json.dumps({"views": views, "gate": gate, "cavity": cavity}, indent=2)
    )
    return report
