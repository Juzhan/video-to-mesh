"""Measured radial-profile regularization; retain observed external appendages separately."""

import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from .assets import load_mesh, export_asset
from .quality import ray_scene, check_views, surface_release_gate
from .depth_proxy import cavity_probe


def measured_inner_floor(points, normals, center, radius, base_z, current_floor_z):
    """Extrapolate a measured, well-supported central horizontal floor plane.

    Do not choose a floor from an isolated deepest point or a one-sided wall patch.
    The circular-body prior allows a supported annulus to span an unobserved centre.
    """
    import open3d as o3d

    radii = np.linalg.norm(points[:, :2] - center, axis=1)
    valid = (
        (radii < radius * 0.45)
        & (points[:, 2] > base_z + 0.002)
        & (points[:, 2] < current_floor_z + 0.003)
        & (normals[:, 2] > 0.85)
    )
    candidates = points[valid]
    if len(candidates) < 100:
        return {"accepted": False, "reason": "too_few_central_floor_observations"}
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(candidates))
    o3d.utility.random.seed(17)
    plane, indices = cloud.segment_plane(0.0015, 3, 300)
    inliers = candidates[indices]
    if abs(plane[2]) < np.cos(np.deg2rad(5)) or len(inliers) < max(
        100, 0.7 * len(candidates)
    ):
        return {
            "accepted": False,
            "reason": "central_measurements_do_not_support_horizontal_floor",
        }
    angles = np.arctan2(inliers[:, 1] - center[1], inliers[:, 0] - center[0])
    sectors = len(
        np.unique(np.floor((angles + np.pi) * 12 / (2 * np.pi)).astype(int) % 12)
    )
    ordered = np.sort(angles)
    max_gap = float(np.max(np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])))
    # Angular-gap coverage is invariant to the arbitrary zero of angle bins.
    if max_gap > np.pi / 2 or np.min(np.ptp(inliers[:, :2], axis=0)) < radius * 0.5:
        return {
            "accepted": False,
            "reason": "floor_support_does_not_span_enough_directions",
        }
    floor_z = float(-(plane[:2] @ center + plane[3]) / plane[2])
    floor_radius = float(
        np.quantile(np.linalg.norm(inliers[:, :2] - center, axis=1), 0.8)
    )
    if not base_z + 0.002 < floor_z < current_floor_z - 0.002:
        return {"accepted": False, "reason": "no_supported_lower_floor"}
    return {
        "accepted": True,
        "z_m": floor_z,
        "radius_m": floor_radius,
        "inlier_points": len(inliers),
        "candidate_points": len(candidates),
        "azimuth_sectors_of_12": sectors,
        "largest_unobserved_azimuth_gap_degrees": float(np.degrees(max_gap)),
        "method": "horizontal plane fitted to central RGB-MVS points; circular flat-floor extrapolation into unseen centre",
    }


def regularize_round_container(
    work: Path, depth_asset: Path, bound_asset: Path, out: Path
):
    import open3d as o3d

    bound_meta = json.loads((bound_asset / "asset.json").read_text())
    depth_meta = json.loads((depth_asset / "asset.json").read_text())
    if not depth_meta.get("cavity", {}).get("detected"):
        raise ValueError("no observed cavity supports a container fit")
    bound = load_mesh(bound_asset / "visual.obj")
    bound.apply_translation(-np.array(bound_meta["centering_translation_m"]))
    measured = load_mesh(depth_asset / "visual.obj")
    measured.apply_translation(-np.array(depth_meta["centering_translation_m"]))
    lo, hi = bound.bounds
    top = bound.vertices[bound.vertices[:, 2] > hi[2] - 0.004, :2]
    rim = top[ConvexHull(top).vertices]
    fit = np.linalg.lstsq(
        np.c_[2 * rim, np.ones(len(rim))], np.sum(rim * rim, axis=1), rcond=None
    )[0]
    center = fit[:2]
    radii = np.linalg.norm(rim - center, axis=1)
    radius = float(np.median(radii))
    if np.percentile(abs(radii - radius), 90) > 0.003:
        raise ValueError("rim is not sufficiently circular for this prior")
    appendages = (
        np.max(np.linalg.norm(bound.vertices[:, :2] - center, axis=1)) > radius + 0.004
    )
    z = np.arange(lo[2] + 0.0015, hi[2] - 0.0005, 0.0015)
    r = np.arange(0.0005, radius + 0.004, 0.001)
    angle = np.arange(48) * 2 * np.pi / 48
    xyz = np.empty((len(z), len(angle), len(r), 3), dtype="f4")
    xyz[:, :, :, 0] = center[0] + np.cos(angle)[None, :, None] * r[None, None, :]
    xyz[:, :, :, 1] = center[1] + np.sin(angle)[None, :, None] * r[None, None, :]
    xyz[:, :, :, 2] = z[:, None, None]
    sample = o3d.core.Tensor(xyz.reshape(-1, 3))
    outer = (
        ray_scene(bound)
        .compute_occupancy(sample, nthreads=2, nsamples=3)
        .numpy()
        .reshape(xyz.shape[:-1])
        > 0.5
    )
    material = (
        ray_scene(measured)
        .compute_occupancy(sample, nthreads=2, nsamples=3)
        .numpy()
        .reshape(xyz.shape[:-1])
        > 0.5
    )
    outer_r = np.max(outer * r, axis=2)
    if (
        np.median(
            np.percentile(outer_r, 75 if appendages else 90, axis=1)
            - np.percentile(outer_r, 25 if appendages else 10, axis=1)
        )
        > 0.004
    ):
        raise ValueError("outer body is not rotationally consistent")
    # Radius samples mark cell centres: half a radial step estimates the boundary.
    exterior = gaussian_filter1d(np.median(outer_r, axis=1) + 0.0005, 1)
    air = outer & ~material & (r[None, None, :] < exterior[:, None, None] - 0.002)
    support = (air.sum(axis=2) >= 4).mean(axis=1)
    inner_r = np.max(air * r, axis=2)
    valid = (support >= 0.7) & (np.median(inner_r, axis=1) > 0.006)
    persistent = np.convolve(valid.astype(int), np.ones(3, dtype=int), "valid") == 3
    if not persistent.any():
        raise ValueError("no sustained multiazimuth interior evidence")
    start = int(np.flatnonzero(persistent)[0])
    valid[:start] = False
    inner = gaussian_filter1d(
        np.interp(z, z[valid], np.median(inner_r, axis=1)[valid] + 0.0005), 1
    )
    inner = np.clip(inner, 0.002, exterior - 0.002)
    # Use measured profiles, with a flat approximate unobserved underside and inner floor.
    outer_indices = list(range(0, len(z), 2))
    inner_indices = list(range(start, len(z), 2))
    if outer_indices[-1] != len(z) - 1:
        outer_indices.append(len(z) - 1)
    if inner_indices[-1] != len(z) - 1:
        inner_indices.append(len(z) - 1)
    profile = [[0, lo[2]], [exterior[0], lo[2]]] + [
        [exterior[i], z[i]] for i in outer_indices
    ]
    profile += [[exterior[-1], hi[2]], [inner[-1], hi[2]]] + [
        [inner[i], z[i]] for i in inner_indices[::-1]
    ]
    from plyfile import PlyData

    observed = PlyData.read(work / "masked_object_points.ply")["vertex"].data
    floor_fit = measured_inner_floor(
        np.c_[observed["x"], observed["y"], observed["z"]],
        np.c_[observed["nx"], observed["ny"], observed["nz"]],
        center,
        radius,
        lo[2],
        z[start],
    )
    floor_fits_wall = floor_fit["accepted"] and floor_fit["radius_m"] < min(
        inner[start], np.interp(floor_fit["z_m"], z, exterior) - 0.002
    )
    if floor_fits_wall:
        profile += [[floor_fit["radius_m"], floor_fit["z_m"]], [0, floor_fit["z_m"]]]
    else:
        if floor_fit["accepted"]:
            floor_fit = {
                "accepted": False,
                "reason": "floor_radius_incompatible_with_observed_wall",
            }
        profile += [[0, z[start]]]
    mesh = trimesh.creation.revolve(np.array(profile), sections=96)
    mesh.apply_translation([*center, 0])
    appendage_mesh = None
    if appendages:
        from .proxy import voxel_mesh

        pitch = 0.001
        origin = bound.bounds[0] - 3 * pitch
        counts = np.ceil((bound.bounds[1] - origin) / pitch).astype(int) + 3
        axes = [origin[i] + (np.arange(counts[i]) + 0.5) * pitch for i in range(3)]
        points = (
            np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype("f4")
        )
        far = bound.vertices[
            np.argmax(np.linalg.norm(bound.vertices[:, :2] - center, axis=1)), :2
        ]
        direction = (far - center) / np.linalg.norm(far - center)
        beyond = (points[:, :2] - center) @ direction > radius - 0.003
        observed = np.zeros(len(points), bool)
        fitted = np.zeros(len(points), bool)
        measured_scene = ray_scene(measured)
        fitted_scene = ray_scene(mesh)
        for begin in range(0, len(points), 150000):
            sample = o3d.core.Tensor(points[begin : begin + 150000])
            observed[begin : begin + 150000] = (
                measured_scene.compute_occupancy(sample, nthreads=2, nsamples=3).numpy()
                > 0.5
            )
            fitted[begin : begin + 150000] = (
                fitted_scene.compute_occupancy(sample, nthreads=2, nsamples=3).numpy()
                > 0.5
            )
        extra = observed & beyond
        appendage_mesh, _ = voxel_mesh(
            extra.reshape(counts), origin, pitch, smoothing=True
        )
        mesh, _ = voxel_mesh(
            (extra | fitted).reshape(counts), origin, pitch, smoothing=True
        )
    views = check_views(mesh, work)
    gate = surface_release_gate(mesh, views)
    report = export_asset(
        mesh,
        out,
        {
            "backend": "measured_round_container_prior",
            "scale_source": "triangulated_board",
            "calibration": depth_meta["provenance"]["calibration"],
            "source_depth_asset": str(depth_asset.resolve()),
            "circular_rim_p90_residual_mm": float(
                np.percentile(abs(radii - radius), 90) * 1000
            ),
            "prior_revision": 3,
            "measured_floor_fit": floor_fit,
            "interior_azimuth_support_min": 0.7,
            "minimum_wall_m": 0.002,
            "limitation": "Assumes rotational symmetry and regularizes noisy measured profiles; inner floor and wall thickness remain approximate. No true mesh/dimensions supplied.",
        },
    )
    shift = np.array(report["centering_translation_m"])
    report["provenance"]["radial_profile_m"] = [
        [float(r), float(z + shift[2])] for r, z in profile
    ]
    report["provenance"]["axis_xy_m"] = (center + shift[:2]).tolist()
    if appendage_mesh is not None:
        appendage_mesh.apply_translation(shift)
        appendage_mesh.export(out / "appendage.obj")
        report["provenance"]["appendage_mesh"] = "appendage.obj"
        report["provenance"]["appendage_source"] = (
            "measured stereo-carved geometry beyond the circular body; retained in the same frame"
        )
    cavity = cavity_probe(load_mesh(out / "visual.obj"))
    if not cavity["detected"]:
        gate["status"] = "rejected"
        gate["reasons"].append("regularization_lost_opening")
    report.update(
        status=gate["status"], quality=gate, cavity=cavity, cavity_required=True
    )
    (out / "asset.json").write_text(json.dumps(report, indent=2))
    return report
