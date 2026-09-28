"""Metric visual hull for approximate packing geometry, using estimated poses and RGB masks.

Silhouettes cannot recover hidden concavities. Keep this proxy separate from measured MVS surfaces.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
import trimesh

from .assets import export_asset
from .progress import event


def voxel_mesh(occupancy, origin, pitch, smoothing=False):
    """Grid cells are centered at origin + (index + .5) * pitch."""
    from skimage.measure import marching_cubes

    if not occupancy.any():
        raise ValueError("empty silhouette volume after height clipping")
    labels, components = ndimage.label(occupancy)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    occupancy = labels == np.argmax(counts)
    truncated = any(
        [
            occupancy[0].any(),
            occupancy[-1].any(),
            occupancy[:, 0].any(),
            occupancy[:, -1].any(),
            occupancy[:, :, -1].any(),
        ]
    )
    if truncated:
        raise ValueError(
            "proxy touches scan-region boundary; enlarge board/ROI or improve masks"
        )
    field = np.pad(occupancy, 1).astype("f4")
    if smoothing:
        field = ndimage.gaussian_filter(field, 0.6)
    vertices, faces, _, _ = marching_cubes(
        field, level=0.5, spacing=(pitch, pitch, pitch)
    )
    vertices += np.asarray(origin) - 0.5 * pitch
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    if not mesh.is_volume:
        # Metric vertex welding can join distinct, very close sheets and break
        # the indexed isosurface. Retry its topology without tolerance welding.
        vertices, faces, _, _ = marching_cubes(
            field, level=0.5, spacing=(pitch, pitch, pitch), allow_degenerate=False
        )
        vertices += np.asarray(origin) - 0.5 * pitch
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh.fix_normals(multibody=True)
    if not mesh.is_volume:
        raise ValueError("proxy is not a closed consistently oriented volume")
    return mesh, components


def proxy_scan_roi(board, measured, pitch):
    roi = np.array(board["suggested_roi_m"], float)
    if board.get("proxy_crop_to_observed"):
        # A small scale reference need not bound a large or off-center object.
        # Conservative observed XY bounds avoid allocating the entire wide table.
        xy = np.column_stack([measured[a] for a in ("x", "y")])
        if not len(xy) or not np.isfinite(xy).all():
            raise ValueError("invalid measured points for adaptive scan region")
        margin = np.maximum(0.02, 0.1 * np.ptp(xy, axis=0))
        roi[0, :2] = np.maximum(roi[0, :2], xy.min(0) - margin)
        roi[1, :2] = np.minimum(roi[1, :2], xy.max(0) + margin)
    measured_height = float(np.quantile(measured["z"], 0.99))
    if not np.isfinite(measured_height) or measured_height <= pitch:
        raise ValueError("insufficient measured height for a packing proxy")
    roi[1, 2] = min(roi[1, 2], measured_height + 3 * pitch)
    return roi, measured_height


def silhouette_observations(camera, cam, mask):
    """Outside the camera image is unknown, never evidence of empty space."""
    xy = camera.img_from_cam(cam) - 0.5
    finite = np.isfinite(xy).all(1)
    xy = np.rint(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
    h, w = mask.shape
    seen = (
        finite
        & (cam[:, 2] > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < h)
    )
    missed = np.zeros(len(cam), bool)
    rows = np.flatnonzero(seen)
    missed[rows] = mask[xy[rows, 1], xy[rows, 0]] == 0
    return seen, missed


def silhouette_consensus(observations, misses, agreement, minimum_views=12):
    return (observations >= minimum_views) & (
        misses <= np.floor((1 - agreement) * observations + 1e-7)
    )


def build_proxy(
    work: Path,
    board: dict,
    out: Path,
    pitch=0.0015,
    agreement=0.97,
    collision=False,
    smoothing=False,
    pixel_margin=1,
):
    import pycolmap
    from plyfile import PlyData

    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("voxel pitch must be positive")
    if not 0 < agreement <= 1:
        raise ValueError("silhouette agreement must be in (0, 1]")
    if not isinstance(pixel_margin, int) or not 0 <= pixel_margin <= 1:
        raise ValueError("pixel margin must be zero or one")
    metric = json.loads((work / "metric.json").read_text())
    inv = np.linalg.inv(np.array(metric["transform_to_m_z_up"]))
    reconstruction = pycolmap.Reconstruction(str(work / "sparse"))
    views = []
    for image_id in reconstruction.reg_image_ids():
        im = reconstruction.images[image_id]
        mask = cv2.imread(
            str(work / "masks" / (im.name + ".png")), cv2.IMREAD_GRAYSCALE
        )
        if mask is not None:
            # Same one-pixel rasterization allowance as point/surface clipping.
            # Quality checks below still use the original, undilated masks.
            if pixel_margin:
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            views.append((im, reconstruction.cameras[im.camera_id], mask))
    if len(views) < 12:
        raise ValueError("not enough masks for a metric packing proxy")
    # Above-only capture leaves a visual hull unbounded above flat tops. Use measured MVS height,
    # robust to isolated stereo outliers. This can miss tiny top protrusions: report the assumption.
    measured = PlyData.read(work / "masked_object_points.ply")["vertex"].data
    roi, measured_height = proxy_scan_roi(board, measured, pitch)
    dimensions = np.ceil((roi[1] - roi[0]) / pitch).astype(int)
    if np.prod(dimensions) > 15000000:
        raise ValueError(
            "proxy grid too large; increase voxel pitch or reduce scan region"
        )
    axes = [roi[0, i] + (np.arange(dimensions[i]) + 0.5) * pitch for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    indices = np.arange(len(points))
    misses = np.zeros(len(points), np.int32)
    observations = np.zeros(len(points), np.int32)
    max_misses = int(np.floor((1 - agreement) * len(views) + 1e-7))
    for count, (im, camera, mask) in enumerate(views, 1):
        p = np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]]) @ inv
        keep = np.empty(len(points), bool)
        for start in range(0, len(points), 200000):
            stop = start + 200000
            cam = points[start:stop] @ p[:3, :3].T + p[:3, 3]
            visible, missed = silhouette_observations(camera, cam, mask)
            observations[start:stop] += visible
            misses[start:stop] += missed
            keep[start:stop] = misses[start:stop] <= max_misses
        points, indices, misses, observations = (
            points[keep],
            indices[keep],
            misses[keep],
            observations[keep],
        )
        if count % 6 == 0:
            print(f"PROXY {count}/{len(views)} views: {len(points)} voxels", flush=True)
        event("advance", count, len(views), "视角")
        if len(points) < 50:
            raise ValueError("masks are inconsistent; no reliable silhouette volume")
    # The threshold applies to actual observations of each voxel, not all frames.
    enough = silhouette_consensus(observations, misses, agreement)
    occupancy = np.zeros(tuple(dimensions), bool)
    occupancy.ravel()[indices[enough]] = True
    occupancy[:, :, axes[2] > measured_height] = False
    mesh, components = voxel_mesh(occupancy, roi[0], pitch, smoothing)
    provenance = {
        "backend": "metric_visual_hull",
        "scale_source": "triangulated_board",
        "calibration": metric,
        "voxel_pitch_m": pitch,
        "silhouette_agreement": agreement,
        "input_views": len(views),
        "components_before_filter": int(components),
        "scan_roi_m": roi.tolist(),
        "crop_to_observed_xy": bool(board.get("proxy_crop_to_observed")),
        "smoothing_sigma_voxels": 0.6 if smoothing else 0,
        "visibility_revision": 2,
        "outside_image_policy": "unknown; ignored, never carved",
        "silhouette_pixel_margin": pixel_margin,
        "minimum_observed_views": 12,
        "height_limit_m": measured_height,
        "height_source": "99th percentile of photo-masked metric MVS points; no true dimensions supplied",
        "limitation": "Silhouette outer volume: hidden cavities (e.g. inside a mug) are filled; not suitable for nesting or cavity-dependent contact.",
    }
    report = export_asset(mesh, out, provenance, collision=collision)
    report["review_reasons"].append("hidden_concavities_filled_by_visual_hull")
    report["review_reasons"].append("robust_height_limit_may_omit_tiny_protrusions")
    report["review_reasons"].append(
        "approximation_not_guaranteed_to_enclose_real_object"
    )
    (out / "asset.json").write_text(json.dumps(report, indent=2))
    return report
