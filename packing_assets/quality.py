"""Geometry checks independent of appearance textures. Never supplies truth to reconstruction."""

import json
from pathlib import Path

import cv2
import numpy as np


def ray_scene(mesh):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32),
    )
    return scene


def silhouette(scene, camera, image, metric_transform, stride=4):
    """Raycast in board metres using the estimated camera and its full distortion model."""
    import open3d as o3d

    yy, xx = np.mgrid[0 : camera.height : stride, 0 : camera.width : stride]
    xy = camera.cam_from_img(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5])
    cam_to_board = metric_transform @ np.linalg.inv(
        np.vstack([image.cam_from_world().matrix(), [0, 0, 0, 1]])
    )
    directions = np.c_[xy, np.ones(len(xy))] @ cam_to_board[:3, :3].T
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    origins = np.broadcast_to(cam_to_board[:3, 3], directions.shape)
    rays = np.c_[origins, directions].astype("f4")
    depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.isfinite(depth).reshape(xx.shape)


def mask_scores(predicted, reference):
    intersection = np.count_nonzero(predicted & reference)
    union = np.count_nonzero(predicted | reference)
    return {
        "iou": intersection / max(union, 1),
        "precision": intersection / max(np.count_nonzero(predicted), 1),
        "recall": intersection / max(np.count_nonzero(reference), 1),
    }


def clean_single_object_mask(mask, max_removed_fraction=0.02):
    """Remove small disconnected SAM islands, never erode connected thin protrusions.

    If fragmentation is substantial, preserve it and flag ambiguity rather than hide it.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 2:
        return mask, {"removed_pixels": 0, "removed_fraction": 0.0, "ambiguous": False}
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    removed = int(areas.sum() - areas.max())
    fraction = removed / max(int(areas.sum()), 1)
    ambiguous = fraction > max_removed_fraction
    return (mask if ambiguous else labels == largest), {
        "removed_pixels": 0 if ambiguous else removed,
        "removed_fraction": 0.0 if ambiguous else fraction,
        "ambiguous": bool(ambiguous),
    }


def convex_mask(mask):
    coords = np.argwhere(mask)
    if len(coords) < 3:
        return mask.copy()
    result = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(result, cv2.convexHull(coords[:, ::-1].astype(np.int32)), 1)
    return result > 0


def silhouette_support_error(predicted, reference, mm_per_pixel):
    directions = np.c_[
        np.cos(np.arange(32) * 2 * np.pi / 32), np.sin(np.arange(32) * 2 * np.pi / 32)
    ]
    values = []
    for mask in [predicted, reference]:
        coords = np.argwhere(mask)[:, ::-1]
        if len(coords) < 3:
            return {"under_mm": 1e6, "over_mm": 1e6}
        hull = cv2.convexHull(coords.astype(np.float32)).reshape(-1, 2)
        values.append((hull @ directions.T).max(0))
    delta = (values[0] - values[1]) * mm_per_pixel
    return {
        "under_mm": float(max(0, -delta.min())),
        "over_mm": float(max(0, delta.max())),
    }


def check_views(mesh_in_board_frame, work: Path, mask_dir=None, convex_reference=False):
    import pycolmap

    rec = pycolmap.Reconstruction(str(work / "sparse"))
    transform = np.array(
        json.loads((work / "metric.json").read_text())["transform_to_m_z_up"]
    )
    scene = ray_scene(mesh_in_board_frame)
    mask_dir = mask_dir or work / "masks"
    rows = []
    for image_id in rec.reg_image_ids():
        im = rec.images[image_id]
        mask = cv2.imread(str(mask_dir / (im.name + ".png")), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        camera = rec.cameras[im.camera_id]
        stride = 2 if convex_reference else 4
        predicted = silhouette(scene, camera, im, transform, stride=stride)
        full_reference = mask > 0
        extra = {}
        if convex_reference:
            full_reference, cleanup = clean_single_object_mask(full_reference)
            extra["mask_cleanup"] = cleanup
        reference = full_reference[::stride, ::stride]
        if convex_reference:
            reference = convex_mask(reference)
            center = mesh_in_board_frame.bounds.mean(0)
            world = (np.linalg.inv(transform) @ np.r_[center, 1])[:3]
            cam = im.cam_from_world().matrix() @ np.r_[world, 1]
            depth_m = cam[2] * np.cbrt(np.linalg.det(transform[:3, :3]))
            mm_per_pixel = (
                1000
                * depth_m
                / ((camera.focal_length_x + camera.focal_length_y) / 2)
                * stride
            )
            extra.update(silhouette_support_error(predicted, reference, mm_per_pixel))
        rows.append({"image": im.name, **mask_scores(predicted, reference), **extra})
    if len(rows) < 12:
        raise ValueError("at least 12 masks are required for view-consistency checks")
    result = {
        "views": rows,
        "median_iou": float(np.median([r["iou"] for r in rows])),
        "median_recall": float(np.median([r["recall"] for r in rows])),
        "p10_recall": float(np.percentile([r["recall"] for r in rows], 10)),
        "limitation": "Consistency with inferred RGB masks is not independent proof of geometry; hidden cavities are untested.",
    }
    if convex_reference:
        result["reference"] = (
            "2D convex hull of single-object masks, removing disconnected islands only when their combined area is at most 2%; cavities intentionally ignored"
        )
        result["ambiguous_mask_views"] = sum(
            r["mask_cleanup"]["ambiguous"] for r in rows
        )
        result["p90_under_mm"] = float(np.percentile([r["under_mm"] for r in rows], 90))
        result["p90_over_mm"] = float(np.percentile([r["over_mm"] for r in rows], 90))
        result["support_scale_note"] = (
            "Pixel error converted at model-center camera depth; approximate, not a physical certification."
        )
    return result


def surface_release_gate(mesh, view_report):
    reasons = []
    if not mesh.is_volume:
        reasons.append("not_closed_or_consistently_oriented")
    if view_report["median_recall"] < 0.95 or view_report["p10_recall"] < 0.90:
        reasons.append("missing_visible_object_area")
    if view_report["median_iou"] < 0.90:
        reasons.append("view_silhouette_disagreement")
    return {
        "status": "rejected" if reasons else "needs_review",
        "reasons": reasons,
        "policy": {
            "median_recall_min": 0.95,
            "p10_recall_min": 0.90,
            "median_iou_min": 0.90,
            "closed_volume_required": True,
        },
        "auto_approved": False,
    }
