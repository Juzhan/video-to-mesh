"""Photo-based GrabCut masks seeded from high object points, then multiview point filtering.

Designed for a single object on the calibration board. Does not consume synthetic masks or truth.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from .metric import transform_points


def isolate_multiview(
    point_path, model_path, images, transform, out_path, mask_dir, compact_seed=False
):
    import pycolmap

    cloud = PlyData.read(point_path)
    data = cloud["vertex"].data
    points = np.c_[data["x"], data["y"], data["z"]]
    source_points = transform_points(points, np.linalg.inv(transform))
    height = float(np.quantile(points[:, 2], 0.99))
    seeds = points[:, 2] > height * 0.70
    rec = pycolmap.Reconstruction(str(model_path))
    inside = np.zeros(len(points), int)
    seen = np.zeros(len(points), int)
    mask_dir.mkdir(exist_ok=True)
    stats = []
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        camera = rec.cameras[image.camera_id]
        rgb = cv2.imread(str(images / image.name))
        p = image.cam_from_world().matrix()
        cam = source_points @ p[:, :3].T + p[:, 3]
        uv = camera.img_from_cam(cam) - 0.5
        uv = np.rint(np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1)).astype(int)
        h, w = rgb.shape[:2]
        valid = (
            (cam[:, 2] > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        if valid.sum() < 500:
            continue
        coords = uv[valid]
        x0, y0 = np.maximum(coords.min(0) - 12, 0)
        x1, y1 = np.minimum(coords.max(0) + 13, [w, h])
        labels = np.zeros((h, w), np.uint8)
        labels[y0:y1, x0:x1] = cv2.GC_PR_BGD
        high = uv[valid & seeds]
        foreground = np.zeros((h, w), np.uint8)
        # An eroded high-point footprint avoids labeling noisy MVS boundary pixels as certain foreground.
        # This is only an initialization for color segmentation, not the exported silhouette.
        hull = cv2.convexHull(high.astype(np.int32))
        cv2.fillConvexPoly(foreground, hull, 255)
        seed_width = max(5, int(min(np.ptp(high[:, 0]), np.ptp(high[:, 1])) * 0.20))
        core = cv2.erode(foreground, np.ones((seed_width, seed_width), np.uint8))
        if not core.any():
            stats.append(
                {"image": image.name, "rejected": "no reliable foreground core"}
            )
            continue
        if compact_seed:
            # A small central seed avoids forcing stereo hull outliers into foreground.
            distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
            seed_y, seed_x = np.unravel_index(np.argmax(distance), distance.shape)
            radius = max(1, int(distance[seed_y, seed_x] * 0.35))
            core = np.zeros_like(foreground)
            cv2.circle(core, (seed_x, seed_y), radius, 255, -1)
        labels[foreground > 0] = cv2.GC_PR_FGD
        foreground = core
        labels[foreground > 0] = cv2.GC_FGD
        if compact_seed:
            from .board import detect

            for corners in detect(rgb).values():
                cv2.fillConvexPoly(
                    labels, np.rint(corners).astype(np.int32), cv2.GC_BGD
                )
        bg = np.zeros((1, 65), np.float64)
        fg = np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, labels, None, bg, fg, 3, cv2.GC_INIT_WITH_MASK)
        mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
        removed = 0
        if compact_seed:
            _, components = cv2.connectedComponents(mask, connectivity=8)
            label = int(components[seed_y, seed_x])
            if label == 0:
                continue
            kept = (components == label).astype(np.uint8)
            removed = int(mask.sum() - kept.sum())
            mask = kept
        fraction = float(mask.mean())
        if not 0.002 < fraction < 0.65:
            stats.append({"image": image.name, "rejected_area_fraction": fraction})
            continue
        # One-pixel allowance preserves thin boundaries and quantization noise.
        relaxed = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        idx = np.flatnonzero(valid)
        inside[idx] += relaxed[uv[idx, 1], uv[idx, 0]]
        seen[idx] += 1
        cv2.imwrite(str(mask_dir / (image.name + ".png")), mask * 255)
        if len(stats) < 4:
            preview = rgb.copy()
            preview[mask == 0] = (preview[mask == 0] * 0.2).astype(np.uint8)
            cv2.imwrite(str(mask_dir / (image.name + ".preview.jpg")), preview)
        stats.append(
            {
                "image": image.name,
                "mask_area_fraction": fraction,
                "disconnected_pixels_removed": removed,
            }
        )
    keep = (seen >= 12) & (inside >= 0.75 * seen)
    if keep.sum() < 500:
        raise ValueError(
            f"Automatic segmentation retained only {keep.sum()} points. Use a clearer background or review masks."
        )
    PlyData([PlyElement.describe(data[keep].copy(), "vertex")], text=False).write(
        out_path
    )
    report = {
        "backend": "GrabCut compact seed" if compact_seed else "GrabCut broad seed",
        "compact_seed": compact_seed,
        "input_points": len(data),
        "retained_points": int(keep.sum()),
        "foreground_seed_height_m": height * 0.7,
        "agreement_fraction": 0.75,
        "masks": stats,
        "limitation": "Automatic color-based masks can fail on similar object/background colors; inspect previews.",
    }
    (mask_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def clip_mesh_to_masks(mesh, model_path, transform, mask_dir, agreement=0.90):
    """Reject Poisson extrapolation unsupported by the RGB foreground silhouettes."""
    import pycolmap

    points = transform_points(mesh.vertices, np.linalg.inv(transform))
    rec = pycolmap.Reconstruction(str(model_path))
    seen = np.zeros(len(points), int)
    hits = np.zeros(len(points), int)
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        mask = cv2.imread(str(mask_dir / (image.name + ".png")), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        camera = rec.cameras[image.camera_id]
        p = image.cam_from_world().matrix()
        cam = points @ p[:, :3].T + p[:, 3]
        xy = camera.img_from_cam(cam) - 0.5
        xy = np.rint(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
        h, w = mask.shape
        valid = (
            (cam[:, 2] > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < w)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < h)
        )
        indices = np.flatnonzero(valid)
        # Quantization allowance, not a physical inflation distance.
        relaxed = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        hits[indices] += relaxed[xy[indices, 1], xy[indices, 0]] > 0
        seen[indices] += 1
    keep = (seen >= 12) & (hits >= agreement * seen)
    original_faces = len(mesh.faces)
    mesh.update_faces(keep[mesh.faces].all(1))
    mesh.remove_unreferenced_vertices()
    return {
        "agreement": agreement,
        "faces_before": original_faces,
        "faces_after": len(mesh.faces),
    }
