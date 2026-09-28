"""Single-object outer silhouettes from independent semantic and color segmentations."""

import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from plyfile import PlyData, PlyElement
from .metric import transform_points


def intersect_outer_masks(semantic, color):
    # Cavities are deliberately outside this mode's scope. One-pixel tolerance for boundaries.
    kernel = np.ones((3, 3), np.uint8)
    a = cv2.dilate(binary_fill_holes(semantic).astype("uint8"), kernel)
    b = cv2.dilate(binary_fill_holes(color).astype("uint8"), kernel)
    combined = a & b
    count, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    if count < 2:
        raise ValueError("semantic and color masks have no common foreground")
    combined = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return binary_fill_holes(combined).astype("uint8")


def isolate_hybrid(
    point_path, model_path, images, transform, out_path, mask_dir, checkpoint
):
    from .sam_masks import isolate_sam
    from .isolate import isolate_multiview

    parent = mask_dir.parent
    isolate_sam(
        point_path,
        model_path,
        images,
        transform,
        parent / "sam_points.ply",
        parent / "masks_sam",
        checkpoint=checkpoint,
    )
    isolate_multiview(
        point_path,
        model_path,
        images,
        transform,
        parent / "color_points.ply",
        parent / "masks_color",
        compact_seed=True,
    )
    return combine_masks(
        point_path,
        model_path,
        images,
        transform,
        out_path,
        mask_dir,
        parent / "masks_sam",
        parent / "masks_color",
    )


def combine_masks(
    point_path,
    model_path,
    images,
    transform,
    out_path,
    mask_dir,
    semantic_dir,
    color_dir,
):
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_path))
    mask_dir.mkdir(parents=True, exist_ok=True)
    data = PlyData.read(point_path)["vertex"].data
    points = np.c_[data["x"], data["y"], data["z"]]
    source = transform_points(points, np.linalg.inv(transform))
    seen = np.zeros(len(points), int)
    hits = np.zeros(len(points), int)
    rows = []
    for image_id in rec.reg_image_ids():
        im = rec.images[image_id]
        camera = rec.cameras[im.camera_id]
        a = cv2.imread(str(semantic_dir / (im.name + ".png")), 0)
        b = cv2.imread(str(color_dir / (im.name + ".png")), 0)
        if a is None or b is None:
            continue
        mask = intersect_outer_masks(a > 0, b > 0)
        cv2.imwrite(str(mask_dir / (im.name + ".png")), mask * 255)
        p = im.cam_from_world().matrix()
        cam = source @ p[:, :3].T + p[:, 3]
        xy = camera.img_from_cam(cam) - 0.5
        h, w = mask.shape
        valid = (
            np.isfinite(xy).all(1)
            & (cam[:, 2] > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < w)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < h)
        )
        ids = np.flatnonzero(valid)
        uv = np.clip(np.rint(xy[ids]).astype(int), [0, 0], [w - 1, h - 1])
        relaxed = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        seen[ids] += 1
        hits[ids] += relaxed[uv[:, 1], uv[:, 0]]
        rows.append(
            {
                "image": im.name,
                "mask_area_fraction": float(mask.mean()),
                "semantic_color_iou": float(
                    np.count_nonzero((a > 0) & (b > 0))
                    / max(np.count_nonzero((a > 0) | (b > 0)), 1)
                ),
            }
        )
    if len(rows) < max(12, 0.8 * rec.num_reg_images()):
        raise ValueError("too few paired segmentation views")
    keep = (seen >= 12) & (hits >= 0.75 * seen)
    if keep.sum() < 500:
        raise ValueError("too few hybrid-filtered surface points")
    PlyData([PlyElement.describe(data[keep].copy(), "vertex")], text=False).write(
        out_path
    )
    report = {
        "backend": "SAM 2.1 tiny intersected with compact-seed GrabCut",
        "input_points": len(data),
        "retained_points": int(keep.sum()),
        "masks": rows,
        "geometry_scope": "outer occupancy only; 2D holes filled; single connected rigid object",
        "limitation": "Methods may share errors or omit the same protrusion; image agreement is not independent certification.",
    }
    (mask_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
