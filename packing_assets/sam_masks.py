"""SAM 2.1 segmentation with automatic prompts from measured geometry, never truth."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from .metric import transform_points
from .progress import event


def object_prompt_cloud(points):
    """Use measured elevated surfaces, excluding low table/printed-board noise.

    Ground projections bound the object's footprint without supplying its size or
    location. They are box hints only, not foreground surface observations.
    """
    cutoff = max(0.002, 0.10 * float(np.quantile(points[:, 2], 0.99)))
    elevated = points[:, 2] > cutoff
    if elevated.sum() < 100:
        raise ValueError(
            "too few elevated object points for automatic foreground prompts"
        )
    high = points[elevated]
    floor = high.copy()
    floor[:, 2] = 0
    return elevated, np.concatenate([high, floor]), cutoff


def support_envelope_cloud(points):
    """Conservative prompt envelope for one connected object on a flat table.

    Coarse depth may lift printed paper and shadow noise several millimetres.
    Use elevated connected measurements; retain nearby disconnected pieces.
    Downward extension accounts for the excluded support-contact band.
    This bounds a mask only; it never becomes the output object's geometry.
    """
    import open3d as o3d

    cutoff = max(0.004, 0.2 * float(np.quantile(points[:, 2], 0.99)))
    high = points[points[:, 2] > cutoff]
    if len(high) < 100:
        raise ValueError("too few elevated measurements for foreground envelope")
    labels = np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(high)).cluster_dbscan(
            0.005, 10
        )
    )
    if not (labels >= 0).any():
        raise ValueError("no coherent foreground measurements for mask envelope")
    counts = np.bincount(labels[labels >= 0])
    primary = high[labels == counts.argmax()]
    lo, hi = np.percentile(primary, [0.5, 99.5], axis=0)
    accepted = []
    for label in np.flatnonzero(counts >= 10):
        center = np.median(high[labels == label], axis=0)
        if ((center >= lo - 0.02) & (center <= hi + 0.02)).all():
            accepted.append(label)
    high = high[np.isin(labels, accepted)]
    lower = high.copy()
    lower[:, 2] = np.maximum(0, lower[:, 2] - cutoff)
    return np.concatenate([high, lower]), {
        "seed_height_m": cutoff,
        "seed_points": len(high),
        "cluster_radius_m": 0.005,
        "nearby_component_margin_m": 0.02,
        "pixel_margin": 12,
        "assumption": "one object with a coherent elevated body; low or widely separated unsupported parts need review",
    }


def trim_to_support_envelope(mask, projected):
    xy = projected[np.isfinite(projected).all(1)]
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
    xy = xy[((xy >= lo) & (xy <= hi)).all(1)]
    if len(xy) < 3:
        raise ValueError("insufficient projected foreground envelope")
    envelope = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(envelope, cv2.convexHull(np.rint(xy).astype("int32")), 1)
    envelope = cv2.dilate(
        envelope, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    clean = mask * envelope
    return clean, float(1 - np.count_nonzero(clean) / max(1, np.count_nonzero(mask)))


def isolate_sam(
    point_path,
    model_path,
    images,
    transform,
    out_path,
    mask_dir,
    checkpoint=None,
    support_envelope=False,
    sparse_prompts=False,
):
    import pycolmap
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = Path(
        checkpoint
        or Path(__file__).resolve().parents[1] / "models/sam2.1_hiera_tiny.pt"
    )
    if not checkpoint.exists():
        raise ValueError("SAM 2 checkpoint missing; run scripts/setup_quality.sh")
    torch.set_num_threads(8)
    model_config = (
        "sam2.1_hiera_b+.yaml"
        if checkpoint.name == "sam2.1_hiera_base_plus.pt"
        else "sam2.1_hiera_t.yaml"
    )
    predictor = SAM2ImagePredictor(
        build_sam2("configs/sam2.1/" + model_config, str(checkpoint), device="cpu")
    )
    data = PlyData.read(point_path)["vertex"].data
    points = np.c_[data["x"], data["y"], data["z"]]
    source = transform_points(points, np.linalg.inv(transform))
    elevated, box_points, seed_height = object_prompt_cloud(points)
    box_source = transform_points(box_points, np.linalg.inv(transform))
    envelope_info = None
    if support_envelope:
        envelope_points, envelope_info = support_envelope_cloud(points)
        envelope_source = transform_points(envelope_points, np.linalg.inv(transform))
        # The elevated cloud can still include scattered lifted floor points.
        # Prompt SAM with the coherent object envelope itself, before prediction.
        # Clipping a floor/board prediction afterwards cannot recover the object.
        box_source = envelope_source
    rec = pycolmap.Reconstruction(str(model_path))
    inside = np.zeros(len(points), int)
    seen = np.zeros(len(points), int)
    mask_dir.mkdir(parents=True, exist_ok=True)
    # A failed/retried view must not silently retain an earlier mask.
    for old in mask_dir.glob("*.png"):
        old.unlink()
    rows = []
    for view_index, image_id in enumerate(rec.reg_image_ids(), 1):
        event("advance", view_index - 1, rec.num_reg_images())
        im = rec.images[image_id]
        camera = rec.cameras[im.camera_id]
        rgb = cv2.imread(str(images / im.name))
        h, w = rgb.shape[:2]
        p = im.cam_from_world().matrix()
        cam = source @ p[:, :3].T + p[:, 3]
        xy = camera.img_from_cam(cam) - 0.5
        valid = (
            np.isfinite(xy).all(1)
            & (cam[:, 2] > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < w)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < h)
        )
        coords = xy[valid]
        if len(coords) < (30 if sparse_prompts else 500):
            continue
        box_cam = box_source @ p[:, :3].T + p[:, 3]
        box_xy = camera.img_from_cam(box_cam) - 0.5
        bv = np.isfinite(box_xy).all(1) & (box_cam[:, 2] > 0)
        hints = box_xy[bv]
        lower = np.maximum(np.percentile(hints, 0.5, axis=0) - 12, 0)
        upper = np.minimum(np.percentile(hints, 99.5, axis=0) + 12, [w - 1, h - 1])
        box = np.r_[lower, upper]
        if sparse_prompts:
            margin = np.maximum(16, (upper - lower) * 0.12)
            box = np.r_[
                np.maximum(lower - margin, 0),
                np.minimum(upper + margin, [w - 1, h - 1]),
            ]
        with torch.inference_mode():
            predictor.set_image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(np.uint8)
        # Retain disconnected object pieces if elevated measured points support
        # them; the adjacent flat paper has no such support.
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        high = xy[valid & elevated]
        ij = np.clip(np.rint(high).astype(int), [0, 0], [w - 1, h - 1])
        votes = np.bincount(labels[ij[:, 1], ij[:, 0]], minlength=int(labels.max()) + 1)
        supported = np.flatnonzero(votes >= 3)
        supported = supported[supported != 0]
        mask = np.isin(labels, supported).astype(np.uint8)
        trimmed = 0.0
        if support_envelope:
            ec = envelope_source @ p[:, :3].T + p[:, 3]
            mask, trimmed = trim_to_support_envelope(
                mask, camera.img_from_cam(ec[ec[:, 2] > 0]) - 0.5
            )
        fraction = float(mask.mean())
        if not 0.002 < fraction < 0.65:
            rows.append({"image": im.name, "rejected_area_fraction": fraction})
            continue
        relaxed = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        indices = np.flatnonzero(valid)
        uv = np.clip(np.rint(coords).astype(int), [0, 0], [w - 1, h - 1])
        inside[indices] += relaxed[uv[:, 1], uv[:, 0]]
        seen[indices] += 1
        cv2.imwrite(str(mask_dir / (im.name + ".png")), mask * 255)
        if len(rows) < 4:
            preview = rgb.copy()
            preview[mask == 0] = (preview[mask == 0] * 0.2).astype(np.uint8)
            cv2.imwrite(str(mask_dir / (im.name + ".preview.jpg")), preview)
        rows.append(
            {
                "image": im.name,
                "mask_area_fraction": fraction,
                "model_score": float(scores[0]),
                "automatic_box": box.tolist(),
                "prompt_type": "coherent_measurements_box"
                if support_envelope
                else "elevated_measurements_box",
                "support_envelope_removed_fraction": trimmed,
            }
        )
        print(f"SAM {len(rows)}/{rec.num_reg_images()}: {im.name}", flush=True)
        event("advance", view_index, rec.num_reg_images())
    keep = (seen >= 12) & (inside >= 0.75 * seen)
    report = {
        "backend": "SAM 2.1 " + ("base plus" if "b+" in model_config else "tiny"),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "input_points": len(data),
        "retained_points": int(keep.sum()),
        "masks": rows,
        "point_prompts": False,
        "prompt_revision": 6 if support_envelope else 4,
        "foreground_seed_height_m": seed_height,
        "support_envelope": envelope_info,
        "limitation": "Learned segmentation is not independent geometry validation; holes and thin edges require review.",
    }
    (mask_dir / "report.json").write_text(json.dumps(report, indent=2))
    if keep.sum() < (30 if sparse_prompts else 500):
        raise ValueError(
            f"too few object points remain after SAM segmentation ({keep.sum()}/{len(data)}); see masks/report.json"
        )
    PlyData([PlyElement.describe(data[keep].copy(), "vertex")], text=False).write(
        out_path
    )
    return report
