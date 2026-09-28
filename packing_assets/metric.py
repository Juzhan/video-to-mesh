"""Recover a metric similarity from triangulated board corners, never pixel ratios."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .board import detect


def similarity(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 3
    ):
        raise ValueError("need matching Nx3 points, N >= 3")
    x, y = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("collinear scale controls")
    u, sigma, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.array([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ np.diag(signs) @ vt
    scale = (sigma * signs).sum() / (x * x).sum() * len(x)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid metric scale")
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target.mean(0) - scale * rotation @ source.mean(0)
    return transform


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def triangulate(observations, max_error_px=2.0):
    """Each observation contains a world-to-camera 3x4 matrix, normalized xy and focal length."""
    if len(observations) < 3:
        raise ValueError("each marker corner needs at least three registered views")
    ps = np.array([o["P"] for o in observations])
    xy = np.array([o["xy"] for o in observations])
    focal = np.array([o["focal"] for o in observations])

    def residual(point):
        camera = ps[:, :, :3] @ point + ps[:, :, 3]
        z = camera[:, 2]
        pred = camera[:, :2] / np.where(abs(z[:, None]) > 1e-12, z[:, None], 1e-12)
        err = (pred - xy) * focal[:, None]
        err[z <= 0] = 1e6
        return err

    # RANSAC over view pairs, then robust reprojection refinement.
    pairs = list(combinations(range(len(ps)), 2))
    rng = np.random.default_rng(17)
    if len(pairs) > 160:
        pairs = [pairs[i] for i in rng.choice(len(pairs), 160, replace=False)]
    best = None
    for i, j in pairs:
        rows = [xy[k, 0] * ps[k, 2] - ps[k, 0] for k in (i, j)] + [
            xy[k, 1] * ps[k, 2] - ps[k, 1] for k in (i, j)
        ]
        _, _, vt = np.linalg.svd(rows)
        if abs(vt[-1, 3]) < 1e-12:
            continue
        point = vt[-1, :3] / vt[-1, 3]
        errors = np.linalg.norm(residual(point), axis=1)
        good = errors <= max_error_px
        score = (int(good.sum()), -float(np.median(errors)))
        if best is None or score > best[0]:
            best = (score, point, good)
    if best is None or best[2].sum() < 3:
        raise ValueError("insufficient consistent marker observations")
    inliers = best[2]
    result = least_squares(
        lambda p: residual(p)[inliers].ravel(), best[1], loss="soft_l1"
    )
    point = result.x
    errors = np.linalg.norm(residual(point), axis=1)
    inliers = errors <= max_error_px
    if inliers.sum() < 3:
        raise ValueError("marker reprojection failed")
    centers = np.array([-p[:, :3].T @ p[:, 3] for p in ps[inliers]])
    rays = point - centers
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    angle = np.degrees(np.arccos(np.clip(np.min(rays @ rays.T), -1, 1)))
    if angle < 3:
        raise ValueError("insufficient triangulation baseline")
    return point, {
        "views": int(inliers.sum()),
        "rms_px": float(np.sqrt(np.mean(errors[inliers] ** 2))),
        "max_ray_angle_deg": float(angle),
    }


def fit_board(points_by_id, board, tolerance_m=0.002):
    """Hold out entire marker IDs to avoid reporting training residual as accuracy."""
    keys = sorted(k for k in points_by_id if k.split(":")[0] in board["markers"])
    ids = sorted(set(k.split(":")[0] for k in keys), key=int)
    if len(ids) < 6:
        raise ValueError("need at least 6 triangulated markers spread across the board")
    held_ids = set(ids[::3])
    train = np.array([k.split(":")[0] not in held_ids for k in keys])
    source = np.array([points_by_id[k] for k in keys])
    target = np.array(
        [board["markers"][k.split(":")[0]][int(k.split(":")[1])] for k in keys]
    )
    page_m = np.array(board["page_mm"]) / 1000
    if np.any(np.ptp(target[:, :2], axis=0) < 0.4 * page_m):
        raise ValueError("marker controls must span both directions of the board")
    # Whole-marker RANSAC prevents one bad marker's four corners from dominating.
    train_ids = [i for i in ids if i not in held_ids]
    candidates = list(combinations(train_ids, 3))
    best = None
    for subset in candidates:
        selected = np.array([k.split(":")[0] in subset for k in keys])
        try:
            t = similarity(source[selected], target[selected])
        except ValueError:
            continue
        error = np.linalg.norm(transform_points(source, t) - target, axis=1)
        inliers = train & (error < tolerance_m)
        score = (int(inliers.sum()), -float(np.median(error[train])))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[1].sum() < 12 or best[1].sum() < train.sum() * 0.75:
        raise ValueError("inconsistent board controls")
    t = similarity(source[best[1]], target[best[1]])
    errors = np.linalg.norm(transform_points(source, t) - target, axis=1)
    held_rms = float(np.sqrt(np.mean(errors[~train] ** 2)))
    if held_rms > tolerance_m or errors[~train].max() > 2 * tolerance_m:
        raise ValueError(
            f"held-out board error {held_rms * 1000:.2f} mm; check print size, focus and reconstruction"
        )
    report = {
        "metric_scale": float(np.cbrt(np.linalg.det(t[:3, :3]))),
        "train_rms_mm": float(np.sqrt(np.mean(errors[best[1]] ** 2)) * 1000),
        "held_out_rms_mm": held_rms * 1000,
        "held_out_marker_ids": sorted(held_ids, key=int),
        "marker_count": len(ids),
        "transform_to_m_z_up": t.tolist(),
        "limitation": "Board residual checks scale consistency, not object shape accuracy or physical print accuracy.",
    }
    return t, report


def from_colmap(model_path: Path, images: Path, board):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path))
    observations = defaultdict(list)
    for image_id in reconstruction.reg_image_ids():
        im = reconstruction.images[image_id]
        image = cv2.imread(str(images / im.name))
        if image is None:
            raise ValueError(f"missing source image: {im.name}")
        camera = reconstruction.cameras[im.camera_id]
        if (image.shape[1], image.shape[0]) != (camera.width, camera.height):
            raise ValueError(f"image size changed after COLMAP: {im.name}")
        found = detect(image)
        for marker_id, corners in found.items():
            if str(marker_id) not in board["markers"]:
                continue
            # COLMAP's pixel center origin differs from OpenCV's by half a pixel.
            normalized = camera.cam_from_img(corners + 0.5)
            for corner, xy in enumerate(normalized):
                if np.isfinite(xy).all():
                    observations[f"{marker_id}:{corner}"].append(
                        {
                            "P": im.cam_from_world().matrix(),
                            "xy": xy,
                            "focal": (camera.focal_length_x + camera.focal_length_y)
                            / 2,
                        }
                    )
    points, diagnostics = {}, {}
    for key, obs in observations.items():
        try:
            points[key], diagnostics[key] = triangulate(obs)
        except ValueError as error:
            diagnostics[key] = {"rejected": str(error)}
    t, report = fit_board(points, board)
    report["triangulation"] = diagnostics
    report["registered_images"] = len(reconstruction.reg_image_ids())
    return t, report
