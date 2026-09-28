"""Camera fallback from a rigid metric marker board; no object/camera truth input."""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def calibrate_frames(frames, board):
    size = (frames[0]["width"], frames[0]["height"])
    usable = []
    for frame in frames:
        if (frame["width"], frame["height"]) != size:
            raise ValueError(
                "board calibration requires shared image size and fixed zoom"
            )
        known = {
            k: np.asarray(v, float)
            for k, v in frame["markers"].items()
            if k in board["markers"]
        }
        train = [k for k in known if int(k) % 3 != 0]
        held = [k for k in known if int(k) % 3 == 0]
        if len(train) < 4 or not held:
            continue
        xyz = np.concatenate([board["markers"][k] for k in train]).astype("f4")
        xy = np.concatenate([known[k] for k in train]).astype("f4")
        usable.append((frame, train, held, xyz, xy))
    if len(usable) < max(12, 0.8 * len(frames)):
        raise ValueError(
            "too few views with four training markers and a held-out marker"
        )
    width, height = size
    k = np.array(
        [
            [max(size), 0, (width - 1) / 2],
            [0, max(size), (height - 1) / 2],
            [0, 0, 1.0],
        ],
        float,
    )
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
        [u[3] for u in usable],
        [u[4] for u in usable],
        size,
        k,
        None,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-9),
    )
    if not (0.3 * max(size) < k[0, 0] < 5 * max(size)) or not (
        0 < k[0, 2] < width and 0 < k[1, 2] < height
    ):
        raise ValueError("implausible calibrated intrinsics")
    radius2 = max(
        ((x - k[0, 2]) / k[0, 0]) ** 2 + ((y - k[1, 2]) / k[1, 1]) ** 2
        for x in (0, width)
        for y in (0, height)
    )
    if 1 + 3 * float(d.flat[0]) * radius2 <= 0:
        raise ValueError("non-monotonic radial camera model")
    views = []
    all_mm = []
    all_px = []
    excluded = []
    for (frame, train, held, xyz, xy), rv, tv in zip(usable, rvecs, tvecs):
        R = cv2.Rodrigues(rv)[0]
        t = tv.ravel()
        center = -R.T @ t
        truth = np.concatenate([board["markers"][h] for h in held])
        observed = np.concatenate([frame["markers"][h] for h in held])
        proj = cv2.projectPoints(truth, rv, tv, k, d)[0].reshape(-1, 2)
        err_px = np.linalg.norm(proj - observed, axis=1)
        rays = cv2.undistortPoints(
            observed.reshape(-1, 1, 2).astype(float), k, d
        ).reshape(-1, 2)
        rays = np.c_[rays, np.ones(len(rays))] @ R
        distance = -center[2] / rays[:, 2]
        intersection = center + rays * distance[:, None]
        err_mm = np.linalg.norm(intersection - truth, axis=1) * 1000
        if center[2] <= 0 or np.any(distance <= 0) or not np.isfinite(err_mm).all():
            raise ValueError("board pose lies behind the observed plane")
        view_rms = float(np.sqrt(np.mean(err_mm**2)))
        if view_rms > 3 or np.percentile(err_mm, 95) > 4:
            excluded.append(
                {
                    "name": frame["name"],
                    "held_out_rms_mm": view_rms,
                    "reason": "board_pose_validation_failed",
                }
            )
            continue
        all_mm.extend(err_mm)
        all_px.extend(err_px)
        views.append(
            {
                "name": frame["name"],
                "R": R.tolist(),
                "t": t.tolist(),
                "train_ids": train,
                "held_out_rms_mm": float(np.sqrt(np.mean(err_mm**2))),
                "held_out_rms_px": float(np.sqrt(np.mean(err_px**2))),
                "train_xy": xy.tolist(),
            }
        )
    if len(views) < max(12, 0.8 * len(frames)):
        raise ValueError(
            f"only {len(views)}/{len(frames)} board poses passed validation"
        )
    held_rms = float(np.sqrt(np.mean(np.array(all_mm) ** 2)))
    if (
        held_rms > 2
        or max(v["held_out_rms_mm"] for v in views) > 3
        or np.percentile(all_mm, 99) > 4
    ):
        raise ValueError(f"board camera held-out error too large: {held_rms:.2f} mm")
    normals = np.array([np.array(v["R"])[:, 2] for v in views])
    if np.max(np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1)))) < 15:
        raise ValueError("insufficient calibration pose diversity")
    report = {
        "method": "opencv_shared_simple_radial_board_calibration",
        "metric_scale": 1.0,
        "transform_to_m_z_up": np.eye(4).tolist(),
        "train_rms_px": float(rms),
        "held_out_rms_mm": held_rms,
        "held_out_rms_px": float(np.sqrt(np.mean(np.array(all_px) ** 2))),
        "held_out_marker_ids": [k for k in board["markers"] if int(k) % 3 == 0],
        "registered_images": len(views),
        "input_images": len(frames),
        "intrinsics_opencv": k.tolist(),
        "radial_k1": float(d.flat[0]),
        "limitation": "Known rigid flat board only. Physical print accuracy, object geometry and camera calibration away from the board are not certified.",
        "excluded_views": excluded,
        "views": [
            {
                k: v
                for k, v in view.items()
                if k not in ("R", "t", "train_xy", "train_ids")
            }
            for view in views
        ],
    }
    return k, float(d.flat[0]), views, report


def write_model(folder, size, k, k1, views, board):
    """Legacy text format; known board controls give MVS image-neighbor tracks."""
    import pycolmap

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    width, height = size
    (folder / "cameras.txt").write_text(
        f"1 SIMPLE_RADIAL {width} {height} {k[0, 0]:.17g} {k[0, 2] + 0.5:.17g} {k[1, 2] + 0.5:.17g} {k1:.17g}\n"
    )
    lines = []
    tracks = {}
    xyzs = {}
    for image_id, view in enumerate(views, 1):
        quat = Rotation.from_matrix(view["R"]).as_quat()[[3, 0, 1, 2]]
        vals = " ".join(f"{v:.17g}" for v in [*quat, *view["t"]])
        lines.append(f"{image_id} {vals} 1 {view['name']}")
        points = []
        for j, marker in enumerate(view["train_ids"]):
            for corner in range(4):
                index = j * 4 + corner
                pid = int(marker) * 4 + corner + 1
                x, y = np.array(view["train_xy"][index]) + 0.5
                points.append(f"{x:.17g} {y:.17g} {pid}")
                xyzs[pid] = board["markers"][marker][corner]
                tracks.setdefault(pid, []).extend([image_id, index])
        lines.append(" ".join(points))
    (folder / "images.txt").write_text("\n".join(lines) + "\n")
    lines = []
    for pid, xyz in xyzs.items():
        lines.append(
            f"{pid} "
            + " ".join(f"{v:.17g}" for v in xyz)
            + " 128 128 128 0 "
            + " ".join(map(str, tracks[pid]))
        )
    (folder / "points3D.txt").write_text("\n".join(lines) + "\n")
    return pycolmap.Reconstruction(str(folder))


def from_capture(capture_path, board, output):
    frames = json.loads(Path(capture_path).read_text())["frames"]
    k, k1, views, report = calibrate_frames(frames, board)
    model = write_model(
        output, (frames[0]["width"], frames[0]["height"]), k, k1, views, board
    )
    return model, report
