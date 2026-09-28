"""Prepare diverse, segmented Meshy inputs from measured cameras. No API calls."""

import hashlib, json, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .metric import transform_points


def choose_views(rows, count=4):
    if len(rows) < count:
        raise ValueError("not enough complete, segmented views")
    first = max(rows, key=lambda r: r["score"])
    chosen = [first]
    for turn in range(1, count):
        target = first["azimuth_rad"] + turn * 2 * np.pi / count
        available = [r for r in rows if r not in chosen]

        # Camera angles, rather than timestamps, keep neighboring video frames
        # from occupying several of the API's four image slots.
        def rank(r):
            delta = abs(
                math.atan2(
                    math.sin(r["azimuth_rad"] - target),
                    math.cos(r["azimuth_rad"] - target),
                )
            )
            elevation_gap = abs(
                r.get("elevation_deg", 0) - first.get("elevation_deg", 0)
            )
            return delta + 0.012 * elevation_gap - 0.04 * r["score"]

        picked = min(available, key=rank)
        distance = min(
            abs(
                math.atan2(
                    math.sin(picked["azimuth_rad"] - r["azimuth_rad"]),
                    math.cos(picked["azimuth_rad"] - r["azimuth_rad"]),
                )
            )
            for r in chosen
        )
        if distance < np.deg2rad(35):
            raise ValueError("insufficient angular coverage for four distinct views")
        chosen.append(picked)
    return chosen


def object_crop(rgb, mask, size=1024):
    y, x = np.nonzero(mask)
    if len(x) < 100:
        raise ValueError("foreground too small")
    if (
        x.min() <= 2
        or y.min() <= 2
        or x.max() >= mask.shape[1] - 3
        or y.max() >= mask.shape[0] - 3
    ):
        raise ValueError("object touches image border")
    foreground = np.where(mask[:, :, None], rgb, 255).astype("uint8")
    crop = Image.fromarray(foreground[y.min() : y.max() + 1, x.min() : x.max() + 1])
    scale = size * 0.86 / max(crop.size)
    crop = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def prepare_meshy_views(result: Path, out: Path):
    import pycolmap
    from plyfile import PlyData

    result = result.resolve()
    out = out.resolve()
    if out.exists():
        raise ValueError(
            "choose a new output folder; prepared API inputs are immutable"
        )
    work = result / "work"
    state = json.loads((result / "reconstruction.json").read_text())
    if state.get("status") == "running":
        raise ValueError("source reconstruction is still running")
    metric = json.loads((work / "metric.json").read_text())
    transform = np.array(metric["transform_to_m_z_up"])
    cloud = PlyData.read(work / "masked_object_points.ply")["vertex"].data
    xyz = np.column_stack([cloud[k] for k in ("x", "y", "z")])
    center = np.median(xyz, axis=0)
    rec = pycolmap.Reconstruction(str(work / "sparse"))
    masks = {
        r["image"]: r
        for r in json.loads((work / "masks/report.json").read_text())["masks"]
    }
    rows = []
    excluded = []
    for image_id in rec.reg_image_ids():
        im = rec.images[image_id]
        mask_path = work / "masks" / (im.name + ".png")
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), 0) > 0
        y, x = np.nonzero(mask)
        if len(x) < 100:
            continue
        if (
            x.min() <= 2
            or y.min() <= 2
            or x.max() >= mask.shape[1] - 3
            or y.max() >= mask.shape[0] - 3
        ):
            excluded.append(
                {"image": im.name, "reason": "foreground touches image border"}
            )
            continue
        rgb = cv2.imread(str(work / "images" / im.name))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_32F)[mask].var())
        origin = transform_points(
            np.asarray(im.projection_center()).reshape(1, 3), transform
        )[0]
        v = origin - center
        azimuth = float(np.arctan2(v[1], v[0]))
        elevation = float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2]))))
        score = float(
            np.log(max(1, len(x)))
            + 0.2 * np.log(max(1, sharp))
            - max(0, elevation - 65) * 0.03
        )
        rows.append(
            {
                "image": im.name,
                "azimuth_rad": azimuth,
                "elevation_deg": elevation,
                "score": score,
                "foreground_pixels": len(x),
                "sharpness": sharp,
                "sam_score": masks.get(im.name, {}).get("model_score"),
            }
        )
    selected = choose_views(rows)
    out.mkdir(parents=True)
    preview = Image.new("RGB", (1200, 350), "white")
    draw = ImageDraw.Draw(preview)
    for k, row in enumerate(selected, 1):
        image_path = work / "images" / row["image"]
        mask_path = work / "masks" / (row["image"] + ".png")
        rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), 0) > 0
        crop = object_crop(rgb, mask)
        filename = f"{k:02d}.png"
        crop.save(out / filename)
        row.update(
            file=filename,
            source_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        )
        preview.paste(crop.resize((300, 300)), ((k - 1) * 300, 30))
        draw.text(((k - 1) * 300 + 5, 5), f"{k}: {row['image']}", fill="black")
    preview.save(out / "preview.jpg")
    report = {
        "status": "prepared_not_submitted",
        "source_reconstruction": str(result),
        "source_video": state["source"],
        "views": selected,
        "excluded_views": excluded,
        "api_settings": {
            "ai_model": "meshy-7.1",
            "geometry_resolution": "standard",
            "should_texture": False,
            "should_remesh": False,
            "image_enhancement": False,
            "auto_size": False,
            "target_formats": ["glb"],
        },
        "primary_view_note": "First image chosen for visibility and sharpness, not semantic front detection. Inspect preview before the pilot.",
        "scale_note": "Crops have no physical scale. Align and uniformly scale the generated model to the metric observations, then validate all views. Do not copy the COLMAP transform directly to Meshy.",
        "coordinate_reference": {
            "up_axis": "z",
            "units": "m",
            "metric_file": str(work / "metric.json"),
        },
        "limitations": "Preparation only; fast mode uses sparse cameras and masks without dense depth. Generation is a separate paid step; scale comes from calibrated RGB observations, not the crops.",
    }
    (out / "prepared.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
