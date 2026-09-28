"""Try local Hunyuan on existing calibrated video frames, without any paid API."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_reference_images(prepared_path, result):
    """Never fit one video's generated shape against a different video's cameras."""
    data = json.loads(prepared_path.read_text())
    source = Path(data["source_reconstruction"]) / "work/images"
    for row in data["views"]:
        a = source / row["image"]
        b = result / "work/images" / row["image"]
        if (
            not b.is_file()
            or hashlib.sha256(a.read_bytes()).digest()
            != hashlib.sha256(b.read_bytes()).digest()
        ):
            raise ValueError("prepared images do not match the measurement reference")


def prepare_inputs(prepared_path, out):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    data = json.loads(prepared_path.read_text())
    work = Path(data["source_reconstruction"]) / "work"
    out.mkdir(parents=True, exist_ok=True)
    preview = Image.new("RGB", (1200, 340), "#eeeeee")
    draw = ImageDraw.Draw(preview)
    rows = data["views"]
    if len(rows) != 4:
        raise ValueError("need four azimuth-ordered measured views")
    angles = np.unwrap([r["azimuth_rad"] for r in rows])
    if not np.all(
        (np.diff(angles) > np.deg2rad(35)) & (np.diff(angles) < np.deg2rad(145))
    ):
        raise ValueError("view order/coverage is unsuitable for four-view Hunyuan")
    for i, (row, tag) in enumerate(zip(rows, ("front", "left", "back", "right"))):
        rgb = Image.open(work / "images" / row["image"]).convert("RGB")
        mask = cv2.imread(str(work / "masks" / (row["image"] + ".png")), 0)
        if mask is None or mask.shape[::-1] != rgb.size:
            raise ValueError("missing or mismatched calibrated foreground mask")
        y, x = np.nonzero(mask)
        if len(x) < 100:
            raise ValueError("empty foreground")
        rgba = rgb.convert("RGBA")
        rgba.putalpha(Image.fromarray(mask))
        crop = rgba.crop(
            (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
        )
        ratio = 512 * 0.85 / max(crop.size)
        crop = crop.resize(
            tuple(max(1, round(s * ratio)) for s in crop.size), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
        canvas.paste(crop, ((512 - crop.width) // 2, (512 - crop.height) // 2))
        row["view_tag"], row["file"] = tag, f"{tag}.png"
        canvas.save(out / row["file"])
        thumb = canvas.resize((300, 300))
        preview.paste(thumb, (i * 300, 30), thumb)
        draw.text(
            (i * 300 + 10, 8),
            f"{tag}: elevation {row['elevation_deg']:.0f} deg",
            fill="black",
        )
    data.pop("api_settings", None)
    data["backend"] = "hunyuan3d-2mv"
    data["input_note"] = (
        "Real perspective frames assigned by calibrated azimuth; elevations are not orthographic canonical views."
    )
    (out / "prepared.json").write_text(json.dumps(data, indent=2))
    preview.save(out / "preview.png")
    return out / "prepared.json"


def main():
    from packing_assets.assets import audit_mesh_file, load_mesh
    from packing_assets.mesh_export import light_visual
    from packing_assets.meshy_dimensions import (
        matching_local_reference,
        refine_dimensions,
    )
    from packing_assets.meshy_fit import refine_photo_pose, register_meshy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="existing calibrated reconstruction directory",
    )
    parser.add_argument(
        "--prepared", type=Path, required=True, help="existing four-view prepared.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--variant", choices=("turbo", "standard"), default="turbo")
    parser.add_argument("--vae-precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    args.result, args.out = args.result.resolve(), args.out.resolve()
    validate_reference_images(args.prepared, args.result)
    if args.out.exists():
        raise ValueError(
            "choose a new experiment output directory; previous models are retained"
        )
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    report = {
        "backend": "hunyuan3d-2mv-" + args.variant,
        "status": "running",
        "reference": str(args.result),
    }
    try:
        prepared = prepare_inputs(args.prepared, args.out / "inputs")
        subprocess.run(
            [
                str(ROOT / "vendor/hunyuan-venv/bin/python"),
                str(ROOT / "scripts/hunyuan_worker.py"),
                "--inputs",
                str(prepared),
                "--out",
                str(args.out / "provider"),
                "--repo",
                str(ROOT / "vendor/Hunyuan3D-2"),
                "--weights",
                str(ROOT / "vendor/hunyuan-models"),
                "--resolution",
                str(args.resolution),
                "--variant",
                args.variant,
                "--vae-precision",
                args.vae_precision,
            ],
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "8", "HF_HUB_OFFLINE": "1"},
        )
        report["generation"] = json.loads(
            (args.out / "provider/generation.json").read_text()
        )
        register_meshy(
            args.out / "provider/source.glb", args.result, args.out / "metric", prepared
        )
        refine_photo_pose(
            args.out / "metric", args.result / "work", args.out / "photo-aligned"
        )
        candidates = []
        for name in ("metric", "photo-aligned"):
            folder = args.out / name
            meta = json.loads((folder / "asset.json").read_text())
            meta["provenance"]["backend"] = "hunyuan3d_2mv_" + args.variant
            (folder / "asset.json").write_text(json.dumps(meta, indent=2))
            quality = json.loads((folder / "quality.json").read_text())
            quality["held_out_from_generator"] = quality.pop("held_out_from_meshy", {})
            if quality["held_out_from_generator"]:
                quality["held_out_from_generator"]["note"] = (
                    "Not submitted to Hunyuan; shared local calibration and masks are not independent ground truth."
                )
            (folder / "quality.json").write_text(json.dumps(quality, indent=2))
            candidates.append(
                (
                    meta["status"] == "needs_review",
                    quality["views"]["median_iou"],
                    folder,
                )
            )
        passed, _, chosen = max(candidates)
        report["quality"] = json.loads((chosen / "quality.json").read_text())
        report["selected"] = str(chosen)
        models = args.out / "models"
        models.mkdir()
        mesh, simplification = light_visual(load_mesh(chosen / "visual.obj"))
        mesh.export(models / "hunyuan.obj")
        report["export"] = {
            "dimensions_mm": (mesh.extents * 1000).tolist(),
            "simplification": simplification,
            "audit": audit_mesh_file(models / "hunyuan.obj"),
        }
        report["status"] = (
            "needs_review" if passed and mesh.is_volume else "retained_unvalidated"
        )
        try:
            dim = refine_dimensions(
                chosen,
                args.result / "work",
                args.out / "dimension-aligned",
                prepared,
                local_reference=matching_local_reference(args.result / "work"),
            )
            report["dimension_calibration"] = dim
            if dim.get("accepted"):
                corrected, _ = light_visual(
                    load_mesh(args.out / "dimension-aligned/visual.obj")
                )
                corrected.export(models / "hunyuan_dimensions.obj")
        except Exception as error:  # noqa: BLE001 - preserve the experimental mesh if optional calibration fails
            report["dimension_error"] = str(error).replace("Meshy", "生成模型")
        (models / "README.txt").write_text(
            "OBJ units: metres; Z up. Experimental Hunyuan shape. See ../report.json for quality status.\n"
            "retained_unvalidated means usable as a visual backup, NOT an accepted physical reconstruction.\n"
        )
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        (args.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
