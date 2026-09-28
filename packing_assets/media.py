import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .board import detect
from .progress import event

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def prepare(
    source: Path, out: Path, target_frames: int = 100, max_photos: int | None = None
):
    """Keep sharp frames across time. Temporal diversity is NOT angular coverage."""
    if target_frames < 4:
        raise ValueError("target_frames must be >= 4")
    if max_photos is not None and max_photos < 4:
        raise ValueError("max_photos must be >= 4")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output folder is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    input_photo_count = None

    def save(frame, origin):
        idx = len(rows)
        name = f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        found = detect(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows.append(
            {
                "name": name,
                "source": origin,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "markers": {str(k): v.tolist() for k, v in found.items()},
            }
        )

    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise ValueError("no images found")
        input_photo_count = len(files)
        if max_photos is not None and len(files) > max_photos:
            indices = np.linspace(0, len(files), max_photos, endpoint=False).astype(int)
            files = [files[i] for i in indices]
        for path in files:
            with Image.open(path) as im:
                frame = cv2.cvtColor(
                    np.array(ImageOps.exif_transpose(im).convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )
            save(frame, str(path.resolve()))
            event("advance", len(rows), len(files))
    else:
        cap = cv2.VideoCapture(str(source))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError(
                    "video unreadable or frame count unknown; convert with ffmpeg first"
                )
            for bin_indices in np.array_split(
                np.arange(count), min(target_frames, count)
            ):
                candidates = bin_indices[
                    np.linspace(
                        0, len(bin_indices) - 1, min(5, len(bin_indices))
                    ).astype(int)
                ]
                best = None
                for index in candidates:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if best is None or score > best[0]:
                        best = (score, frame, int(index))
                if best is not None:
                    save(best[1], f"{source.resolve()}#frame={best[2]}")
                event("advance", len(rows), min(target_frames, count))
        finally:
            cap.release()
    if not rows:
        raise ValueError("no frames decoded")
    report = {
        "frames": rows,
        "frame_count": len(rows),
        "input_photo_count": input_photo_count,
        "photo_limit": max_photos,
        "photo_selection": "uniform across filename order"
        if max_photos
        else "all supplied photos",
        "frames_with_3_markers": sum(len(r["markers"]) >= 3 for r in rows),
        "status": "capture_prepared_not_geometry_validated",
        "notes": [
            "No automatic object masks yet.",
            "Sharpness includes background; inspect contact views before batch use.",
            "Keep full frames and markers for reconstruction; object-only crops for generative APIs are separate inputs.",
        ],
    }
    (out.parent / f"{out.name}_capture.json").write_text(json.dumps(report, indent=2))
    return report
