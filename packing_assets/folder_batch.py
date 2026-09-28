"""Discover one object per child photo folder or video file."""

import hashlib
import json
import re
from pathlib import Path

from .media import IMAGE_EXT
from .pipeline import run_batch

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def select_objects(objects, names=None):
    """Exact input filename first; otherwise an unambiguous stem or stable ID."""
    if names is None:
        return objects
    if not names:
        raise ValueError("object selection cannot be empty")
    chosen = set()
    for name in names:
        matches = [item for item in objects if Path(item["input"]).name == name]
        if not matches:
            matches = [
                item
                for item in objects
                if Path(item["input"]).stem == name or item["id"] == name
            ]
        if len(matches) != 1:
            reason = "ambiguous" if matches else "unknown"
            available = ", ".join(Path(item["input"]).name for item in objects)
            raise ValueError(
                f"{reason} object {name!r}; use an exact input filename. Available: {available}"
            )
        chosen.add(matches[0]["id"])
    return [item for item in objects if item["id"] in chosen]


def inferred_scope(label):
    return (
        "open_container"
        if re.search(r"杯|碗|\b(?:mug|cup|bowl)(?=\W|\d|_|$)", label.lower())
        else "outer_shape"
    )


def discover(source: Path, board: Path, quality="standard", scope="outer_shape"):
    objects = []
    for path in sorted(source.resolve().iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            if not any(
                p.is_file() and p.suffix.lower() in IMAGE_EXT for p in path.iterdir()
            ):
                continue
        elif path.suffix.lower() not in VIDEO_EXT:
            continue
        # Stable across reruns and supports Chinese names without collisions.
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "object"
        item_id = label + "_" + hashlib.sha256(path.name.encode()).hexdigest()[:8]
        objects.append(
            {
                "id": item_id,
                "backend": "colmap_rgb",
                "input": str(path),
                "board": str(board.resolve()),
                "quality": quality,
                "scope": inferred_scope(path.name) if scope == "outer_shape" else scope,
            }
        )
    if not objects:
        raise ValueError("no child photo folders or video files found")
    return {"objects": objects}


def run_folder(
    source: Path, board: Path, out: Path, quality="standard", scope="outer_shape"
):
    source, out = source.resolve(), out.resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError("output must be outside the input batch folder")
    manifest = discover(source, board, quality, scope)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return run_batch(manifest_path, out)
