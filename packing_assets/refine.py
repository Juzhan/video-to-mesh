"""Reuse verified RGB reconstruction caches when only segmentation changes."""

import copy
import json
from pathlib import Path

from .reconstruct import input_digest, reconstruct, reconstruction_options, save_json


def refine_segmentation(
    result: Path, board: Path, out: Path, segmentation="sam2-base", scope=None
):
    import pycolmap

    result, board, out = result.resolve(), board.resolve(), out.resolve()
    state = json.loads((result / "reconstruction.json").read_text())
    if state["status"] == "running":
        raise ValueError("source reconstruction is still running")
    if out.exists():
        raise ValueError("refinement output must be a new directory")
    source = Path(state["source"])
    old = state["options"]
    if input_digest(source, board, old) != state["fingerprint"]:
        raise ValueError(
            "source images or board changed; cannot reuse measured geometry"
        )
    early = [
        "prepare",
        "features",
        "matching",
        "camera_estimation",
        "metric_scale",
        "undistort",
        "multi_view_stereo",
        "fusion",
        "remove_board",
    ]
    if any(not state["stages"].get(k, {}).get("done") for k in early):
        raise ValueError(
            "refinement requires completed dense reconstruction and object isolation"
        )
    required = [
        "images",
        "images_capture.json",
        "sparse",
        "dense",
        "metric.json",
        "object_points.ply",
        "isolation.json",
    ]
    if any(not (result / "work" / k).exists() for k in required):
        raise ValueError("required cache files are missing")
    scope = scope or old["scope"]
    options, _ = reconstruction_options(
        old["quality"], old["collision"], segmentation, scope, pycolmap.__version__
    )
    if old["engine"] != options["engine"]:
        raise ValueError("COLMAP version changed; rerun instead of reusing caches")
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    # Early-stage files are read only during refinement. New masks/surfaces are written here.
    for name in required:
        (work / name).symlink_to(
            result / "work" / name,
            target_is_directory=(result / "work" / name).is_dir(),
        )
    fresh = {
        "fingerprint": input_digest(source, board, options),
        "stages": {k: copy.deepcopy(state["stages"][k]) for k in early},
        "status": "prepared",
        "source": str(source),
        "options": options,
        "cache_reused_from": str(result),
    }
    for key in ["registered_images", "input_images", "camera_method"]:
        if key in state:
            fresh[key] = state[key]
    save_json(out / "reconstruction.json", fresh)
    return reconstruct(
        source, board, out, old["quality"], old["collision"], segmentation, scope
    )
