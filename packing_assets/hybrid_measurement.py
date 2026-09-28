"""Complete local dimensional measurements from the Meshy frontend's shared work."""

import json
import shutil
import time
from pathlib import Path

from .progress import event
from .reconstruct import input_digest, reconstruct, reconstruction_options, save_json


def seed_measurement(result, source, board):
    """Copy completed checkpoints only, keeping submitted images/masks immutable.

    No hardlinks: local recovery/mask upgrades must never mutate the Meshy branch.
    """
    import pycolmap

    result, source, board = map(Path, (result, source, board))
    target = result / "local-measurement"
    state = json.loads((result / "reconstruction.json").read_text())
    options, _ = reconstruction_options(
        "fast",
        False,
        state["options"]["segmentation"],
        "outer_shape",
        pycolmap.__version__,
    )
    original = dict(state["options"])
    original.pop("meshy_frontend_revision", None)
    if (
        original != options
        or input_digest(source, board, state["options"]) != state["fingerprint"]
    ):
        raise ValueError("shared measurement input/board/options changed")
    expected = input_digest(source, board, options)
    if (target / "reconstruction.json").exists():
        cached = json.loads((target / "reconstruction.json").read_text())
        if cached["fingerprint"] != expected:
            raise ValueError(
                "local measurement cache changed; choose a new output directory"
            )
        return target
    if target.exists():
        target.rename(result / f"local-measurement-incomplete-{time.time_ns()}")
    work = target / "work"
    work.mkdir(parents=True)
    mapping = {
        "prepare": ["images", "images_capture.json"],
        "features": ["database.db"],
        "matching": [],
        "camera_estimation": ["sparse", "board_metric.json"],
        "metric_scale": ["metric.json"],
        # Incomplete stereo files are never seeded into a resumable workspace.
        "multi_view_stereo": ["dense"],
        "fusion": [],
        "remove_board": ["object_points.ply", "isolation.json"],
        "photo_masks_v3": ["masks", "masked_object_points.ply"],
    }
    stages = {}
    for stage, names in mapping.items():
        if not state["stages"].get(stage, {}).get("done"):
            continue
        for name in names:
            path = result / "work" / name
            if path.is_dir():
                shutil.copytree(path, work / name)
            elif path.is_file():
                shutil.copy2(path, work / name)
        stages[stage] = state["stages"][stage]
    if "multi_view_stereo" in stages:
        stages["undistort"] = state["stages"]["undistort"]
    local = {
        "fingerprint": expected,
        "options": options,
        "source": str(source.resolve()),
        "stages": stages,
        "shared_stages": list(stages),
        "shared_from": str(result.resolve()),
    }
    for key in ["camera_method", "registered_images", "input_images"]:
        if key in state:
            local[key] = state[key]
    save_json(target / "reconstruction.json", local)
    return target


def ensure_measurement(result, source, board):
    from .automation import closed_candidate
    from .meshy_dimensions import matching_local_reference

    result = Path(result)
    start = time.perf_counter()
    report = {"status": "running"}
    event("stage", "hybrid_measurement")
    try:
        existing = matching_local_reference(result / "work")
        if existing:
            report.update(status="ready", reference=str(existing), reused_existing=True)
        else:
            target = seed_measurement(result, source, board)
            reconstruct(
                source,
                board,
                target,
                quality="fast",
                segmentation=json.loads((result / "reconstruction.json").read_text())[
                    "options"
                ]["segmentation"],
            )
            reference = closed_candidate(target)
            report.update(
                status="ready",
                reference=str(reference.resolve()),
                reused_existing=False,
                shared_stages=json.loads(
                    (target / "reconstruction.json").read_text()
                ).get("shared_stages", []),
            )
    except Exception as error:  # noqa: BLE001 -- optional measurement must retain Meshy backup
        # The user's generated backup remains deliverable if measurement fails.
        report.update(
            status="failed", error=str(error), fallback="photo_axis_refinement"
        )
    report["seconds"] = round(time.perf_counter() - start, 2)
    save_json(result / "hybrid-measurement.json", report)
    return report
