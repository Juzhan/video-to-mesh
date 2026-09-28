"""Fast RGB camera/scale frontend with optional dense foreground recovery.

The closed measured envelope is ONLY a registration initializer, never the delivered shape.
"""

import json
import time
from pathlib import Path

import numpy as np
import trimesh

from .assets import export_asset
from .metric import transform_points


class SparseForegroundError(ValueError):
    """Sparse observations cannot reliably locate the foreground."""


def reuse_legacy_generation(root, item):
    """Reuse paid geometry only after verifying the original video/board fingerprint.

    Preserve exact submitted crops and provider request fingerprint. Fast masks
    are different observations used for fitting, not a new paid request.
    """
    import shutil

    from .reconstruct import input_digest, save_json

    old = root / "meshy" / item["id"]
    new = root / "meshy-fast" / item["id"]
    state_path = root / "reconstruction" / item["id"] / "reconstruction.json"
    if (new / "provider").exists() or not state_path.exists():
        return
    task_file = old / "provider/meshy-task.json"
    if not task_file.exists() or not (old / "inputs/prepared.json").exists():
        return
    task = json.loads(task_file.read_text())
    if not (task.get("task_id") or task.get("submission_started")):
        return
    state = json.loads(state_path.read_text())
    expected = input_digest(Path(item["input"]), Path(item["board"]), state["options"])
    if expected != state["fingerprint"]:
        raise ValueError(
            "Legacy Meshy input changed; choose a new output directory instead of reusing paid geometry"
        )
    new.mkdir(parents=True, exist_ok=True)
    # Inputs first, provider last: interrupted copies must not look reusable.
    import tempfile

    with tempfile.TemporaryDirectory(dir=new, prefix=".reuse-") as temp:
        temp = Path(temp)
        shutil.copytree(old / "inputs", temp / "inputs")
        shutil.copytree(old / "provider", temp / "provider")
        save_json(
            new / "generation-reuse.json",
            {
                "source_case": str(old.resolve()),
                "original_input_fingerprint": expected,
                "note": "Exact original submitted crops/task reused for the same verified video and board. Fast masks used only for metric fitting; no new API submission.",
            },
        )
        if not (new / "inputs").exists():
            (temp / "inputs").rename(new / "inputs")
        (temp / "provider").rename(new / "provider")


def sparse_object_points(rec, transform, board):
    import open3d as o3d

    points = [
        p.xyz for p in rec.points3D.values() if p.error < 2 and p.track.length() >= 3
    ]
    if len(points) < 100:
        raise SparseForegroundError("Too few sparse object features")
    points = transform_points(np.asarray(points), transform)
    roi = np.asarray(board["suggested_roi_m"], float)
    keep = ((points >= roi[0]) & (points <= roi[1])).all(1) & (points[:, 2] > 0.004)
    points = points[keep]
    if len(points) < 100:
        raise SparseForegroundError("Too few elevated sparse features")
    labels = np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).cluster_dbscan(
            0.012, 5
        )
    )
    if not (labels >= 0).any():
        raise SparseForegroundError("No coherent sparse object cluster")
    counts = np.bincount(labels[labels >= 0])
    primary = points[labels == counts.argmax()]
    lo, hi = np.percentile(primary, [0.5, 99.5], axis=0)
    accepted = []
    for label in np.flatnonzero(counts >= 5):
        center = np.median(points[labels == label], axis=0)
        if ((center >= lo - 0.025) & (center <= hi + 0.025)).all():
            accepted.append(label)
    points = points[np.isin(labels, accepted)]
    if len(points) < 100:
        raise SparseForegroundError("Sparse foreground unreliable")
    return points


def write_points(path, points):
    from plyfile import PlyData, PlyElement

    data = np.empty(len(points), dtype=[(k, "f4") for k in "xyz"])
    for i, k in enumerate("xyz"):
        data[k] = points[:, i]
    PlyData([PlyElement.describe(data, "vertex")]).write(path)


def finish_sparse_frontend(
    out,
    work,
    board,
    transform,
    checkpoint,
    state,
    stage,
    started,
    *,
    dense_prompts=False,
):
    import pycolmap
    from plyfile import PlyData

    from .reconstruct import save_json
    from .sam_masks import isolate_sam

    def seeds():
        rec = pycolmap.Reconstruction(str(work / "sparse"))
        points = sparse_object_points(rec, transform, board)
        write_points(work / "object_points.ply", points)

    if not dense_prompts:
        stage("sparse_object_prompts", seeds)
        stage(
            "sparse_photo_masks",
            lambda: isolate_sam(
                work / "object_points.ply",
                work / "sparse",
                work / "images",
                transform,
                work / "masked_object_points.ply",
                work / "masks",
                checkpoint=checkpoint,
                sparse_prompts=True,
            ),
        )

    def reference():
        data = PlyData.read(work / "masked_object_points.ply")["vertex"].data
        points = np.column_stack([data[k] for k in "xyz"])
        floor = points.copy()
        floor[:, 2] = 0
        envelope = trimesh.convex.convex_hull(np.concatenate([points, floor]))
        export_asset(
            envelope,
            out / "reference",
            {
                "backend": "sparse_registration_initializer",
                "scale_source": "A4_metric_dense_RGB_points"
                if dense_prompts
                else "A4_metric_sparse_RGB_points",
                "prompt_source": "dense" if dense_prompts else "sparse",
                "calibration": json.loads((work / "metric.json").read_text()),
                "warning": "Measured convex envelope is ONLY an initializer for Meshy scale and pose, NOT a deliverable reconstruction or collision shape.",
            },
        )

    stage("sparse_registration_reference", reference)
    selection = {
        "status": "needs_review",
        "scope": "outer_shape",
        "candidate_for_review": "reference/asset.json",
    }
    save_json(out / "selection.json", selection)
    state.update(
        status="needs_review",
        frontend="meshy_fast",
        frontend_prompt_source="dense" if dense_prompts else "sparse",
        selection=selection,
        last_run_seconds=round(time.monotonic() - started, 2),
        total_stage_seconds=round(
            sum(s["seconds"] for s in state["stages"].values()), 2
        ),
    )
    state.pop("active_stage", None)
    save_json(out / "reconstruction.json", state)
    return state


def metric_reference(result):
    state = (
        json.loads((Path(result) / "reconstruction.json").read_text())
        if (Path(result) / "reconstruction.json").exists()
        else {}
    )
    if state.get("frontend") == "meshy_fast":
        return Path(result) / "reference"
    from .automation import closed_candidate

    return closed_candidate(Path(result))
