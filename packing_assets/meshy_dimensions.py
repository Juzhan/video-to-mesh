"""Conservative axis-scale variant of an existing, metric Meshy asset.

No generation requests. Fits calibrated silhouettes, not known physical dimensions.
"""

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

from .assets import audit_mesh_file, export_asset, load_mesh
from .quality import check_views, ray_scene, surface_release_gate
from .reconstruct import save_json

REVISION = 2


def axis_transform(parameters, basis, center, tied=False):
    scales = np.exp(np.asarray(parameters[:3]) / 100)
    if tied:
        scales[1] = scales[0]
    matrix = np.eye(4)
    matrix[:3, :3] = basis @ np.diag(scales) @ basis.T
    matrix[:3, 3] = center - matrix[:3, :3] @ center + np.asarray(parameters[3:]) / 1000
    return matrix, scales


def scores(hit, ref):
    intersection = np.count_nonzero(hit & ref)
    union = np.count_nonzero(hit | ref)
    iou = intersection / max(1, union)
    if not hit.any() or not ref.any():
        return {"iou": iou, "edge": 1.0, "extent": 1.0}
    kernel = np.ones((3, 3), np.uint8)
    boundaries = [
        m & ~cv2.erode(m.astype("uint8"), kernel).astype(bool) for m in [hit, ref]
    ]
    distances = [
        cv2.distanceTransform((~b).astype("uint8"), cv2.DIST_L2, 3) for b in boundaries
    ]
    diag = max(1, np.linalg.norm(np.ptp(np.argwhere(ref), axis=0)))
    edge = (distances[0][boundaries[1]].mean() + distances[1][boundaries[0]].mean()) / (
        2 * diag
    )
    boxes = [np.r_[np.argwhere(m).min(0), np.argwhere(m).max(0)] for m in [hit, ref]]
    extent = np.mean(np.abs(boxes[0] - boxes[1])) / diag
    return {"iou": float(iou), "edge": float(edge), "extent": float(extent)}


def validation_reasons(before, after, scales, gate):
    reasons = []
    if gate["status"] != "needs_review":
        reasons.append("failed_existing_geometry_gate")
    if (
        after["mean_iou"] < before["mean_iou"] - 0.001
        or after["p10_iou"] < before["p10_iou"] - 0.005
    ):
        reasons.append("held_out_overlap_regressed")
    if (
        after["edge"] > before["edge"] + 0.0005
        or after["extent"] > before["extent"] + 0.0005
    ):
        reasons.append("held_out_boundary_regressed")
    improved = after["mean_iou"] >= before["mean_iou"] + 0.002 or (
        after["edge"] < before["edge"] * 0.97 and after["extent"] <= before["extent"]
    )
    if not improved:
        reasons.append("no_clear_held_out_improvement")
    if np.max(np.abs(np.log(scales))) >= 0.099:
        reasons.append("axis_scale_at_search_limit")
    return reasons


def refine_dimensions(asset, work, out, prepared, local_reference=None):
    import open3d as o3d
    import pycolmap

    asset, work, out = map(Path, [asset, work, out])
    digest = hashlib.sha256(str(REVISION).encode())
    paths = [
        asset / "visual.obj",
        asset / "asset.json",
        work / "metric.json",
        Path(prepared),
    ]
    if local_reference:
        local_reference = Path(local_reference)
        paths += [local_reference / "visual.obj", local_reference / "asset.json"]
    paths += sorted((work / "sparse").glob("*.bin")) + sorted(
        (work / "masks").glob("*.png")
    )
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    report_file = out / "dimension-report.json"
    if report_file.exists():
        report = json.loads(report_file.read_text())
        if report.get("signature") == signature:
            audit_mesh_file(out / "visual.obj", require_volume=True)
            return report
    if out.exists():
        out.rename(out.with_name(out.name + "-previous-" + str(time.time_ns())))
    started = time.perf_counter()
    meta = json.loads((asset / "asset.json").read_text())
    mesh = load_mesh(asset / "visual.obj")
    mesh.apply_translation(-np.asarray(meta["centering_translation_m"]))
    original = mesh.copy()
    # Use the generated model's calibrated up-axis, then its horizontal principal axes.
    initial = np.asarray(meta["provenance"]["board_from_generated_z_up"])[:3, :3]
    u, _, vt = np.linalg.svd(initial)
    basis = u @ vt
    samples, _ = trimesh.sample.sample_surface(mesh, 5000, seed=104)
    local = samples @ basis
    values, vectors = np.linalg.eigh(np.cov(local[:, :2].T))
    angle = np.arctan2(vectors[1, -1], vectors[0, -1])
    basis = basis @ trimesh.transformations.rotation_matrix(angle, [0, 0, 1])[:3, :3]
    tied = values[-1] / max(values[0], 1e-12) < 1.15
    local_dimensions = None
    if local_reference:
        # Both assets are in the same verified A4 board frame; use its axes,
        # not independently estimated principal axes or noisy sparse bounds.
        measured = load_mesh(local_reference / "visual.obj")
        local_dimensions = measured.extents
        basis = np.eye(3)
        tied = False
    center = mesh.bounds.mean(0)
    rec = pycolmap.Reconstruction(str(work / "sparse"))
    metric = np.asarray(
        json.loads((work / "metric.json").read_text())["transform_to_m_z_up"]
    )
    available = sorted(
        [
            im
            for im in rec.images.values()
            if (work / "masks" / (im.name + ".png")).exists()
        ],
        key=lambda im: im.name,
    )
    fit_names = set(meta["provenance"].get("fitting_views", []))
    if not fit_names:
        fit_names = {
            available[i].name
            for i in np.linspace(
                0, len(available) - 1, min(12, len(available)), dtype=int
            )
        }
    submitted = {v["image"] for v in json.loads(Path(prepared).read_text())["views"]}
    fit = [im for im in available if im.name in fit_names]
    held = [im for im in available if im.name not in fit_names | submitted]
    if len(fit) < 6 or len(held) < 6:
        raise ValueError(
            "size calibration needs at least six fitting and six held-out views"
        )
    scene = ray_scene(mesh)

    def observations(images):
        result = []
        for im in images:
            camera = rec.cameras[im.camera_id]
            mask = cv2.imread(str(work / "masks" / (im.name + ".png")), 0) > 0
            y, x = np.nonzero(mask)
            pad = max(40, int(max(np.ptp(y), np.ptp(x)) * 0.18))
            yy, xx = np.mgrid[
                max(0, y.min() - pad) : min(camera.height, y.max() + pad) : 3,
                max(0, x.min() - pad) : min(camera.width, x.max() + pad) : 3,
            ]
            xy = camera.cam_from_img(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5])
            pose = metric @ np.linalg.inv(
                np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]])
            )
            directions = np.c_[xy, np.ones(len(xy))] @ pose[:3, :3].T
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            result.append(
                (
                    np.c_[np.broadcast_to(pose[:3, 3], directions.shape), directions],
                    mask[yy, xx],
                )
            )
        return result

    fit_obs, held_obs = observations(fit), observations(held)

    def evaluate(parameters, observations):
        matrix, _ = axis_transform(parameters, basis, center, tied)
        inv = np.linalg.inv(matrix)
        rows = []
        for rays, ref in observations:
            origins = rays[:, :3] @ inv[:3, :3].T + inv[:3, 3]
            directions = rays[:, 3:] @ inv[:3, :3].T
            hit = np.isfinite(
                scene.cast_rays(
                    o3d.core.Tensor(np.c_[origins, directions].astype("f4")), nthreads=4
                )["t_hit"].numpy()
            ).reshape(ref.shape)
            rows.append(scores(hit, ref))
        return {
            "mean_iou": float(np.mean([v["iou"] for v in rows])),
            "p10_iou": float(np.percentile([v["iou"] for v in rows], 10)),
            "edge": float(np.mean([v["edge"] for v in rows])),
            "extent": float(np.mean([v["extent"] for v in rows])),
        }

    best = [np.inf, np.zeros(6)]
    calls = 0

    def cost(parameters):
        nonlocal calls
        s = evaluate(parameters, fit_obs)
        # A small deformation penalty favors the existing shape when evidence is weak.
        loss = (
            1
            - s["mean_iou"]
            + 2 * s["edge"]
            + 2 * s["extent"]
            + 0.002 * np.mean((parameters[:3] / 10) ** 2)
        )
        calls += 1
        if loss < best[0]:
            best[:] = [loss, np.array(parameters)]
        return loss

    initial_parameters = np.zeros(6)
    bounds = [(-10, 10), (0, 0) if tied else (-10, 10), (-10, 10)] + [(-3, 3)] * 3
    if local_dimensions is not None:
        initial_parameters[:3] = 100 * np.log(local_dimensions / original.extents)
        if np.max(np.abs(initial_parameters[:3])) >= 9.9:
            raise ValueError(
                "同视频本地尺寸与 Meshy 相差超过约 10%，保留原版并请人工核验"
            )
        # Local RGB geometry also has discretization/noise. Permit only 0.5%
        # around that reference; the usual photo-only search is much wider.
        bounds[:3] = [
            (max(-9.8, float(v) - 0.49), min(9.8, float(v) + 0.49))
            for v in initial_parameters[:3]
        ]
    cost(initial_parameters)
    minimize(
        cost,
        initial_parameters,
        method="Powell",
        bounds=bounds,
        options={"maxfev": 420, "xtol": 0.03, "ftol": 0.0001},
    )
    matrix, scales = axis_transform(best[1], basis, center, tied)
    before, after = evaluate(np.zeros(6), held_obs), evaluate(best[1], held_obs)
    mesh.apply_transform(matrix)
    views = check_views(mesh, work)
    gate = surface_release_gate(mesh, views)
    reasons = validation_reasons(before, after, scales, gate)
    provenance = dict(
        meta["provenance"],
        dimension_refinement_revision=REVISION,
        dimension_refinement_delta=matrix.tolist(),
        dimension_axis_scales=scales.tolist(),
        no_axis_stretching=False,
    )
    provenance["board_from_generated_z_up"] = (
        matrix @ np.asarray(provenance["board_from_generated_z_up"])
    ).tolist()
    result = export_asset(mesh, out, provenance)
    result.update(status=gate["status"], quality=gate)
    save_json(out / "asset.json", result)
    save_json(out / "quality.json", {"views": views, "gate": gate})
    report = {
        "signature": signature,
        "revision": REVISION,
        "accepted": not reasons,
        "method": "same_video_local_dimensions"
        if local_reference
        else "photo_axis_refinement",
        "local_reference": str(local_reference) if local_reference else None,
        "local_dimensions_m": local_dimensions.tolist()
        if local_dimensions is not None
        else None,
        "local_dimension_adjustment_limit": 0.005 if local_reference else None,
        "reasons": reasons,
        "axis_scales": scales.tolist(),
        "tied_horizontal_axes": bool(tied),
        "axes_in_board": basis.tolist(),
        "before_dimensions_m": original.extents.tolist(),
        "after_dimensions_m": mesh.extents.tolist(),
        "held_out_before": before,
        "held_out_after": after,
        "fitting_views": [im.name for im in fit],
        "held_out_views": [im.name for im in held],
        "calls": calls,
        "seconds": round(time.perf_counter() - started, 2),
        "note": "Held out from dimension/pose fitting and Meshy input, but shares estimated cameras and SAM masks; not independent physical dimensional accuracy. Original model remains the default.",
    }
    save_json(report_file, report)
    return report


def attach_dimension_variant(row, case, out):
    """Publish a separate visual variant; keep default/collision geometry unchanged."""
    row["dimension_calibration"] = case.get("dimension_calibration", {})
    source = case.get("dimension_asset")
    if not source:
        return
    source = Path(source)
    source_meta = json.loads((source / "asset.json").read_text())
    base_meta = json.loads((Path(row["source_asset"]) / "asset.json").read_text())
    original = Path(out) / row["visual"]
    digest = hashlib.sha256(
        (source / "asset.json").read_bytes() + original.read_bytes()
    ).hexdigest()[:20]
    target = Path(out) / "assets" / f"{row['id']}-dimensions-{digest}"
    target.mkdir(parents=True, exist_ok=True)
    # Apply the measured affine change to the existing low-poly mesh. Do not
    # reintroduce the provider's hundreds of thousands of triangles or decimate twice.
    mesh = load_mesh(original)
    mesh.apply_translation(-np.asarray(base_meta["centering_translation_m"]))
    mesh.apply_transform(
        np.asarray(source_meta["provenance"]["dimension_refinement_delta"])
    )
    mesh.apply_translation(np.asarray(source_meta["centering_translation_m"]))
    mesh.export(target / "visual.obj")
    row["dimension_calibration"] = dict(
        row["dimension_calibration"],
        exported_dimensions_m=mesh.extents.tolist(),
        exported_faces=len(mesh.faces),
    )
    save_json(target / "dimension-report.json", row["dimension_calibration"])
    audit_mesh_file(target / "visual.obj", require_volume=True)
    row["dimension_visual"] = str((target / "visual.obj").relative_to(out))


def matching_local_reference(work):
    """Find a completed same-video local reconstruction, verifying board and cache.

    Search this batch and (for the phone portal) completed local queue peers only.
    Never treat a filename or an arbitrary neighboring model as a size reference.
    """
    import sqlite3

    from .automation import closed_candidate
    from .reconstruct import input_digest

    work = Path(work).resolve()
    result = work.parent
    manifest_path = result.parent.parent / "input-manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    item = next((i for i in manifest["objects"] if i["id"] == result.name), None)
    if not item:
        return None
    source = Path(item["input"])
    board = Path(item["board"])
    if not source.is_file():
        return None
    video_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    own_measurement = result / "local-measurement/reconstruction.json"
    candidates = ([own_measurement] if own_measurement.exists() else []) + list(
        (result.parent.parent / "reconstruction").glob("*/reconstruction.json")
    )
    # jobs/<job>/output/reconstruction-*/<object>/work
    for root in result.parents:
        database = root / "queue.sqlite3"
        if database.is_file() and (root / "jobs").is_dir():
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
                rows = db.execute(
                    "SELECT id FROM jobs WHERE backend='local' AND state='done' AND sha256=? ORDER BY updated DESC",
                    (video_hash,),
                ).fetchall()
            for (uid,) in rows:
                if len(uid) == 24 and all(c in "0123456789abcdef" for c in uid):
                    candidates.extend(
                        (root / "jobs" / uid / "output/reconstruction").glob(
                            "*/reconstruction.json"
                        )
                    )
            break
    for state_path in candidates:
        state = json.loads(state_path.read_text())
        if state.get("status") not in (None, "needs_review"):
            continue
        local_video = Path(state["source"])
        if (
            not local_video.is_file()
            or hashlib.sha256(local_video.read_bytes()).hexdigest() != video_hash
        ):
            continue
        if input_digest(local_video, board, state["options"]) != state["fingerprint"]:
            continue
        try:
            reference = closed_candidate(state_path.parent)
            meta = json.loads((reference / "asset.json").read_text())
            if meta.get("status") != "needs_review":
                continue
            audit_mesh_file(reference / "visual.obj", require_volume=True)
        except ValueError:
            continue
        return reference
    return None
