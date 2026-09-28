"""Uniform metric registration of a generated mesh, with photo-based rejection."""

import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from .assets import load_mesh, rotate_up, export_asset
from .quality import check_views, surface_release_gate


def fit_upright_similarity(source, target, sparse=False):
    """Source and target are Z-up samples. Search yaw, then similarity ICP.

    The metric target is an approximate RGB reference, never independent truth.
    """
    import open3d as o3d

    source = np.asarray(source, float)
    target = np.asarray(target, float)
    src_center = source.mean(0)
    dst_center = target.mean(0)
    source_span = np.sqrt(
        np.mean(np.sum((source[:, :2] - src_center[:2]) ** 2, axis=1))
    )
    target_span = np.sqrt(
        np.mean(np.sum((target[:, :2] - dst_center[:2]) ** 2, axis=1))
    )
    if min(source_span, target_span) <= 1e-9:
        raise ValueError("insufficient extent for metric alignment")
    scale = target_span / source_span
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    distance = max(0.005, np.linalg.norm(np.ptp(target, axis=0)) * 0.15)
    tree = cKDTree(target)
    best = None
    for yaw in np.arange(0, 360, 30):
        r = trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 0, 1])[:3, :3]
        matrix = np.eye(4)
        matrix[:3, :3] = scale * r
        matrix[:3, 3] = dst_center - scale * r @ src_center
        for threshold in [distance, distance * 0.3]:
            result = o3d.pipelines.registration.registration_icp(
                src,
                dst,
                threshold,
                matrix,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=not sparse
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
            )
            matrix = result.transformation.copy()
        if sparse:
            # Partial sparse envelopes can make free-scale ICP collapse into one
            # small surface patch. Keep the measured span scale and upright pose;
            # photos, not ICP, refine physical scale afterwards.
            yaw_fit = np.arctan2(matrix[1, 0], matrix[0, 0])
            upright = trimesh.transformations.rotation_matrix(yaw_fit, [0, 0, 1])[
                :3, :3
            ]
            matrix[:3, :3] = scale * upright
            matrix[:3, 3] = dst_center - matrix[:3, :3] @ src_center
            moved = source @ matrix[:3, :3].T + matrix[:3, 3]
            matrix[2, 3] -= moved[:, 2].min()
        singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
        factor = float(singular.mean())
        rotation = matrix[:3, :3] / factor
        if (
            np.linalg.det(rotation) < 0
            or rotation[2, 2] < 0.94
            or np.ptp(singular) > factor * 1e-5
        ):
            continue
        moved = source @ matrix[:3, :3].T + matrix[:3, 3]
        distances = np.r_[tree.query(moved)[0], cKDTree(moved).query(target)[0]]
        score = float(np.mean(np.minimum(distances, np.quantile(distances, 0.95)) ** 2))
        if best is None or score < best[0]:
            best = (score, matrix)
    if best is None:
        raise ValueError("no physically upright uniform-scale registration found")
    return best[1], {
        "method": "fixed sparse-span scale, upright yaw ICP initialization; photos refine scale"
        if sparse
        else "upright yaw search and uniform similarity ICP against metric RGB candidate",
        "uniform_scale": float(np.cbrt(np.linalg.det(best[1][:3, :3]))),
        "trimmed_sample_rms_mm": float(np.sqrt(best[0]) * 1000),
        "limitation": "Reference geometry is approximate; registration does not guarantee true dimensions or recover generated shape errors.",
    }


def register_meshy(
    source: Path, result: Path, out: Path, prepared: Path, reference_asset=None
):
    from .meshy_frontend import metric_reference

    source = source.resolve()
    result = result.resolve()
    out = out.resolve()
    if out.exists():
        raise ValueError("use a new registration output directory")
    mesh = load_mesh(source)
    rotate_up(mesh, "y")
    reference_asset = (
        Path(reference_asset) if reference_asset else metric_reference(result)
    )
    reference = load_mesh(reference_asset / "visual.obj")
    meta = json.loads((reference_asset / "asset.json").read_text())
    reference.apply_translation(-np.array(meta["centering_translation_m"]))
    s, _ = trimesh.sample.sample_surface(mesh, 8000, seed=81)
    t, _ = trimesh.sample.sample_surface(reference, 8000, seed=82)
    sparse = (
        meta.get("provenance", {}).get("backend") == "sparse_registration_initializer"
    )
    matrix, registration = fit_upright_similarity(s, t, sparse=sparse)
    mesh.apply_transform(matrix)
    views = check_views(mesh, result / "work")
    gate = surface_release_gate(mesh, views)
    used = {r["image"] for r in json.loads(prepared.read_text())["views"]}
    heldout = [r for r in views["views"] if r["image"] not in used]
    withheld = {
        "views": len(heldout),
        "median_iou": float(np.median([r["iou"] for r in heldout])),
        "median_recall": float(np.median([r["recall"] for r in heldout])),
        "note": "These photos were not sent to Meshy, but contribute to local metric reference and estimated masks.",
    }
    calibration = json.loads((result / "work/metric.json").read_text())
    report = export_asset(
        mesh,
        out,
        {
            "backend": "meshy_multi_image",
            "scale_source": "uniform_registration_to_A4_metric_RGB_reference",
            "calibration": calibration,
            "reference_asset": str(reference_asset),
            "registration": registration,
            "board_from_generated_z_up": matrix.tolist(),
            "no_axis_stretching": True,
            "metric_registration_profile": "sparse-fixed-v1" if sparse else "full-v1",
        },
    )
    report["status"] = gate["status"]
    report["quality"] = gate
    report["review_reasons"].append(
        "AI_generated_shape_may_invent_or_omit_geometry; metric_reference_is_not_ground_truth"
    )
    (out / "asset.json").write_text(json.dumps(report, indent=2))
    (out / "quality.json").write_text(
        json.dumps(
            {
                "views": views,
                "gate": gate,
                "held_out_from_meshy": withheld,
                "registration": registration,
            },
            indent=2,
        )
    )
    return report


def refine_photo_pose(asset: Path, work: Path, out: Path):
    """Fit only a similarity transform to a subset of calibrated photos, never truth.

    Keep quality thresholds unchanged and validate against all masks afterwards.
    """
    import cv2, time, pycolmap, open3d as o3d
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    from .quality import ray_scene

    if out.exists():
        raise ValueError("use a new photo-refinement directory")
    meta = json.loads((asset / "asset.json").read_text())
    state_path = work.parent / "reconstruction.json"
    sparse_frontend = (
        state_path.exists()
        and json.loads(state_path.read_text()).get("frontend") == "meshy_fast"
    )
    mesh = load_mesh(asset / "visual.obj")
    mesh.apply_translation(-np.array(meta["centering_translation_m"]))
    center = mesh.bounds.mean(0)
    scene = ray_scene(mesh)
    rec = pycolmap.Reconstruction(str(work / "sparse"))
    metric = np.array(
        json.loads((work / "metric.json").read_text())["transform_to_m_z_up"]
    )
    available = sorted(
        (
            im
            for im in rec.images.values()
            if (work / "masks" / (im.name + ".png")).is_file()
        ),
        key=lambda i: i.name,
    )
    if len(available) < 12:
        raise ValueError("need at least twelve valid masks for photo refinement")
    images = [available[i] for i in np.linspace(0, len(available) - 1, 12, dtype=int)]
    rays = []
    references = []
    sizes = []
    for im in images:
        cam = rec.cameras[im.camera_id]
        mask = cv2.imread(str(work / "masks" / (im.name + ".png")), 0) > 0
        y, x = np.nonzero(mask)
        pad = 50
        yy, xx = np.mgrid[
            max(0, y.min() - pad) : min(cam.height, y.max() + pad) : 4,
            max(0, x.min() - pad) : min(cam.width, x.max() + pad) : 4,
        ]
        xy = cam.cam_from_img(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5])
        pose = metric @ np.linalg.inv(
            np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]])
        )
        dirs = np.c_[xy, np.ones(len(xy))] @ pose[:3, :3].T
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        rays.append(np.c_[np.broadcast_to(pose[:3, 3], dirs.shape), dirs])
        references.append(mask[yy, xx].ravel())
        sizes.append(len(xy))
    rays = np.concatenate(rays)
    ref = np.concatenate(references)
    cut = np.r_[0, np.cumsum(sizes)]
    best = [1, None]
    calls = 0
    started = time.perf_counter()

    def delta(p):
        r = Rotation.from_euler("xyz", p[3:6], degrees=True).as_matrix()
        s = np.exp(p[6] / 100)
        m = np.eye(4)
        m[:3, :3] = s * r
        m[:3, 3] = center + np.array(p[:3]) / 1000 - s * r @ center
        return m

    def cost(p):
        nonlocal calls
        m = delta(p)
        inv = np.linalg.inv(m)
        origins = rays[:, :3] @ inv[:3, :3].T + inv[:3, 3]
        dirs = rays[:, 3:] @ inv[:3, :3].T
        hit = np.isfinite(
            scene.cast_rays(
                o3d.core.Tensor(np.c_[origins, dirs].astype("f4")), nthreads=4
            )["t_hit"].numpy()
        )
        intersection = np.add.reduceat((hit & ref).astype(int), cut[:-1])
        union = np.add.reduceat((hit | ref).astype(int), cut[:-1])
        loss = 1 - float(np.mean(intersection / np.maximum(union, 1)))
        calls += 1
        if loss < best[0]:
            best[:] = [loss, np.array(p)]
        if calls % 40 == 0:
            print(
                calls,
                "best IoU",
                1 - best[0],
                "seconds",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
        return loss

    initial = np.zeros(7)
    print("initial", 1 - cost(initial), flush=True)
    # Sparse envelopes may omit untextured sides: allow a broader uniform fit,
    # keeping all original shape gates and never stretching individual axes.
    bounds = (
        ([(-30, 30)] * 3 + [(-15, 15), (-15, 15), (-45, 45), (-35, 35)])
        if sparse_frontend
        else [(-8, 8)] * 3 + [(-5, 5), (-5, 5), (-8, 8), (-8, 8)]
    )
    result = minimize(
        cost,
        initial,
        method="Powell",
        bounds=bounds,
        options={
            "maxfev": 600 if sparse_frontend else 300,
            "xtol": 0.05,
            "ftol": 0.0002,
        },
    )
    m = delta(best[1])
    mesh.apply_transform(m)
    views = check_views(mesh, work)
    gate = surface_release_gate(mesh, views)
    provenance = dict(
        meta["provenance"],
        photo_refinement_delta=m.tolist(),
        fitting_views=[i.name for i in images],
        photo_refinement_profile="sparse-wide-v2" if sparse_frontend else "full-v1",
    )
    provenance["board_from_generated_z_up"] = (
        m @ np.array(provenance["board_from_generated_z_up"])
    ).tolist()
    report = export_asset(mesh, out, provenance)
    report["status"] = gate["status"]
    report["quality"] = gate
    (out / "asset.json").write_text(json.dumps(report, indent=2))
    (out / "quality.json").write_text(
        json.dumps(
            {
                "views": views,
                "gate": gate,
                "pose_params": best[1].tolist(),
                "calls": calls,
                "seconds": time.perf_counter() - started,
                "fit_views": [i.name for i in images],
            },
            indent=2,
        )
    )
    return report
