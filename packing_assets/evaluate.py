"""Independent evaluation against an optional ground-truth mesh. Never used to reconstruct."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .assets import load_mesh


def bottom_center(mesh):
    mesh.apply_translation(
        [-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), -mesh.bounds[0, 2]]
    )
    return mesh


def volume_overlap(predicted, truth, pitch=0.0015):
    if not predicted.is_volume or not truth.is_volume:
        return None
    import open3d as o3d
    from .quality import ray_scene

    lo = np.minimum(predicted.bounds[0], truth.bounds[0])
    hi = np.maximum(predicted.bounds[1], truth.bounds[1])
    axes = [
        lo[i] + (np.arange(int(np.ceil((hi[i] - lo[i]) / pitch))) + 0.5) * pitch
        for i in range(3)
    ]
    # Cap evaluator memory for large assets; report the actual resolution.
    count = np.prod([len(a) for a in axes])
    if count > 8000000:
        return volume_overlap(predicted, truth, pitch * 1.5)
    points = (
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype("f4")
    )
    # Jitter avoids a regular grid reporting perfect overlap for sub-voxel boundary errors.
    points += (
        np.random.default_rng(23)
        .uniform(-pitch / 2, pitch / 2, points.shape)
        .astype("f4")
    )
    pscene, tscene = ray_scene(predicted), ray_scene(truth)
    intersection = union = truth_count = pred_count = 0
    for start in range(0, len(points), 200000):
        sample = o3d.core.Tensor(points[start : start + 200000])
        p = pscene.compute_occupancy(sample, nthreads=8, nsamples=3).numpy() > 0.5
        t = tscene.compute_occupancy(sample, nthreads=8, nsamples=3).numpy() > 0.5
        intersection += int(np.count_nonzero(p & t))
        union += int(np.count_nonzero(p | t))
        pred_count += int(p.sum())
        truth_count += int(t.sum())
    return {
        "iou": intersection / max(union, 1),
        "pitch_mm": pitch * 1000,
        "stratified_jitter_seed": 23,
        "excess_fraction_of_truth": (pred_count - intersection) / max(truth_count, 1),
        "missing_fraction_of_truth": (truth_count - intersection) / max(truth_count, 1),
    }


def support_errors(predicted, truth):
    # Shared board axes; no fitted rotation or scale. Fibonacci directions plus the six axes.
    i = np.arange(642)
    z = 1 - 2 * (i + 0.5) / 642
    r = np.sqrt(1 - z * z)
    a = i * np.pi * (3 - np.sqrt(5))
    directions = np.r_[np.c_[r * np.cos(a), r * np.sin(a), z], np.eye(3), -np.eye(3)]
    p = predicted.convex_hull.vertices @ directions.T
    t = truth.convex_hull.vertices @ directions.T
    delta = (p.max(0) - t.max(0)) * 1000
    # For convex sets, distance to the other set attains its maximum at a vertex.
    # This catches shallow top-plane errors that a finite directional grid can miss.
    import open3d as o3d
    from .quality import ray_scene

    phull, thull = predicted.convex_hull, truth.convex_hull
    over = (
        ray_scene(thull)
        .compute_signed_distance(
            o3d.core.Tensor(np.asarray(phull.vertices, dtype="f4")),
            nthreads=8,
            nsamples=3,
        )
        .numpy()
    )
    under = (
        ray_scene(phull)
        .compute_signed_distance(
            o3d.core.Tensor(np.asarray(thull.vertices, dtype="f4")),
            nthreads=8,
            nsamples=3,
        )
        .numpy()
    )
    return {
        "method": "maximum convex-vertex distance to the other closed convex set; float32 mesh queries",
        "sampled_directions": len(directions),
        "sampled_support_under_mm": float(max(0, -delta.min())),
        "sampled_support_over_mm": float(max(0, delta.max())),
        "max_under_mm": float(max(0, under.max()) * 1000),
        "max_over_mm": float(max(0, over.max()) * 1000),
    }


def evaluate(asset: Path, truth_path: Path, out: Path, scope="full_geometry"):
    predicted = bottom_center(load_mesh(asset))
    truth = load_mesh(truth_path)
    if scope == "outer_convex":
        truth = truth.convex_hull
    truth = bottom_center(truth)
    # Translation only. No ICP, axis stretch or scale fitting to the expected object.
    a, _ = trimesh.sample.sample_surface(predicted, 100000, seed=13)
    b, _ = trimesh.sample.sample_surface(truth, 100000, seed=17)
    da = cKDTree(b).query(a, workers=4)[0]
    db = cKDTree(a).query(b, workers=4)[0]
    errors = abs(predicted.extents - truth.extents)
    limit = np.maximum(0.003, 0.03 * truth.extents)
    both = np.r_[da, db]
    overlap = volume_overlap(predicted, truth)
    checks = {
        "dimensions": bool(np.all(errors <= limit)),
        "surface_p95_at_most_3mm": bool(np.percentile(both, 95) <= 0.003),
        "closed_volume": bool(predicted.is_volume),
        "volume_iou_at_least_90_percent": bool(
            overlap is not None and overlap["iou"] >= 0.90
        ),
    }
    support = None
    if scope == "outer_convex":
        support = support_errors(predicted, truth)
        checks.update(
            {
                "convex_output": bool(predicted.is_convex),
                "extremities_under_at_most_3mm": support["max_under_mm"] <= 3,
                "envelope_over_at_most_5mm": support["max_over_mm"] <= 5,
            }
        )
    report = {
        "evaluation": "independent_Blender_truth_not_given_to_reconstruction",
        "scope": scope,
        "support_errors": support,
        "alignment": "bottom center translation only; no fitted rotation or scale",
        "true_dimensions_mm": (truth.extents * 1000).tolist(),
        "reconstructed_dimensions_mm": (predicted.extents * 1000).tolist(),
        "dimension_error_mm": (errors * 1000).tolist(),
        "dimension_gate_passed": bool(np.all(errors <= limit)),
        "sampled_surface_distance_median_mm": float(np.median(both) * 1000),
        "sampled_surface_distance_p95_mm": float(np.percentile(both, 95) * 1000),
        "surface_fraction_within_2mm": float(np.mean(both <= 0.002)),
        "predicted_watertight": bool(predicted.is_watertight),
        "volume_overlap": overlap,
        "benchmark_gate": {
            "status": "passed" if all(checks.values()) else "rejected",
            "checks": checks,
            "scope": f"synthetic {scope} only; not a real-world certification",
        },
        "notes": [
            "Surface distances are finite-sample bidirectional nearest-point estimates, not exact Hausdorff distance.",
            "Synthetic renders; not proof of real-world accuracy. Some cases deliberately have weak texture.",
            "Dimension gate is max(3 mm, 3%) per axis; critical cavities still require separate review.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report
