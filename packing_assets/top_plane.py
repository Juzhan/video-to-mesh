"""Constrain convex occupancy by a broad measured upper plane when evidence supports it."""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import trimesh


def fit_supported_top(points, normals, tolerance=0.001):
    points = np.asarray(points)
    normals = np.asarray(normals)
    high = (points[:, 2] > 0.5 * np.quantile(points[:, 2], 0.99)) & (
        normals[:, 2] > 0.94
    )
    top = points[high]
    if len(top) < 100:
        return None, {"reason": "not_enough_upward_surface"}
    rng = np.random.default_rng(19)
    sample = top[rng.choice(len(top), min(len(top), 3000), replace=False)]
    design = np.c_[sample[:, :2], np.ones(len(sample))]
    best = None
    for _ in range(400):
        subset = rng.choice(len(sample), 3, replace=False)
        try:
            plane = np.linalg.solve(design[subset], sample[subset, 2])
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(10)):
            continue
        errors = np.abs(design @ plane - sample[:, 2])
        good = errors < tolerance
        score = int(good.sum())
        if best is None or score > best[0]:
            best = (score, plane, good)
    if best is None:
        return None, {"reason": "no_broad_upper_plane"}
    plane = np.linalg.lstsq(design[best[2]], sample[best[2], 2], rcond=None)[0]
    errors = np.abs(np.c_[top[:, :2], np.ones(len(top))] @ plane - top[:, 2])
    good = errors < tolerance
    span = np.ptp(top[good, :2], axis=0) / np.maximum(
        np.ptp(points[:, :2], axis=0), 1e-9
    )
    excess = points[:, 2] - np.c_[points[:, :2], np.ones(len(points))] @ plane
    report = {
        "upward_plane_inlier_fraction": float(good.mean()),
        "all_point_support_fraction": float(good.sum() / len(points)),
        "xy_span_fraction": span.tolist(),
        "p999_point_above_plane_mm": float(np.quantile(excess, 0.999) * 1000),
        "plane_z_ax_by_c": plane.tolist(),
        "tolerance_m": tolerance,
        "assumption": "broad measured planar upper envelope; unseen protrusions remain unverified",
    }
    if (
        good.mean() < 0.80
        or good.sum() < 0.15 * len(points)
        or np.any(span < 0.60)
        or np.quantile(excess, 0.999) > 0.0015
    ):
        return None, {"reason": "upper_plane_not_supported", **report}
    return plane, report


def cap_hull(hull, plane):
    a, b, c = plane
    equations = ConvexHull(hull.vertices).equations
    halfspaces = np.r_[equations, [[-a, -b, 1.0, -c]]]
    A = np.c_[halfspaces[:, :3], np.linalg.norm(halfspaces[:, :3], axis=1)]
    result = linprog(
        [0, 0, 0, -1],
        A_ub=A,
        b_ub=-halfspaces[:, 3],
        bounds=[(None, None)] * 3 + [(0, None)],
        method="highs",
    )
    if not result.success or result.x[3] < 1e-6:
        raise ValueError("upper plane leaves no robust volume")
    points = HalfspaceIntersection(halfspaces, result.x[:3]).intersections
    return trimesh.convex.convex_hull(points)
