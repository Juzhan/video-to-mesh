"""Optional box candidate fitted to five measured planes, with explicit support-plane closure."""

import numpy as np
import trimesh


def fit_supported_box(cloud):
    import open3d as o3d

    points = np.asarray(cloud.points)
    total = len(points)
    if total < 1000:
        return None, {"applicable": False, "reason": "not enough measured points"}
    o3d.utility.random.seed(17)
    remaining = cloud
    planes = []
    for _ in range(10):
        if len(remaining.points) < max(100, total * 0.015):
            break
        equation, indices = remaining.segment_plane(
            distance_threshold=0.0015, ransac_n=3, num_iterations=2000
        )
        if len(indices) < max(100, total * 0.02):
            break
        xyz = np.asarray(remaining.points)[indices].copy()
        planes.append((np.array(equation), xyz))
        remaining = remaining.select_by_index(indices, invert=True)
    top = [(eq, p) for eq, p in planes if abs(eq[2]) > 0.996 and len(p) > 0.3 * total]
    # Thin faces have noisy fitted normals; actual point-to-box distances below remain decisive.
    sides = [(eq, p) for eq, p in planes if abs(eq[2]) < 0.174]
    failure = {
        "applicable": False,
        "reason": "five supported orthogonal planes not found",
        "plane_counts": [len(p) for eq, p in planes],
    }
    if not top or len(sides) < 4:
        return None, failure
    top = max(top, key=lambda item: len(item[1]))
    height = float(np.median(top[1][:, 2]))
    if height <= 0.003:
        return None, failure
    u = sides[0][0][:2]
    u = u / np.linalg.norm(u)
    if abs(u[1]) > abs(u[0]):
        u = np.array([u[1], -u[0]])
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    limits = []
    support_counts = []
    for axis in [u, v]:
        faces = []
        for eq, p in sides:
            normal = eq[:2] / np.linalg.norm(eq[:2])
            if abs(normal @ axis) > 0.996 and np.ptp(p[:, 2]) > height * 0.35:
                faces.append((float(np.median(p[:, :2] @ axis)), len(p)))
        if len(faces) != 2:
            return None, failure
        faces.sort()
        limits.append([faces[0][0], faces[1][0]])
        support_counts.extend([faces[0][1], faces[1][1]])
    limits = np.array(limits)
    extent = np.r_[np.diff(limits, axis=1).ravel(), height]
    if np.any(extent <= 0.003):
        return None, failure
    center_xy = u * limits[0].mean() + v * limits[1].mean()
    transform = np.eye(4)
    transform[:2, 0] = u
    transform[:2, 1] = v
    transform[:3, 3] = [*center_xy, height / 2]
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    q = abs(local) - extent / 2
    distances = abs(
        np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
    )
    fraction = float(np.mean(distances <= 0.0015))
    if fraction < 0.90:
        return None, {
            **failure,
            "reason": "measured points do not sufficiently support a box",
            "surface_inlier_fraction": fraction,
        }
    mesh = trimesh.creation.box(extent, transform=transform)
    return mesh, {
        "applicable": True,
        "backend": "five_measured_planes_supported_box",
        "top_plane_points": len(top[1]),
        "side_plane_points": support_counts,
        "surface_inlier_fraction": fraction,
        "point_distance_p95_mm": float(np.percentile(distances, 95) * 1000),
        "assumption": "Flat underside on board closes the sixth face. Edges and unobserved underside are regularized, not measured.",
        "true_dimensions_supplied": False,
    }
