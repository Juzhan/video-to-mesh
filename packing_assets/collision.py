"""Compound convex collisions in one shared metric frame, with union occupancy validation."""

import json
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from .assets import load_mesh
from .quality import ray_scene


def union_overlap(parts, reference, pitch=0.0015):
    import open3d as o3d

    lo = np.minimum(reference.bounds[0], np.min([p.bounds[0] for p in parts], axis=0))
    hi = np.maximum(reference.bounds[1], np.max([p.bounds[1] for p in parts], axis=0))
    counts = np.ceil((hi - lo) / pitch).astype(int)
    if np.prod(counts) > 4000000:
        return union_overlap(parts, reference, pitch * 1.5)
    axes = [lo[i] + (np.arange(counts[i]) + 0.5) * pitch for i in range(3)]
    points = (
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype("f4")
    )
    points += (
        np.random.default_rng(31)
        .uniform(-pitch / 2, pitch / 2, points.shape)
        .astype("f4")
    )
    target = ray_scene(reference)
    truth = np.zeros(len(points), bool)
    prediction = np.zeros(len(points), bool)
    for start in range(0, len(points), 100000):
        truth[start : start + 100000] = (
            target.compute_occupancy(
                o3d.core.Tensor(points[start : start + 100000]), nthreads=2, nsamples=3
            ).numpy()
            > 0.5
        )
    # Build one acceleration structure at a time; hundreds of simultaneous scenes
    # can exhaust native resources on highly fragmented reconstruction meshes.
    for part in parts:
        scene = ray_scene(part)
        indices = np.flatnonzero(
            (points >= part.bounds[0]).all(1)
            & (points <= part.bounds[1]).all(1)
            & ~prediction
        )
        for start in range(0, len(indices), 100000):
            selected = indices[start : start + 100000]
            prediction[selected] |= (
                scene.compute_occupancy(
                    o3d.core.Tensor(points[selected]), nthreads=2, nsamples=3
                ).numpy()
                > 0.5
            )
        del scene
    intersection = int((prediction & truth).sum())
    union = int((prediction | truth).sum())
    truth_count = int(truth.sum())
    pred_count = int(prediction.sum())
    return {
        "iou": intersection / max(union, 1),
        "excess_fraction_of_reference": (pred_count - intersection)
        / max(truth_count, 1),
        "missing_fraction_of_reference": (truth_count - intersection)
        / max(truth_count, 1),
        "pitch_mm": pitch * 1000,
        "method": "OR of per-part occupancy; overlapping collision pieces are not double-counted",
    }


def collision_input(mesh, threshold_m):
    """Bound tiny near-surface noise removal; retain substantial disconnected parts."""
    import open3d as o3d

    components = sorted(
        mesh.split(only_watertight=False), key=lambda p: abs(p.volume), reverse=True
    )
    main = components[0]
    surface = ray_scene(main)
    kept = [main]
    removed = 0.0
    count = 0
    total = sum(abs(p.volume) for p in components)
    for part in components[1:]:
        volume = abs(part.volume)
        if volume < total * 0.0001 and removed + volume <= total * 0.001:
            distance = surface.compute_distance(
                o3d.core.Tensor(np.asarray(part.vertices, dtype="f4")), nthreads=2
            ).numpy()
            if float(distance.max()) <= threshold_m:
                removed += volume
                count += 1
                continue
        kept.append(part)
    combined = trimesh.util.concatenate(kept)
    before = len(combined.faces)
    if before > 8000:
        native = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(combined.vertices),
            o3d.utility.Vector3iVector(combined.faces),
        )
        reduced = native.simplify_quadric_decimation(8000)
        candidate = trimesh.Trimesh(
            np.asarray(reduced.vertices), np.asarray(reduced.triangles), process=True
        )
        if candidate.is_volume:
            combined = candidate
    return combined, {
        "source_components": len(components),
        "removed_near_surface_tiny_components": count,
        "removed_volume_fraction": removed / max(total, 1e-20),
        "removed_component_max_distance_m": threshold_m,
        "input_faces": before,
        "decomposition_faces": len(combined.faces),
        "note": "Only collision input is simplified. Final union is compared to the full original reconstruction.",
    }


def pad_convex(parts, margin):
    """Small outward Minkowski offset; shared frame and all original vertices retained."""
    offsets = np.r_[np.eye(3), -np.eye(3)] * margin
    return [
        trimesh.convex.convex_hull(
            (p.vertices[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        )
        for p in parts
    ]


def build_collision(mesh_path: Path, out: Path, threshold_m=0.003, max_hulls=64):
    import coacd

    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    if out.exists() and any(out.iterdir()):
        raise ValueError("collision output must be a fresh folder")
    mesh = load_mesh(mesh_path)
    if not mesh.is_volume:
        raise ValueError("use a closed occupancy proxy for collision decomposition")
    simplified, preparation = collision_input(mesh, threshold_m)
    coacd.set_log_level("warn")
    attempts = []
    padding = 0.0
    for effective_threshold in [threshold_m, threshold_m / 2, threshold_m / 4]:
        # Input is already a closed volume; coarse voxel remeshing can inflate thin items.
        raw = coacd.run_coacd(
            coacd.Mesh(simplified.vertices, simplified.faces),
            threshold=effective_threshold,
            real_metric=True,
            max_convex_hull=max_hulls,
            preprocess_mode="off",
            decimate=True,
            max_ch_vertex=64,
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=150,
            mcts_max_depth=3,
            seed=17,
        )
        if len(raw) > max(128, max_hulls):
            raise ValueError(
                f"excessive collision fragmentation: {len(raw)} parts; inspect source geometry"
            )
        parts = [
            trimesh.Trimesh(vertices=v, faces=f, process=True).convex_hull
            for v, f in raw
        ]
        if not parts or not all(p.is_volume and p.is_convex for p in parts):
            raise ValueError("invalid convex decomposition")
        scores = union_overlap(parts, mesh)
        attempts.append(
            {
                "effective_threshold_m": effective_threshold,
                "parts": len(parts),
                **scores,
            }
        )
        # Low-vertex hulls may slightly shrink curved surfaces. Recover that loss
        # with a bounded outward offset only when the full union check approves it.
        if scores["missing_fraction_of_reference"] > 0.02 and scores["iou"] >= 0.93:
            for margin in [0.00025, 0.0005]:
                padded = pad_convex(parts, margin)
                padded_scores = union_overlap(padded, mesh)
                attempts.append(
                    {
                        "effective_threshold_m": effective_threshold,
                        "padding_m": margin,
                        "parts": len(padded),
                        **padded_scores,
                    }
                )
                if (
                    padded_scores["iou"] >= 0.95
                    and padded_scores["missing_fraction_of_reference"] <= 0.02
                ):
                    parts, scores, padding = padded, padded_scores, margin
                    break
        if scores["iou"] >= 0.95 and scores["missing_fraction_of_reference"] <= 0.02:
            break
    if scores["iou"] < 0.95 or scores["missing_fraction_of_reference"] > 0.02:
        raise ValueError(
            f"collision simplification failed occupancy check: IoU={scores['iou']:.3f}, missing={scores['missing_fraction_of_reference']:.3f}"
        )
    out.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    colors = [
        [100, 170, 225, 255],
        [242, 176, 93, 255],
        [119, 198, 160, 255],
        [207, 142, 196, 255],
        [183, 193, 95, 255],
    ]
    for i, part in enumerate(parts):
        (out / f"part_{i:03d}.obj").write_text(
            trimesh.exchange.obj.export_obj(part, include_texture=False, digits=16)
        )
        display = part.copy()
        display.visual.face_colors = colors[i % len(colors)]
        display.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )
        scene.add_geometry(display, node_name=f"collision_{i:03d}")
    scene.export(out / "compound.glb")
    report = {
        "units": "m",
        "obj_up_axis": "z",
        "glb_up_axis": "y",
        "origin": "same as source; do not recenter parts independently",
        "source": str(mesh_path.resolve()),
        "source_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "backend": "CoACD",
        "threshold_m": threshold_m,
        "real_metric": True,
        "parts": len(parts),
        "total_faces": sum(len(p.faces) for p in parts),
        "seed": 17,
        "source_volume_comparison": scores,
        "preparation": preparation,
        "attempts": attempts,
        "outward_padding_m": padding,
        "pipeline_version": 3,
        "status": "needs_review",
        "auto_approved": False,
        "limitation": "Threshold is an algorithm parameter, not a certified Hausdorff bound. Comparison is against reconstructed geometry, not the real item. Mass and friction remain unspecified.",
    }
    (out / "collision.json").write_text(json.dumps(report, indent=2))
    return report
