"""RGB -> COLMAP SfM/MVS -> board scale -> support-plane removal -> object mesh.

Reconstruction algorithms are the official COLMAP implementation, not custom SfM/MVS.
Only images and a physical board definition are read. No Blender truth is accepted.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from .progress import event

from .assets import export_asset, load_mesh
from .media import IMAGE_EXT, prepare
from .metric import from_colmap, transform_points


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def recover_interrupted_stereo(work, state):
    """COLMAP skips existing map files, including truncated files after a reboot.

    Only a completed stereo stage is a usable checkpoint. Preserve partial data
    outside the active cache and retain intact images, cameras and configuration.
    """
    if state.get("active_stage") != "multi_view_stereo" or state.get("stages", {}).get(
        "multi_view_stereo", {}
    ).get("done"):
        return []
    import shutil

    archive = work / "recovery" / f"interrupted-stereo-{time.time_ns()}"
    moved = []
    for name in ("depth_maps", "normal_maps", "consistency_graphs"):
        source = work / "dense/stereo" / name
        if source.is_dir() and any(source.iterdir()):
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(archive / name))
            source.mkdir()
            moved.append(str(archive / name))
    if moved:
        state.setdefault("recovery", []).append(
            {
                "reason": "incomplete stereo stage is not a reusable checkpoint",
                "preserved_at": moved,
            }
        )
    return moved


def upgrade_mask_cache(out, work, state, segmentation, required_revision=4):
    """Keep measured geometry, rerun descendants of old broad-box SAM masks."""
    if segmentation not in ("sam2", "sam2-base") or not state.get("stages", {}).get(
        "photo_masks_v3", {}
    ).get("done"):
        return False
    report = work / "masks/report.json"
    try:
        revision = json.loads(report.read_text()).get("prompt_revision", 0)
    except (OSError, ValueError):
        revision = 0
    if revision >= required_revision:
        return False
    import shutil

    archive = work / "recovery" / f"before-reference-aware-masks-{time.time_ns()}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("masks", "masked_object_points.ply", "object_masked.ply"):
        path = work / name
        if path.exists():
            shutil.move(str(path), str(archive / name))
    for name in (
        "packing_proxy",
        "depth_proxy",
        "round_container",
        "regularized_box",
        "outer_convex",
        "asset.json",
        "selection.json",
        "surface_quality.json",
    ):
        path = out / name
        if path.exists():
            shutil.move(str(path), str(archive / name))
    early = {
        "prepare",
        "features",
        "matching",
        "camera_estimation",
        "metric_scale",
        "undistort",
        "multi_view_stereo",
        "fusion",
        "remove_board",
    }
    state["stages"] = {k: v for k, v in state["stages"].items() if k in early}
    state.setdefault("recovery", []).append(
        {
            "reason": f"reference-aware SAM prompt revision {required_revision}",
            "preserved_at": str(archive),
        }
    )
    return True


def input_digest(source, board_path, options):
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    digest.update(board_path.read_bytes())
    files = (
        sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if source.is_dir()
        else [source]
    )
    if not files:
        raise ValueError("no input images")
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def upgrade_proxy_cache(out, work, state):
    """Recompute occupancy after fixing out-of-frame votes; preserve RGB/MVS."""
    path = out / "packing_proxy/asset.json"
    if not path.exists():
        return False
    try:
        revision = (
            json.loads(path.read_text())
            .get("provenance", {})
            .get("visibility_revision", 0)
        )
    except (OSError, ValueError):
        revision = 0
    if revision >= 2:
        return False
    import shutil

    archive = work / "recovery" / f"before-visibility-fix-{time.time_ns()}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in (
        "packing_proxy",
        "depth_proxy",
        "round_container",
        "selection.json",
        "asset.json",
    ):
        source = out / name
        if source.exists():
            shutil.move(str(source), str(archive / name))
    for name in ("packing_proxy_v2", "depth_proxy_v2", "round_container_v3"):
        state["stages"].pop(name, None)
    state.setdefault("recovery", []).append(
        {
            "reason": "out-of-frame observations are unknown; occupancy revision 2",
            "preserved_at": str(archive),
        }
    )
    return True


def select_object_cloud(fused, transform, board, output, support_cut_m=0.001):
    from plyfile import PlyData, PlyElement

    cloud = PlyData.read(fused)
    data = cloud["vertex"].data
    xyz = np.column_stack([data[a] for a in ("x", "y", "z")])
    xyz = transform_points(xyz, transform)
    roi = np.array(board["suggested_roi_m"], float)
    roi[0, 2] = support_cut_m
    good = np.isfinite(xyz).all(1) & (xyz >= roi[0]).all(1) & (xyz <= roi[1]).all(1)
    points = xyz[good]
    selected = data[good].copy()
    if len(points) < 500:
        raise ValueError(
            f"only {len(points)} object points after board crop; capture more textured views"
        )
    # Remove isolated MVS noise with a data-driven nearest-neighbor distance gate.
    distances, _ = cKDTree(points).query(points, k=9, workers=4)
    local = distances[:, -1]
    cutoff = max(
        float(np.median(local) + 4 * np.median(abs(local - np.median(local)))), 0.001
    )
    keep = local < cutoff
    points, selected = points[keep], selected[keep]
    for i, a in enumerate(("x", "y", "z")):
        selected[a] = points[:, i]
    normals = np.column_stack([selected[a] for a in ("nx", "ny", "nz")])
    rotation = transform[:3, :3] / np.cbrt(np.linalg.det(transform[:3, :3]))
    normals = normals @ rotation.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    for i, a in enumerate(("nx", "ny", "nz")):
        selected[a] = normals[:, i]
    PlyData([PlyElement.describe(selected, "vertex")], text=False).write(output)
    return {
        "scene_points": len(data),
        "object_points": len(points),
        "support_cut_m": support_cut_m,
        "roi_m": roi.tolist(),
        "outlier_cutoff_m": cutoff,
    }


def surface_mesh(point_path, output, quality):
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(point_path))
    if len(cloud.points) < 500 or not cloud.has_normals():
        raise ValueError("surface reconstruction requires enough oriented points")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=9 if quality in ("draft", "fast") else 11,
        scale=1.08,
        linear_fit=True,
        n_threads=8,
    )
    density = np.asarray(densities)
    mesh.remove_vertices_by_mask(density < np.quantile(density, 0.03))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError("surface reconstruction produced nonfinite vertices")
    o3d.io.write_triangle_mesh(str(output), mesh)


FAST_PROFILE = {
    "version": 3,
    "max_frames": 48,
    "max_image_size": 640,
    "iterations": 2,
    "sift_features": 10000,
    "feature_image_size": 1600,
    "stereo_neighbors": 5,
    "proxy_agreement": 0.97,
}


def reconstruction_options(quality, collision, segmentation, scope, engine):
    if quality not in ("draft", "fast", "standard"):
        raise ValueError("unknown reconstruction quality")
    options = {
        "quality": quality,
        "version": 5,
        "engine": engine,
        "collision": collision,
        "segmentation": segmentation,
        "scope": scope,
    }
    if quality == "fast":
        options["fast_profile"] = dict(FAST_PROFILE)
    checkpoint = None
    if segmentation in ("sam2", "sam2-base", "hybrid"):
        model_name = (
            "sam2.1_hiera_base_plus.pt"
            if segmentation == "sam2-base"
            else "sam2.1_hiera_tiny.pt"
        )
        checkpoint = Path(__file__).resolve().parents[1] / "models" / model_name
        if not checkpoint.exists():
            raise ValueError(
                "quality workflow requires SAM 2: run scripts/setup_quality.sh"
            )
        options["segmentation_checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
    return options, checkpoint


def retain_surface_components(mesh):
    """A gap in reconstruction is not evidence that a pump/nozzle is unrelated noise."""
    parts = mesh.split(only_watertight=False)
    if not len(parts):
        raise ValueError("empty mesh from surface reconstruction")
    return trimesh.util.concatenate(parts), len(parts)


def select_candidate(report, scope):
    candidates = [("surface", {"status": report["status"], "path": "asset.json"})]
    if scope == "outer_shape":
        candidates = [
            ("round_container", report.get("round_container", {})),
            ("depth_proxy", report.get("depth_proxy", {})),
            ("regularized_box", report.get("regularized_box", {})),
        ]
    elif scope == "open_container":
        candidates = [
            ("round_container", report.get("round_container", {})),
            ("depth_proxy", report.get("depth_proxy", {})),
        ]
    elif scope == "outer_convex":
        candidates = [("outer_convex", report.get("outer_convex", {}))]
    elif scope != "full_geometry":
        raise ValueError("unknown geometry scope")
    usable = [(kind, c) for kind, c in candidates if c.get("status") == "needs_review"]
    return {
        "status": "needs_review" if usable else "rejected",
        "auto_approved": False,
        "candidate_for_review": usable[0][1]["path"] if usable else None,
        "candidate_kind": usable[0][0] if usable else None,
        "scope": scope,
        "note": "Visible exterior recesses must be preserved; hidden cavities are unverified. A review candidate is not certified geometry.",
    }


def reconstruct(
    source: Path,
    board_path: Path,
    out: Path,
    quality="standard",
    collision=False,
    segmentation="sam2",
    scope="outer_shape",
    frontend_only=False,
    measure_dimensions=False,
):
    import pycolmap

    if not pycolmap.has_cuda and not frontend_only:
        raise RuntimeError(
            "dense reconstruction needs pycolmap-cuda12; run scripts/setup_local.sh"
        )
    source, board_path, out = source.resolve(), board_path.resolve(), out.resolve()
    if frontend_only:
        quality = "fast"
    options, checkpoint = reconstruction_options(
        quality, collision, segmentation, scope, pycolmap.__version__
    )
    if frontend_only:
        options["meshy_frontend_revision"] = 1
        if scope != "outer_shape":
            raise ValueError("Meshy fast frontend currently supports outer_shape only")
    fingerprint = input_digest(source, board_path, options)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "reconstruction.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"fingerprint": fingerprint, "stages": {}}
    )
    if state["fingerprint"] != fingerprint:
        raise ValueError(
            "inputs, board or settings changed; choose a new output directory"
        )
    state["options"] = options
    state["source"] = str(source)
    state["status"] = "running"
    state.pop("error", None)
    save_json(state_path, state)
    started = time.monotonic()
    work = out / "work"
    work.mkdir(exist_ok=True)
    if recover_interrupted_stereo(work, state):
        save_json(state_path, state)
        print(
            "RECOVERY: quarantined incomplete stereo maps; reusing completed camera/scale stages",
            flush=True,
        )
    # Completed Meshy observations can already be bound to paid input crops.
    # Upgrade masks only in the separate local-measurement workspace.
    frozen_frontend = frontend_only and state.get("frontend") == "meshy_fast"
    if not frozen_frontend and upgrade_mask_cache(
        out, work, state, segmentation, 6 if quality == "fast" else 4
    ):
        save_json(state_path, state)
        print(
            "UPGRADE: rerunning masks and dependent meshes; reusing RGB camera and depth caches",
            flush=True,
        )
    if upgrade_proxy_cache(out, work, state):
        save_json(state_path, state)
        print(
            "UPGRADE: rebuilding occupancy with correct image visibility; reusing masks and depth",
            flush=True,
        )

    def stage(name, function):
        if state["stages"].get(name, {}).get("done"):
            event("stage", name, cached=True)
            print(f"RESUME {name}", flush=True)
            return
        state["active_stage"] = name
        save_json(state_path, state)
        print(f"START {name}", flush=True)
        event("stage", name)
        start = time.monotonic()
        function()
        state["stages"][name] = {
            "done": True,
            "seconds": round(time.monotonic() - start, 2),
        }
        save_json(state_path, state)
        print(f"DONE {name}", flush=True)

    images = work / "images"
    sparse = work / "sparse"
    dense = work / "dense"
    db = work / "database.db"
    board = json.loads(board_path.read_text())
    try:

        def preparation():
            # A partially written prepare folder cannot be reused as a finished capture.
            if images.exists():
                import shutil

                shutil.rmtree(images)
            report = (
                prepare(
                    source,
                    images,
                    FAST_PROFILE["max_frames"],
                    max_photos=FAST_PROFILE["max_frames"],
                )
                if quality == "fast"
                else prepare(source, images, 100)
            )
            if report["frame_count"] < 12:
                raise ValueError(
                    "need at least 12 overlapping views; 36-100 is a better starting point"
                )

        stage("prepare", preparation)
        image_count = len(list(images.glob("*.jpg")))
        stage(
            "features",
            lambda: pycolmap.extract_features(
                db,
                images,
                camera_mode=pycolmap.CameraMode.SINGLE,
                reader_options={"camera_model": "SIMPLE_RADIAL"},
                extraction_options={
                    "max_image_size": FAST_PROFILE["feature_image_size"]
                    if quality == "fast"
                    else 1600,
                    "num_threads": 8,
                    "sift": {
                        "max_num_features": FAST_PROFILE["sift_features"]
                        if quality == "fast"
                        else 10000
                    },
                },
                device=pycolmap.Device.cpu,
            ),
        )
        stage(
            "matching",
            lambda: pycolmap.match_exhaustive(
                db, matching_options={"num_threads": 8}, device=pycolmap.Device.cpu
            ),
        )

        def mapping():
            sparse.mkdir(exist_ok=True)
            maps = pycolmap.incremental_mapping(
                db, images, work / "maps", options={"num_threads": 8}
            )
            best = (
                max(maps.values(), key=lambda m: m.num_reg_images()) if maps else None
            )
            if best is None or best.num_reg_images() < max(12, 0.8 * image_count):
                from .board_pose import from_capture

                sfm_count = best.num_reg_images() if best is not None else 0
                print(
                    f"BOARD POSE FALLBACK: SfM registered {sfm_count}/{image_count}",
                    flush=True,
                )
                best, board_report = from_capture(
                    work / "images_capture.json", board, work / "board_pose_model"
                )
                board_report["sfm_registered_before_fallback"] = sfm_count
                save_json(work / "board_metric.json", board_report)
                state["camera_method"] = "metric_board_calibration"
            else:
                state["camera_method"] = "colmap_sfm"
            best.write(sparse)
            state["registered_images"] = best.num_reg_images()
            state["input_images"] = image_count

        stage("camera_estimation", mapping)

        def metric():
            if state.get("camera_method") == "metric_board_calibration":
                report = json.loads((work / "board_metric.json").read_text())
            else:
                transform, report = from_colmap(sparse, images, board)
            save_json(work / "metric.json", report)

        stage("metric_scale", metric)
        metric_report = json.loads((work / "metric.json").read_text())
        transform = np.array(metric_report["transform_to_m_z_up"])
        if frontend_only:
            from .meshy_frontend import SparseForegroundError, finish_sparse_frontend

            # Existing fast observations may already have been submitted to Meshy.
            # Keep those immutable; extra measurement gets its own work directory.
            new_hybrid = measure_dimensions and state.get("frontend") != "meshy_fast"
            if (
                not state.get("frontend_fallback")
                and not new_hybrid
                and state.get("frontend_prompt_source") != "dense"
            ):
                try:
                    return finish_sparse_frontend(
                        out, work, board, transform, checkpoint, state, stage, started
                    )
                except SparseForegroundError as error:
                    # Keep the calibrated cameras and scale. Only recover foreground
                    # measurements, never build an unnecessary local final surface.
                    state["frontend_fallback"] = {
                        "from": "sparse",
                        "to": "dense_prompts",
                        "reason": str(error),
                    }
                    save_json(state_path, state)
            print(
                "DENSE FOREGROUND: hybrid measurement or sparse recovery; reusing cameras and A4 scale",
                flush=True,
            )
            if not pycolmap.has_cuda:
                raise RuntimeError(
                    "稀疏物体特征不足，自动补充深度需要 CUDA；请运行 scripts/setup_local.sh 配置本机重建依赖"
                )
        size = (
            FAST_PROFILE["max_image_size"]
            if quality == "fast"
            else 800
            if quality == "draft"
            else 1600
        )
        stage(
            "undistort",
            lambda: pycolmap.undistort_images(
                dense,
                sparse,
                images,
                num_patch_match_src_images=FAST_PROFILE["stereo_neighbors"]
                if quality == "fast"
                else 8,
                undistort_options={"max_image_size": size},
                num_threads=8,
            ),
        )
        stage(
            "multi_view_stereo",
            lambda: pycolmap.patch_match_stereo(
                dense,
                options={
                    "max_image_size": size,
                    "num_iterations": FAST_PROFILE["iterations"]
                    if quality == "fast"
                    else 3
                    if quality == "draft"
                    else 5,
                    "geom_consistency": True,
                    "filter_min_num_consistent": 3,
                    "cache_size": 8.0,
                    "gpu_index": "0",
                },
            ),
        )
        stage(
            "fusion",
            lambda: pycolmap.stereo_fusion(
                dense / "fused.ply",
                dense,
                options={"min_num_pixels": 3, "num_threads": 8},
                output_type="ply",
            ),
        )

        def isolate():
            report = select_object_cloud(
                dense / "fused.ply", transform, board, work / "object_points.ply"
            )
            save_json(work / "isolation.json", report)

        stage("remove_board", isolate)

        def multiview_isolation():
            if segmentation == "hybrid":
                if scope != "outer_convex":
                    raise ValueError(
                        "hybrid fills visible 2D holes; use sam2 for shape-preserving output"
                    )
                from .hybrid_masks import isolate_hybrid

                isolate_hybrid(
                    work / "object_points.ply",
                    sparse,
                    images,
                    transform,
                    work / "masked_object_points.ply",
                    work / "masks",
                    checkpoint,
                )
            elif segmentation in ("sam2", "sam2-base"):
                from .sam_masks import isolate_sam

                isolate_sam(
                    work / "object_points.ply",
                    sparse,
                    images,
                    transform,
                    work / "masked_object_points.ply",
                    work / "masks",
                    checkpoint=checkpoint,
                    support_envelope=quality == "fast",
                )
            else:
                from .isolate import isolate_multiview

                isolate_multiview(
                    work / "object_points.ply",
                    sparse,
                    images,
                    transform,
                    work / "masked_object_points.ply",
                    work / "masks",
                    compact_seed=segmentation == "grabcut-core",
                )

        stage("photo_masks_v3", multiview_isolation)
        if frontend_only:
            return finish_sparse_frontend(
                out,
                work,
                board,
                transform,
                checkpoint,
                state,
                stage,
                started,
                dense_prompts=True,
            )
        stage(
            "masked_surface_v3",
            lambda: surface_mesh(
                work / "masked_object_points.ply", work / "object_masked.ply", quality
            ),
        )
        mesh = load_mesh(work / "object_masked.ply")
        event("stage", "surface_check")
        from .isolate import clip_mesh_to_masks

        silhouette_check = clip_mesh_to_masks(mesh, sparse, transform, work / "masks")
        mesh, component_count = retain_surface_components(mesh)
        # Bound Poisson extrapolation using the physical scan region, not true object dimensions.
        roi = np.array(board["suggested_roi_m"])
        v = mesh.vertices
        inside = (v >= roi[0]).all(1) & (v <= roi[1]).all(1)
        mesh.update_faces(inside[mesh.faces].all(1))
        mesh.remove_unreferenced_vertices()
        if len(mesh.faces) < 100:
            raise ValueError("too little reconstructed surface")
        provenance = {
            "backend": "colmap_rgb",
            "engine_version": pycolmap.__version__,
            "surface_engine": "Open3D screened Poisson; lower 3% density trimmed",
            "scale_source": state.get("camera_method", "colmap_sfm") + "_board",
            "calibration": metric_report,
            "isolation": json.loads((work / "isolation.json").read_text()),
            "photo_masks": json.loads((work / "masks/report.json").read_text()),
            "silhouette_clipping": silhouette_check,
            "components_preserved": component_count,
            "support_plane_assumption": "object rests on the same support plane as the reference board; lowest 1 mm is not observed by this crop",
            "geometry_source": "RGB multiview stereo; no supplied object mesh, dimensions or camera poses",
        }
        from .quality import check_views, surface_release_gate

        view_report = check_views(mesh, work)
        surface_gate = surface_release_gate(mesh, view_report)
        save_json(
            out / "surface_quality.json",
            {"view_consistency": view_report, "release_gate": surface_gate},
        )
        report = export_asset(
            mesh,
            out,
            provenance,
            collision=collision and surface_gate["status"] != "rejected",
        )
        report["surface_quality"] = surface_gate
        report["status"] = surface_gate["status"]
        report["review_reasons"].extend(
            [
                "support_contact_and_unseen_bottom_need_review",
                "disconnected_reconstructed_parts_preserved_require_review",
            ]
            if component_count > 1
            else ["support_contact_and_unseen_bottom_need_review"]
        )
        save_json(out / "asset.json", report)
        # Keep the measured surface and closed packing approximation as distinct assets.
        from .proxy import build_proxy

        try:
            stage(
                "packing_proxy_v2",
                lambda: build_proxy(
                    work,
                    board,
                    out / "packing_proxy",
                    pitch=0.001,
                    smoothing=True,
                    agreement=FAST_PROFILE["proxy_agreement"]
                    if quality == "fast"
                    else 0.97,
                ),
            )
            proxy_path = out / "packing_proxy/asset.json"
            event("stage", "proxy_check")
            proxy_report = json.loads(proxy_path.read_text())
            proxy_mesh = load_mesh(out / "packing_proxy/visual.obj")
            proxy_mesh.apply_translation(
                -np.array(proxy_report["centering_translation_m"])
            )
            proxy_views = check_views(proxy_mesh, work)
            proxy_gate = surface_release_gate(proxy_mesh, proxy_views)
            proxy_report["status"] = proxy_gate["status"]
            proxy_report["geometry_scope"] = (
                "outer_occupancy_only; cavities, nesting and small protrusions are not certified"
            )
            proxy_report["quality"] = proxy_gate
            save_json(proxy_path, proxy_report)
            save_json(
                out / "packing_proxy/quality.json",
                {"view_consistency": proxy_views, "release_gate": proxy_gate},
            )
            report["packing_proxy"] = {
                "status": proxy_gate["status"],
                "path": "packing_proxy/asset.json",
                "geometry_scope": proxy_report["geometry_scope"],
            }
        except Exception as proxy_error:
            report["packing_proxy"] = {"status": "failed", "error": str(proxy_error)}
            report["review_reasons"].append("packing_proxy_generation_failed")
        if scope in ("outer_shape", "open_container"):
            try:
                from .depth_proxy import build_depth_proxy

                stage(
                    "depth_proxy_v2",
                    lambda: build_depth_proxy(
                        work,
                        out / "packing_proxy",
                        out / "depth_proxy",
                        min_views=2,
                        require_cavity=scope == "open_container",
                    ),
                )
                depth_report = json.loads((out / "depth_proxy/asset.json").read_text())
                report["depth_proxy"] = {
                    "status": depth_report["status"],
                    "path": "depth_proxy/asset.json",
                    "cavity": depth_report["cavity"],
                }
                if depth_report["cavity"]["detected"]:
                    try:
                        from .round_container import regularize_round_container

                        stage(
                            "round_container_v3",
                            lambda: regularize_round_container(
                                work,
                                out / "depth_proxy",
                                out / "packing_proxy",
                                out / "round_container",
                            ),
                        )
                        regular = json.loads(
                            (out / "round_container/asset.json").read_text()
                        )
                        report["round_container"] = {
                            "status": regular["status"],
                            "path": "round_container/asset.json",
                            "cavity": regular["cavity"],
                        }
                    except Exception as regular_error:
                        report["round_container"] = {
                            "status": "not_applicable",
                            "reason": str(regular_error),
                        }
            except Exception as depth_error:
                report["depth_proxy"] = {"status": "failed", "error": str(depth_error)}
        # Fit a regular solid only when five observed planes and the photos support that prior.
        import open3d as o3d

        event("stage", "candidate_check")
        from .planar import fit_supported_box

        box, box_fit = fit_supported_box(
            o3d.io.read_point_cloud(str(work / "masked_object_points.ply"))
        )
        if box is not None:
            box_views = check_views(box, work)
            box_gate = surface_release_gate(box, box_views)
            box_report = export_asset(
                box,
                out / "regularized_box",
                {
                    "scale_source": "triangulated_board",
                    "calibration": metric_report,
                    **box_fit,
                },
            )
            box_report["status"] = box_gate["status"]
            box_report["quality"] = box_gate
            box_report["review_reasons"].append(
                "five_plane_box_prior_and_flat_unobserved_underside"
            )
            save_json(out / "regularized_box/asset.json", box_report)
            save_json(
                out / "regularized_box/quality.json",
                {"view_consistency": box_views, "release_gate": box_gate},
            )
            report["regularized_box"] = {
                "status": box_gate["status"],
                "path": "regularized_box/asset.json",
            }
        else:
            report["regularized_box"] = {
                "status": "not_applicable",
                "diagnostic": box_fit,
            }
        if scope == "outer_convex":
            from .convex import build_outer_convex

            report["outer_convex"] = build_outer_convex(out, work)
        selection = select_candidate(report, scope)
        report["selection"] = selection
        save_json(out / "selection.json", selection)
        (out / "README.md").write_text(
            "# 重建诊断缓存，不是最终交付目录\n\n"
            "`visual.obj` 是原始测量表面，可能有开边和缺失的底面。"
            "不要把它当成最终水密模型或直接用于碰撞。\n\n"
            "原始表面状态：" + report["status"] + "。\n\n"
            "当前候选：" + str(selection.get("candidate_for_review")) + "。"
            "最终普通版和平滑版请打开批处理输出的 `models/原视频名/`，"
            "其 `status.json` 记录重新读取最终 OBJ 后的水密检查。\n"
        )
        save_json(out / "asset.json", report)
        state["status"] = selection["status"]
        state["surface_status"] = report["status"]
        state.pop("error", None)
        state.pop("active_stage", None)
        state["last_run_seconds"] = round(time.monotonic() - started, 2)
        state["total_stage_seconds"] = round(
            sum(s["seconds"] for s in state["stages"].values()), 2
        )
        save_json(state_path, state)
        return report
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        save_json(state_path, state)
        raise
