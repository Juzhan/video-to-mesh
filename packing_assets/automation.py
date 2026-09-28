"""Batch RGB reconstruction, checked candidate export, and a shared metric scene."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from .assets import load_mesh, audit_mesh_file
from .folder_batch import discover, inferred_scope, VIDEO_EXT, select_objects
from .pipeline import run_batch
from .progress import event, ProgressSession
from contextlib import nullcontext

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def isolated_export(source, destination, mass, threshold, smooth=True):
    """A native geometry-library crash must not abort other objects in the batch."""
    command = [
        sys.executable,
        "-m",
        "packing_assets.cli",
        "export-pybullet",
        "--asset",
        str(source),
        "--out",
        str(destination),
        "--mass-kg",
        str(mass),
        "--threshold-mm",
        str(threshold * 1000),
    ]
    if not smooth:
        command.append("--no-smooth")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
    )
    if result.returncode:
        raise RuntimeError(
            f"collision/export process exited {result.returncode}: "
            + result.stdout[-1800:]
        )


def closed_candidate(result):
    """Never substitute a rejected candidate or a global convex hull."""
    selection = json.loads((result / "selection.json").read_text())
    if selection["status"] == "rejected":
        raise ValueError("reconstruction rejected: no candidate can be exported")
    paths = [selection.get("candidate_for_review")]
    if selection.get("scope") == "outer_shape":
        paths += [
            "round_container/asset.json",
            "depth_proxy/asset.json",
            "regularized_box/asset.json",
        ]
    for relative in dict.fromkeys(p for p in paths if p):
        path = (result / relative).resolve()
        if not path.is_relative_to(result.resolve()) or "outer_convex" in path.parts:
            continue
        if not path.is_file():
            continue
        meta = json.loads(path.read_text())
        if meta.get("provenance", {}).get("backend") == "metric_visual_hull":
            continue
        if selection.get("scope") == "open_container" and not meta.get(
            "cavity", {}
        ).get("detected"):
            continue
        if meta.get("status") != "needs_review":
            continue
        if load_mesh(path.parent / "visual.obj").is_volume:
            return path.parent
    raise ValueError(
        "no admitted closed exterior model; inspect reconstruction or recapture"
    )


def layout(items, width=1.0, gap=0.075):
    """Translate only. Actual bounds, including origin offsets, govern spacing."""
    x = y = row_depth = 0.0
    for item in items:
        lo, hi = np.array(item["bounds_m"])
        dx, dy, _ = hi - lo
        if x and x + dx > width:
            x = 0.0
            y += row_depth + gap
            row_depth = 0.0
        item["position_m"] = [float(x - lo[0]), float(y - lo[1]), float(-lo[2])]
        x += dx + gap
        row_depth = max(row_depth, dy)
    return items


def named_models(rows, errors, out, incremental=False, name_context=None):
    """Publish current outputs under original object names, including failures."""
    entries = [*rows, *[e for e in errors if e.get("id")]]
    stems = {
        e["id"]: Path(e.get("label", e["id"])).stem
        if Path(e.get("label", e["id"])).suffix.lower() in VIDEO_EXT
        else e.get("label", e["id"])
        for e in (name_context or entries)
    }
    names = list(stems.values())
    root = out / "models"
    root.mkdir(exist_ok=True)
    old_index = root / "index.json"
    previous_entries = json.loads(old_index.read_text()) if old_index.exists() else []
    updated_ids = {e["id"] for e in entries}
    if old_index.exists():
        for previous in previous_entries:
            if incremental and previous["id"] not in updated_ids:
                continue
            for key in ("visual", "smooth_visual", "dimension_visual", "local_visual"):
                if previous.get(key):
                    path = (out / previous[key]).resolve()
                    if path.is_relative_to(root.resolve()):
                        path.unlink(missing_ok=True)
    index = (
        [e for e in previous_entries if e["id"] not in updated_ids]
        if incremental
        else []
    )
    for entry in entries:
        name = stems[entry["id"]]
        if name in ("", ".", "..") or "/" in name or "\\" in name:
            name = entry["id"]
        if names.count(name) > 1:
            name += "-" + entry["id"]
        folder = root / name
        folder.mkdir(exist_ok=True)
        # Only remove outputs this function owns; a failed rerun must not leave
        # an old model looking like a new successful result.
        for filename in (
            name + ".obj",
            name + "_smooth.obj",
            name + "_dimensions.obj",
            name + "_local.obj",
            "dimension-report.json",
            "local-report.json",
        ):
            (folder / filename).unlink(missing_ok=True)
        status = {
            "id": entry["id"],
            "input_name": entry.get("label", entry["id"]),
            "units": "m",
            "up_axis": "z",
            "status": entry.get("quality_status", "failed"),
            "quality_check": entry.get("quality_check", {}),
        }
        if "visual" in entry:
            checks = {}
            for key, suffix in [
                ("visual", ""),
                ("smooth_visual", "_smooth"),
                ("dimension_visual", "_dimensions"),
                ("local_visual", "_local"),
            ]:
                if not entry.get(key):
                    continue
                destination = folder / (name + suffix + ".obj")
                temporary = destination.with_suffix(".tmp.obj")
                shutil.copy2(out / entry[key], temporary)
                checks[key] = audit_mesh_file(temporary, require_volume=True)
                temporary.replace(destination)
                checks[key]["file"] = destination.name
                status[key] = str(destination.relative_to(out))
            status.update(
                {
                    "dimensions_m": entry["dimensions_m"],
                    "urdf": entry["urdf"],
                    "exported_mesh_checks": checks,
                    "note": "Geometry needs review. See scene and independent evaluation when available.",
                }
            )
            entry["named_visual"] = status["visual"]
            if entry.get("local_backup"):
                status["local_backup"] = entry["local_backup"]
                write_json(folder / "local-report.json", entry["local_backup"])
            if entry.get("dimension_calibration"):
                status["dimension_calibration"] = entry["dimension_calibration"]
                write_json(
                    folder / "dimension-report.json", entry["dimension_calibration"]
                )
        else:
            status["error"] = entry.get("error", "no admitted model")
        write_json(folder / "status.json", status)
        index.append(status)
    write_json(root / "index.json", index)
    return index


def export_collision_item(item, out, threshold_mm):
    item_started = time.perf_counter()
    asset = Path(item["asset"]).resolve()
    meta = json.loads((asset / "asset.json").read_text())
    if meta.get("status") != "needs_review":
        raise ValueError("source candidate is rejected or failed")
    cavity_required = item.get(
        "cavity_required",
        inferred_scope(item.get("label", item["id"])) == "open_container",
    )
    if cavity_required and not meta.get("cavity", {}).get("detected"):
        raise ValueError(
            "open container requires an observed cavity; solid proxy is forbidden"
        )
    if "outer_convex" in asset.parts:
        raise ValueError("global hulls are excluded from the exterior-shape scene")
    mass = float(item.get("mass_kg", 0.1))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass_kg must be positive")
    mesh = load_mesh(asset / "visual.obj")
    smooth = item.get("smooth", True)
    signature = {"mass_kg": mass, "threshold_mm": threshold_mm, "version": 3}
    if not smooth:
        signature["smooth"] = False
    if meta.get("cavity", {}).get("detected"):
        signature["cavity"] = meta["cavity"]
    fingerprint = hashlib.sha256(
        (asset / "visual.obj").read_bytes()
        + json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()[:20]
    target = out / "assets" / f"{item['id']}-{fingerprint}"
    if not (target / "package.json").exists():
        target.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=target.parent, prefix=".building-"
        ) as temp:
            built = Path(temp) / "package"
            isolated_export(asset, built, mass, threshold_mm / 1000, smooth=smooth)
            built.rename(target)
    package = json.loads((target / "package.json").read_text())
    from .smooth import ensure_smooth_variant

    if smooth:
        package = ensure_smooth_variant(target)
    else:
        audit_mesh_file(target / "visual.obj", require_volume=True)
    mass_source = item.get(
        "mass_source",
        "caller supplied" if "mass_kg" in item else "preview default, not measured",
    )
    package["mass_source"] = mass_source
    write_json(target / "package.json", package)
    return {
        "id": item["id"],
        "label": item.get("label", item["id"]),
        "urdf": str((target / "asset.urdf").relative_to(out)),
        "visual": str((target / "visual.obj").relative_to(out)),
        "smooth_visual": str((target / "visual_smooth.obj").relative_to(out))
        if smooth
        else None,
        "smooth_urdf": str((target / "asset_smooth.urdf").relative_to(out))
        if smooth
        else None,
        "cavity": package.get("cavity", {}),
        "dimensions_m": mesh.extents.tolist(),
        "bounds_m": mesh.bounds.tolist(),
        "visual_simplification": package.get("visual_simplification"),
        "mass_kg": mass,
        "mass_source": mass_source,
        "collision_parts": package["collision_parts"],
        "collision_union_check": package["collision_union_check"],
        "source_asset": str(asset),
        "quality_status": meta["status"],
        "export_seconds": round(time.perf_counter() - item_started, 2),
    }


def assemble(
    items,
    out,
    threshold_mm=3.0,
    render=False,
    prior_errors=None,
    physics_test=False,
    prepared_rows=None,
):
    started = time.perf_counter()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids
    ):
        raise ValueError("unique safe object IDs required")
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError("positive threshold_mm required")
    # Reports and previews always describe this run, never stale successful output.
    for name in (
        "scene.blend",
        "scene.png",
        "scene-smooth.blend",
        "scene-smooth.png",
        "scene.json",
        "scene-package.zip",
        "pybullet-scene.png",
        "physics-check.json",
        "blender-check.json",
        "blender-smooth-check.json",
    ):
        (out / name).unlink(missing_ok=True)
    write_json(
        out / "run-report.json",
        {"status": "running", "stage": "export", "planned": len(items)},
    )
    rows = list(prepared_rows or [])
    errors = list(prior_errors or [])
    if prepared_rows is None:
        event("phase", "export", len(items))
    for index, item in enumerate(items if prepared_rows is None else [], 1):
        event("item", item.get("label", item["id"]), index)
        event("stage", "collision_export")
        item_started = time.perf_counter()
        print(f"[export] {item['id']}", flush=True)
        try:
            rows.append(export_collision_item(item, out, threshold_mm))
        except Exception as error:
            errors.append(
                {
                    "id": item["id"],
                    "label": item.get("label", item["id"]),
                    "stage": "export",
                    "error": str(error),
                }
            )
            print(f"[failed] {item['id']}: {error}", flush=True)
        event(
            "item_done",
            "failed" if errors and errors[-1].get("id") == item["id"] else "done",
        )
    named_models(rows, errors, out)
    scene = {
        "units": "m",
        "up_axis": "z",
        "global_scaling": 1,
        "items": layout(rows),
        "errors": errors,
        "note": "Same metric scale; translations only. Geometry needs visual review. Default masses are previews.",
    }
    write_json(out / "scene.json", scene)
    shutil.copy2(PROJECT / "scripts/view_scene.py", out / "view_scene.py")
    report = {
        "status": "complete_needs_review"
        if rows and not errors
        else "partial"
        if rows
        else "failed",
        "exported": len(rows),
        "retained_unvalidated": sum(
            r.get("quality_status") == "retained_unvalidated" for r in rows
        ),
        "failed": len(errors),
        "errors": errors,
        "scene": "scene.json",
        "physics_test": "not_run",
        "blender": "not_run",
    }
    write_json(out / "run-report.json", report)
    event("phase", "scene", (1 + 2 * int(render) + int(physics_test)) if rows else 0)
    if rows:
        for stage, command in [
            (
                "physics_test",
                [sys.executable, str(out / "view_scene.py"), "--headless"],
            ),
            (
                "blender",
                [
                    "blender",
                    "-b",
                    "--python",
                    str(PROJECT / "scripts/build_scene.py"),
                    "--",
                    "--scene",
                    str(out / "scene.json"),
                ],
            ),
            (
                "blender_smooth",
                [
                    "blender",
                    "-b",
                    "--python",
                    str(PROJECT / "scripts/build_scene.py"),
                    "--",
                    "--scene",
                    str(out / "scene.json"),
                    "--smooth",
                ],
            ),
        ]:
            stage_started = time.perf_counter()
            if stage == "physics_test" and not physics_test:
                report[stage] = "not_requested"
                continue
            if stage == "physics_test" and any(not r.get("urdf") for r in rows):
                report[stage] = "skipped_unvalidated_geometry"
                report["physics_note"] = (
                    "Retained Meshy candidates have visual OBJ only; review shape before collision/physics export."
                )
                event("item_done", report[stage])
                continue
            if stage.startswith("blender") and not render:
                report[stage] = "skipped"
                continue
            if stage == "blender_smooth" and not any(
                r.get("smooth_visual") for r in rows
            ):
                report[stage] = "skipped"
                event("item_done", "skipped")
                continue
            print(f"[scene] {stage}", flush=True)
            event("stage", stage)
            try:
                with (out / f"{stage}.log").open("w") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                report[stage] = "passed"
            except (OSError, subprocess.CalledProcessError) as error:
                report[stage] = "failed"
                report["status"] = "partial"
                report["errors"].append({"stage": stage, "error": str(error)})
            report.setdefault("stage_seconds", {})[stage] = round(
                time.perf_counter() - stage_started, 2
            )
            write_json(out / "run-report.json", report)
            event("item_done", report[stage])
    report["failed"] = len(report["errors"])
    if rows:
        report["portable_bundle"] = "scene-package.zip"
    report["assembly_seconds_before_zip"] = round(time.perf_counter() - started, 2)
    write_json(out / "run-report.json", report)
    if rows:
        event("stage", "package")
        lines = [
            "# 米制重建场景",
            "",
            f"执行状态：`{report['status']}`。所有模型保持统一米制尺度，地面网格 10 cm。",
            "",
            "若启用 --with-blender，可打开 scene.blend 或 scene-smooth.blend。安装可选 pybullet 后运行 python view_scene.py 查看物理场景，加 --headless 可测试。未启用仿真时，URDF 导出成功不代表已通过物理检查。",
            "",
            "**质量未实拍认证。未测量的质量、均匀密度惯量及默认摩擦仅用于预览。** 参见 `scene.json`、`package.json` 与运行检查报告。",
            "",
            "按视频原名整理的 OBJ 位于 `models/物品名/物品名.obj`；平滑版带 `_smooth` 后缀。失败物品只有 `status.json`，总索引为 `models/index.json`。",
            "retained_unvalidated 的 Meshy 模型保留 OBJ 和平滑版，但不生成碰撞体；本批含此类模型时跳过整体物理检查。",
            "",
            "| 模型 | 外尺寸 mm | 视觉三角面 | 凸碰撞块 | 碰撞/原重建体积 IoU |",
            "|---|---|---:|---:|---:|",
        ]
        for item in rows:
            size = " × ".join(f"{v * 1000:.1f}" for v in item["dimensions_m"])
            faces = (item.get("visual_simplification") or {}).get("export_faces", "—")
            iou = item.get("collision_union_check", {}).get("iou")
            iou_text = f"{iou:.1%}" if iou is not None else "未生成"
            lines.append(
                f"| {item['id']} | {size} | {faces} | {item.get('collision_parts', '未生成')} | {iou_text} |"
            )
        (out / "README.md").write_text("\n".join(lines) + "\n")
        files = [
            out / name
            for name in (
                "README.md",
                "scene.json",
                "view_scene.py",
                "scene.blend",
                "scene.png",
                "scene-smooth.blend",
                "scene-smooth.png",
                "blender-smooth-check.json",
                "pybullet-scene.png",
                "physics-check.json",
                "blender-check.json",
                "run-report.json",
            )
            if (out / name).is_file()
        ]
        for item in rows:
            files.extend(
                p for p in (out / Path(item["visual"]).parent).rglob("*") if p.is_file()
            )
        files.extend(p for p in (out / "models").rglob("*") if p.is_file())
        with zipfile.ZipFile(
            out / "scene-package.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in files:
                archive.write(path, path.relative_to(out))
        event("item_done")
    return report


def run_automatic(
    source,
    board,
    out,
    settings_path=None,
    quality="standard",
    render=False,
    obj_only=True,
    progress_mode=None,
    verbose=False,
    backend="local",
    key_file=None,
    physics_test=False,
    objects=None,
    meshy_frontend="hybrid",
    meshy_smooth=False,
):
    started = time.perf_counter()
    if obj_only and physics_test:
        raise ValueError("PyBullet test requires collision/URDF export")
    source, out = source.resolve(), out.resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError("output must be outside input folder")
    settings = json.loads(settings_path.read_text()) if settings_path else {}
    if backend not in {"local", "meshy"}:
        raise ValueError("backend must be local or meshy")
    if meshy_frontend not in {"hybrid", "fast", "full"}:
        raise ValueError("meshy_frontend must be hybrid, fast or full")
    base = settings_path.resolve().parent if settings_path else Path.cwd()
    manifest = discover(source, board, quality, "outer_shape")
    labels = {}
    backends = {}
    overrides = settings.get("objects", {})
    actual_labels = {Path(item["input"]).name for item in manifest["objects"]}
    if set(overrides) - actual_labels:
        raise ValueError(
            "settings names do not match inputs: "
            + ", ".join(sorted(set(overrides) - actual_labels))
        )
    discovered_count = len(manifest["objects"])
    manifest["objects"] = select_objects(manifest["objects"], objects)
    cache_out = out
    if objects is not None:
        ids = sorted(item["id"] for item in manifest["objects"])
        selection_id = (
            ids[0]
            if len(ids) == 1
            else "batch-" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
        )
        out = out / "selected" / selection_id
    for item in manifest["objects"]:
        label = Path(item["input"]).name
        labels[item["id"]] = label
        override = overrides.get(label, {})
        selected_backend = override.get("backend", backend)
        if selected_backend not in {"local", "meshy"}:
            raise ValueError(f"unknown backend for {label}: {selected_backend}")
        backends[item["id"]] = selected_backend
        if selected_backend == "meshy" and meshy_frontend in {"hybrid", "fast"}:
            item["meshy_frontend"] = meshy_frontend
        if "board" in override:
            item["board"] = str((base / override["board"]).resolve())
        if "geometry" in override:
            if override["geometry"] not in ("outer_shape", "open_container"):
                raise ValueError("geometry must be outer_shape or open_container")
            item["scope"] = override["geometry"]
        if not Path(item["board"]).is_file():
            raise ValueError(f"board definition missing for {label}: {item['board']}")
        mass = float(override.get("mass_kg", settings.get("default_mass_kg", 0.1)))
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"mass_kg must be positive for {label}")
    threshold = float(settings.get("threshold_mm", 3.0))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("positive threshold_mm required")
    out.mkdir(parents=True, exist_ok=True)
    # One output run at a time; subprocesses do not inherit this descriptor.
    import fcntl

    with (cache_out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another run is already using this output directory")
        from .meshy import preflight_credentials

        if meshy_frontend in {"hybrid", "fast"}:
            from .meshy_frontend import reuse_legacy_generation

            for item in manifest["objects"]:
                if backends[item["id"]] == "meshy":
                    reuse_legacy_generation(cache_out, item)

        preflight_credentials(
            [
                cache_out
                / ("meshy-fast" if meshy_frontend in {"hybrid", "fast"} else "meshy")
                / name
                / "provider"
                for name, branch in backends.items()
                if branch == "meshy"
            ],
            key_file,
        )
        with (
            ProgressSession(out, progress_mode, verbose)
            if progress_mode
            else nullcontext()
        ):
            write_json(out / "input-manifest.json", manifest)
            write_json(
                out / "run-report.json",
                {
                    "status": "running",
                    "stage": "reconstruction",
                    "planned": len(manifest["objects"]),
                    "mode": "obj_only"
                    if obj_only
                    else "pybullet"
                    if physics_test
                    else "collision",
                    "backend": backend,
                    "object_backends": backends,
                },
            )
            existing_cache = [
                item["id"]
                for item in manifest["objects"]
                if (
                    cache_out
                    / (
                        "reconstruction-meshy-fast"
                        if item.get("meshy_frontend") in {"hybrid", "fast"}
                        else "reconstruction"
                    )
                    / item["id"]
                    / "reconstruction.json"
                ).exists()
            ]
            print(
                "[scan] reconstructing each object; successful stages are cached",
                flush=True,
            )
            items, rows, errors = [], [], []
            meshy_cases = {}
            name_context = [
                {"id": item["id"], "label": labels[item["id"]]}
                for item in manifest["objects"]
            ]
            named_models(
                [],
                [
                    dict(item, quality_status="pending", error="waiting for this run")
                    for item in name_context
                ],
                out,
            )
            completed = 0

            def deliver(result):
                nonlocal completed
                name = result["id"]
                label = labels[name]
                selected_backend = backends[name]
                stage = "reconstruction"
                status = "failed"
                try:
                    if result["status"] in ("failed", "rejected"):
                        raise ValueError(result.get("error", "reconstruction rejected"))
                    asset = None
                    if selected_backend == "meshy":
                        from .meshy_workflow import prepare_candidate

                        stage = "meshy"
                        asset, meshy_cases[name] = prepare_candidate(
                            cache_out
                            / (
                                "reconstruction-meshy-fast"
                                if meshy_frontend in {"hybrid", "fast"}
                                else "reconstruction"
                            )
                            / name,
                            cache_out
                            / (
                                "meshy-fast"
                                if meshy_frontend in {"hybrid", "fast"}
                                else "meshy"
                            )
                            / name,
                            key_file,
                        )
                    else:
                        asset = closed_candidate(cache_out / "reconstruction" / name)
                    override = settings.get("objects", {}).get(label, {})
                    item = {
                        "id": name,
                        "label": label,
                        "asset": str(asset),
                        "smooth": selected_backend != "meshy" or meshy_smooth,
                        "retain_unvalidated_meshy": meshy_cases.get(name, {}).get(
                            "retained_unvalidated", False
                        ),
                        "mass_kg": override.get(
                            "mass_kg", settings.get("default_mass_kg", 0.1)
                        ),
                        "mass_source": "caller supplied"
                        if "mass_kg" in override
                        else "batch default, not individually measured",
                    }
                    stage = "export"
                    event("stage", "mesh_export" if obj_only else "collision_export")
                    if obj_only or item["retain_unvalidated_meshy"]:
                        from .visual_batch import export_visual_item

                        row = export_visual_item(item, out)
                    else:
                        row = export_collision_item(item, out, threshold)
                    if selected_backend == "meshy":
                        from .meshy_dimensions import attach_dimension_variant

                        attach_dimension_variant(row, meshy_cases[name], out)
                        from .local_backup import attach_local_backup

                        try:
                            attach_local_backup(row, meshy_cases[name], out)
                        except Exception as error:  # noqa: BLE001 -- local backup must not discard the paid mesh
                            meshy_cases[name]["local_backup_error"] = str(error)
                    named_models(
                        [row], [], out, incremental=True, name_context=name_context
                    )
                    rows.append(row)
                    items.append(item)
                    status = row.get("quality_status", "needs_review")
                    print(f"[ready] {label}: {out / row['named_visual']}", flush=True)
                except Exception as error:
                    failure = {
                        "id": name,
                        "label": label,
                        "stage": stage,
                        "error": str(error),
                    }
                    errors.append(failure)
                    named_models(
                        [], [failure], out, incremental=True, name_context=name_context
                    )
                    print(f"[failed] {label}: {error}", flush=True)
                completed += 1
                write_json(
                    out / "run-report.json",
                    {
                        "status": "running",
                        "stage": "per_object_delivery",
                        "completed": completed,
                        "planned": len(manifest["objects"]),
                        "exported": len(rows),
                        "failed": len(errors),
                        "errors": errors,
                        "output_dir": str(out),
                        "object_backends": backends,
                        "ready_objects": [r["label"] for r in rows],
                        "note": "Each ready model is already published; shared scene/package follows the batch.",
                    },
                )
                return status

            try:
                results = run_batch(
                    out / "input-manifest.json",
                    cache_out / "reconstruction",
                    on_result=deliver,
                )
            except BaseException as error:
                write_json(
                    out / "run-report.json",
                    {
                        "status": "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed",
                        "completed": completed,
                        "planned": len(manifest["objects"]),
                        "exported": len(rows),
                        "failed": len(errors),
                        "errors": errors,
                        "ready_objects": [r["label"] for r in rows],
                        "output_dir": str(out),
                        "error": str(error),
                        "note": "Already published models remain available; resume the same command.",
                    },
                )
                raise
            for result in results:
                if result.get("delivery_error"):
                    errors.append(
                        {
                            "id": result["id"],
                            "label": labels[result["id"]],
                            "stage": "publish",
                            "error": result["delivery_error"],
                        }
                    )
            if obj_only:
                from .visual_batch import assemble_visual

                report = assemble_visual(items, out, render, errors, prepared_rows=rows)
            else:
                report = assemble(
                    items,
                    out,
                    settings.get("threshold_mm", 3.0),
                    render,
                    errors,
                    physics_test=physics_test,
                    prepared_rows=rows,
                )
            report["total_run_seconds"] = round(time.perf_counter() - started, 2)
            report["input_count"] = len(manifest["objects"])
            report["discovered_input_count"] = discovered_count
            report["selected_objects"] = [
                labels[item["id"]] for item in manifest["objects"]
            ]
            report["selection_requested"] = objects
            report["output_dir"] = str(out)
            report["quality"] = quality
            report["backend"] = backend
            report["object_backends"] = backends
            report["meshy_cases"] = meshy_cases
            report["meshy_frontend"] = meshy_frontend
            report["meshy_smoothing"] = meshy_smooth
            report["mesh_branch_note"] = (
                "Meshy hybrid completes local dimension measurement from shared A4 camera/depth caches; fast uses sparse/photo fitting without actively building local measurement. Paid generation and original input masks are preserved across hybrid/fast switches. Full retains legacy reconstruction."
            )
            report["cache_present_before_run_ids"] = existing_cache
            report["timing_note"] = (
                "total_run_seconds measures this invocation; existing reconstruction caches may be reused. Do not treat a resumed run as cold-start timing."
            )
            write_json(out / "run-report.json", report)
            # Keep the portable report identical to the on-disk report.
            archive = out / "scene-package.zip"
            if archive.exists():
                temporary = archive.with_suffix(".tmp.zip")
                with (
                    zipfile.ZipFile(archive) as old,
                    zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as new,
                ):
                    for entry in old.infolist():
                        if entry.filename != "run-report.json":
                            new.writestr(entry, old.read(entry.filename))
                    new.write(out / "run-report.json", "run-report.json")
                temporary.replace(archive)
            event("finish", report["status"])
            return report
