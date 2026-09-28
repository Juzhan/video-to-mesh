"""Low-poly OBJ batch delivery without expensive physics collision decomposition."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

from .assets import load_mesh
from .automation import PROJECT, write_json, layout, named_models
from .mesh_export import light_visual
from .smooth import ensure_smooth_variant
from .progress import event


def export_visual_item(item, out):
    item_started = time.perf_counter()
    source = Path(item["asset"])
    meta = json.loads((source / "asset.json").read_text())
    retained = bool(
        item.get("retain_unvalidated_meshy")
        and meta.get("provenance", {}).get("backend") == "meshy_multi_image"
        and meta.get("status") == "rejected"
    )
    if meta.get("status") != "needs_review" and not retained:
        raise ValueError("source geometry rejected")
    quality_status = "retained_unvalidated" if retained else "needs_review"
    smooth = item.get("smooth", True)
    mesh = load_mesh(source / "visual.obj")
    if not mesh.is_volume:
        raise ValueError("source is not a closed material volume")
    fingerprint = hashlib.sha256((source / "visual.obj").read_bytes()).hexdigest()[:20]
    target = (
        out
        / "assets"
        / f"{item['id']}-visual-{fingerprint}{'' if smooth else '-plain'}"
    )
    target.mkdir(parents=True, exist_ok=True)
    if not (target / "package.json").exists():
        visual, simplification = light_visual(mesh)
        visual.export(target / "visual.obj")
        write_json(
            target / "package.json",
            {
                "status": quality_status,
                "units": "m",
                "up_axis": "z",
                "cavity": meta.get("cavity", {}),
                "visual_simplification": simplification,
                "collision_generated": False,
                "source_asset": str(source),
            },
        )
    # Geometry cache and review policy are separate: do not inherit stale labels.
    package = json.loads((target / "package.json").read_text())
    package.update(status=quality_status, quality_check=meta.get("quality", {}))
    write_json(target / "package.json", package)
    if smooth:
        package = ensure_smooth_variant(target)
    row = {
        "id": item["id"],
        "label": item.get("label", item["id"]),
        "visual": str((target / "visual.obj").relative_to(out)),
        "smooth_visual": str((target / "visual_smooth.obj").relative_to(out))
        if smooth
        else None,
        "urdf": None,
        "dimensions_m": mesh.extents.tolist(),
        "bounds_m": mesh.bounds.tolist(),
        "cavity": package.get("cavity", {}),
        "visual_simplification": package["visual_simplification"],
        "quality_status": quality_status,
        "quality_check": meta.get("quality", {}),
        "source_asset": str(source),
        "mass_source": "visual-only output; physics not generated",
        "export_seconds": round(time.perf_counter() - item_started, 2),
    }
    if item.get("dimension_case"):
        from .meshy_dimensions import attach_dimension_variant

        attach_dimension_variant(row, item["dimension_case"], out)
    return row


def assemble_visual(items, out, render=False, prior_errors=None, prepared_rows=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = list(prepared_rows or [])
    errors = list(prior_errors or [])
    for filename in (
        "scene.blend",
        "scene.png",
        "scene-smooth.blend",
        "scene-smooth.png",
        "scene-package.zip",
        "physics-check.json",
        "pybullet-scene.png",
        "view_scene.py",
        "blender-check.json",
        "blender-smooth-check.json",
    ):
        (out / filename).unlink(missing_ok=True)
    write_json(
        out / "run-report.json",
        {"status": "running", "stage": "visual_export", "planned": len(items)},
    )
    if prepared_rows is None:
        event("phase", "export", len(items))
    for index, item in enumerate(items if prepared_rows is None else [], 1):
        event("item", item.get("label", item["id"]), index)
        event("stage", "mesh_export")
        item_started = time.perf_counter()
        try:
            rows.append(export_visual_item(item, out))
        except Exception as error:
            errors.append(
                {
                    "id": item["id"],
                    "label": item.get("label", item["id"]),
                    "stage": "visual_export",
                    "error": str(error),
                }
            )
        event(
            "item_done",
            "failed" if errors and errors[-1].get("id") == item["id"] else "done",
        )
    named_models(rows, errors, out)
    write_json(
        out / "scene.json",
        {
            "items": layout(rows),
            "units": "m",
            "up_axis": "z",
            "global_scaling": 1,
            "errors": errors,
            "note": "Visual-only batch. No collision meshes or physics execution; no per-object rescaling.",
        },
    )
    report = {
        "status": "complete_needs_review"
        if rows and not errors
        else "partial"
        if rows
        else "failed",
        "mode": "obj_only",
        "exported": len(rows),
        "retained_unvalidated": sum(
            r.get("quality_status") == "retained_unvalidated" for r in rows
        ),
        "errors": errors,
        "scene": "scene.json",
        "physics_test": "not_requested",
        "blender": "skipped",
        "blender_smooth": "skipped",
        "stage_seconds": {},
    }
    has_smooth = any(r.get("smooth_visual") for r in rows)
    event("phase", "scene", (2 + int(has_smooth)) if rows and render else 1)
    if rows and render:
        for stage, suffix in [("blender", []), ("blender_smooth", ["--smooth"])]:
            if stage == "blender_smooth" and not has_smooth:
                continue
            stage_started = time.perf_counter()
            event("stage", stage)
            try:
                with (out / f"{stage}.log").open("w") as log:
                    subprocess.run(
                        [
                            "blender",
                            "-b",
                            "--threads",
                            "8",
                            "--python",
                            str(PROJECT / "scripts/build_scene.py"),
                            "--",
                            "--scene",
                            str(out / "scene.json"),
                            *suffix,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                report[stage] = "passed"
            except (OSError, subprocess.CalledProcessError) as error:
                report[stage] = "failed"
                report["status"] = "partial"
                errors.append({"stage": stage, "error": str(error)})
            report["stage_seconds"][stage] = round(
                time.perf_counter() - stage_started, 2
            )
            event("item_done", report[stage])
    report["failed"] = len(errors)
    report["assembly_seconds_before_zip"] = round(time.perf_counter() - started, 2)
    event("stage", "package")
    report["portable_bundle"] = "scene-package.zip"
    write_json(out / "run-report.json", report)
    lines = [
        "# 视频批量 OBJ 结果",
        "",
        f"状态：{report['status']}。单位米，Z 向上，同尺度摆放。",
        "",
        "模型位于 `models/原视频名/原视频名.obj`。local 默认附带 `_smooth.obj`；Meshy 仅启用 --with-smooth 时附带平滑版。逐件状态见 `models/index.json`。",
        "retained_unvalidated 表示 Meshy 形状检查未通过但模型已保留；不是质量合格。原始检查结果保存在 status.json 的 quality_check。",
        "本次生成几何与场景布局数据，未生成碰撞体，也未运行物理仿真。仅启用 --with-blender 时生成展示场景和图片；几何仍需检查，不能把导出成功当作真实精度保证。",
        "",
        "需要碰撞块/URDF 或仿真检查时，安装对应可选依赖，给原批处理命令添加 --with-collision 或 --pybullet，保留原有 backend、objects 和质量选项以复用正确缓存。",
        "",
        "| 输入 | 尺寸 mm | 面数 |",
        "|---|---|---:|",
    ]
    for row in rows:
        dims = " × ".join(f"{v * 1000:.1f}" for v in row["dimensions_m"])
        lines.append(
            f"| {row['label']} | {dims} | {row['visual_simplification']['export_faces']} |"
        )
    (out / "README.md").write_text("\n".join(lines) + "\n")
    files = [
        out / n
        for n in [
            "README.md",
            "scene.json",
            "run-report.json",
            "scene.blend",
            "scene-smooth.blend",
            "scene.png",
            "scene-smooth.png",
            "blender-check.json",
            "blender-smooth-check.json",
        ]
        if (out / n).is_file()
    ]
    files.extend(p for p in (out / "models").rglob("*") if p.is_file())
    for row in rows:
        files.extend(p for p in (out / row["visual"]).parent.rglob("*") if p.is_file())
        for key in ("dimension_visual", "local_visual"):
            if row.get(key):
                files.extend(
                    p for p in (out / row[key]).parent.rglob("*") if p.is_file()
                )
    with zipfile.ZipFile(
        out / "scene-package.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(out))
    event("item_done")
    return report
