"""Optional paid Meshy branch sharing the local metric frontend and export gates."""

import fcntl
import hashlib
import json
import time
from pathlib import Path

from .automation import closed_candidate
from .assets import audit_mesh_file, load_mesh
from .meshy import generate, write_json, preflight_credentials
from .meshy_views import prepare_meshy_views
from .meshy_fit import register_meshy, refine_photo_pose
from .progress import event, ProgressSession
from .visual_batch import assemble_visual


def reference_signature(result, reference):
    """Hash geometry/observations, not volatile timings or progress metadata."""
    paths = [
        reference / "visual.obj",
        reference / "asset.json",
        result / "work/metric.json",
    ]
    for folder in ("sparse", "images", "masks"):
        paths.extend(
            p
            for p in (result / "work" / folder).iterdir()
            if p.is_file()
            and p.suffix.lower() in {".bin", ".txt", ".png", ".jpg", ".jpeg"}
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(result)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare_candidate(result, out, key_file=None):
    """Generate/register one exterior candidate. Never silently fall back to local."""
    result, out = Path(result).resolve(), Path(out).resolve()
    if out == result or out.is_relative_to(result) or result.is_relative_to(out):
        raise ValueError("keep Meshy output separate from the measured reference")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("this Meshy output already has an active run")
        started = time.perf_counter()
        report = {
            "status": "running",
            "backend": "meshy",
            "stage_seconds": {},
            "reference": str(result),
            "cached_stages": [],
            "scale_note": "A4 metric RGB reference plus calibrated photos; no axis stretching; reference is not independent truth",
        }

        def stage(name, cached=False):
            report["stage"] = name
            if cached:
                report["cached_stages"].append(name)
            write_json(out / "case-report.json", report)
            event("stage", name, cached=cached)

        def timed(name, cached, action):
            stage(name, cached)
            before = time.perf_counter()
            value = action()
            report["stage_seconds"][name] = round(time.perf_counter() - before, 3)
            return value

        try:
            stage("meshy_reference")
            measurement = result / "hybrid-measurement.json"
            if measurement.exists():
                report["hybrid_measurement"] = json.loads(measurement.read_text())
            selection = json.loads((result / "selection.json").read_text())
            state_file = result / "reconstruction.json"
            frontend_state = (
                json.loads(state_file.read_text()) if state_file.exists() else {}
            )
            fast_frontend = frontend_state.get("frontend") == "meshy_fast"
            reference = (
                result / "reference" if fast_frontend else closed_candidate(result)
            )
            report["frontend"] = "fast_sparse" if fast_frontend else "full"
            if (
                fast_frontend
                and frontend_state.get("frontend_prompt_source") == "dense"
            ):
                report["frontend"] = "fast_dense_prompts"
                report["frontend_fallback"] = frontend_state.get("frontend_fallback")
            meta = json.loads((reference / "asset.json").read_text())
            if selection.get("scope") != "outer_shape" or meta.get("cavity", {}).get(
                "detected"
            ):
                raise ValueError(
                    "Meshy branch currently validates exterior shape only; use local for open_container/cavities"
                )
            preflight_credentials([out / "provider"], key_file)
            signature = reference_signature(result, reference)
            state_path = out / "input-state.json"
            if (
                state_path.exists()
                and json.loads(state_path.read_text())["signature"] != signature
            ):
                raise ValueError(
                    "metric reference changed; choose a new output directory"
                )
            cached_inputs = (out / "inputs/prepared.json").exists()
            stage("meshy_views", cached_inputs)
            if cached_inputs:
                prepared = json.loads((out / "inputs/prepared.json").read_text())
                reused = out / "generation-reuse.json"
                validation_root = result
                if reused.exists():
                    report["generation_reuse"] = json.loads(reused.read_text())
                    # Submitted crops retain the old masks; validate against their
                    # recorded reconstruction, while input-state hashes fast observations.
                    validation_root = Path(prepared["source_reconstruction"])
                for row in prepared["views"]:
                    for path, field in [
                        (
                            validation_root / "work/images" / row["image"],
                            "source_sha256",
                        ),
                        (
                            validation_root / "work/masks" / (row["image"] + ".png"),
                            "mask_sha256",
                        ),
                    ]:
                        if hashlib.sha256(path.read_bytes()).hexdigest() != row[field]:
                            raise ValueError(
                                "prepared source images changed; choose a new output directory"
                            )
            else:
                # Preserve interrupted work for diagnosis; provider tasks are never removed.
                if (out / "inputs").exists():
                    (out / "inputs").rename(out / f"inputs-incomplete-{time.time_ns()}")
                prepared = prepare_meshy_views(result, out / "inputs")
            write_json(state_path, {"signature": signature, "reference": str(result)})
            task_file = out / "provider/meshy-task.json"
            task = json.loads(task_file.read_text()) if task_file.exists() else {}
            downloaded = bool(
                task.get("downloaded") and (out / "provider/source.glb").is_file()
            )
            report["provider_cache_present_before"] = downloaded
            report["provider_task_present_before"] = bool(task.get("task_id"))
            generated = timed(
                "meshy_generate",
                downloaded,
                lambda: generate(
                    [out / "inputs" / v["file"] for v in prepared["views"]],
                    out / "provider",
                    key_file=key_file,
                ),
            )
            # Keep a usable raw OBJ even if metric fitting or later checks fail.
            # Provider coordinates have no measured physical scale.
            if Path(generated).is_file():
                retained = out / "retained"
                retained.mkdir(exist_ok=True)
                report["retained_raw_glb"] = str(generated)
                try:
                    load_mesh(generated).export(retained / "raw.obj")
                    report["retained_raw_obj"] = str(retained / "raw.obj")
                except Exception as error:
                    report["raw_obj_error"] = str(error)
                write_json(
                    retained / "status.json",
                    {
                        "status": "retained_unvalidated",
                        "units": "unknown_provider_scale",
                        "up_axis": "y",
                        "raw_glb": str(generated),
                        "note": "Raw provider geometry; NOT calibrated metres. See metric/photo-aligned for measured candidates.",
                    },
                )
                (retained / "README.md").write_text(
                    "# Meshy 原始备用模型\n\nraw.obj 是服务端原始几何，Y 向上，"
                    "尚未校准真实尺寸，不能按米直接用于仿真。原始 GLB 保存在 ../provider/source.glb。\n"
                )
            for folder, name, action in [
                (
                    "metric",
                    "meshy_metric",
                    lambda: register_meshy(
                        generated, result, out / "metric", out / "inputs/prepared.json"
                    ),
                ),
                (
                    "photo-aligned",
                    "meshy_photo",
                    lambda: refine_photo_pose(
                        out / "metric", result / "work", out / "photo-aligned"
                    ),
                ),
            ]:
                target = out / folder
                cached = all(
                    (target / p).is_file()
                    for p in ("asset.json", "quality.json", "visual.obj")
                )
                if cached and fast_frontend and folder == "photo-aligned":
                    cached = (
                        json.loads((target / "asset.json").read_text())
                        .get("provenance", {})
                        .get("photo_refinement_profile")
                        == "sparse-wide-v2"
                    )
                if cached and fast_frontend and folder == "metric":
                    cached = (
                        json.loads((target / "asset.json").read_text())
                        .get("provenance", {})
                        .get("metric_registration_profile")
                        == "sparse-fixed-v1"
                    )
                if target.exists() and not cached:
                    target.rename(out / f"{folder}-incomplete-{time.time_ns()}")
                timed(name, cached, (lambda: None) if cached else action)
            candidate_folders = ["metric", "photo-aligned"]
            candidate_work = result / "work"
            # Upgrading a cached fast job must actually refit against its new
            # dense measurement. The old sparse fit can be far outside the
            # small axis-correction range; preserve it as a separate candidate.
            if report.get("hybrid_measurement", {}).get("status") == "ready":
                from .meshy_dimensions import matching_local_reference

                local_reference = matching_local_reference(result / "work")
                if local_reference:
                    local_result = local_reference.parent
                    suffix = reference_signature(local_result, local_reference)[:16]
                    metric_name, photo_name = (
                        f"hybrid-metric-{suffix}",
                        f"hybrid-photo-{suffix}",
                    )
                    try:
                        for folder, action in [
                            (
                                metric_name,
                                lambda: register_meshy(
                                    generated,
                                    local_result,
                                    out / metric_name,
                                    out / "inputs/prepared.json",
                                    reference_asset=local_reference,
                                ),
                            ),
                            (
                                photo_name,
                                lambda: refine_photo_pose(
                                    out / metric_name,
                                    local_result / "work",
                                    out / photo_name,
                                ),
                            ),
                        ]:
                            target = out / folder
                            cached = all(
                                (target / name).is_file()
                                for name in ("asset.json", "quality.json", "visual.obj")
                            )
                            if target.exists() and not cached:
                                target.rename(
                                    out / f"{folder}-incomplete-{time.time_ns()}"
                                )
                            timed(
                                "meshy_metric"
                                if folder == metric_name
                                else "meshy_photo",
                                cached,
                                (lambda: None) if cached else action,
                            )
                        # Sparse prompts may have segmented only a reflective
                        # label/printed patch. Evaluate all new candidates using
                        # the independently recomputed dense masks. Submitted
                        # provider crops and the original masks remain immutable.
                        candidate_folders = [metric_name, photo_name]
                        candidate_work = local_result / "work"
                        report["hybrid_registration_reference"] = str(local_reference)
                        report["validation_work"] = str(candidate_work)
                    except Exception as error:  # noqa: BLE001 -- retain the existing paid mesh if optional refit fails
                        report["hybrid_registration_error"] = str(error)
            stage("meshy_check")
            candidates = []
            retained_candidates = []
            audit_errors = []
            for folder in candidate_folders:
                target = out / folder
                meta = json.loads((target / "asset.json").read_text())
                quality = json.loads((target / "quality.json").read_text())
                try:
                    audit_mesh_file(target / "visual.obj", require_volume=True)
                except ValueError as error:
                    audit_errors.append(str(error))
                    continue
                retained_candidates.append((quality["views"]["median_iou"], target))
                if meta["status"] == "needs_review":
                    candidates.append((quality["views"]["median_iou"], target))
            if not retained_candidates:
                raise ValueError(
                    "Meshy candidates are not a closed volume; raw generation retained. "
                    + "; ".join(audit_errors)
                )
            selected = max(candidates or retained_candidates, key=lambda row: row[0])[1]
            selected_meta = json.loads((selected / "asset.json").read_text())
            retained_unvalidated = not bool(candidates)
            # A separately delivered variant must never prevent retaining generation.
            stage("meshy_dimensions")
            try:
                from .meshy_dimensions import (
                    matching_local_reference,
                    refine_dimensions,
                )

                dimension = refine_dimensions(
                    selected,
                    candidate_work,
                    out / "dimension-aligned",
                    out / "inputs/prepared.json",
                    local_reference=matching_local_reference(result / "work"),
                )
                report["dimension_calibration"] = dimension
                if dimension["accepted"]:
                    report["dimension_asset"] = str(out / "dimension-aligned")
            except Exception as error:
                report["dimension_calibration"] = {
                    "accepted": False,
                    "error": str(error),
                }
            stage("meshy_check")
            report.update(
                status="retained_unvalidated"
                if retained_unvalidated
                else "needs_review",
                retained_unvalidated=retained_unvalidated,
                quality_check=selected_meta.get("quality", {}),
                warning="Shape checks failed; retained for manual review, not quality-approved."
                if retained_unvalidated
                else None,
                selected_asset=str(selected),
                label=Path(prepared["source_video"]).name,
                current_run_seconds=round(time.perf_counter() - started, 2),
            )
            write_json(out / "case-report.json", report)
            return selected, report
        except BaseException as error:
            report.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                error=str(error),
                current_run_seconds=round(time.perf_counter() - started, 2),
            )
            write_json(out / "case-report.json", report)
            raise


def run_case(
    result,
    out,
    key_file=None,
    render=False,
    progress_mode="auto",
    verbose=False,
    smooth=False,
):
    out = Path(out).resolve()
    # Outer lock also protects delivery/progress/log files. The inner lock protects provider state.
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".delivery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another Meshy run is using this delivery folder")
        with ProgressSession(out, progress_mode, verbose):
            event("phase", "meshy", 1)
            event("item", Path(result).name, 1)
            try:
                selected, case = prepare_candidate(result, out, key_file)
                event("item_done")
                report = assemble_visual(
                    [
                        {
                            "id": Path(result).name,
                            "label": case["label"],
                            "asset": str(selected),
                            "smooth": smooth,
                            "dimension_case": case,
                            "retain_unvalidated_meshy": case.get(
                                "retained_unvalidated", False
                            ),
                        }
                    ],
                    out / "delivery",
                    render=render,
                )
                report.update(backend="meshy", case=case)
            except Exception as error:
                event("item_done", "failed")
                report = assemble_visual(
                    [],
                    out / "delivery",
                    render=False,
                    prior_errors=[{"id": Path(result).name, "error": str(error)}],
                )
                report.update(backend="meshy", error=str(error))
            write_json(out / "run-report.json", report)
            event("finish", report["status"])
            return report
