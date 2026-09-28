import json
from pathlib import Path
import re

import numpy as np

from .assets import export_asset, load_mesh, rotate_up, scale_height
from .meshy import generate
from .metric import from_colmap
from .progress import event


def run_batch(manifest: Path, out: Path, on_result=None):
    data = json.loads(manifest.read_text())
    base = manifest.resolve().parent
    items = data["objects"]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids
    ):
        raise ValueError("object IDs must be unique, using letters, digits, _ or -")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    event("phase", "objects" if on_result else "reconstruction", len(items))

    def finished(result):
        # Reconstruction is a distinct milestone; only the callback publishes final files.
        if on_result:
            try:
                result["delivery_status"] = on_result(result)
            except Exception as error:
                result.update(delivery_status="failed", delivery_error=str(error))
        temporary = out / "batch.tmp.json"
        temporary.write_text(json.dumps(results, indent=2))
        temporary.replace(out / "batch.json")
        event("item_done", result.get("delivery_status", result["status"]))

    for index, item in enumerate(items, 1):
        event("item", Path(item.get("input", item["id"])).name, index)
        folder = out / item["id"]
        if item.get("meshy_frontend") in {"hybrid", "fast"}:
            folder = out.parent / "reconstruction-meshy-fast" / item["id"]
            folder.parent.mkdir(parents=True, exist_ok=True)
        folder.mkdir(exist_ok=True)
        try:
            backend = item["backend"]
            if backend == "colmap_rgb":
                from .reconstruct import reconstruct

                report = reconstruct(
                    base / item["input"],
                    base / item["board"],
                    folder,
                    item.get("quality", "standard"),
                    item.get("collision", False),
                    item.get("segmentation", "sam2"),
                    item.get("scope", "outer_shape"),
                    **(
                        {
                            "frontend_only": True,
                            **(
                                {"measure_dimensions": True}
                                if item.get("meshy_frontend") == "hybrid"
                                else {}
                            ),
                        }
                        if item.get("meshy_frontend") in {"hybrid", "fast"}
                        else {}
                    ),
                )
                if item.get("meshy_frontend") == "hybrid":
                    from .hybrid_measurement import ensure_measurement

                    ensure_measurement(
                        folder, base / item["input"], base / item["board"]
                    )
                results.append(
                    {
                        "id": item["id"],
                        "status": report.get("selection", {}).get(
                            "status", report["status"]
                        ),
                        "surface_status": report["status"],
                        "selection": report.get("selection"),
                        "packing_proxy": report.get("packing_proxy"),
                        "asset": str(
                            (
                                folder
                                / (
                                    "reference/asset.json"
                                    if item.get("meshy_frontend") in {"hybrid", "fast"}
                                    else "asset.json"
                                )
                            ).resolve()
                        ),
                    }
                )
                (folder / "failure.json").unlink(missing_ok=True)
                finished(results[-1])
                continue
            elif backend == "meshy":
                # Validate scale BEFORE creating a billable task.
                height = item.get("height_m")
                if (
                    not isinstance(height, (float, int))
                    or not np.isfinite(height)
                    or height <= 0
                ):
                    raise ValueError(
                        "Meshy route requires a measured height_m; board scaling is not transferable automatically"
                    )
                source = generate(
                    [base / p for p in item["images"]],
                    folder / "provider",
                    item.get("model", "meshy-7.1"),
                    geometry_resolution=item.get("geometry_resolution", "standard"),
                )
                mesh = load_mesh(source)
                rotate_up(mesh, item.get("up_axis", "y"))
                scale = scale_height(mesh, height)
                provenance = {
                    "backend": backend,
                    "scale_source": "measured_height",
                    "height_m": height,
                    "height_source": item.get("height_source", "user_measurement"),
                    "uniform_scale": scale,
                    "note": "Matching declared height is not an independent accuracy test.",
                }
            elif backend == "colmap_mesh":
                board = json.loads((base / item["board"]).read_text())
                transform, scale_report = from_colmap(
                    base / item["colmap_model"], base / item["images_dir"], board
                )
                mesh = load_mesh(base / item["mesh"])
                mesh.apply_transform(transform)
                provenance = {
                    "backend": backend,
                    "scale_source": "triangulated_board",
                    "calibration": scale_report,
                }
                # The mesh must be from the SAME COLMAP reconstruction and already object-isolated.
            elif backend == "known_mesh":
                mesh = load_mesh(base / item["mesh"])
                factor = float(item["meters_per_unit"])
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("meters_per_unit must be positive")
                rotate_up(mesh, item.get("up_axis", "z"))
                mesh.apply_scale(factor)
                provenance = {
                    "backend": backend,
                    "scale_source": "declared_units",
                    "meters_per_unit": factor,
                }
            else:
                raise ValueError(f"unsupported backend: {backend}")
            report = export_asset(
                mesh,
                folder,
                provenance,
                item.get("expected_extents_m"),
                item.get("collision", False),
            )
            (folder / "failure.json").unlink(missing_ok=True)
            results.append(
                {
                    "id": item["id"],
                    "status": report["status"],
                    "asset": str((folder / "asset.json").resolve()),
                }
            )
        except Exception as error:
            failure = {"id": item["id"], "status": "failed", "error": str(error)}
            (folder / "failure.json").write_text(json.dumps(failure, indent=2))
            prior_asset = folder / "asset.json"
            if prior_asset.exists():
                prior = json.loads(prior_asset.read_text())
                prior["status"] = "failed_current_run"
                prior["latest_error"] = str(error)
                prior_asset.write_text(json.dumps(prior, indent=2))
            results.append(failure)
        finished(results[-1])
    (out / "batch.json").write_text(json.dumps(results, indent=2))
    return results
