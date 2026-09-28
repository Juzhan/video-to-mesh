"""Publish a local backup when Meshy geometry or dimension fitting needs review."""

import hashlib
import json
from pathlib import Path

from .assets import audit_mesh_file, load_mesh
from .mesh_export import light_visual


def attach_local_backup(row, case, out):
    dimension_error = case.get("dimension_calibration", {}).get("error")
    if not case.get("retained_unvalidated") and not dimension_error:
        return
    measurement = case.get("hybrid_measurement", {})
    if measurement.get("status") != "ready":
        return
    source = Path(measurement["reference"])
    meta = json.loads((source / "asset.json").read_text())
    if meta.get("status") != "needs_review":
        raise ValueError("local backup did not pass its reconstruction gates")
    audit_mesh_file(source / "visual.obj", require_volume=True)
    digest = hashlib.sha256((source / "visual.obj").read_bytes()).hexdigest()[:20]
    target = Path(out) / "assets" / f"{row['id']}-local-{digest}"
    target.mkdir(parents=True, exist_ok=True)
    mesh, simplification = light_visual(load_mesh(source / "visual.obj"))
    mesh.export(target / "visual.obj")
    check = audit_mesh_file(target / "visual.obj", require_volume=True)
    info = {
        "status": "needs_review",
        "backend": "local_rgb_reconstruction",
        "backup_trigger": "meshy_geometry_rejected"
        if case.get("retained_unvalidated")
        else "meshy_dimension_fit_error",
        "dimension_fit_error": dimension_error,
        "source_asset": str(source),
        "dimensions_m": mesh.extents.tolist(),
        "mesh_check": check,
        "simplification": simplification,
        "quality_check": meta.get("quality", {}),
        "cavity": meta.get("cavity", {}),
        "note": "Separate measured local candidate. Passing automatic checks does not certify physical accuracy or hidden geometry. Original Meshy model is retained.",
    }
    (target / "local-report.json").write_text(json.dumps(info, indent=2))
    row["local_visual"] = str((target / "visual.obj").relative_to(out))
    row["local_backup"] = info
