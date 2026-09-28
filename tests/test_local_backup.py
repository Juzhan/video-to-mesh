import json

import pytest
import trimesh

from packing_assets.assets import export_asset
from packing_assets.automation import named_models
from packing_assets.local_backup import attach_local_backup


@pytest.mark.parametrize("dimension_error", [None, "local and Meshy dimensions differ by more than 10%"])
def test_local_backup_is_separate_and_published_under_original_name(tmp_path, dimension_error):
    source = tmp_path / "local"
    export_asset(trimesh.creation.box([0.1, 0.08, 0.05]), source, {})
    meta = json.loads((source / "asset.json").read_text())
    meta["status"] = "needs_review"
    (source / "asset.json").write_text(json.dumps(meta))
    out = tmp_path / "output"
    out.mkdir()
    original = out / "original.obj"
    trimesh.creation.box([0.11, 0.09, 0.05]).export(original)
    before = original.read_bytes()
    row = {
        "id": "box",
        "label": "Box.mp4",
        "visual": "original.obj",
        "quality_status": "needs_review" if dimension_error else "retained_unvalidated",
        "dimensions_m": [0.11, 0.09, 0.05],
        "urdf": None,
    }
    case = {
        "retained_unvalidated": not bool(dimension_error),
        "dimension_calibration": {"error": dimension_error} if dimension_error else {},
        "hybrid_measurement": {"status": "ready", "reference": str(source)},
    }
    attach_local_backup(row, case, out)
    named_models([row], [], out)
    assert original.read_bytes() == before
    assert (out / "models/Box/Box_local.obj").is_file()
    status = json.loads((out / "models/Box/status.json").read_text())
    assert status["status"] == row["quality_status"]
    assert status["local_backup"]["backup_trigger"] == (
        "meshy_dimension_fit_error" if dimension_error else "meshy_geometry_rejected"
    )
    assert status["local_backup"]["status"] == "needs_review"
    assert status["local_backup"]["mesh_check"]["closed_volume"]
    row.pop("local_visual")
    row.pop("local_backup")
    named_models([row], [], out)
    assert not (out / "models/Box/Box_local.obj").exists()


def test_local_backup_never_exports_rejected_geometry(tmp_path):
    source = tmp_path / "rejected"
    source.mkdir()
    (source / "asset.json").write_text('{"status":"rejected"}')
    case = {
        "retained_unvalidated": True,
        "hybrid_measurement": {"status": "ready", "reference": str(source)},
    }
    with pytest.raises(ValueError, match="did not pass"):
        attach_local_backup({"id": "x"}, case, tmp_path / "output")


def test_no_backup_for_harmless_dimension_refinement_rejection(tmp_path):
    row = {"id": "x"}
    attach_local_backup(row, {
        "dimension_calibration": {"accepted": False, "reasons": ["no_clear_held_out_improvement"]},
        "hybrid_measurement": {"status": "ready", "reference": "must-not-be-read"},
    }, tmp_path)
    assert row == {"id": "x"}
