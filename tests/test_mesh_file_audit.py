import json
import pytest
import trimesh
from packing_assets.assets import audit_mesh_file, export_asset
from packing_assets.smooth import ensure_smooth_variant
from packing_assets.visual_batch import assemble_visual


def missing_bottom():
    mesh = trimesh.creation.box([0.1, 0.2, 0.03])
    mesh.update_faces(mesh.face_normals[:, 2] > -0.9)
    mesh.remove_unreferenced_vertices()
    return mesh


def test_serialized_open_bottom_is_rejected_without_repair(tmp_path):
    path = tmp_path / "open.obj"
    missing_bottom().export(path)
    original = path.read_bytes()
    report = audit_mesh_file(path)
    assert report["boundary_edges"] == 4 and not report["closed_volume"]
    with pytest.raises(ValueError, match="not a closed volume"):
        audit_mesh_file(path, require_volume=True)
    assert path.read_bytes() == original


def test_cached_smooth_mesh_is_rechecked_even_if_report_claims_success(tmp_path):
    trimesh.creation.box([0.1, 0.2, 0.03]).export(tmp_path / "visual.obj")
    missing_bottom().export(tmp_path / "visual_smooth.obj")
    (tmp_path / "package.json").write_text(
        json.dumps({"visual_smoothing": {"method": "cached"}})
    )
    with pytest.raises(ValueError, match="visual_smooth.obj"):
        ensure_smooth_variant(tmp_path)


def test_rejected_cached_obj_is_not_published_as_a_success(tmp_path):
    source = tmp_path / "source"
    export_asset(trimesh.creation.box([0.1, 0.2, 0.03]), source, {})
    out = tmp_path / "out"
    items = [{"id": "item", "asset": str(source)}]
    assert assemble_visual(items, out, render=False)["exported"] == 1
    target = next((out / "assets").iterdir())
    missing_bottom().export(target / "visual.obj")
    result = assemble_visual(items, out, render=False)
    assert result["exported"] == 0 and result["failed"] == 1
    assert not (out / "models/item/item.obj").exists()
    assert (
        "not a closed volume"
        in json.loads((out / "models/item/status.json").read_text())["error"]
    )
