import json
import trimesh
from packing_assets.assets import export_asset
from packing_assets.visual_batch import assemble_visual


def test_rejected_meshy_export_is_explicit_and_preserves_failed_check(tmp_path):
    source = tmp_path / "source"
    export_asset(
        trimesh.creation.box([0.04, 0.06, 0.1]),
        source,
        {"backend": "meshy_multi_image"},
    )
    meta = json.loads((source / "asset.json").read_text())
    meta.update(
        status="rejected",
        quality={"status": "rejected", "reasons": ["view_silhouette_disagreement"]},
    )
    original = json.dumps(meta)
    (source / "asset.json").write_text(original)
    item = {"id": "plane", "label": "plane.mp4", "asset": str(source)}
    blocked = assemble_visual([item], tmp_path / "blocked")
    assert blocked["exported"] == 0
    item["retain_unvalidated_meshy"] = True
    out = tmp_path / "out"
    report = assemble_visual([item], out)
    assert report["exported"] == 1 and report["retained_unvalidated"] == 1
    status = json.loads((out / "models/plane/status.json").read_text())
    assert status["status"] == "retained_unvalidated"
    assert status["quality_check"]["status"] == "rejected"
    assert (out / "models/plane/plane.obj").exists()
    assert (out / "models/plane/plane_smooth.obj").exists()
    assert (source / "asset.json").read_text() == original
    # Retention policy never bypasses the actual closed-volume check.
    open_mesh = trimesh.creation.box()
    open_mesh.update_faces(range(11))
    open_mesh.export(source / "visual.obj")
    broken = assemble_visual([item], tmp_path / "open")
    assert broken["exported"] == 0


def test_collision_assembly_retains_visual_only_without_claiming_physics(tmp_path):
    from packing_assets.automation import assemble
    from packing_assets.visual_batch import export_visual_item

    source = tmp_path / "source"
    export_asset(
        trimesh.creation.box([0.04, 0.06, 0.1]),
        source,
        {"backend": "meshy_multi_image"},
    )
    meta = json.loads((source / "asset.json").read_text())
    meta["status"] = "rejected"
    (source / "asset.json").write_text(json.dumps(meta))
    item = {"id": "plane", "asset": str(source), "retain_unvalidated_meshy": True}
    out = tmp_path / "out"
    row = export_visual_item(item, out)
    report = assemble([item], out, physics_test=True, prepared_rows=[row])
    assert report["exported"] == 1 and report["retained_unvalidated"] == 1
    assert report["physics_test"] == "skipped_unvalidated_geometry"
    assert (out / "scene-package.zip").exists()


def test_obj_batch_keeps_names_scale_and_continues_failures_without_physics(tmp_path):
    source = tmp_path / "source"
    export_asset(
        trimesh.creation.box([0.04, 0.06, 0.1]),
        source,
        {"scale_source": "test fixture"},
    )
    out = tmp_path / "out"
    report = assemble_visual(
        [
            {"id": "bottle", "label": "瓶子.mp4", "asset": str(source)},
            {"id": "bad", "label": "坏视频.mov", "asset": str(tmp_path / "missing")},
        ],
        out,
        render=False,
    )
    assert (
        report["exported"] == 1
        and report["failed"] == 1
        and report["status"] == "partial"
    )
    assert report["physics_test"] == "not_requested"
    assert (out / "models/瓶子/瓶子.obj").is_file()
    assert (out / "models/瓶子/瓶子_smooth.obj").is_file()
    assert not list(out.rglob("*.urdf"))
    scene = json.loads((out / "scene.json").read_text())
    assert scene["items"][0]["dimensions_m"] == [0.04, 0.06, 0.1]
    assert scene["items"][0]["urdf"] is None
    assert (out / "scene-package.zip").is_file()
