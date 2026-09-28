import io
import json
from pathlib import Path

import pytest
import trimesh

from packing_assets import automation, meshy_workflow
from packing_assets.assets import export_asset
from packing_assets.progress import ProgressSession


@pytest.fixture(autouse=True)
def fake_credential(monkeypatch):
    monkeypatch.delenv("MESHY_API_KEY_FILE", raising=False)
    monkeypatch.setenv("MESHY_API_KEY", "test-placeholder")


def test_mixed_backends_route_and_isolate_paid_failure(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    for name in ("bottle.mp4", "broken.mp4", "bowl.mp4"):
        (source / name).write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"objects": {"bowl.mp4": {"backend": "local"}}}))
    calls = []

    def fake_batch(manifest, out, on_result=None):
        objects = json.loads(manifest.read_text())["objects"]
        assert all(obj["backend"] == "colmap_rgb" for obj in objects)
        results = [{"id": obj["id"], "status": "needs_review"} for obj in objects]
        for result in results:
            on_result(result)
        return results

    def fake_candidate(result, out, key_file):
        calls.append(result.name)
        if result.name.startswith("broken_"):
            raise ValueError("provider failed, no local fallback")
        return out / "photo-aligned", {"status": "needs_review"}

    def fake_assemble(items, out, render, errors, **kwargs):
        assert len(items) == 2 and len(errors) == 1
        assert errors[0]["stage"] == "meshy"
        assets = {item["label"]: item["asset"] for item in items}
        assert "meshy" in assets["bottle.mp4"]
        assert "reconstruction" in assets["bowl.mp4"]
        return {"status": "partial", "exported": 2, "failed": 1, "errors": errors}

    monkeypatch.setattr(
        "packing_assets.visual_batch.export_visual_item",
        lambda item, *a: dict(item, named_visual="fixture.obj"),
    )
    monkeypatch.setattr("packing_assets.automation.named_models", lambda *a, **k: [])
    monkeypatch.setattr(automation, "run_batch", fake_batch)
    monkeypatch.setattr(automation, "closed_candidate", lambda p: p / "depth_proxy")
    monkeypatch.setattr(meshy_workflow, "prepare_candidate", fake_candidate)
    monkeypatch.setattr("packing_assets.visual_batch.assemble_visual", fake_assemble)
    out = tmp_path / "out"
    report = automation.run_automatic(
        source, board, out, settings, obj_only=True, backend="meshy"
    )
    assert len(calls) == 2 and report["backend"] == "meshy"
    assert sorted(report["object_backends"].values()) == ["local", "meshy", "meshy"]
    assert json.loads((out / "run-report.json").read_text())["status"] == "partial"


def test_local_default_never_calls_paid_workflow(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    (source / "bottle.mp4").write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")

    def fake_batch(manifest, out, on_result=None):
        obj = json.loads(manifest.read_text())["objects"][0]
        result = {"id": obj["id"], "status": "needs_review"}
        on_result(result)
        return [result]

    monkeypatch.setattr(
        "packing_assets.visual_batch.export_visual_item",
        lambda item, *a: dict(item, named_visual="fixture.obj"),
    )
    monkeypatch.setattr("packing_assets.automation.named_models", lambda *a, **k: [])
    monkeypatch.setattr(automation, "run_batch", fake_batch)
    monkeypatch.setattr(automation, "closed_candidate", lambda p: p / "depth_proxy")
    monkeypatch.setattr(
        meshy_workflow,
        "prepare_candidate",
        lambda *a, **k: pytest.fail("unrequested paid branch"),
    )
    monkeypatch.setattr(
        "packing_assets.visual_batch.assemble_visual",
        lambda *a, **k: {"status": "complete_needs_review"},
    )
    assert (
        automation.run_automatic(source, board, tmp_path / "out", obj_only=True)[
            "backend"
        ]
        == "local"
    )


def test_open_container_rejected_before_any_upload(tmp_path, monkeypatch):
    source = tmp_path / "reference"
    source.mkdir()
    (source / "selection.json").write_text(json.dumps({"scope": "open_container"}))
    asset = source / "closed"
    asset.mkdir()
    (asset / "asset.json").write_text("{}")
    monkeypatch.setattr(meshy_workflow, "closed_candidate", lambda p: asset)
    monkeypatch.setattr(
        meshy_workflow, "generate", lambda *a, **k: pytest.fail("must not upload")
    )
    with pytest.raises(ValueError, match="exterior shape only"):
        meshy_workflow.prepare_candidate(source, tmp_path / "out")
    report = json.loads((tmp_path / "out/case-report.json").read_text())
    assert report["status"] == "failed" and report["stage"] == "meshy_reference"


@pytest.mark.parametrize("shape_status", ["needs_review", "rejected"])
def test_cached_geometry_reaudited_and_progress_reports_cache(
    tmp_path, monkeypatch, shape_status
):
    source, out = tmp_path / "reference", tmp_path / "out"
    source.mkdir()
    (source / "selection.json").write_text(json.dumps({"scope": "outer_shape"}))
    reference = source / "closed"
    export_asset(trimesh.creation.box([0.03, 0.04, 0.1]), reference, {})
    monkeypatch.setattr(meshy_workflow, "closed_candidate", lambda p: reference)
    monkeypatch.setattr(meshy_workflow, "reference_signature", lambda *a: "stable")
    (out / "inputs").mkdir(parents=True)
    (out / "inputs/prepared.json").write_text(
        json.dumps({"source_video": "bottle.mp4", "views": []})
    )
    for folder in ("metric", "photo-aligned"):
        export_asset(trimesh.creation.box([0.03, 0.04, 0.1]), out / folder, {})
        meta_path = out / folder / "asset.json"
        meta = json.loads(meta_path.read_text())
        meta["status"] = shape_status
        meta_path.write_text(json.dumps(meta))
        (out / folder / "quality.json").write_text(
            json.dumps({"views": {"median_iou": 0.95}})
        )
    generated = out / "source.glb"
    trimesh.creation.box().export(generated)
    monkeypatch.setattr(meshy_workflow, "generate", lambda *a, **k: generated)
    monkeypatch.setattr(
        meshy_workflow, "register_meshy", lambda *a: pytest.fail("should reuse metric")
    )
    monkeypatch.setattr(
        meshy_workflow, "refine_photo_pose", lambda *a: pytest.fail("should reuse pose")
    )
    with ProgressSession(out, "off", verbose=True, stream=io.StringIO()):
        selected, report = meshy_workflow.prepare_candidate(source, out)
    assert selected.is_dir() and "meshy_metric" in report["cached_stages"]
    assert report["retained_unvalidated"] == (shape_status == "rejected")
    assert json.loads((selected / "asset.json").read_text())["status"] == shape_status
    assert (out / "retained/raw.obj").exists()
    assert (
        json.loads((out / "retained/status.json").read_text())["units"]
        == "unknown_provider_scale"
    )
    assert json.loads((out / "progress.json").read_text())["stage"] == "meshy_check"
    # A stale admitted flag must not let an edited open model through.
    for folder in ("metric", "photo-aligned"):
        mesh = trimesh.creation.box()
        mesh.update_faces(range(len(mesh.faces) - 1))
        mesh.export(out / folder / "visual.obj")
    with pytest.raises(ValueError, match="closed volume"):
        meshy_workflow.prepare_candidate(source, out)
    assert (out / "retained/raw.obj").exists()
    assert json.loads((out / "case-report.json").read_text())[
        "retained_raw_glb"
    ] == str(generated)


def test_failed_single_case_does_not_leave_old_model_as_success(tmp_path, monkeypatch):
    out = tmp_path / "out"
    models = out / "delivery/models/bottle"
    models.mkdir(parents=True)
    old = models / "bottle.obj"
    old.write_text("old output")
    (models.parent / "index.json").write_text(
        json.dumps([{"visual": "models/bottle/bottle.obj"}])
    )

    def reject(*a, **k):
        raise ValueError("reference changed")

    monkeypatch.setattr(meshy_workflow, "prepare_candidate", reject)
    report = meshy_workflow.run_case(
        tmp_path / "ref", out, render=False, progress_mode="off", verbose=True
    )
    assert report["status"] == "failed" and not old.exists()
    assert json.loads((out / "progress.json").read_text())["status"] == "failed"


def test_hybrid_upgrade_refits_and_validates_with_dense_masks(tmp_path, monkeypatch):
    from packing_assets import meshy_dimensions

    source, out, local = tmp_path / "reference", tmp_path / "out", tmp_path / "local"
    source.mkdir()
    (source / "selection.json").write_text('{"scope":"outer_shape"}')
    (source / "hybrid-measurement.json").write_text('{"status":"ready"}')
    original_reference, dense_reference = source / "closed", local / "depth_proxy"
    for asset in (original_reference, dense_reference):
        export_asset(trimesh.creation.box([0.1, 0.2, 0.03]), asset, {})
    monkeypatch.setattr(
        meshy_workflow, "closed_candidate", lambda p: original_reference
    )
    monkeypatch.setattr(meshy_workflow, "reference_signature", lambda *a: "stable")
    monkeypatch.setattr(
        meshy_dimensions, "matching_local_reference", lambda p: dense_reference
    )
    (out / "inputs").mkdir(parents=True)
    prepared = out / "inputs/prepared.json"
    prepared.write_text('{"source_video":"book.mp4","views":[]}')
    original_inputs = prepared.read_bytes()

    def write_candidate(target, iou):
        export_asset(trimesh.creation.box([0.1, 0.2, 0.03]), target, {})
        p = target / "asset.json"
        meta = json.loads(p.read_text())
        meta["status"] = "needs_review"
        p.write_text(json.dumps(meta))
        (target / "quality.json").write_text(json.dumps({"views": {"median_iou": iou}}))

    for name in ("metric", "photo-aligned"):
        write_candidate(out / name, 0.99)
    generated = out / "source.glb"
    trimesh.creation.box().export(generated)
    monkeypatch.setattr(meshy_workflow, "generate", lambda *a, **k: generated)
    calls = []

    def fit(generated, result, target, prepared, reference_asset=None):
        assert result == local and reference_asset == dense_reference
        calls.append("fit")
        write_candidate(target, 0.92)

    def photos(asset, work, target):
        assert work == local / "work"
        calls.append("photos")
        write_candidate(target, 0.93)

    def dimensions(asset, work, *a, **k):
        assert work == local / "work" and k["local_reference"] == dense_reference
        return {"accepted": False}

    monkeypatch.setattr(meshy_workflow, "register_meshy", fit)
    monkeypatch.setattr(meshy_workflow, "refine_photo_pose", photos)
    monkeypatch.setattr(meshy_dimensions, "refine_dimensions", dimensions)
    selected, report = meshy_workflow.prepare_candidate(source, out)
    assert selected.name.startswith("hybrid-photo-")
    assert report["validation_work"] == str(local / "work")
    assert calls == ["fit", "photos"]
    meshy_workflow.prepare_candidate(source, out)
    assert calls == ["fit", "photos"]
    assert prepared.read_bytes() == original_inputs
