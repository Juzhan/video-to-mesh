import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from packing_assets.assets import export_asset
from packing_assets.automation import (
    closed_candidate,
    layout,
    assemble,
    run_automatic,
    isolated_export,
)
from packing_assets.automation import named_models


def make_asset(path):
    export_asset(
        trimesh.creation.box([0.04, 0.06, 0.1]), path, {"scale_source": "test fixture"}
    )
    return path


def test_candidate_fallback_uses_admitted_closed_proxy(tmp_path):
    make_asset(tmp_path / "depth_proxy")
    (tmp_path / "asset.json").write_text(json.dumps({"status": "rejected"}))
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "status": "needs_review",
                "scope": "outer_shape",
                "candidate_for_review": "asset.json",
            }
        )
    )
    assert closed_candidate(tmp_path) == tmp_path / "depth_proxy"
    d = json.loads((tmp_path / "depth_proxy/asset.json").read_text())
    d["status"] = "rejected"
    (tmp_path / "depth_proxy/asset.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="no admitted"):
        closed_candidate(tmp_path)


def test_rejected_run_cannot_use_stale_successful_candidate(tmp_path):
    make_asset(tmp_path / "packing_proxy")
    (tmp_path / "selection.json").write_text(json.dumps({"status": "rejected"}))
    with pytest.raises(ValueError, match="rejected"):
        closed_candidate(tmp_path)


def test_layout_keeps_sizes_and_separates_asymmetric_bounds():
    items = [
        {"bounds_m": [[-0.3, -0.1, -0.01], [0.3, 0.1, 0.1]]},
        {"bounds_m": [[0.1, 0.2, 0], [0.6, 0.4, 0.25]]},
        {"bounds_m": [[-0.05, -0.04, 0], [0.05, 0.04, 0.05]]},
    ]
    result = layout(items)
    boxes = [np.array(i["bounds_m"]) + i["position_m"] for i in result]
    for i, a in enumerate(boxes):
        assert a[0, 2] == 0
        for b in boxes[:i]:
            assert any(
                a[0, k] >= b[1, k] + 0.074 or b[0, k] >= a[1, k] + 0.074 for k in [0, 1]
            )


def test_assemble_continues_failure_and_reuses_success(tmp_path, monkeypatch):
    asset = make_asset(tmp_path / "source")
    out = tmp_path / "out"
    calls = []

    def fake_export(source, destination, mass, threshold, smooth=True):
        calls.append((mass, threshold))
        destination.mkdir()
        (destination / "package.json").write_text(
            json.dumps({"collision_parts": 1, "collision_union_check": {"iou": 1}})
        )
        import shutil

        shutil.copy2(source / "visual.obj", destination / "visual.obj")
        (destination / "asset.urdf").write_text(
            '<robot><link><visual><geometry><mesh filename="visual.obj"/></geometry></visual></link></robot>'
        )

    monkeypatch.setattr("packing_assets.automation.isolated_export", fake_export)
    monkeypatch.setattr(
        "packing_assets.automation.subprocess.run", lambda *a, **k: None
    )
    items = [
        {"id": "bad", "asset": str(tmp_path / "missing")},
        {"id": "good", "asset": str(asset)},
    ]
    report = assemble(items, out, render=False)
    assert (
        report["status"] == "partial"
        and report["exported"] == 1
        and report["failed"] == 1
    )
    first = json.loads((out / "scene.json").read_text())["items"][0]
    assert "not measured" in first["mass_source"]
    assemble(items, out, render=False)
    assert len(calls) == 1
    assemble(items, out, threshold_mm=2, render=False)
    assert len(calls) == 2


def test_raw_run_routes_mixed_boards_and_mass_without_manual_selection(
    tmp_path, monkeypatch
):
    source = tmp_path / "photos"
    (source / "键盘").mkdir(parents=True)
    (source / "键盘/a.jpg").write_bytes(b"photo")
    board = tmp_path / "board.json"
    board.write_text("{}")
    other = tmp_path / "large.json"
    other.write_text("{}")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"objects": {"键盘": {"board": "large.json", "mass_kg": 0.7}}})
    )
    asset = make_asset(tmp_path / "candidate")

    def fake_batch(manifest, out, on_result=None):
        item = json.loads(manifest.read_text())["objects"][0]
        assert item["board"] == str(other) and item["scope"] == "outer_shape"
        result = {"id": item["id"], "status": "needs_review"}
        on_result(result)
        return [result]

    def fake_assemble(items, out, *args, **kwargs):
        assert (
            items[0]["mass_kg"] == 0.7 and items[0]["mass_source"] == "caller supplied"
        )
        assert items[0]["asset"] == str(asset)
        return {"status": "complete_needs_review"}

    monkeypatch.setattr(
        "packing_assets.automation.export_collision_item",
        lambda item, *a: dict(item, named_visual="fixture.obj"),
    )
    monkeypatch.setattr("packing_assets.automation.named_models", lambda *a, **k: [])
    monkeypatch.setattr("packing_assets.automation.run_batch", fake_batch)
    monkeypatch.setattr("packing_assets.automation.closed_candidate", lambda p: asset)
    monkeypatch.setattr("packing_assets.automation.assemble", fake_assemble)
    assert (
        run_automatic(source, board, tmp_path / "out", settings, obj_only=False)[
            "status"
        ]
        == "complete_needs_review"
    )


def test_unknown_settings_name_is_not_silently_ignored(tmp_path):
    source = tmp_path / "input"
    (source / "bottle").mkdir(parents=True)
    (source / "bottle/a.jpg").write_bytes(b"photo")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"objects": {"botle": {}}}))
    with pytest.raises(ValueError, match="do not match"):
        run_automatic(source, tmp_path / "board.json", tmp_path / "out", settings)


def test_native_crash_is_converted_to_per_object_error(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(
        "packing_assets.automation.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], -11, stdout="native failure"),
    )
    with pytest.raises(RuntimeError, match="exited -11"):
        isolated_export(tmp_path / "source", tmp_path / "out", 0.1, 0.003)


def test_named_video_models_preserve_names_and_remove_failed_rerun(tmp_path):
    trimesh.creation.box([0.1, 0.1, 0.2]).export(tmp_path / "visual.obj")
    trimesh.creation.box([0.1, 0.1, 0.2]).export(tmp_path / "smooth.obj")
    row = {
        "id": "object_1",
        "label": "杯子.mp4",
        "visual": "visual.obj",
        "smooth_visual": "smooth.obj",
        "dimensions_m": [0.1, 0.1, 0.2],
        "urdf": "asset.urdf",
        "quality_status": "needs_review",
    }
    named_models([row], [], tmp_path)
    assert (tmp_path / "models/杯子/杯子.obj").read_bytes() == (
        tmp_path / "visual.obj"
    ).read_bytes()
    assert (tmp_path / "models/杯子/杯子_smooth.obj").read_bytes() == (
        tmp_path / "smooth.obj"
    ).read_bytes()
    status = json.loads((tmp_path / "models/杯子/status.json").read_text())
    assert status["exported_mesh_checks"]["visual"]["boundary_edges"] == 0
    named_models(
        [],
        [{"id": "object_1", "label": "杯子.mp4", "error": "recapture needed"}],
        tmp_path,
    )
    assert not (tmp_path / "models/杯子/杯子.obj").exists()
    assert (
        json.loads((tmp_path / "models/杯子/status.json").read_text())["error"]
        == "recapture needed"
    )


def test_same_stem_videos_do_not_overwrite_each_other(tmp_path):
    entries = [
        {"id": "one", "label": "瓶子.mp4", "error": "first"},
        {"id": "two", "label": "瓶子.mov", "error": "second"},
    ]
    named_models([], entries, tmp_path)
    assert (
        json.loads((tmp_path / "models/瓶子-one/status.json").read_text())["error"]
        == "first"
    )
    assert (
        json.loads((tmp_path / "models/瓶子-two/status.json").read_text())["error"]
        == "second"
    )
