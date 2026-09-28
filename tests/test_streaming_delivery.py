import json
from pathlib import Path

import pytest
import trimesh

from packing_assets.assets import export_asset
from packing_assets.automation import run_automatic, named_models


@pytest.fixture(autouse=True)
def fake_credential(monkeypatch):
    monkeypatch.delenv("MESHY_API_KEY_FILE", raising=False)
    monkeypatch.setenv("MESHY_API_KEY", "test-placeholder")


@pytest.mark.parametrize("backend", ["local", "meshy"])
def test_first_model_is_published_before_second_reconstruction(
    tmp_path, monkeypatch, backend
):
    pytest.importorskip("open3d")
    source = tmp_path / "input"
    source.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (source / name).write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    out = tmp_path / "output"
    calls = []
    ready = out / "models/a/a.obj"

    def reconstruct(source, board, folder, *args, **kwargs):
        assert bool(kwargs.get("frontend_only")) == (backend == "meshy")
        calls.append(source.name)
        if source.name != "a.mp4":
            assert ready.is_file(), "a must be delivered before starting b"
            assert (out / "models/a/a_smooth.obj").is_file() == (backend == "local")
            assert (
                json.loads((out / "models/a/status.json").read_text())["status"]
                == "needs_review"
            )
            assert json.loads((out / "run-report.json").read_text())["exported"] == 1
        if source.name == "b.mp4":
            raise ValueError("bad capture")
        if source.name == "c.mp4":
            assert (
                json.loads((out / "models/b/status.json").read_text())["status"]
                == "failed"
            )
            assert not (out / "models/b/b.obj").exists()
        report = export_asset(trimesh.creation.box([0.03, 0.04, 0.1]), folder, {})
        selection = {
            "status": "needs_review",
            "scope": "outer_shape",
            "candidate_for_review": "asset.json",
        }
        (folder / "selection.json").write_text(json.dumps(selection))
        report["selection"] = selection
        return report

    monkeypatch.setattr("packing_assets.reconstruct.reconstruct", reconstruct)
    # Stub only the provider, retaining the actual per-item export and final assembly.
    monkeypatch.setattr(
        "packing_assets.meshy_workflow.prepare_candidate",
        lambda result, *a: (result, {"status": "needs_review"}),
    )
    report = run_automatic(source, board, out, backend=backend)
    assert calls == ["a.mp4", "b.mp4", "c.mp4"]
    assert report["status"] == "partial" and report["exported"] == 2
    assert ready.is_file() and (out / "models/c/c.obj").is_file()


def test_interrupt_keeps_already_delivered_model(tmp_path, monkeypatch):
    pytest.importorskip("open3d")
    source = tmp_path / "input"
    source.mkdir()
    for name in ["a.mp4", "b.mp4"]:
        (source / name).write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    out = tmp_path / "output"

    def reconstruct(source, board, folder, *args):
        if source.name == "b.mp4":
            raise KeyboardInterrupt()
        report = export_asset(trimesh.creation.box([0.03, 0.04, 0.1]), folder, {})
        selection = {
            "status": "needs_review",
            "scope": "outer_shape",
            "candidate_for_review": "asset.json",
        }
        (folder / "selection.json").write_text(json.dumps(selection))
        report["selection"] = selection
        return report

    monkeypatch.setattr("packing_assets.reconstruct.reconstruct", reconstruct)
    with pytest.raises(KeyboardInterrupt):
        run_automatic(source, board, out)
    assert (out / "models/a/a.obj").is_file()
    report = json.loads((out / "run-report.json").read_text())
    assert report["status"] == "interrupted" and report["exported"] == 1


def test_incremental_names_do_not_collide_or_delete_previous_models(tmp_path):
    context = [{"id": "a", "label": "cup.mp4"}, {"id": "b", "label": "cup.mov"}]
    named_models([], [dict(i, quality_status="pending") for i in context], tmp_path)
    mesh = trimesh.creation.box([0.03, 0.04, 0.1])
    mesh.export(tmp_path / "fixture.obj")
    row = dict(
        context[0],
        visual="fixture.obj",
        smooth_visual="fixture.obj",
        quality_status="needs_review",
        dimensions_m=[0.03, 0.04, 0.1],
        urdf=None,
    )
    named_models([row], [], tmp_path, incremental=True, name_context=context)
    first = tmp_path / "models/cup-a/cup-a.obj"
    expected = first.read_bytes()
    named_models(
        [],
        [dict(context[1], error="failed")],
        tmp_path,
        incremental=True,
        name_context=context,
    )
    assert first.read_bytes() == expected
    assert (
        json.loads((tmp_path / "models/cup-b/status.json").read_text())["status"]
        == "failed"
    )
