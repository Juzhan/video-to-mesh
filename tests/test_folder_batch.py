from packing_assets.folder_batch import discover


def test_folder_discovery_uses_stable_unique_ids_and_ignores_other_files(tmp_path):
    (tmp_path / "杯子").mkdir()
    (tmp_path / "杯子" / "a.JPG").write_bytes(b"photo")
    (tmp_path / "杯子.mp4").write_bytes(b"video")
    (tmp_path / "notes.txt").write_text("not an object")
    (tmp_path / "empty").mkdir()
    board = tmp_path / "board.json"
    board.write_text("{}")
    items = discover(tmp_path, board)["objects"]
    assert len(items) == 2
    assert len({i["id"] for i in items}) == 2
    assert items == discover(tmp_path, board)["objects"]


def test_selection_defaults_all_and_rejects_ambiguous_names():
    import pytest
    from packing_assets.folder_batch import select_objects

    objects = [
        {"id": "a", "input": "keyboard.mp4"},
        {"id": "b", "input": "keyboard.mov"},
        {"id": "c", "input": "杯子.mp4"},
    ]
    assert select_objects(objects) == objects
    assert select_objects(objects, ["杯子", "keyboard.mov", "杯子.mp4"]) == [
        objects[1],
        objects[2],
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        select_objects(objects, ["keyboard"])
    with pytest.raises(ValueError, match="unknown"):
        select_objects(objects, ["missing"])


def test_selected_run_reuses_shared_cache_and_preserves_full_delivery(
    tmp_path, monkeypatch
):
    import json
    from packing_assets.automation import run_automatic

    source = tmp_path / "input"
    source.mkdir()
    for name in ["keyboard.mp4", "bottle.mp4"]:
        (source / name).write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    out = tmp_path / "output"
    (out / "models").mkdir(parents=True)
    (out / "models/previous.obj").write_text("previous delivery")
    (out / "run-report.json").write_text("previous report")

    def batch(manifest, cache, on_result=None):
        assert cache == out / "reconstruction"
        jobs = json.loads(manifest.read_text())["objects"]
        assert len(jobs) == 1 and jobs[0]["input"].endswith("keyboard.mp4")
        result = {"id": jobs[0]["id"], "status": "needs_review"}
        on_result(result)
        return [result]

    def assemble(items, delivery, *args, **kwargs):
        assert delivery.is_relative_to(out / "selected")
        assert len(items) == 1
        return {"status": "complete_needs_review", "exported": 1, "failed": 0}

    monkeypatch.setattr(
        "packing_assets.visual_batch.export_visual_item",
        lambda item, *a: dict(item, named_visual="fixture.obj"),
    )
    monkeypatch.setattr("packing_assets.automation.named_models", lambda *a, **k: [])
    monkeypatch.setattr("packing_assets.automation.run_batch", batch)
    monkeypatch.setattr(
        "packing_assets.automation.closed_candidate", lambda p: p / "depth_proxy"
    )
    monkeypatch.setattr("packing_assets.visual_batch.assemble_visual", assemble)
    report = run_automatic(source, board, out, objects=["keyboard"])
    assert report["input_count"] == 1 and report["discovered_input_count"] == 2
    assert report["selected_objects"] == ["keyboard.mp4"]
    assert (out / "run-report.json").read_text() == "previous report"
    assert (out / "models/previous.obj").read_text() == "previous delivery"
