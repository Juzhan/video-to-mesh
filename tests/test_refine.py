import json
from pathlib import Path
import pycolmap
import pytest
from packing_assets.reconstruct import input_digest, reconstruction_options
from packing_assets import refine


def fixture(tmp_path):
    source = tmp_path / "photos"
    source.mkdir()
    (source / "one.jpg").write_bytes(b"fixture")
    board = tmp_path / "board.json"
    board.write_text("{}")
    old, _ = reconstruction_options(
        "draft", False, "grabcut", "outer_convex", pycolmap.__version__
    )
    result = tmp_path / "original"
    work = result / "work"
    work.mkdir(parents=True)
    required = [
        "images",
        "images_capture.json",
        "sparse",
        "dense",
        "metric.json",
        "object_points.ply",
        "isolation.json",
    ]
    for k in required:
        (work / k).write_text("immutable fixture")
    early = [
        "prepare",
        "features",
        "matching",
        "camera_estimation",
        "metric_scale",
        "undistort",
        "multi_view_stereo",
        "fusion",
        "remove_board",
    ]
    state = {
        "source": str(source),
        "options": old,
        "fingerprint": input_digest(source, board, old),
        "status": "rejected",
        "stages": {k: {"done": True} for k in early},
        "camera_method": "colmap_sfm",
    }
    (result / "reconstruction.json").write_text(json.dumps(state))
    return result, board


def test_refine_verifies_inputs_and_keeps_new_writes_separate(tmp_path, monkeypatch):
    result, board = fixture(tmp_path)
    out = tmp_path / "refined"

    def fake_reconstruct(source, board, out, *args):
        (out / "work/masks").mkdir()
        (out / "work/masks/new.png").write_text("new")
        return {"selection": {"status": "needs_review"}}

    monkeypatch.setattr(refine, "reconstruct", fake_reconstruct)
    refine.refine_segmentation(result, board, out, "grabcut-core")
    assert (out / "work/images").is_symlink()
    assert not (result / "work/masks").exists()
    assert (result / "work/object_points.ply").read_text() == "immutable fixture"
    state = json.loads((out / "reconstruction.json").read_text())
    assert "photo_masks_v3" not in state["stages"]


def test_refine_rejects_changed_board_without_writing_output(tmp_path):
    result, board = fixture(tmp_path)
    out = tmp_path / "refined"
    board.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        refine.refine_segmentation(result, board, out, "grabcut-core")
    assert not out.exists()
