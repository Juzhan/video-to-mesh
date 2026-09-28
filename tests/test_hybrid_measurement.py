import json

import pytest

from packing_assets.hybrid_measurement import ensure_measurement, seed_measurement
from packing_assets.reconstruct import input_digest


def test_seed_copies_completed_stages_without_sharing_writable_files(
    tmp_path, monkeypatch
):
    pytest.importorskip("pycolmap")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    result = tmp_path / "frontend"
    work = result / "work"
    (work / "images").mkdir(parents=True)
    (work / "images/a.jpg").write_bytes(b"image")
    (work / "database.db").write_bytes(b"database")
    (work / "masks").mkdir()
    (work / "masks/a.png").write_bytes(b"paid mask")
    (work / "dense").mkdir()
    (work / "dense/incomplete.bin").write_bytes(b"partial")
    options = {"segmentation": "sam2"}
    monkeypatch.setattr(
        "packing_assets.hybrid_measurement.reconstruction_options",
        lambda *a: (dict(options), None),
    )
    fast = dict(options, meshy_frontend_revision=1)
    state = {
        "fingerprint": input_digest(source, board, fast),
        "options": fast,
        "stages": {
            k: {"done": True, "seconds": 1}
            for k in ["prepare", "features", "matching", "sparse_photo_masks"]
        },
        "active_stage": "multi_view_stereo",
    }
    (result / "reconstruction.json").write_text(json.dumps(state))
    target = seed_measurement(result, source, board)
    cached = json.loads((target / "reconstruction.json").read_text())
    assert set(cached["stages"]) == {"prepare", "features", "matching"}
    assert not (target / "work/dense").exists()
    assert not (target / "work/masks").exists()
    (target / "work/images/a.jpg").write_bytes(b"changed image")
    assert (work / "images/a.jpg").read_bytes() == b"image"
    assert seed_measurement(result, source, board) == target
    source.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="changed"):
        seed_measurement(result, source, board)


def test_optional_measurement_failure_preserves_photo_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "packing_assets.meshy_dimensions.matching_local_reference", lambda *a: None
    )

    def fail(*a):
        raise RuntimeError("measurement could not complete")

    monkeypatch.setattr("packing_assets.hybrid_measurement.seed_measurement", fail)
    report = ensure_measurement(tmp_path, tmp_path / "video", tmp_path / "board")
    assert (
        report["status"] == "failed" and report["fallback"] == "photo_axis_refinement"
    )
    assert (tmp_path / "hybrid-measurement.json").exists()
