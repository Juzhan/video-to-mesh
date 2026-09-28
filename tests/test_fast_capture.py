from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from packing_assets.media import prepare
from packing_assets.reconstruct import reconstruction_options


def test_fast_photos_cover_whole_sequence_and_keep_scale_reference(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    for i in range(48):
        Image.new("RGB", (80, 60), (i, 40, 90)).save(source / f"{i:03d}.png")
    report = prepare(source, tmp_path / "selected", 24, max_photos=24)
    assert report["input_photo_count"] == 48 and report["frame_count"] == 24
    assert [Path(row["source"]).stem for row in report["frames"]] == [
        f"{i:03d}" for i in range(0, 48, 2)
    ]
    assert all((r["width"], r["height"]) == (80, 60) for r in report["frames"])
    assert len(list(source.glob("*.png"))) == 48


def test_fast_profile_has_distinct_cache_options_without_changing_standard():
    fast, _ = reconstruction_options("fast", False, "grabcut", "outer_shape", "test")
    standard, _ = reconstruction_options(
        "standard", False, "grabcut", "outer_shape", "test"
    )
    assert fast["fast_profile"]["max_frames"] == 48
    assert "fast_profile" not in standard and fast != standard
    with pytest.raises(ValueError, match="unknown"):
        reconstruction_options("invalid", False, "grabcut", "outer_shape", "test")
