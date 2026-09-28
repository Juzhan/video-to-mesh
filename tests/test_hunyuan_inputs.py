import json

import numpy as np
import pytest
from PIL import Image

from scripts.run_hunyuan_case import prepare_inputs, validate_reference_images


def make_inputs(tmp_path):
    work = tmp_path / "source/work"
    (work / "images").mkdir(parents=True)
    (work / "masks").mkdir()
    rows = []
    for i in range(4):
        name = f"{i}.png"
        Image.new("RGB", (100, 80), "red").save(work / "images" / name)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[10:70, 35:65] = 255
        Image.fromarray(mask).save(work / "masks" / (name + ".png"))
        rows.append({"image": name, "azimuth_rad": i * np.pi / 2, "elevation_deg": 30})
    path = tmp_path / "prepared.json"
    path.write_text(
        json.dumps({"source_reconstruction": str(work.parent), "views": rows})
    )
    return path


def test_preserves_alpha_and_assigns_calibrated_views(tmp_path):
    source = make_inputs(tmp_path)
    result = prepare_inputs(source, tmp_path / "out")
    rows = json.loads(result.read_text())["views"]
    assert [r["view_tag"] for r in rows] == ["front", "left", "back", "right"]
    for row in rows:
        rgba = np.asarray(Image.open(result.parent / row["file"]))
        assert rgba.shape == (512, 512, 4)
        assert rgba[0, 0, 3] == 0
        assert rgba[256, 256, 3] == 255
        assert tuple(rgba[256, 256, :3]) == (255, 0, 0)


def test_rejects_unsuitable_camera_order(tmp_path):
    source = make_inputs(tmp_path)
    data = json.loads(source.read_text())
    data["views"][1]["azimuth_rad"] = 0.1
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="order/coverage"):
        prepare_inputs(source, tmp_path / "out")


def test_measurement_must_use_matching_frames(tmp_path):
    source = make_inputs(tmp_path)
    validate_reference_images(source, tmp_path / "source")
    wrong = tmp_path / "wrong/work/images"
    wrong.mkdir(parents=True)
    Image.new("RGB", (100, 80), "blue").save(wrong / "0.png")
    with pytest.raises(ValueError, match="do not match"):
        validate_reference_images(source, tmp_path / "wrong")
