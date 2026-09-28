import json

import numpy as np
from plyfile import PlyData, PlyElement

from packing_assets.board import create_board
from packing_assets.metric import transform_points
from packing_assets.reconstruct import input_digest, select_object_cloud


def test_cache_digest_ignores_evaluation_but_tracks_rgb_and_board(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    photo = images / "a.jpg"
    photo.write_bytes(b"original-image")
    board = tmp_path / "board.json"
    board.write_text("{}")
    before = input_digest(images, board, {"quality": "draft"})
    evaluation = images / "evaluation_only"
    evaluation.mkdir()
    (evaluation / "truth.obj").write_text("deliberately different answers")
    assert before == input_digest(images, board, {"quality": "draft"})
    photo.write_bytes(b"changed-image")
    assert before != input_digest(images, board, {"quality": "draft"})
    photo.write_bytes(b"original-image")
    board.write_text('{"scale":2}')
    assert before != input_digest(images, board, {"quality": "draft"})


def test_cloud_scale_rotation_and_table_removal(tmp_path):
    board = create_board(tmp_path / "board")
    rng = np.random.default_rng(22)
    obj = rng.uniform([-0.03, -0.03, 0.02], [0.03, 0.03, 0.1], (2500, 3))
    table = rng.uniform([-0.14, -0.08, 0], [0.14, 0.08, 0], (300, 3))
    world = np.r_[obj, table]
    transform = np.eye(4)
    transform[:3, :3] *= 0.05
    transform[:3, 3] = [0.2, -0.3, 0.1]
    raw = transform_points(world, np.linalg.inv(transform))
    data = np.zeros(
        len(raw), dtype=[(a, "f4") for a in ["x", "y", "z", "nx", "ny", "nz"]]
    )
    for i, a in enumerate(["x", "y", "z"]):
        data[a] = raw[:, i]
    data["nz"] = 1
    source = tmp_path / "fused.ply"
    target = tmp_path / "object.ply"
    PlyData([PlyElement.describe(data, "vertex")]).write(source)
    report = select_object_cloud(source, transform, board, target)
    got = PlyData.read(target)["vertex"].data
    assert report["object_points"] >= 2000
    assert got["z"].min() > 0.019
    assert got["z"].max() < 0.101
    np.testing.assert_allclose(got["nz"], 1)


def test_visibility_cache_upgrade_preserves_masks_and_stereo(tmp_path):
    from packing_assets.reconstruct import upgrade_proxy_cache

    work = tmp_path / "work"
    work.mkdir()
    for name in ("masks", "dense"):
        (work / name).mkdir()
        (work / name / "keep").write_text("measured")
    (tmp_path / "packing_proxy").mkdir()
    (tmp_path / "packing_proxy/asset.json").write_text(json.dumps({"provenance": {}}))
    (tmp_path / "depth_proxy").mkdir()
    (tmp_path / "depth_proxy/visual.obj").write_text("old mesh")
    state = {
        "stages": {
            k: {"done": True}
            for k in (
                "multi_view_stereo",
                "photo_masks_v3",
                "masked_surface_v3",
                "packing_proxy_v2",
                "depth_proxy_v2",
            )
        }
    }
    assert upgrade_proxy_cache(tmp_path, work, state)
    assert set(state["stages"]) == {
        "multi_view_stereo",
        "photo_masks_v3",
        "masked_surface_v3",
    }
    assert (work / "masks/keep").read_text() == "measured" and (
        work / "dense/keep"
    ).read_text() == "measured"
    assert (
        not (tmp_path / "packing_proxy").exists()
        and not (tmp_path / "depth_proxy").exists()
    )
    assert not upgrade_proxy_cache(tmp_path, work, state)
