import json
import numpy as np
from packing_assets.sam_masks import (
    object_prompt_cloud,
    support_envelope_cloud,
    trim_to_support_envelope,
)
from packing_assets.reconstruct import upgrade_mask_cache


def test_adjacent_low_paper_points_do_not_expand_object_prompt():
    rng = np.random.default_rng(2)
    paper = np.c_[
        rng.uniform(-0.148, 0.148, 500),
        rng.uniform(-0.105, 0.105, 500),
        rng.uniform(0.001, 0.002, 500),
    ]
    bottle = np.c_[
        rng.uniform(-0.04, 0.04, 800),
        rng.uniform(0.17, 0.23, 800),
        rng.uniform(0.025, 0.22, 800),
    ]
    high, box, cut = object_prompt_cloud(np.r_[paper, bottle])
    assert not high[:500].any() and high[500:].all()
    assert box[:, 1].min() > 0.16 and box[:, 2].min() == 0
    assert 0.02 < cut < 0.025


def test_old_masks_invalidate_meshes_but_keep_depth_and_cameras(tmp_path):
    work = tmp_path / "work"
    (work / "masks").mkdir(parents=True)
    (work / "masks/report.json").write_text(json.dumps({"point_prompts": False}))
    (work / "dense").mkdir()
    (work / "dense/depth").write_bytes(b"valid expensive cache")
    (tmp_path / "depth_proxy").mkdir()
    (tmp_path / "depth_proxy/asset.json").write_text("old admitted candidate")
    state = {
        "stages": {
            k: {"done": True}
            for k in (
                "metric_scale",
                "multi_view_stereo",
                "photo_masks_v3",
                "depth_proxy_v2",
            )
        }
    }
    assert upgrade_mask_cache(tmp_path, work, state, "sam2")
    assert (work / "dense/depth").read_bytes() == b"valid expensive cache"
    assert not (tmp_path / "depth_proxy").exists()
    assert set(state["stages"]) == {"metric_scale", "multi_view_stereo"}


def test_support_envelope_keeps_nearby_piece_and_excludes_remote_table_noise():
    grid = np.stack(
        np.meshgrid(
            np.arange(10) * 0.002,
            np.arange(10) * 0.002,
            np.arange(10) * 0.002,
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)
    body = grid + [0, 0.2, 0.025]
    nearby = grid[:200] + [0.03, 0.2, 0.025]
    distant = grid[:200] + [0.3, 0, 0.004]
    cloud, report = support_envelope_cloud(np.r_[body, nearby, distant])
    assert cloud[:, 0].max() < 0.1 and cloud[:, 0].max() > 0.03
    assert cloud[:, 1].min() >= 0.2 and cloud[:, 2].min() >= 0
    assert report["seed_points"] == len(body) + len(nearby)


def test_support_envelope_removes_far_mask_and_preserves_holes():
    mask = np.zeros((160, 200), np.uint8)
    mask[30:90, 30:90] = 1
    mask[50:70, 50:70] = 0
    mask[110:150, 120:180] = 1
    xy = np.stack(np.meshgrid(np.arange(30, 91), np.arange(30, 91)), -1).reshape(-1, 2)
    cleaned, removed = trim_to_support_envelope(mask, xy)
    assert cleaned[30:90, 30:90].sum() == mask[30:90, 30:90].sum()
    assert not cleaned[50:70, 50:70].any() and not cleaned[110:150, 120:180].any()
    assert removed > 0


def test_lifted_floor_noise_is_excluded_from_coherent_prompt():
    # Fast stereo on a real floor produces elevated fragments above the old
    # 10%-height cutoff. They must not enlarge the box sent to SAM.
    grid = np.stack(
        np.meshgrid(
            np.arange(30) * 0.002,
            np.arange(15) * 0.002,
            np.arange(8) * 0.002,
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)
    keyboard = grid + [0.05, 0.25, 0.015]
    floor = grid[::10] + [-0.3, -0.2, -0.003]
    points = np.r_[keyboard, floor]
    _, old_box, _ = object_prompt_cloud(points)
    coherent, _ = support_envelope_cloud(points)
    assert old_box[:, 0].min() < -0.2
    assert coherent[:, 0].min() >= 0.05
    assert coherent[:, 1].min() >= 0.25


def test_revision_five_masks_rerun_without_repeating_stereo(tmp_path):
    work = tmp_path / "work"
    (work / "masks").mkdir(parents=True)
    (work / "masks/report.json").write_text(json.dumps({"prompt_revision": 5}))
    state = {
        "stages": {
            k: {"done": True}
            for k in ("multi_view_stereo", "photo_masks_v3", "masked_surface_v3")
        }
    }
    assert upgrade_mask_cache(tmp_path, work, state, "sam2", required_revision=6)
    assert set(state["stages"]) == {"multi_view_stereo"}


def test_sam_receives_coherent_box_before_prediction_and_keeps_failure_report(
    tmp_path, monkeypatch
):
    import cv2, pytest
    from types import SimpleNamespace
    from plyfile import PlyData, PlyElement
    import pycolmap
    from sam2 import build_sam, sam2_image_predictor
    from packing_assets.sam_masks import isolate_sam

    grid = np.stack(
        np.meshgrid(
            np.arange(30) * 0.002,
            np.arange(15) * 0.002,
            np.arange(8) * 0.002,
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)
    points = np.r_[grid + [0.05, 0.25, 0.015], grid[::10] + [-0.3, -0.2, -0.003]]
    data = np.zeros(len(points), dtype=[(k, "f4") for k in ("x", "y", "z")])
    for i, k in enumerate(("x", "y", "z")):
        data[k] = points[:, i]
    ply = tmp_path / "cloud.ply"
    PlyData([PlyElement.describe(data, "vertex")]).write(ply)
    camera = SimpleNamespace(
        img_from_cam=lambda p: p[:, :2] / p[:, 2, None] * 1000 + 500
    )
    im = SimpleNamespace(
        name="frame.jpg",
        camera_id=1,
        cam_from_world=lambda: SimpleNamespace(
            matrix=lambda: np.c_[np.eye(3), [0, 0, 1]]
        ),
    )
    rec = SimpleNamespace(
        reg_image_ids=lambda: [1],
        num_reg_images=lambda: 1,
        images={1: im},
        cameras={1: camera},
    )
    monkeypatch.setattr(pycolmap, "Reconstruction", lambda _: rec)
    monkeypatch.setattr(build_sam, "build_sam2", lambda *a, **k: None)
    boxes = []

    class Predictor:
        def __init__(self, *a):
            pass

        def set_image(self, rgb):
            pass

        def predict(self, box, **kw):
            boxes.append(box.copy())
            return np.zeros((1, 1024, 1024), np.uint8), np.array([0.9]), None

    monkeypatch.setattr(sam2_image_predictor, "SAM2ImagePredictor", Predictor)
    cv2.imwrite(str(tmp_path / "frame.jpg"), np.zeros((1024, 1024, 3), np.uint8))
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"test")
    with pytest.raises(ValueError, match="too few object points"):
        isolate_sam(
            ply,
            tmp_path,
            tmp_path,
            np.eye(4),
            tmp_path / "out.ply",
            tmp_path / "masks",
            weight,
            True,
        )
    assert len(boxes) == 1 and boxes[0][0] > 530 and boxes[0][1] > 720
    report = json.loads((tmp_path / "masks/report.json").read_text())
    assert report["prompt_revision"] == 6 and report["retained_points"] == 0
