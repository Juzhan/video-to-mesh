import json

import cv2
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from packing_assets.assets import export_asset
from packing_assets.meshy_dimensions import refine_dimensions, validation_reasons
from packing_assets.quality import ray_scene, silhouette


@pytest.mark.parametrize("with_local", [False, True])
def test_known_box_recovers_axis_proportions_and_cache(tmp_path, with_local):
    pycolmap = pytest.importorskip("pycolmap")
    pytest.importorskip("open3d")
    work = tmp_path / "scan/work"
    sparse = work / "sparse"
    sparse.mkdir(parents=True)
    (work / "masks").mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 320 240 380 380 160 120\n")
    names = [f"frame_{i:04}.jpg" for i in range(24)]
    lines = []
    for i, name in enumerate(names):
        a = i * 2 * np.pi / 24
        position = np.array([0.45 * np.cos(a), 0.45 * np.sin(a), 0.29])
        forward = np.array([0, 0, 0.045]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        r = np.array([right, down, forward])
        t = -r @ position
        q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
        lines += [f"{i + 1} " + " ".join(map(str, [*q, *t])) + f" 1 {name}", ""]
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")
    (sparse / "points3D.txt").write_text("")
    rec = pycolmap.Reconstruction(str(sparse))
    rec.write(str(sparse))
    (work / "metric.json").write_text(
        json.dumps({"transform_to_m_z_up": np.eye(4).tolist()})
    )
    truth = trimesh.creation.box([0.16, 0.09, 0.10])
    truth.apply_translation([0, 0, 0.05])
    scene = ray_scene(truth)
    for im in rec.images.values():
        mask = silhouette(scene, rec.cameras[im.camera_id], im, np.eye(4), stride=1)
        cv2.imwrite(
            str(work / "masks" / (im.name + ".png")), mask.astype("uint8") * 255
        )
    base = truth.copy()
    base.apply_scale([1.055, 0.96, 0.95])
    asset = tmp_path / "base"
    export_asset(
        base,
        asset,
        {
            "backend": "meshy_multi_image",
            "board_from_generated_z_up": np.eye(4).tolist(),
            "fitting_views": names[::2],
        },
    )
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"views": [{"image": n} for n in names[::6]]}))
    old = (asset / "visual.obj").read_bytes()
    out = tmp_path / "corrected"
    local = tmp_path / "local" if with_local else None
    if local:
        export_asset(truth.copy(), local, {})
    report = refine_dimensions(asset, work, out, prepared, local_reference=local)
    assert report["accepted"], report
    original_error = np.linalg.norm(
        np.asarray(report["before_dimensions_m"]) - truth.extents
    )
    corrected_error = np.linalg.norm(
        np.asarray(report["after_dimensions_m"]) - truth.extents
    )
    assert corrected_error < 0.5 * original_error
    assert not set(report["fitting_views"]) & set(report["held_out_views"])
    assert (asset / "visual.obj").read_bytes() == old
    if local:
        np.testing.assert_allclose(
            report["after_dimensions_m"], truth.extents, rtol=0.005
        )
    assert (
        refine_dimensions(asset, work, out, prepared, local_reference=local) == report
    )


def test_held_out_regression_and_excessive_stretch_are_rejected():
    before = {"mean_iou": 0.96, "p10_iou": 0.94, "edge": 0.01, "extent": 0.01}
    after = dict(before, mean_iou=0.94, edge=0.009)
    assert "held_out_overlap_regressed" in validation_reasons(
        before, after, [1, 1, 1], {"status": "needs_review"}
    )
    assert "axis_scale_at_search_limit" in validation_reasons(
        before, before, [np.exp(0.10), 1, 1], {"status": "needs_review"}
    )


def test_dimension_variant_preserves_default_and_low_face_count(tmp_path):
    from packing_assets.visual_batch import assemble_visual

    source = tmp_path / "source"
    variant = tmp_path / "variant"
    base = trimesh.creation.box([0.1, 0.05, 0.08])
    export_asset(base.copy(), source, {})
    correction = np.diag([1.03, 0.97, 1.02, 1])
    moved = base.copy()
    moved.apply_transform(correction)
    export_asset(moved, variant, {"dimension_refinement_delta": correction.tolist()})
    (variant / "dimension-report.json").write_text("{}")
    item = {
        "id": "item",
        "asset": str(source),
        "smooth": False,
        "dimension_case": {
            "dimension_asset": str(variant),
            "dimension_calibration": {"accepted": True},
        },
    }
    out = tmp_path / "out"
    result = assemble_visual([item], out)
    assert result["exported"] == 1
    original = out / "models/item/item.obj"
    extra = out / "models/item/item_dimensions.obj"
    frozen = original.read_bytes()
    assert len(trimesh.load(extra, force="mesh").faces) == 12
    assert trimesh.load(extra, force="mesh").is_volume
    item["dimension_case"] = {
        "dimension_calibration": {
            "accepted": False,
            "reasons": ["no_clear_held_out_improvement"],
        }
    }
    assemble_visual([item], out)
    assert original.read_bytes() == frozen
    assert not extra.exists()
    assert not json.loads((out / "models/item/dimension-report.json").read_text())[
        "accepted"
    ]


def test_matching_local_requires_same_content_and_board(tmp_path, monkeypatch):
    from packing_assets.meshy_dimensions import matching_local_reference
    from packing_assets.reconstruct import input_digest

    root = tmp_path / "out"
    work = root / "reconstruction-meshy-fast/new/work"
    work.mkdir(parents=True)
    source = tmp_path / "new.mp4"
    source.write_bytes(b"same video")
    previous = tmp_path / "renamed.mp4"
    previous.write_bytes(b"same video")
    board = tmp_path / "board.json"
    board.write_text('{"size": 1}')
    (root / "input-manifest.json").write_text(
        json.dumps(
            {"objects": [{"id": "new", "input": str(source), "board": str(board)}]}
        )
    )
    local = root / "reconstruction/old"
    local.mkdir(parents=True)
    (local / "reconstruction.json").write_text(
        json.dumps(
            {
                "source": str(previous),
                "options": {},
                "fingerprint": input_digest(previous, board, {}),
            }
        )
    )
    asset = local / "candidate"
    export_asset(trimesh.creation.box(), asset, {})
    monkeypatch.setattr("packing_assets.automation.closed_candidate", lambda _: asset)
    assert matching_local_reference(work) == asset
    previous.write_bytes(b"different video")
    assert matching_local_reference(work) is None
    previous.write_bytes(b"same video")
    board.write_text('{"size": 2}')
    assert matching_local_reference(work) is None
