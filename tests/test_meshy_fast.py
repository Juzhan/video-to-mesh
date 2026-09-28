import json

import pytest
import trimesh

from packing_assets.assets import export_asset
from packing_assets.meshy_frontend import reuse_legacy_generation
from packing_assets.reconstruct import input_digest


@pytest.mark.parametrize("downloaded", [True, False])
def test_raw_generation_reuse_requires_same_video_and_board(tmp_path, downloaded):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"original-video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    item = {"id": "object", "input": str(source), "board": str(board)}
    old = tmp_path / "meshy/object"
    for directory in [
        old / "provider",
        old / "inputs",
        tmp_path / "reconstruction/object",
    ]:
        directory.mkdir(parents=True)
    if downloaded:
        (old / "provider/source.glb").write_bytes(b"glTFalready-paid")
    (old / "provider/meshy-task.json").write_text(
        json.dumps({"downloaded": downloaded, "task_id": "existing"})
    )
    (old / "inputs/prepared.json").write_text("{}")
    state = {"options": {}, "fingerprint": input_digest(source, board, {})}
    (tmp_path / "reconstruction/object/reconstruction.json").write_text(
        json.dumps(state)
    )
    source.write_bytes(b"different-video")
    with pytest.raises(ValueError, match="input changed"):
        reuse_legacy_generation(tmp_path, item)
    assert not (tmp_path / "meshy-fast/object/provider").exists()
    source.write_bytes(b"original-video")
    reuse_legacy_generation(tmp_path, item)
    if downloaded:
        assert (
            tmp_path / "meshy-fast/object/provider/source.glb"
        ).read_bytes() == b"glTFalready-paid"
    assert (
        json.loads(
            (tmp_path / "meshy-fast/object/provider/meshy-task.json").read_text()
        )["task_id"]
        == "existing"
    )
    assert (tmp_path / "meshy-fast/object/generation-reuse.json").exists()


def test_fast_reconstruction_stops_before_stereo_without_cuda(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip("pycolmap")
    from packing_assets import reconstruct as module

    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    out = tmp_path / "reference"
    (out / "work/images").mkdir(parents=True)
    (out / "work/metric.json").write_text(
        json.dumps(
            {
                "transform_to_m_z_up": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            }
        )
    )
    options = {"meshy_frontend_revision": 1}
    (out / "reconstruction.json").write_text(
        json.dumps(
            {
                "fingerprint": input_digest(source, board, options),
                "stages": {
                    k: {"done": True, "seconds": 0}
                    for k in [
                        "prepare",
                        "features",
                        "matching",
                        "camera_estimation",
                        "metric_scale",
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(module, "reconstruction_options", lambda *a: ({}, None))
    monkeypatch.setattr(pycolmap, "has_cuda", False)
    for name in ["undistort_images", "patch_match_stereo", "stereo_fusion"]:
        monkeypatch.setattr(
            pycolmap, name, lambda *a, **kw: pytest.fail("dense stage called")
        )
    monkeypatch.setattr(
        "packing_assets.meshy_frontend.finish_sparse_frontend",
        lambda *a: {"frontend": "meshy_fast"},
    )
    assert (
        module.reconstruct(source, board, out, frontend_only=True)["frontend"]
        == "meshy_fast"
    )


def test_plain_visual_does_not_run_smoothing_or_publish_stale_smooth(
    tmp_path, monkeypatch
):
    from packing_assets import visual_batch

    source = tmp_path / "source"
    export_asset(trimesh.creation.box([0.1, 0.05, 0.02]), source, {})
    out = tmp_path / "out"
    (out / "models/keyboard").mkdir(parents=True)
    (out / "models/keyboard/keyboard_smooth.obj").write_text("old")
    monkeypatch.setattr(
        visual_batch,
        "ensure_smooth_variant",
        lambda *a: pytest.fail("smoothing called"),
    )
    monkeypatch.setattr(
        visual_batch,
        "light_visual",
        lambda mesh: (mesh, {"export_faces": len(mesh.faces)}),
    )
    r = visual_batch.assemble_visual(
        [{"id": "keyboard", "asset": str(source), "smooth": False}], out
    )
    assert r["exported"] == 1
    assert not (out / "models/keyboard/keyboard_smooth.obj").exists()
    assert not list((out / "assets").rglob("*smooth*"))
    assert "smooth_visual" not in json.loads(
        (out / "models/keyboard/status.json").read_text()
    )


def test_sparse_alignment_cannot_collapse_scale_into_small_patch():
    pytest.importorskip("open3d")
    import numpy as np

    from packing_assets.meshy_fit import fit_upright_similarity

    mesh = trimesh.creation.box([2, 1, 0.5])
    source, _ = trimesh.sample.sample_surface(mesh, 800, seed=14)
    target = source[source[:, 0] > -0.5] * 0.1 + [0.3, -0.2, 0.025]
    expected = np.std(target[:, :2], axis=0).dot(np.std(target[:, :2], axis=0)) ** 0.5
    expected /= np.std(source[:, :2], axis=0).dot(np.std(source[:, :2], axis=0)) ** 0.5
    matrix, _ = fit_upright_similarity(source, target, sparse=True)
    assert np.allclose(np.linalg.svd(matrix[:3, :3], compute_uv=False), expected)
    assert np.isclose((source @ matrix[:3, :3].T + matrix[:3, 3])[:, 2].min(), 0)


@pytest.mark.parametrize("cuda", [True, False])
@pytest.mark.parametrize("hybrid", [True, False])
def test_sparse_failure_recovers_depth_only_and_resumes(
    tmp_path, monkeypatch, cuda, hybrid
):
    pycolmap = pytest.importorskip("pycolmap")
    from packing_assets import reconstruct as module
    from packing_assets.meshy_frontend import SparseForegroundError

    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    out = tmp_path / "reference"
    work = out / "work"
    (work / "images").mkdir(parents=True)
    (work / "metric.json").write_text(
        json.dumps(
            {
                "transform_to_m_z_up": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            }
        )
    )
    (out / "reconstruction.json").write_text(
        json.dumps(
            {
                "fingerprint": input_digest(
                    source, board, {"meshy_frontend_revision": 1}
                ),
                "stages": {
                    k: {"done": True, "seconds": 0}
                    for k in [
                        "prepare",
                        "features",
                        "matching",
                        "camera_estimation",
                        "metric_scale",
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(module, "reconstruction_options", lambda *a: ({}, None))
    monkeypatch.setattr(pycolmap, "has_cuda", cuda)
    calls = []
    for name in ["undistort_images", "patch_match_stereo", "stereo_fusion"]:
        monkeypatch.setattr(
            pycolmap, name, lambda *a, _name=name, **kw: calls.append(_name)
        )
    for name in ["extract_features", "match_exhaustive", "incremental_mapping"]:
        monkeypatch.setattr(
            pycolmap, name, lambda *a, **kw: pytest.fail("camera cache lost")
        )
    monkeypatch.setattr(module, "select_object_cloud", lambda *a: {})

    def masks(*args, **kwargs):
        calls.append("masks")
        (work / "masks").mkdir(exist_ok=True)
        (work / "masks/report.json").write_text('{"prompt_revision": 6}')

    monkeypatch.setattr("packing_assets.sam_masks.isolate_sam", masks)
    monkeypatch.setattr(
        module, "surface_mesh", lambda *a: pytest.fail("local surface built")
    )

    def finish(*args, dense_prompts=False):
        if not dense_prompts:
            assert not hybrid
            calls.append("sparse_attempt")
            raise SparseForegroundError("fragmented reflective foreground")
        if not hybrid:
            assert args[5]["frontend_fallback"]["to"] == "dense_prompts"
        # Persist the same routing metadata as the real frontend. A hybrid
        # retry must not fall back into sparse masks and overwrite paid inputs.
        args[5].update(frontend="meshy_fast", frontend_prompt_source="dense")
        (out / "reconstruction.json").write_text(json.dumps(args[5]))
        return args[5]

    monkeypatch.setattr("packing_assets.meshy_frontend.finish_sparse_frontend", finish)
    if not cuda:
        with pytest.raises(RuntimeError, match="CUDA"):
            module.reconstruct(
                source, board, out, frontend_only=True, measure_dimensions=hybrid
            )
        assert calls == ([] if hybrid else ["sparse_attempt"])
        return
    for _ in range(2):
        assert (
            module.reconstruct(
                source, board, out, frontend_only=True, measure_dimensions=hybrid
            )["frontend_prompt_source"]
            == "dense"
        )
    assert calls == ([] if hybrid else ["sparse_attempt"]) + [
        "undistort_images",
        "patch_match_stereo",
        "stereo_fusion",
        "masks",
    ]
