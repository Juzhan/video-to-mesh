import json
from pathlib import Path

import numpy as np
import pytest
import requests
import trimesh

from packing_assets.assets import load_mesh, export_asset, rotate_up, scale_height
from packing_assets.pipeline import run_batch
from packing_assets.meshy import generate


def test_unit_conversion_scene_transforms_and_glb_axes(tmp_path):
    box = trimesh.creation.box([100, 50, 200])
    scene = trimesh.Scene()
    t = np.eye(4)
    t[:3, :3] *= 2
    t[:3, 3] = [600, 700, 800]
    scene.add_geometry(box, transform=t)
    source = tmp_path / "scene.glb"
    scene.export(source)
    mesh = load_mesh(source)
    np.testing.assert_allclose(mesh.extents, [200, 100, 400])
    mesh.apply_scale(0.001)
    report = export_asset(
        mesh, tmp_path / "out", {"scale_source": "declared_units"}, [0.2, 0.1, 0.4]
    )
    assert report["dimensions"]["passed"]
    assert report["status"] == "needs_review"
    obj = load_mesh(tmp_path / "out/visual.obj")
    glb = load_mesh(tmp_path / "out/visual.glb")
    np.testing.assert_allclose(obj.extents, [0.2, 0.1, 0.4], atol=1e-7)
    np.testing.assert_allclose(glb.extents, [0.2, 0.4, 0.1], atol=1e-7)
    assert obj.bounds[0, 2] == 0


def test_uniform_scale_does_not_hide_wrong_proportions(tmp_path):
    mesh = trimesh.creation.box([1, 2, 3])
    rotate_up(mesh, "y")
    scale_height(mesh, 0.2)
    report = export_asset(
        mesh, tmp_path, {"scale_source": "measured_height"}, [0.1, 0.1, 0.2]
    )
    assert not report["dimensions"]["passed"]
    assert not report["dimensions"]["independent_check"]
    assert "dimension_check_failed" in report["review_reasons"]


def test_batch_continues_on_failure(tmp_path):
    trimesh.creation.box([100, 50, 200]).export(tmp_path / "box.obj")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "id": "bad",
                        "backend": "known_mesh",
                        "mesh": "missing.obj",
                        "meters_per_unit": 0.001,
                    },
                    {
                        "id": "good",
                        "backend": "known_mesh",
                        "mesh": "box.obj",
                        "meters_per_unit": 0.001,
                    },
                ]
            }
        )
    )
    result = run_batch(manifest, tmp_path / "output")
    assert [x["status"] for x in result] == ["failed", "needs_review"]


def test_meshy_ambiguous_submission_is_never_reposted(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHY_API_KEY", "test-placeholder")
    image = tmp_path / "photo.png"
    image.write_bytes(b"fixture")

    class Session:
        count = 0

        def post(self, *args, **kwargs):
            self.count += 1
            raise requests.Timeout("lost response")

    session = Session()
    with pytest.raises(RuntimeError, match="uncertain"):
        generate([image], tmp_path / "provider", session=session)
    with pytest.raises(RuntimeError, match="unknown"):
        generate([image], tmp_path / "provider", session=session)
    assert session.count == 1


def test_meshy_resume_download_has_no_authorization_header(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHY_API_KEY", "test-placeholder")
    image = tmp_path / "photo.png"
    image.write_bytes(b"fixture")

    class Response:
        status_code = 200
        content = trimesh.creation.box().export(file_type="glb")

        def __init__(self, data=None):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class Session:
        posts = 0

        def post(self, *args, **kwargs):
            self.posts += 1
            return Response({"result": "task-123"})

        def get(self, url, **kwargs):
            if "asset.invalid" in url:
                assert "headers" not in kwargs
                return Response()
            return Response(
                {
                    "status": "SUCCEEDED",
                    "model_urls": {"glb": "https://asset.invalid/test.glb"},
                }
            )

    session = Session()
    out = tmp_path / "provider"
    generate([image], out, session=session)
    generate([image], out, session=session)
    assert session.posts == 1


def test_meshy_key_file_is_used_without_persisting_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHY_API_KEY", "obsolete-placeholder")
    image = tmp_path / "photo.png"
    image.write_bytes(b"fixture")
    key_file = tmp_path / "credential.txt"
    key_file.write_text("test-file-key-placeholder\n")

    class Session:
        def post(self, *args, **kwargs):
            assert (
                kwargs["headers"]["Authorization"] == "Bearer test-file-key-placeholder"
            )
            raise requests.Timeout("lost response")

    out = tmp_path / "provider"
    with pytest.raises(RuntimeError, match="uncertain"):
        generate([image], out, key_file=key_file, session=Session())
    assert "test-file-key-placeholder" not in (out / "meshy-task.json").read_text()


def test_meshy_multiline_key_rejected_before_submission(tmp_path, monkeypatch):
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    image = tmp_path / "photo.png"
    image.write_bytes(b"fixture")
    key_file = tmp_path / "credential.txt"
    key_file.write_text("first\nsecond")

    class Session:
        def post(self, *args, **kwargs):
            pytest.fail("must not submit malformed credentials")

    with pytest.raises(ValueError, match="one line"):
        generate([image], tmp_path / "provider", key_file=key_file, session=Session())


def test_no_measured_scale_no_meshy_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "packing_assets.pipeline.generate",
        lambda *a, **k: pytest.fail("must validate before upload"),
    )
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps({"objects": [{"id": "x", "backend": "meshy", "images": []}]})
    )
    result = run_batch(manifest, tmp_path / "out")
    assert result[0]["status"] == "failed"
