import hashlib
import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import trimesh

from packing_assets.collision import union_overlap, collision_input, pad_convex
from packing_assets.pybullet_asset import export_pybullet, light_visual


def make_asset(tmp_path):
    asset = tmp_path / "source"
    asset.mkdir()
    mesh = trimesh.creation.box([0.04, 0.06, 0.1])
    mesh.apply_translation([0.01, 0, 0.05])
    mesh.export(asset / "visual.obj")
    (asset / "asset.json").write_text(
        json.dumps({"units": "m", "obj_up_axis": "z", "status": "needs_review"})
    )
    cached = asset / "collision"
    cached.mkdir()
    mesh.export(cached / "part_000.obj")
    (cached / "collision.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(
                    (asset / "visual.obj").read_bytes()
                ).hexdigest(),
                "threshold_m": 0.003,
                "pipeline_version": 3,
                "parts": 1,
                "source_volume_comparison": {"iou": 1.0},
            }
        )
    )
    return asset, mesh


def test_overlap_of_duplicate_hulls_uses_union_not_parity():
    pytest.importorskip("open3d")
    box = trimesh.creation.box([0.03, 0.04, 0.05])
    result = union_overlap([box, box.copy()], box, pitch=0.004)
    assert result["iou"] == 1
    assert result["excess_fraction_of_reference"] == 0
    assert result["missing_fraction_of_reference"] == 0


def test_collision_cleanup_keeps_disconnected_pump_and_bounds_noise_removal():
    pytest.importorskip("open3d")
    body = trimesh.creation.box([0.1, 0.1, 0.1])
    pump = trimesh.creation.box([0.02, 0.01, 0.01])
    pump.apply_translation([0, 0, 0.057])
    noise = trimesh.creation.box([0.001, 0.001, 0.001])
    noise.apply_translation([0.0508, 0, 0])
    result, report = collision_input(
        trimesh.util.concatenate([body, pump, noise]), 0.003
    )
    assert report["removed_near_surface_tiny_components"] == 1
    assert report["removed_volume_fraction"] < 0.001
    assert len(result.split()) == 2
    assert result.bounds[1, 2] == pytest.approx(pump.bounds[1, 2])


def test_visual_reduction_limits_faces_without_resizing():
    pytest.importorskip("open3d")
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    mesh.apply_translation([0.012, 0, 0.05])
    result, report = light_visual(mesh, target_faces=1000)
    assert len(result.faces) <= 1000
    assert report["p95_distance_mm"] <= 1
    assert np.max(abs(result.bounds - mesh.bounds)) <= 0.001


def test_convex_padding_is_outward_and_bounded_in_same_frame():
    mesh = trimesh.creation.box([0.02, 0.03, 0.04])
    mesh.apply_translation([0.01, 0.02, 0.03])
    padded = pad_convex([mesh], 0.00025)[0]
    np.testing.assert_allclose(
        padded.bounds, mesh.bounds + [[-0.00025] * 3, [0.00025] * 3]
    )
    assert padded.is_convex and padded.is_volume and padded.volume > mesh.volume


def test_portable_urdf_preserves_shared_frame_and_mass(tmp_path):
    asset, mesh = make_asset(tmp_path)
    out = tmp_path / "package"
    report = export_pybullet(asset, out, 0.32)
    root = ET.parse(out / "asset.urdf").getroot()
    assert len(root.findall("link")) == 1 and not root.findall("joint")
    assert float(root.find("link/inertial/mass").get("value")) == 0.32
    np.testing.assert_allclose(report["estimated_center_of_mass_m"], mesh.center_mass)
    for node in root.findall(".//mesh"):
        assert node.get("scale") == "1 1 1"
        assert (out / node.get("filename")).is_file()
    collision = trimesh.load_mesh(out / "collision/part_000.obj")
    np.testing.assert_allclose(collision.bounds, mesh.bounds)


def test_changed_threshold_invalidates_collision_cache(tmp_path, monkeypatch):
    asset, _ = make_asset(tmp_path)

    def rebuild(mesh_path, out, threshold):
        assert threshold == 0.001
        raise RuntimeError("rebuild required")

    monkeypatch.setattr("packing_assets.pybullet_asset.build_collision", rebuild)
    with pytest.raises(RuntimeError, match="rebuild required"):
        export_pybullet(asset, tmp_path / "package", 0.32, threshold_m=0.001)


@pytest.mark.parametrize("mass", [0, -1, float("nan")])
def test_mass_must_be_explicit_positive_value(tmp_path, mass):
    with pytest.raises(ValueError, match="mass_kg"):
        export_pybullet(tmp_path / "missing", tmp_path / "out", mass)
