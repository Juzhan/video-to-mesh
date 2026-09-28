import numpy as np
import trimesh
from packing_assets.reconstruct import retain_surface_components, select_candidate
from packing_assets.folder_batch import discover


def test_disconnected_pump_is_retained_with_body():
    body = trimesh.creation.box([0.08, 0.06, 0.18])
    body.apply_translation([0, 0, 0.09])
    nozzle = trimesh.creation.box([0.06, 0.012, 0.012])
    nozzle.apply_translation([0.018, 0, 0.216])
    source = trimesh.util.concatenate([body, nozzle])
    retained, count = retain_surface_components(source)
    assert count == 2
    np.testing.assert_allclose(retained.bounds, source.bounds)
    assert retained.bounds[1, 2] > 0.22


def test_outer_shape_never_silently_chooses_global_convex_hull():
    report = {
        "status": "rejected",
        "depth_proxy": {"status": "needs_review", "path": "depth_proxy/asset.json"},
        "packing_proxy": {"status": "needs_review", "path": "packing_proxy/asset.json"},
        "outer_convex": {"status": "needs_review", "path": "outer_convex/asset.json"},
    }
    selected = select_candidate(report, "outer_shape")
    assert selected["candidate_for_review"] == "depth_proxy/asset.json"
    report["depth_proxy"]["status"] = "rejected"
    assert select_candidate(report, "outer_shape")["status"] == "rejected"
    assert select_candidate(report, "outer_convex")["candidate_kind"] == "outer_convex"


def test_full_geometry_cannot_admit_cavity_filling_proxy():
    report = {
        "status": "rejected",
        "packing_proxy": {"status": "needs_review", "path": "packing_proxy/asset.json"},
    }
    assert select_candidate(report, "full_geometry")["status"] == "rejected"


def test_batch_default_preserves_external_shape(tmp_path):
    (tmp_path / "pump").mkdir()
    (tmp_path / "pump/photo.jpg").write_bytes(b"photo")
    assert (
        discover(tmp_path, tmp_path / "board.json")["objects"][0]["scope"]
        == "outer_shape"
    )
