import numpy as np
import trimesh

from packing_assets.evaluate import evaluate, volume_overlap
from packing_assets.quality import mask_scores, surface_release_gate


def test_same_bounds_do_not_hide_filled_cavity():
    hollow = trimesh.creation.annulus(r_min=0.03, r_max=0.04, height=0.1, sections=64)
    solid = trimesh.creation.cylinder(radius=0.04, height=0.1, sections=64)
    np.testing.assert_allclose(hollow.extents, solid.extents)
    overlap = volume_overlap(solid, hollow)
    assert overlap["iou"] < 0.5
    assert overlap["excess_fraction_of_truth"] > 1


def test_subvoxel_height_error_is_not_reported_as_perfect_overlap():
    a = trimesh.creation.box([0.18, 0.12, 0.028])
    a.apply_translation([0, 0, 0.014])
    b = trimesh.creation.box([0.18, 0.12, 0.0288])
    b.apply_translation([0, 0, 0.0144])
    overlap = volume_overlap(a, b)
    assert 0.95 < overlap["iou"] < 0.985


def test_independent_benchmark_accepts_identical_closed_shape(tmp_path):
    mesh = trimesh.creation.box([0.06, 0.04, 0.02])
    path = tmp_path / "box.obj"
    mesh.export(path)
    report = evaluate(path, path, tmp_path / "evaluation.json")
    assert report["benchmark_gate"]["status"] == "passed"


def test_incomplete_surface_is_rejected_even_with_matching_bounds():
    mesh = trimesh.creation.box([0.1, 0.1, 0.1])
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    report = surface_release_gate(
        mesh, {"median_iou": 1, "median_recall": 1, "p10_recall": 1}
    )
    assert report["status"] == "rejected"
    assert "not_closed_or_consistently_oriented" in report["reasons"]
    assert report["auto_approved"] is False


def test_visible_missing_geometry_fails_even_when_volume_closed():
    reference = np.ones((10, 10), bool)
    predicted = reference.copy()
    predicted[:5] = False
    assert mask_scores(predicted, reference)["recall"] == 0.5
    gate = surface_release_gate(
        trimesh.creation.box(),
        {"median_iou": 0.5, "median_recall": 0.5, "p10_recall": 0.4},
    )
    assert gate["status"] == "rejected"
