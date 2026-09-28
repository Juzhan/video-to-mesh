import numpy as np
import trimesh
from packing_assets.evaluate import evaluate, support_errors
from packing_assets.quality import convex_mask, silhouette_support_error
from packing_assets.convex import convex_gate


def test_convex_scope_allows_cavity_but_full_geometry_rejects(tmp_path):
    truth = trimesh.creation.annulus(r_min=0.03, r_max=0.04, height=0.08, sections=64)
    predicted = truth.convex_hull
    truth.export(tmp_path / "truth.obj")
    predicted.export(tmp_path / "predicted.obj")
    assert (
        evaluate(
            tmp_path / "predicted.obj", tmp_path / "truth.obj", tmp_path / "full.json"
        )["benchmark_gate"]["status"]
        == "rejected"
    )
    assert (
        evaluate(
            tmp_path / "predicted.obj",
            tmp_path / "truth.obj",
            tmp_path / "convex.json",
            "outer_convex",
        )["benchmark_gate"]["status"]
        == "passed"
    )


def test_extremity_error_is_detected_when_main_body_matches():
    body = trimesh.creation.box([0.06, 0.06, 0.15])
    tip = trimesh.creation.box([0.012, 0.01, 0.012])
    tip.apply_translation([0.032, 0, 0.08])
    truth = trimesh.util.concatenate([body, tip]).convex_hull
    error = support_errors(body, truth)
    assert error["max_under_mm"] > 8


def test_convex_reference_fills_hole_without_inflating_exterior():
    mask = np.zeros((40, 40), bool)
    mask[5:35, 7:30] = True
    mask[12:25, 13:25] = False
    hull = convex_mask(mask)
    assert hull[12:25, 13:25].all()
    assert not hull[:5].any()


def test_small_visible_tip_cannot_hide_in_area_score():
    a = np.zeros((100, 100), bool)
    a[20:80, 20:80] = True
    b = a.copy()
    b[25:28, 80:90] = True
    error = silhouette_support_error(a, b, 1)
    assert error["under_mm"] >= 9
    views = {
        "median_iou": 0.99,
        "median_recall": 0.99,
        "p10_recall": 0.99,
        "p90_under_mm": error["under_mm"],
        "p90_over_mm": 0,
    }
    assert convex_gate(trimesh.creation.box(), views)["status"] == "rejected"


def test_noise_cleanup_preserves_connected_thin_tip_and_flags_large_fragments():
    from packing_assets.quality import clean_single_object_mask

    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    mask[25:28, 80:90] = True
    mask[2, 2] = True
    clean, info = clean_single_object_mask(mask)
    assert not clean[2, 2] and clean[25:28, 80:90].all()
    assert info["removed_pixels"] == 1 and not info["ambiguous"]
    mask[0:10, 0:10] = True
    clean, info = clean_single_object_mask(mask)
    assert info["ambiguous"] and np.array_equal(clean, mask)


def test_hybrid_outer_mask_keeps_common_tip_and_fills_inner_hole():
    from packing_assets.hybrid_masks import intersect_outer_masks

    a = np.zeros((100, 100), bool)
    a[20:80, 20:80] = True
    a[30:34, 80:92] = True
    b = a.copy()
    b[20:80, 10:20] = True
    a[40:60, 40:60] = False
    result = intersect_outer_masks(a, b)
    assert result[40:60, 40:60].all() and result[30:34, 80:92].all()
    assert not result[:, 0:10].any()
