import json
from xml.etree import ElementTree as ET
import numpy as np
import pytest
import trimesh
from packing_assets.depth_proxy import cavity_probe
from packing_assets.smooth import ensure_smooth_variant
from packing_assets.folder_batch import inferred_scope
from packing_assets.reconstruct import select_candidate


def cup():
    return trimesh.creation.revolve(
        np.array(
            [[0, 0], [0.04, 0], [0.04, 0.1], [0.03, 0.1], [0.03, 0.01], [0, 0.01]]
        ),
        sections=64,
    )


def test_cup_cavity_is_detected_but_solid_cylinder_is_not():
    pytest.importorskip("open3d")
    result = cavity_probe(cup())
    assert result["detected"] and result["depth_m"] > 0.085
    assert len(result["probe_columns"]) == 5
    assert not cavity_probe(trimesh.creation.cylinder(0.04, 0.1))["detected"]


def test_container_mode_never_falls_back_to_filled_silhouette():
    report = {
        "status": "needs_review",
        "packing_proxy": {"status": "needs_review", "path": "packing_proxy/asset.json"},
        "depth_proxy": {"status": "rejected", "path": "depth_proxy/asset.json"},
    }
    assert select_candidate(report, "open_container")["status"] == "rejected"
    assert inferred_scope("蓝色杯子") == "open_container"
    assert inferred_scope("bowl_02.mp4") == "open_container"
    assert inferred_scope("cupboard") == "outer_shape"


def test_smooth_visual_retains_open_cavity_and_exact_collision_xml(tmp_path):
    pytest.importorskip("open3d")
    mesh = cup()
    mesh.export(tmp_path / "visual.obj")
    (tmp_path / "package.json").write_text(json.dumps({"cavity": cavity_probe(mesh)}))
    original = '<robot><link><inertial><mass value="0.2"/></inertial><visual><geometry><mesh filename="visual.obj"/></geometry></visual><collision><geometry><mesh filename="part.obj"/></geometry></collision></link></robot>'
    (tmp_path / "asset.urdf").write_text(original)
    report = ensure_smooth_variant(tmp_path)
    assert report["visual_smoothing"]["collision_unchanged"]
    assert report["visual_smoothing"]["cavity_check"]["detected"]
    a = ET.fromstring(original)
    b = ET.parse(tmp_path / "asset_smooth.urdf").getroot()
    assert ET.tostring(a.find("link/collision")) == ET.tostring(
        b.find("link/collision")
    )
    assert ET.tostring(a.find("link/inertial")) == ET.tostring(b.find("link/inertial"))


def test_radial_partition_covers_material_and_excludes_inner_air():
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    from packing_assets.radial_collision import convex_profile_pieces

    profile = [[0, 0], [0.04, 0], [0.04, 0.1], [0.03, 0.1], [0.03, 0.01], [0, 0.01]]
    pieces = convex_profile_pieces(profile)
    assert unary_union(pieces).symmetric_difference(Polygon(profile)).area < 1e-12
    assert not any(p.covers(Point(0.015, 0.08)) for p in pieces)
    assert all(abs(p.area - p.convex_hull.area) < 1e-12 for p in pieces)


def test_automatic_scene_rejects_filled_bowl(tmp_path):
    from packing_assets.assets import export_asset
    from packing_assets.automation import assemble

    source = tmp_path / "solid"
    export_asset(
        trimesh.creation.cylinder(0.08, 0.06), source, {"backend": "metric_visual_hull"}
    )
    result = assemble(
        [{"id": "bowl", "asset": str(source)}], tmp_path / "out", render=False
    )
    assert result["status"] == "failed" and result["exported"] == 0
    assert "observed cavity" in result["errors"][0]["error"]
