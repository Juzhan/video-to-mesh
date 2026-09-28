"""External convex occupancy assets; explicit scope, never certified inner geometry."""

import json
from pathlib import Path
import numpy as np

from .assets import load_mesh, export_asset
from .quality import check_views, surface_release_gate


def convex_gate(mesh, views):
    gate = surface_release_gate(mesh, views)
    if views.get("ambiguous_mask_views", 0) > 0:
        gate["reasons"].append("fragmented_object_segmentation_requires_review")
    if views["p90_under_mm"] > 3:
        gate["reasons"].append("visible_extremities_underestimated")
    if views["p90_over_mm"] > 5:
        gate["reasons"].append("outer_envelope_excess_too_large")
    gate["policy"].update({"p90_view_under_mm_max": 3, "p90_view_over_mm_max": 5})
    gate["status"] = "rejected" if gate["reasons"] else "needs_review"
    return gate


def build_outer_convex(folder: Path, work: Path):
    options = []
    import open3d as o3d
    from .top_plane import fit_supported_top, cap_hull

    cloud = o3d.io.read_point_cloud(str(work / "masked_object_points.ply"))
    upper_plane, upper_report = fit_supported_top(
        np.asarray(cloud.points), np.asarray(cloud.normals)
    )
    for kind, base in [
        ("surface", folder),
        ("regularized_box", folder / "regularized_box"),
        ("packing_proxy", folder / "packing_proxy"),
    ]:
        if not (base / "asset.json").exists():
            continue
        asset = json.loads((base / "asset.json").read_text())
        mesh = load_mesh(base / "visual.obj")
        mesh.apply_translation(-np.array(asset["centering_translation_m"]))
        try:
            hull = mesh.convex_hull
            if upper_plane is not None:
                hull = cap_hull(hull, upper_plane)
        except Exception:
            continue
        views = check_views(hull, work, convex_reference=True)
        gate = convex_gate(hull, views)
        options.append((kind, hull, views, gate, asset))
    if not options:
        raise ValueError("no geometry for an external convex candidate")
    # Choose by photo consistency only. Blender truth is never used in this decision.
    ranked = sorted(
        options,
        key=lambda v: (v[3]["status"] == "needs_review", v[2]["median_iou"]),
        reverse=True,
    )
    kind, hull, views, gate, source = ranked[0]
    out = folder / "outer_convex"
    report = export_asset(
        hull,
        out,
        {
            "backend": "external_convex_hull",
            "source_candidate": kind,
            "scale_source": source["provenance"].get("scale_source"),
            "calibration": source["provenance"].get("calibration"),
            "geometry_scope": "external convex occupancy only; all cavities, handle gaps and concavities filled",
            "source_assumptions": source.get("review_reasons", []),
            "upper_plane_constraint": upper_report,
            "upper_plane_applied": upper_plane is not None,
        },
    )
    report["status"] = gate["status"]
    report["quality"] = gate
    report["review_reasons"] += [
        "convex_envelope_is_not_exact_contact_geometry",
        "visible_extremities_and_height_require_review",
    ]
    (out / "asset.json").write_text(json.dumps(report, indent=2))
    alternatives = [
        {
            "source": k,
            "gate": g,
            "median_iou": v["median_iou"],
            "p90_under_mm": v["p90_under_mm"],
            "p90_over_mm": v["p90_over_mm"],
        }
        for k, m, v, g, a in options
    ]
    (out / "quality.json").write_text(
        json.dumps(
            {
                "view_consistency": views,
                "release_gate": gate,
                "alternatives": alternatives,
            },
            indent=2,
        )
    )
    return {
        "status": gate["status"],
        "path": "outer_convex/asset.json",
        "scope": "external_convex_occupancy",
    }
