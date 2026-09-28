"""Blender: actual reconstructed assets at unchanged metric scale, with a 10 cm grid."""

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import bpy
from mathutils import Vector


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Roughness"].default_value = 0.7
    return mat


def text_obj(body, position, size, mat):
    curve = bpy.data.curves.new(body, "FONT")
    curve.body = body
    curve.size = size
    curve.align_x = "CENTER"
    obj = bpy.data.objects.new(body, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.data.materials.append(mat)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--smooth", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    root = args.scene.resolve().parent
    data = json.loads(args.scene.read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1050
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.world.color = (0.22, 0.22, 0.22)
    colors = [
        (0.13, 0.42, 0.67),
        (0.78, 0.35, 0.13),
        (0.22, 0.58, 0.39),
        (0.53, 0.32, 0.69),
    ]
    dark = material("Text and 10 cm grid", (0.13, 0.17, 0.22))
    floor_mat = material("Warm floor", (0.84, 0.85, 0.83))
    limits = []
    audit = []
    for index, item in enumerate(data["items"]):
        selected = (
            (item.get("smooth_visual") or item["visual"])
            if args.smooth
            else item["visual"]
        )
        bpy.ops.wm.obj_import(
            filepath=str(root / selected), forward_axis="Y", up_axis="Z"
        )
        obj = bpy.context.object
        obj.name = item["id"]
        obj.location = item["position_m"]
        obj.data.materials.clear()
        obj.data.materials.append(material(item["id"], colors[index % len(colors)]))
        if args.smooth:
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
        obj["source"] = item["visual"]
        obj["unit"] = "m"
        obj["mass_source"] = item["mass_source"]
        obj["quality_status"] = item.get("quality_status", "needs_review")
        expected = item["dimensions_m"]
        actual = list(obj.dimensions)
        if max(abs(a - b) for a, b in zip(actual, expected)) > (
            0.004 if args.smooth else 0.002
        ):
            raise ValueError(
                f"Blender import changed dimensions of {item['id']}: {actual} vs {expected}"
            )
        audit.append(
            {"id": item["id"], "dimensions_m": actual, "object_scale": list(obj.scale)}
        )
        bounds = [Vector(b) + obj.location for b in item["bounds_m"]]
        limits.extend(
            Vector(v) + obj.location for v in product(*zip(*item["bounds_m"]))
        )
        center_x = (bounds[0].x + bounds[1].x) / 2
        # Text lies on the floor in world XY, preserving a readable scale diagram.
        text_obj(item["id"], (center_x, bounds[0].y - 0.020, 0.001), 0.011, dark)
        dim = " x ".join(str(round(v * 1000)) for v in expected) + " mm"
        text_obj(dim, (center_x, bounds[0].y - 0.034, 0.001), 0.0075, dark)
        if item.get("quality_status") == "retained_unvalidated":
            text_obj(
                "SHAPE CHECK FAILED - RETAINED",
                (center_x, bounds[0].y - 0.047, 0.001),
                0.006,
                dark,
            )
    lo = Vector(tuple(min(v[a] for v in limits) for a in range(3)))
    hi = Vector(tuple(max(v[a] for v in limits) for a in range(3)))
    x0, y0 = -0.10, -0.13
    x1, y1 = hi.x + 0.10, hi.y + 0.15
    bpy.ops.mesh.primitive_plane_add(
        size=2, location=((x0 + x1) / 2, (y0 + y1) / 2, -0.002)
    )
    floor = bpy.context.object
    floor.name = "Floor - 10 cm grid"
    floor.scale = ((x1 - x0) / 2, (y1 - y0) / 2, 1)
    floor.data.materials.append(floor_mat)
    vertices = []
    edges = []
    for axis, start, end, other0, other1 in [(0, x0, x1, y0, y1), (1, y0, y1, x0, x1)]:
        for tick in range(math.ceil(start * 10), math.floor(end * 10) + 1):
            a = [tick * 0.1, other0, 0]
            b = [tick * 0.1, other1, 0]
            if axis == 1:
                a[0], a[1] = a[1], a[0]
                b[0], b[1] = b[1], b[0]
            vertices.extend([a, b])
            edges.append((len(vertices) - 2, len(vertices) - 1))
    curve = bpy.data.curves.new("Grid lines", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.00025
    curve.bevel_resolution = 0
    for a, b in edges:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (*vertices[a], 1)
        spline.points[1].co = (*vertices[b], 1)
    grid = bpy.data.objects.new("10 cm grid", curve)
    bpy.context.collection.objects.link(grid)
    curve.materials.append(dark)
    text_obj(
        "METRIC RECONSTRUCTIONS  |  grid = 10 cm",
        ((x0 + x1) / 2, y0 + 0.025, 0.001),
        0.014,
        dark,
    )
    target = Vector(((x0 + x1) / 2, (y0 + y1) / 2, hi.z / 2))
    span = max(x1 - x0, y1 - y0, hi.z * 1.6)
    bpy.ops.object.camera_add(location=target + Vector((0.10, -1.15, 1.45)) * span)
    cam = bpy.context.object
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = span * 1.3
    scene.camera = cam
    frame = limits + [Vector((x, y, 0)) for x, y in product((x0, x1), (y0, y1))]
    rotation = cam.rotation_euler.to_matrix()
    projected = [rotation.transposed() @ (v - cam.location) for v in frame]
    left, right = min(v.x for v in projected), max(v.x for v in projected)
    bottom, top = min(v.y for v in projected), max(v.y for v in projected)
    cam.location += rotation @ Vector(((left + right) / 2, (bottom + top) / 2, 0))
    cam.data.ortho_scale = max(right - left, (top - bottom) * 1600 / 1050) * 1.08
    bpy.ops.object.light_add(
        type="AREA", location=target + Vector((-0.3, -0.4, 1.5)) * span
    )
    bpy.context.object.data.energy = 40 * span * span
    bpy.context.object.data.size = span * 1.5
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    scene["notes"] = (
        "RGB reconstructions, true shared metric scale. No per-object resizing. See scene.json for quality and assumed masses."
    )
    suffix = "-smooth" if args.smooth else ""
    bpy.ops.wm.save_as_mainfile(filepath=str(root / f"scene{suffix}.blend"))
    scene.render.filepath = str(root / f"scene{suffix}.png")
    bpy.ops.render.render(write_still=True)
    (root / f"blender{suffix}-check.json").write_text(
        json.dumps({"status": "passed", "units": "m", "items": audit}, indent=2)
    )


if __name__ == "__main__":
    main()
