"""Run with blender --background --python this_file -- --out ... --board ...

RGB images are the only reconstruction input. Mesh truth lives in a separate evaluation directory.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def material(name, base, textured=True):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base, 1)
    bsdf.inputs["Roughness"].default_value = 0.8
    if textured:
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 35
        tex.inputs["Detail"].default_value = 2
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.3
        ramp.color_ramp.elements[0].color = (*[c * 0.12 for c in base], 1)
        ramp.color_ramp.elements[1].position = 0.7
        ramp.color_ramp.elements[1].color = (*[min(1, c * 1.5 + 0.1) for c in base], 1)
        links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def lathe(name, profile, segments=96, yscale=1):
    verts, faces = [], []
    for radius, z in profile:
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append((radius * math.cos(a), radius * math.sin(a) * yscale, z))
    for i in range(len(profile) - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # Weld the axis poles and recalculate surface orientation.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-7)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def object_case(name):
    if name == "book":
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.014))
        obj = bpy.context.object
        obj.name = "book"
        obj.dimensions = (0.18, 0.12, 0.028)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bevel = obj.modifiers.new("Rounded edges", "BEVEL")
        bevel.width = 0.001
        bevel.segments = 2
        bpy.ops.object.modifier_apply(modifier=bevel.name)
        obj.data.materials.append(material("Printed blue cover", (0.08, 0.27, 0.7)))
    elif name == "bottle":
        obj = lathe(
            "bottle",
            [
                (0, 0),
                (0.033, 0),
                (0.038, 0.008),
                (0.038, 0.15),
                (0.027, 0.175),
                (0.016, 0.185),
                (0.016, 0.2),
                (0.020, 0.2),
                (0.020, 0.225),
                (0, 0.225),
            ],
            yscale=0.75,
        )
        obj.data.materials.append(material("Textured orange bottle", (0.8, 0.3, 0.06)))
    elif name == "mug":
        obj = lathe(
            "mug",
            [
                (0, 0),
                (0.036, 0),
                (0.04, 0.005),
                (0.04, 0.105),
                (0.033, 0.105),
                (0.033, 0.009),
                (0, 0.009),
            ],
        )
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.031,
            minor_radius=0.008,
            major_segments=64,
            minor_segments=16,
            location=(0.044, 0, 0.057),
            rotation=(math.pi / 2, 0, 0),
        )
        handle = bpy.context.object
        bpy.context.view_layer.objects.active = obj
        union = obj.modifiers.new("Solid handle union", "BOOLEAN")
        union.operation = "UNION"
        union.object = handle
        bpy.ops.object.modifier_apply(modifier=union.name)
        bpy.data.objects.remove(handle, do_unlink=True)
        # Intentionally include a difficult, untextured cup alongside the textured cases.
        obj.data.materials.clear()
        obj.data.materials.append(
            material("Plain white ceramic", (0.8, 0.8, 0.8), False)
        )
        for polygon in obj.data.polygons:
            polygon.material_index = 0
    else:
        raise ValueError(name)
    return obj


def save_truth(obj, folder):
    folder.mkdir(parents=True, exist_ok=True)
    deps = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(deps)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
    lines = ["v %.9f %.9f %.9f" % tuple(v) for v in vertices]
    lines += [
        "f %d %d %d" % tuple(i + 1 for i in tri.vertices) for tri in mesh.loop_triangles
    ]
    (folder / "truth.obj").write_text("\n".join(lines) + "\n")
    bounds = [
        [min(v[i] for v in vertices) for i in range(3)],
        [max(v[i] for v in vertices) for i in range(3)],
    ]
    (folder / "truth.json").write_text(
        json.dumps(
            {
                "units": "m",
                "up_axis": "z",
                "bounds_m": bounds,
                "extents_m": [bounds[1][i] - bounds[0][i] for i in range(3)],
                "use": "evaluation only; not read by reconstruction",
            },
            indent=2,
        )
    )
    evaluated.to_mesh_clear()


def scene_setup(board):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    # CPU rendering is independent of CUDA reconstruction and works headlessly.
    scene.cycles.device = "CPU"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = -3.5
    scene.world.color = (0.25, 0.25, 0.25)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.0003))
    plane = bpy.context.object
    plane.name = "metric_board"
    plane.scale = (0.420, 0.297, 1)
    mat = bpy.data.materials.new("Calibration print")
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = 1
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(board.resolve()))
    tex.interpolation = "Closest"
    mat.node_tree.links.new(tex.outputs["Color"], shader.inputs["Base Color"])
    plane.data.materials.append(mat)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.001))
    bpy.context.object.data.materials.append(
        material("Neutral table", (0.22, 0.23, 0.25), False)
    )
    for location, power, size in [
        ((0.25, -0.3, 1.0), 55, 0.7),
        ((-0.45, 0.2, 0.7), 35, 0.6),
    ]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = power
        light.data.shape = "DISK"
        light.data.size = size
        light.rotation_euler = (
            (Vector((0, 0, 0.08)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 36
    camera.data.sensor_width = 36
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10
    scene.camera = camera
    return scene, camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["book", "bottle", "mug"])
    parser.add_argument("--views-per-ring", type=int, default=16)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    for name in args.cases:
        images = args.out / name / "images"
        images.mkdir(parents=True, exist_ok=True)
        settings = {
            "case": name,
            "views_per_ring": args.views_per_ring,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "board_sha256": hashlib.sha256(args.board.read_bytes()).hexdigest(),
        }
        settings_file = images.parent / "capture-settings.json"
        if any(images.glob("*.png")):
            if (
                not settings_file.exists()
                or json.loads(settings_file.read_text()) != settings
            ):
                raise RuntimeError(
                    f"Existing render settings differ or are unknown: {images}. Choose a new output directory."
                )
        settings_file.write_text(json.dumps(settings, indent=2))
        scene, camera = scene_setup(args.board)
        obj = object_case(name)
        save_truth(obj, args.out / name / "evaluation_only")
        bpy.ops.wm.save_as_mainfile(
            filepath=str((args.out / name / "evaluation_only/scene.blend").resolve())
        )
        index = 0
        for radius, height in [(0.48, 0.26), (0.48, 0.48), (0.27, 0.65)]:
            for j in range(args.views_per_ring):
                angle = 2 * math.pi * j / args.views_per_ring
                camera.location = (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    height,
                )
                camera.rotation_euler = (
                    (Vector((0, 0, 0.07)) - camera.location)
                    .to_track_quat("-Z", "Y")
                    .to_euler()
                )
                scene.render.filepath = str(
                    (images / f"view_{index:03d}.png").resolve()
                )
                if not Path(scene.render.filepath).exists():
                    bpy.ops.render.render(write_still=True)
                index += 1
        print(f"CASE_COMPLETE {name} {index} RGB frames", flush=True)


if __name__ == "__main__":
    main()
