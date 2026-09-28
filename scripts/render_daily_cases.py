"""New everyday-object RGB capture suite. Truth is stored separately for evaluation only."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_blender_cases import material, lathe, scene_setup, save_truth


def box(name, size, location, mat, bevel=0.001):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new("Rounded manufactured edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.materials.append(mat)
    return obj


def text_label(text, location, size, mat):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.data.body = text
    obj.data.size = size
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.extrude = 0.00003
    obj.data.materials.append(mat)
    bpy.ops.object.convert(target="MESH")
    return bpy.context.object


def join(parts, name):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    return obj


def everyday_object(name):
    dark = material("matte charcoal", (0.045, 0.05, 0.058), False)
    pale = material("light plastic", (0.65, 0.69, 0.72), False)
    ink = material("printed white legends", (0.95, 0.95, 0.95), False)
    if name == "keyboard":
        # Full-size wireless keyboard; deliberate steps and key gaps, not a plain box.
        parts = [box("keyboard base", (0.44, 0.145, 0.018), (0, 0, 0.009), dark, 0.004)]
        for row in range(6):
            for col in range(22):
                x = (col - 10.5) * 0.0193
                y = (row - 2.5) * 0.020
                h = 0.009 + row * 0.0012
                keymat = pale if (row == 0 or col > 17) else dark
                parts.append(
                    box("key", (0.017, 0.017, h), (x, y, 0.018 + h / 2), keymat, 0.001)
                )
                legend = chr(65 + (row * 22 + col) % 26)
                parts.append(text_label(legend, (x, y, 0.018 + h + 0.0001), 0.006, ink))
        return join(parts, name)
    if name == "remote":
        parts = [
            box("remote housing", (0.052, 0.17, 0.016), (0, 0, 0.008), dark, 0.005)
        ]
        for row in range(8):
            for col in range(3):
                x = (col - 1) * 0.014
                y = (row - 3.5) * 0.018
                mat = (
                    material("red power", (0.65, 0.02, 0.01), False)
                    if row == 7 and col == 0
                    else pale
                )
                parts.append(
                    box("button", (0.010, 0.012, 0.0035), (x, y, 0.0172), mat, 0.002)
                )
                parts.append(
                    text_label(str((row * 3 + col) % 10), (x, y, 0.0191), 0.006, dark)
                )
        return join(parts, name)
    if name == "mouse":
        obj = lathe(
            "mouse shell",
            [
                (0, 0),
                (0.040, 0),
                (0.055, 0.003),
                (0.060, 0.012),
                (0.059, 0.023),
                (0.048, 0.037),
                (0.028, 0.044),
                (0, 0.046),
            ],
            yscale=0.59,
        )
        obj.data.materials.append(pale)
        # Mostly smooth/weak texture, with only a wheel and a small printed logo.
        parts = [
            obj,
            box("wheel", (0.022, 0.007, 0.004), (-0.015, 0, 0.043), dark, 0.002),
            text_label("M", (0.025, 0, 0.0425), 0.009, dark),
        ]
        return join(parts, name)
    if name == "pump_bottle":
        obj = lathe(
            "labelled bottle",
            [
                (0, 0),
                (0.035, 0),
                (0.040, 0.008),
                (0.040, 0.145),
                (0.030, 0.165),
                (0.013, 0.176),
                (0.013, 0.19),
                (0, 0.19),
            ],
            yscale=0.70,
        )
        obj.data.materials.append(material("mottled printed label", (0.04, 0.36, 0.25)))
        parts = [
            obj,
            box("pump stem", (0.012, 0.012, 0.024), (0, 0, 0.198), dark, 0.001),
            box("pump head", (0.036, 0.026, 0.012), (0.006, 0, 0.216), dark, 0.004),
            box("pump nozzle", (0.038, 0.010, 0.008), (0.034, 0, 0.215), dark, 0.002),
        ]
        return join(parts, name)
    if name == "bowl":
        # Thin bowl with flared rim and foot; pure color is an intentional difficult case.
        obj = lathe(
            "plain bowl",
            [
                (0, 0),
                (0.040, 0),
                (0.044, 0.006),
                (0.047, 0.010),
                (0.057, 0.018),
                (0.070, 0.033),
                (0.081, 0.050),
                (0.090, 0.070),
                (0.094, 0.075),
                (0.094, 0.079),
                (0.089, 0.079),
                (0.087, 0.073),
                (0.076, 0.050),
                (0.065, 0.032),
                (0.052, 0.018),
                (0.035, 0.011),
                (0, 0.011),
            ],
            segments=128,
        )
        obj.data.materials.append(
            material("plain blue ceramic", (0.08, 0.23, 0.50), False)
        )
        return obj
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["keyboard", "remote", "mouse", "pump_bottle", "bowl"],
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    for name in args.cases:
        folder = args.out / name
        images = folder / "images"
        images.mkdir(parents=True, exist_ok=True)
        if any(images.glob("*.png")):
            raise ValueError("do not overwrite a capture; use a fresh folder")
        scale = 2 if name == "keyboard" else 1
        board = Path("calibration/a1" if scale == 2 else "calibration/a3-daily")
        scene, camera = scene_setup(board / "board-preview.png")
        if scale == 2:
            bpy.data.objects["metric_board"].scale.x *= 2
            bpy.data.objects["metric_board"].scale.y *= 2
            for light in [o for o in scene.objects if o.type == "LIGHT"]:
                light.location *= 2
                light.data.energy *= 4
                light.data.size *= 2
                light.rotation_euler = (
                    (Vector((0, 0, 0.06)) - light.location)
                    .to_track_quat("-Z", "Y")
                    .to_euler()
                )
        obj = everyday_object(name)
        save_truth(obj, folder / "evaluation_only")
        bpy.ops.wm.save_as_mainfile(
            filepath=str((folder / "evaluation_only/scene.blend").resolve())
        )
        rings = (
            [(0.8, 0.18), (0.8, 0.5), (0.50, 0.98)]
            if scale == 2
            else [(0.48, 0.17), (0.48, 0.36), (0.27, 0.64)]
        )
        index = 0
        for radius, height in rings:
            for j in range(12):
                a = 2 * math.pi * (j + 0.25) / 12
                camera.location = (radius * math.cos(a), radius * math.sin(a), height)
                target = (
                    0.03 if name in ["keyboard", "remote", "mouse", "bowl"] else 0.075
                )
                camera.rotation_euler = (
                    (Vector((0, 0, target)) - camera.location)
                    .to_track_quat("-Z", "Y")
                    .to_euler()
                )
                scene.render.filepath = str(
                    (images / f"view_{index:03d}.png").resolve()
                )
                bpy.ops.render.render(write_still=True)
                index += 1
        (folder / "capture.json").write_text(
            json.dumps(
                {
                    "case": name,
                    "rgb_views": index,
                    "board": str((board / "board.json").resolve()),
                    "board_scale": scale,
                    "script_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "scope": "independent RGB-only input; object dimensions/poses/depth are evaluation-only",
                },
                indent=2,
            )
        )
        print("CASE_COMPLETE", name, flush=True)


if __name__ == "__main__":
    main()
