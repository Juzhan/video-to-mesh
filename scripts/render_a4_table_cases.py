"""Blender RGB-only cases: white table, one adjacent real-size dense A4 reference.

Run from the project root. Evaluation meshes and camera metadata are kept outside
the batch input. The reconstruction receives only PNG photos and board.json.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_blender_cases import scene_setup, object_case, material, save_truth
from render_daily_cases import everyday_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["pump_bottle", "keyboard", "bowl", "mug_textured"],
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    for name in args.cases:
        folder = args.out / "input" / name
        folder.mkdir(parents=True, exist_ok=True)
        settings = {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "board_sha256": hashlib.sha256(
                Path("calibration/a4-dense/board.json").read_bytes()
            ).hexdigest(),
            "board_texture_sha256": hashlib.sha256(
                Path("calibration/a4-dense/board-preview.png").read_bytes()
            ).hexdigest(),
        }
        settings_path = args.out / "evaluation_only" / name / "capture.json"
        if any(folder.glob("*.png")) and (
            not settings_path.exists()
            or json.loads(settings_path.read_text())["settings"] != settings
        ):
            raise ValueError("capture settings changed; use a fresh output directory")
        scene, camera = scene_setup(Path("calibration/a4-dense/board-preview.png"))
        scene.render.resolution_x = 1600
        scene.render.resolution_y = 1200
        scene.cycles.samples = 12
        board = bpy.data.objects["metric_board"]
        board.scale = (0.297, 0.210, 1)
        board.location.z = 0
        table = next(o for o in scene.objects if o.type == "MESH" and o != board)
        table.name = "white_matte_table"
        table.location.z = -0.0002
        table.data.materials.clear()
        table.data.materials.append(
            material("white matte tabletop", (0.85, 0.85, 0.85), False)
        )
        if name == "mug_textured":
            obj = object_case("mug")
            obj.data.materials.clear()
            obj.data.materials.append(
                material(
                    "speckled ceramic inside and outside", (0.18, 0.45, 0.65), True
                )
            )
            for p in obj.data.polygons:
                p.material_index = 0
        else:
            obj = everyday_object(name)
        # Keep at least 20 mm of clear table between the paper and the object.
        obj.location.y += 0.22 if name == "bowl" else 0.20
        bpy.context.view_layer.update()
        truth = args.out / "evaluation_only" / name
        save_truth(obj, truth)
        bpy.ops.wm.save_as_mainfile(filepath=str((truth / "capture.blend").resolve()))
        center = Vector((0, 0.11, 0.045 if name in ("keyboard", "bowl") else 0.075))
        # The same framing includes both object and paper. Upper rings see cavities.
        rings = [(0.68, 0.28), (0.62, 0.48), (0.40, 0.68), (0.18, 0.72)]
        (truth / "capture.json").write_text(
            json.dumps(
                {
                    "settings": settings,
                    "views": 48,
                    "case": name,
                    "conditions": "white matte table, one adjacent A4 dense reference, fixed object and paper",
                    "geometry_and_camera_truth": "evaluation only, not reconstruction input",
                    "material": "weak-texture plain ceramic"
                    if name == "bowl"
                    else "same textured/printed object as previous cases",
                    "object_translation_m": list(obj.location),
                },
                indent=2,
            )
        )
        index = 0
        for radius, height in rings:
            for j in range(12):
                a = 2 * math.pi * (j + 0.25) / 12
                camera.location = (
                    center.x + radius * math.cos(a),
                    center.y + radius * math.sin(a),
                    height,
                )
                camera.rotation_euler = (
                    (center - camera.location).to_track_quat("-Z", "Y").to_euler()
                )
                scene.render.filepath = str(
                    (folder / f"view_{index:03d}.png").resolve()
                )
                if not Path(scene.render.filepath).is_file():
                    bpy.ops.render.render(write_still=True)
                index += 1
        print("CASE_COMPLETE", name, index, flush=True)


if __name__ == "__main__":
    main()
