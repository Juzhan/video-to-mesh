"""Independently audit OBJ topology in Blender and show the last mesh underside."""

import argparse, hashlib, json, sys
from pathlib import Path
import bpy, bmesh
from mathutils import Vector

p = argparse.ArgumentParser()
p.add_argument("--mesh", type=Path, action="append", required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
a.out.mkdir(parents=True, exist_ok=True)
rows = []
for path in a.mesh:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=str(path.resolve()), forward_axis="Y", up_axis="Z")
    obj = bpy.context.object
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    rows.append(
        {
            "file": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "vertices": len(bm.verts),
            "faces": len(bm.faces),
            "boundary_edges": sum(e.is_boundary for e in bm.edges),
            "nonmanifold_edges": sum(not e.is_manifold for e in bm.edges),
            "noncontiguous_edges": sum(
                e.is_manifold and not e.is_contiguous for e in bm.edges
            ),
            "signed_volume_m3": bm.calc_volume(signed=True),
            "dimensions_m": list(obj.dimensions),
        }
    )
    bm.free()
(a.out / "blender-topology.json").write_text(
    json.dumps({"backend": "Blender bmesh, reimported OBJ", "meshes": rows}, indent=2)
)
obj.name = a.mesh[-1].stem
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.resolution_x = 1400
scene.render.resolution_y = 850
scene.render.resolution_percentage = 100
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
    0.18,
    0.18,
    0.18,
    1,
)
mat = bpy.data.materials.new("Verified exported mesh")
mat.diffuse_color = (0.18, 0.5, 0.7, 1)
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs[
    "Base Color"
].default_value = mat.diffuse_color
mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.65
obj.data.materials.clear()
obj.data.materials.append(mat)
for f in obj.data.polygons:
    f.use_smooth = True
center = sum((Vector(v) for v in obj.bound_box), Vector()) / 8
span = max(obj.dimensions)
bpy.ops.object.camera_add(location=center + Vector((0.25, -0.4, -1.1)) * span)
cam = bpy.context.object
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.type = "ORTHO"
cam.data.ortho_scale = span * 1.3
scene.camera = cam
bpy.ops.object.light_add(type="AREA", location=center + Vector((-0.3, -0.4, -1)) * span)
bpy.context.object.data.energy = 30
bpy.context.object.data.size = span
scene["inspection"] = (
    "Underside of final OBJ; no closing geometry added by this script."
)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
bpy.ops.wm.save_as_mainfile(filepath=str((a.out / "underside.blend").resolve()))
scene.render.filepath = str((a.out / "underside.png").resolve())
bpy.ops.render.render(write_still=True)
