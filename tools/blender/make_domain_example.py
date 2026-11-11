"""Create the water-tank authoring example in an isolated factory-startup process."""

import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
import fluidity
from fluidity import grid_asset

fluidity.register()
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
assert bpy.ops.fluidity.create_setup() == {"FINISHED"}
scene = bpy.context.scene
bpy.ops.object.camera_add(location=(3, -5, 2.5))
camera = bpy.context.object
camera.rotation_euler = (Vector((0, 0, 0.45)) - camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.type, camera.data.ortho_scale = "ORTHO", 2.5
scene.camera = camera
scene.render.resolution_x, scene.render.resolution_y = 320, 320
scene.render.resolution_percentage = 100
scene.render.engine = "CYCLES"
scene.cycles.samples = 8
scene.frame_set(0)
for selected in bpy.context.selected_objects:
    selected.select_set(False)
scene.fluidity.domain.select_set(True)
bpy.context.view_layer.objects.active = scene.fluidity.domain
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "PROPERTIES":
            area.spaces.active.context = "PHYSICS"
fluidity.playback.create_points_asset()
if grid_asset.supported():
    grid_asset.create_grid_asset()
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
