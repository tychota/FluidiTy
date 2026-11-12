"""Verify separate Blender objects and materials from an existing mixed point cache."""

import sys
from pathlib import Path

import bpy

repo, build, cache, path = map(Path, sys.argv[sys.argv.index("--") + 1:])
sys.path.insert(0, str(repo / "blender"))
import fluidity
from fluidity import jobs, playback

fluidity.register()
jobs.BIN_DIR = build
scene = bpy.context.scene
scene.render.fps = 24
scene.render.fps_base = 1
last = playback.committed_last(cache)
bpy.ops.wm.save_as_mainfile(filepath=str(path))
assert playback.attach(scene, cache) == last
water, snow = scene.fluidity.playback_object, scene.fluidity.snow_playback_object
assert water != snow
for obj, expected in ((water, "FluidiTy Water Points"), (snow, "FluidiTy Snow Points")):
    modifier = obj.modifiers[-1]
    assert getattr(modifier.properties.inputs, modifier.node_group["fluidity_material_socket"]).value.name == expected
assert playback.attach(scene, cache) == last
assert (scene.fluidity.playback_object, scene.fluidity.snow_playback_object) == (water, snow)
bpy.ops.wm.save_as_mainfile(filepath=str(path))
bpy.ops.wm.open_mainfile(filepath=str(path))
scene = bpy.context.scene
scene.frame_set(last)
for obj in (scene.fluidity.playback_object, scene.fluidity.snow_playback_object):
    obj.modifiers[-1].show_viewport = False
graph = bpy.context.evaluated_depsgraph_get()
points = playback.inspection(cache, last)["points"]
for substance, obj in (("water", scene.fluidity.playback_object), ("snow", scene.fluidity.snow_playback_object)):
    count = sum(point["substance"] == substance for point in points)
    assert count > 0 and len(obj.evaluated_get(graph).data.points) == count
print("FLUIDITY_M5_TWO_CHANNELS_PASS")
