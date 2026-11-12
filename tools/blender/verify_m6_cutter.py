"""Exercise Cutter assignment and time-zero slab resolution in Blender."""

import sys
import tempfile
from pathlib import Path

import bpy
from bpy_extras.anim_utils import action_get_channelbag_for_slot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))

import fluidity  # noqa: E402
from fluidity.scene_inputs import SceneError, resolve_cutter  # noqa: E402


def expect_scene_error(operation, message):
    try:
        operation()
    except SceneError as error:
        assert error.args[0] == "cutter", error
        assert message in error.args[1], error
    else:
        raise AssertionError("Expected Cutter preflight failure")


fluidity.register()
assert bpy.ops.fluidity.create_setup() == {"FINISHED"}
scene = bpy.context.scene
scene.frame_end = 24
settings = scene.fluidity
liquid_packet = fluidity.scene_inputs(scene)

cutter = fluidity.authoring.box("Animated blade", (-0.5, 0, 0.5), (0.125, 2, 2))
cutter.keyframe_insert(data_path="location", frame=0)
cutter.location = (0.5, 0.125, 0.375)
cutter.keyframe_insert(data_path="location", frame=24)
data = cutter.animation_data
bag = action_get_channelbag_for_slot(data.action, data.action_slot)
for curve in bag.fcurves:
    curve.keyframe_points[0].interpolation = "LINEAR"

assert bpy.ops.fluidity.assign_role(role="cutter") == {"FINISHED"}
assert settings.cutter == cutter
scene.frame_set(7, subframe=0.25)
slab = resolve_cutter(scene)
assert slab.unit_normal == (1, 0, 0)
assert slab.start_center_coordinate == -0.5
assert slab.half_width == 0.0625
assert slab.wall_velocity == (1, -0.125, -0.125)
assert (scene.frame_current, scene.frame_subframe) == (7, 0.25)
expect_scene_error(lambda: fluidity.scene_inputs(scene), "elastic")
x_curve = next(curve for curve in bag.fcurves if curve.array_index == 0)
x_curve.keyframe_points[0].interpolation = "BEZIER"
expect_scene_error(lambda: resolve_cutter(scene), cutter.name)
x_curve.keyframe_points[0].interpolation = "LINEAR"
authored_scale_y = cutter.scale.y
cutter.scale.y = 0.5
expect_scene_error(lambda: resolve_cutter(scene), "across")
assert (scene.frame_current, scene.frame_subframe) == (7, 0.25)
cutter.scale.y = authored_scale_y
with tempfile.TemporaryDirectory() as directory:
    project = str(Path(directory) / "Cutter scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=project)
    bpy.ops.wm.open_mainfile(filepath=project)
    scene = bpy.context.scene
    settings = scene.fluidity
    assert settings.cutter.name == "Animated blade"
    assert resolve_cutter(scene) == slab
    settings.cutter = None
    assert fluidity.scene_inputs(scene) == liquid_packet
print("PASS: Cutter role, slab resolution, frame restoration and liquid preflight")
