"""Create the versioned M1-B example scene using Blender's data API."""

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def create_example():
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.frame_start, scene.frame_end = 0, 12
    scene.render.fps, scene.render.fps_base = 24, 1.0
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1

    def box(name, center, dimensions, display):
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        obj = bpy.context.object
        obj.name, obj.dimensions, obj.display_type = name, dimensions, display
        obj.hide_render = True
        return obj

    scene.fluidity.domain = box("Domain", (0, 0, -0.25), (2, 1, 2), "WIRE")
    scene.fluidity.liquid = box("Initial Liquid", (0, 0, 0.25), (0.25, 0.25, 0.25), "WIRE")
    paddle = box("Animated Paddle", (-0.28, 0, -0.25), (0.1, 1.2, 2.2), "SOLID")
    paddle.keyframe_insert(data_path="location", frame=0)
    paddle.location.x = 0.08
    paddle.keyframe_insert(data_path="location", frame=12)
    from bpy_extras.anim_utils import action_get_channelbag_for_slot

    data = paddle.animation_data
    for curve in action_get_channelbag_for_slot(data.action, data.action_slot).fcurves:
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"
    scene.fluidity.paddle = paddle
    bpy.ops.object.camera_add(location=(3, -6, 2))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, -0.45)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type, camera.data.ortho_scale = "ORTHO", 2.5
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 192, 192, 100
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.frame_set(0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
    import fluidity

    fluidity.register()
    create_example()
    fluidity.playback.create_points_asset()
    bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
