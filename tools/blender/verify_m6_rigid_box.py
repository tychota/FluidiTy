"""Verify the authored Blender rigid box packet and frame-zero transform."""

import math
from pathlib import Path
import sys
import tempfile
import time

import bpy
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))

import fluidity  # noqa: E402
from fluidity.scene_inputs import SceneError, rigid_scene_packets, scene_inputs  # noqa: E402


fluidity.register()
assert bpy.ops.fluidity.create_setup() == {"FINISHED"}
scene = bpy.context.scene
settings = scene.fluidity
scene.frame_end = 4
settings.backend = "cpu"
settings.friction = 0
settings.surface_tension = 0
settings.boundary_mode = "TANK"
settings.planar_fill = True
settings.fill_height = 0.75
settings.domain = fluidity.authoring.box("Tank", (1, -1, 1), (1.75, 1.75, 1.75))
source = settings.liquid_sources.add()
source.geometry = fluidity.authoring.box("Initial water", (0.375, -0.375, 0.375), (0.25, 0.25, 0.25))
box = fluidity.authoring.box("Float", (0.5, -0.5, 0.625), (0.25, 0.5, 0.375))
box.rotation_euler.y = 0.4
assert bpy.ops.fluidity.assign_role(role="rigid_box") == {"FINISHED"}
assert settings.rigid_box == box
settings.rigid_mass = 0.5
settings.rigid_substeps = 8

scene.frame_set(2, subframe=0.25)
liquid, rigid = rigid_scene_packets(scene)
assert (scene.frame_current, scene.frame_subframe) == (2, 0.25)
assert liquid.startswith("FLUIDITY-BLENDER-4\n")
version, *values = rigid.split()
assert version == "FLUIDITY-RIGID-1"
substeps, mass, ix, iy, iz, ex, ey, ez, x, y, z, qw, qx, qy, qz, *velocities = map(float, values)
assert (substeps, mass) == (8, 0.5)
assert all(abs(actual - expected) < 1e-7 for actual, expected in zip((ex, ey, ez), (0.125, 0.25, 0.1875)))
assert (x, y, z) == (0.5, 0.625, 0.5)
assert math.isclose(ix, mass * (ey**2 + ez**2) / 3)
assert math.isclose(iy, mass * (ex**2 + ez**2) / 3)
assert math.isclose(iz, mass * (ex**2 + ey**2) / 3)
assert abs(math.hypot(qw, qx, qy, qz) - 1) < 1e-15
assert velocities == [0] * 6
for axis in (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))):
    world = box.rotation_euler.to_quaternion() @ axis
    assert (Quaternion((qw, qx, qy, qz)) @ axis - Vector((world.x, world.z, -world.y))).length < 1e-6

try:
    scene_inputs(scene)
except SceneError as error:
    assert error.args[0] == "rigid_box"
else:
    raise AssertionError("Ordinary liquid bake accepted a driven box")

box.keyframe_insert(data_path="location", frame=0)
try:
    rigid_scene_packets(scene)
except SceneError as error:
    assert error.args[0] == "rigid_box"
else:
    raise AssertionError("Driven box accepted conflicting transform animation")

print("PASS authored rigid box, native pose and inertia, and transform conflicts")

if "--" in sys.argv:
    solver = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    fluidity.jobs.BIN_DIR = solver.parent
    box.animation_data_clear()
    settings.cell_width = 1
    scene.use_gravity = False
    settings.initial_velocity = (0, 0, 0.125)
    scene.frame_end = 2
    for index, (x, y) in enumerate(((1.375, -0.375), (0.375, -1.375), (1.375, -1.375))):
        source = settings.liquid_sources.add()
        source.geometry = fluidity.authoring.box(f"Water {index}", (x, y, 0.375), (0.25, 0.25, 0.25))

    def poll_until(condition):
        deadline = time.monotonic() + 30
        while not condition() and time.monotonic() < deadline:
            fluidity.jobs.tick()
            time.sleep(0.01)
        assert condition(), (settings.message, settings.diagnostic)

    with tempfile.TemporaryDirectory(prefix="fluidity-blender-rigid-") as directory:
        root = Path(directory)
        cache = root / "Bake"
        settings.bake_cache = str(cache)
        settings.preview_cache = str(root / "Preview")
        bpy.ops.wm.save_as_mainfile(filepath=str(root / "Rigid bake.blend"))
        assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, (settings.message, settings.diagnostic)
        assert not cache.exists()
        assert bpy.ops.fluidity.bake() == {"FINISHED"}, (settings.message, settings.diagnostic)
        poll_until(lambda: settings.committed_frame == 1 or not fluidity.jobs.is_running(scene))
        assert settings.committed_frame == 1, settings.diagnostic
        assert bpy.ops.fluidity.cancel() == {"FINISHED"}
        poll_until(lambda: not fluidity.jobs.is_running(scene))
        first = fluidity.jobs.rigid_checkpoint(cache, 1)
        second = fluidity.jobs.rigid_checkpoint(cache, 2)
        first_bytes = first.read_bytes()
        assert not second.exists()

        project = root / "Interrupted rigid bake.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(project))
        bpy.ops.wm.open_mainfile(filepath=str(project))
        scene = bpy.context.scene
        settings = scene.fluidity
        settings.rigid_mass = 1
        assert bpy.ops.fluidity.resume() == {"CANCELLED"}
        assert "settings changed" in settings.message
        settings.rigid_mass = 0.5
        assert bpy.ops.fluidity.resume() == {"FINISHED"}, (settings.message, settings.diagnostic)
        poll_until(lambda: not fluidity.jobs.is_running(scene))
        assert settings.committed_frame == 2, settings.diagnostic
        assert first.read_bytes() == first_bytes
        receipt = fluidity.jobs.inspect_rigid(cache, 2)
        assert receipt["frame"] == 2
        scene.frame_set(2)
        assert bpy.ops.fluidity.inspect() == {"FINISHED"}, settings.message
        assert settings.inspect_frame == 2 and "Rigid Box" in settings.point_detail
        replay = root / "Replay.chk"
        paths = tuple(cache / name for name in ("scene.flrs", "state.flst", "rigid.txt"))
        fluidity.jobs.checked("simulate", "rigid-frame", *paths, first, replay, event="rigid_frame")
        assert replay.read_bytes() == second.read_bytes()
        box = settings.rigid_box
        authored = box.matrix_world.copy()
        before = set(scene.objects)
        assert bpy.ops.fluidity.attach() == {"FINISHED"}, settings.message
        display, = set(scene.objects) - before
        assert display.data == box.data and box.animation_data is None
        display_name = display.name
        x, y, z = receipt["position"]
        expected = Vector((x, -z, y))
        scene.frame_set(2)
        assert (display.location - expected).length < 1e-7
        axes = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
        expected_rotation = (axes @ Quaternion(receipt["orientation"]).to_matrix()).to_quaternion()
        assert abs(display.rotation_quaternion.dot(expected_rotation)) > 1 - 1e-6
        scene.frame_set(0)
        assert (display.location - authored.translation).length < 1e-7
        assert (display.location - expected).length > 1e-7
        assert box.matrix_world == authored

        baked_project = root / "Rigid motion.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(baked_project))
        bpy.ops.wm.open_mainfile(filepath=str(baked_project))
        scene = bpy.context.scene
        scene.frame_set(2)
        assert (scene.objects[display_name].location - expected).length < 1e-7
        assert scene.fluidity.rigid_box.matrix_world == authored

    print("PASS rigid bake, byte-identical resume, and saved native timeline playback")
