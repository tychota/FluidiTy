"""Exercise Cutter assignment and time-zero slab resolution in Blender."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
from bpy_extras.anim_utils import action_get_channelbag_for_slot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))

import fluidity  # noqa: E402
from fluidity.scene_inputs import SceneError, cutter_frame_packet, resolve_cutter  # noqa: E402


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

if "--" in sys.argv:
    solver = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    scene.frame_end = 2
    scene.frame_set(0)
    settings.domain.location = (1.0, -0.5, 0.5)
    settings.domain.dimensions = (4, 3, 3)
    settings.elastic_solid = fluidity.authoring.box("Elastic seed box", (1.5, -0.5, 0.5), (3, 1, 1))
    settings.cell_width = 1
    settings.elastic_density = 2
    settings.elastic_shear_modulus = 2
    settings.elastic_first_lame_parameter = 1
    settings.elastic_substeps = 1

    blade = fluidity.authoring.box("Bake blade", (0.359375, -0.5, 0.5), (0.125, 4, 4))
    blade.keyframe_insert(data_path="location", frame=0)
    blade.location.x = 0.421875
    blade.keyframe_insert(data_path="location", frame=2)
    data = blade.animation_data
    bag = action_get_channelbag_for_slot(data.action, data.action_slot)
    for curve in bag.fcurves:
        curve.keyframe_points[0].interpolation = "LINEAR"
    assert bpy.ops.fluidity.assign_role(role="cutter") == {"FINISHED"}
    packet_text = cutter_frame_packet(scene)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        packet = root / "Cutter inputs.txt"
        first = root / "Frame one.chk"
        second = root / "Frame two.chk"
        replay = root / "Frame two replay.chk"
        rejected = root / "Rejected frame.chk"
        packet.write_text(packet_text, encoding="ascii", newline="\n")

        def run(previous, output):
            return subprocess.run([solver, "cutter-frame", packet, previous, output], capture_output=True, check=False)

        result = run("-", first)
        assert result.returncode == 0, result.stderr
        first_frame = json.loads(result.stdout)
        assert first_frame["accepted_step"] == 1
        assert first_frame["blade_work"] == 0
        result = run(first, second)
        assert result.returncode == 0, result.stderr
        second_frame = json.loads(result.stdout)
        assert second_frame["accepted_step"] == 2
        assert second_frame["particle_impulse"] == [0, 0, 0], second_frame
        assert second_frame["blade_work"] == 0, second_frame
        assert second_frame["sides"] == [{"particle_id": 1, "sign": 1}, {"particle_id": 2, "sign": 1}, {"particle_id": 3, "sign": 1}]
        result = run(first, replay)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == second_frame
        assert replay.read_bytes() == second.read_bytes()
        assert run(first, second).returncode == 0

        first_bytes = first.read_bytes()
        settings.elastic_density = 3
        packet.write_text(cutter_frame_packet(scene), encoding="ascii", newline="\n")
        result = run(first, rejected)
        assert result.returncode != 0 and b"different scene" in result.stderr, result.stderr
        assert not rejected.exists() and first.read_bytes() == first_bytes

        settings.elastic_density = 2
        for curve in bag.fcurves:
            if curve.array_index == 0:
                for key in curve.keyframe_points:
                    key.co.y += 0.03125
                curve.update()
        packet.write_text(cutter_frame_packet(scene), encoding="ascii", newline="\n")
        before_impact = root / "Before impact.chk"
        assert run("-", before_impact).returncode == 0
        before_impact_bytes = before_impact.read_bytes()
        impact_output = root / "Impact.chk"
        impact_replay = root / "Impact replay.chk"
        result = run(before_impact, impact_output)
        assert result.returncode == 0, result.stderr
        impact = json.loads(result.stdout)
        assert impact["particle_impulse"] == [1.5, 0, 0], impact
        assert impact["blade_work"] == 1.125, impact
        assert impact["sides"] == first_frame["sides"]
        assert abs(impact["cut_work"]) <= 1e-6
        result = run(before_impact, impact_replay)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == impact
        assert impact_output.read_bytes() == impact_replay.read_bytes()
        assert before_impact.read_bytes() == before_impact_bytes

    print("PASS: Blender Cutter packet, persistent sides, byte-identical restart and accepted impact")
