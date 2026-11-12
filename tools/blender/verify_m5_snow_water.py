"""Bake, resume, attach, and reopen an authored mixed snow and water scene in Blender."""

import json
import subprocess
import sys
from pathlib import Path

import bpy
from bpy_extras.anim_utils import action_get_channelbag_for_slot


def box(name, center, dimensions):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    result = bpy.context.object
    result.name = name
    result.scale = dimensions
    return result


def run(executable, *arguments):
    process = subprocess.run([str(executable), *map(str, arguments)], capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, (process.stdout, process.stderr)
    return [json.loads(line) for line in process.stdout.splitlines()]


def verify(repo, binaries, output, backend):
    sys.path.insert(0, str(repo / "blender"))
    import fluidity
    from fluidity import jobs, playback
    from fluidity.scene_inputs import scene_inputs

    fluidity.register()
    jobs.BIN_DIR = binaries
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, 20
    scene.render.fps = 24
    scene.render.fps_base = 1
    scene.use_gravity = False
    settings = scene.fluidity
    settings.boundary_mode = "SNOW_PLATEN"
    settings.cell_width = 0.5
    settings.backend = backend
    settings.surface_tension = 0
    settings.maximum_step = 0.01
    settings.initial_velocity = (0.05, 0, 0)
    settings.domain = box("Domain", (2, 0, 0.75), (4, 3, 1.5))
    settings.snow_mound = box("Snow Mound", (2.125, 0, 0.45), (1.0, 1.0, 0.9))
    settings.collider = box("Animated Platen", (2.125, 0, 1.15), (1.25, 1.25, 0.2))
    for frame, height in ((0, 1.15), (20, 0.8)):
        settings.collider.location.z = height
        settings.collider.keyframe_insert(data_path="location", frame=frame)
    animation = settings.collider.animation_data
    for curve in action_get_channelbag_for_slot(animation.action, animation.action_slot).fcurves:
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"
    assert bpy.ops.fluidity.liquid_source() == {"FINISHED"}
    source = settings.liquid_sources[0]
    source.geometry = box("Water Emitter", (1.25, 0.25, 0.25), (0.24, 0.24, 0.24))
    source.emission_rate = 3
    source.emitted_volume = 0.125
    output.mkdir()
    blend = output / "snow-water.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    resolved, initial = jobs.prepare(scene_inputs(scene), backend, output)
    cache = output / "cache"
    solver = jobs.binary("simulate")
    command = ["run", "--stage", "simulate", "--scene", resolved, "--cache", cache]
    result = run(solver, *command, "--initial-state", initial, "--frame-end", 20)[-1]
    assert result["source_next_child_ordinals"] == [2]
    before = {p.name: p.read_bytes() for p in (cache / "blobs").glob("*.flsi")}
    restarted = run(solver, *command, "--resume", "--frame-start", 10, "--frame-end", 20)[-1]
    assert result["state_digest"] == restarted["state_digest"]
    assert before == {p.name: p.read_bytes() for p in (cache / "blobs").glob("*.flsi")}
    assert playback.attach(scene, cache) == 20
    water, snow = settings.playback_object, settings.snow_playback_object
    assert water and snow and water != snow
    for obj, material_name in ((water, "FluidiTy Water Points"), (snow, "FluidiTy Snow Points")):
        modifier = obj.modifiers[-1]
        material = getattr(modifier.properties.inputs, modifier.node_group["fluidity_material_socket"]).value
        assert material.name == material_name
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    scene.frame_set(20)
    graph = bpy.context.evaluated_depsgraph_get()
    points = playback.inspection(cache, 20)["points"]
    for substance, obj in (("water", scene.fluidity.playback_object), ("snow", scene.fluidity.snow_playback_object)):
        count = sum(point["substance"] == substance for point in points)
        display = obj.modifiers[-1]
        display.show_viewport = False
        graph = bpy.context.evaluated_depsgraph_get()
        try:
            assert count > 0 and len(obj.evaluated_get(graph).data.points) == count
        finally:
            display.show_viewport = True
    print("FLUIDITY_M5_BLENDER_PASS", backend, blend)


if __name__ == "__main__":
    repo, binaries, output, backend = sys.argv[sys.argv.index("--") + 1:]
    verify(Path(repo), Path(binaries), Path(output), backend)
