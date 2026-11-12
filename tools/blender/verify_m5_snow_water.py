"""Bake, resume, attach, and reopen an authored mixed snow and water scene in Blender."""

import json
import math
import subprocess
import sys
import time
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
    settings.bake_cache = "//cache"
    settings.surface_tension = 0
    settings.maximum_step = 0.01
    settings.initial_velocity = (0.05, 0, 0)
    settings.domain = box("Domain", (2, 0, 0.75), (4, 3, 1.5))
    settings.snow_mound = box("Snow Mound", (2.125, 0, 0.45), (1.0, 1.0, 0.9))
    settings.snow_density = 250
    assert bpy.ops.fluidity.apply_snow_preset() == {"FINISHED"}
    assert settings.snow_density == 300 and settings.surface_tension == 0
    settings.collider = box("Animated Platen", (2.125, 0, 1.15), (1.25, 1.25, 0.2))
    for frame, height in ((0, 1.15), (20, 0.68)):
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
    assert scene_inputs(scene).startswith("FLUIDITY-BLENDER-5\n")
    assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, settings.message
    assert bpy.ops.fluidity.bake() == {"FINISHED"}, settings.message
    deadline = time.monotonic() + 120
    while fluidity.is_running(scene):
        assert time.monotonic() < deadline, settings.message
        fluidity.tick()
        time.sleep(0.005)
    assert settings.committed_frame == 20, settings.message
    cache = output / "cache"
    solver = jobs.binary("simulate")
    command = ["run", "--stage", "simulate", "--scene", cache / "resolved.flrs", "--cache", cache]
    result = run(solver, "replay", "--cache", cache, "--frame", 20, "--substep", 0)[-1]
    assert result["source_next_child_ordinals"] == [2]
    before = {p.name: p.read_bytes() for p in (cache / "blobs").glob("*.flsi")}
    restarted = run(solver, *command, "--resume", "--frame-start", 10, "--frame-end", 20)[-1]
    assert result["state_digest"] == restarted["state_digest"]
    assert before == {p.name: p.read_bytes() for p in (cache / "blobs").glob("*.flsi")}
    assert bpy.ops.fluidity.attach() == {"FINISHED"}, settings.message
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
    start_points = playback.inspection(cache, 0)["points"]
    start_snow = [point for point in start_points if point["substance"] == "snow"]
    final_snow = [point for point in points if point["substance"] == "snow"]
    emitted_water = [point for point in points if point["substance"] == "water" and point["id"] > 1]
    platen_y = 0.58
    assert max(point["position"][2] for point in start_snow) > platen_y
    assert max(point["position"][2] for point in final_snow) < platen_y
    volume_ratio = (min(point["radius"] for point in final_snow) / start_snow[0]["radius"]) ** 3
    assert 0 < volume_ratio < 0.9, volume_ratio
    snow_support, water_support = 0.875 * 0.5, 0.9 * 0.5
    closest = min(math.dist(water["position"], snow["position"]) for water in emitted_water for snow in final_snow)
    assert snow_support <= closest < snow_support + water_support, closest
    plastic = json.loads(jobs.checked("blender_bridge", "inspect-snow", cache, 20))
    assert plastic["minimum_log_plastic_jacobian"] < -0.05, plastic
    print("FLUIDITY_M5_MEASUREMENTS", json.dumps({
        "minimum_volume_ratio": volume_ratio,
        "minimum_log_plastic_jacobian": plastic["minimum_log_plastic_jacobian"],
        "emitted_gap_m": closest,
    }))
    print("FLUIDITY_M5_BLENDER_PASS", backend, blend)


if __name__ == "__main__":
    repo, binaries, output, backend = sys.argv[sys.argv.index("--") + 1:]
    verify(Path(repo), Path(binaries), Path(output), backend)
