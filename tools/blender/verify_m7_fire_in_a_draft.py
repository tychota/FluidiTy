"""Build and qualify the Blender 5.2 Fire in a draft acceptance scene."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import bpy
from mathutils import Vector
import openvdb


def command(executable, *args):
    run = subprocess.run([str(executable), *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    return [json.loads(line) for line in run.stdout.splitlines()]


def snapshot(paths):
    return {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "blender"))
    import fluidity
    from fluidity import jobs
    from fluidity.gas_volume import GAS_GRID_NAMES

    fluidity.register()
    jobs.BIN_DIR = args.solver.resolve().parent

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "bake"
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene.render.fps = 24
    scene.frame_start = 0
    scene.frame_end = 12

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, -0.5, 0.75))
    domain = bpy.context.object
    domain.name = "Gas Domain"
    domain.dimensions = (1.0, 1.0, 1.5)
    domain.display_type = "WIRE"
    domain.hide_render = True

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.375, -0.5, 0.1875))
    source = bpy.context.object
    source.name = "Keyframed Fuel and Heat"
    source.dimensions = (0.25, 0.25, 0.25)
    source.display_type = "WIRE"
    source.hide_render = True
    for frame, fuel, heat in ((0, 0.0, 0.0), (3, 0.3, 240000.0), (8, 0.3, 180000.0), (12, 0.02, 30000.0)):
        source["Fuel Rate"] = fuel
        source["Heat Rate"] = heat
        source.keyframe_insert(data_path='["Fuel Rate"]', frame=frame)
        source.keyframe_insert(data_path='["Heat Rate"]', frame=frame)
    bpy.ops.object.effector_add(type="WIND")
    field = bpy.context.object
    field.name = "Keyframed Cross Draft"
    field.rotation_euler.y = math.pi / 2.0
    for frame, strength in ((0, 0.0), (4, 0.3), (12, 0.85)):
        field.field.strength = strength
        field.field.keyframe_insert(data_path="strength", frame=frame)

    settings = scene.fluidity
    settings.gas_enabled = True
    settings.domain = domain
    settings.gas_source = source
    settings.gas_wind = field
    settings.backend = args.backend
    settings.bake_cache = "//bake"
    settings.preview_cache = "//preview"
    bpy.context.preferences.filepaths.save_version = 0
    blend = output / "Fire in a draft.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), relative_remap=False)
    assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, (settings.message, settings.diagnostic)
    assert bpy.ops.fluidity.bake() == {"FINISHED"}, (settings.message, settings.diagnostic)
    deadline = time.monotonic() + 180
    while jobs.is_running(scene):
        if time.monotonic() >= deadline:
            raise RuntimeError("Gas bake did not finish")
        jobs.tick()
        time.sleep(0.01)
    assert settings.committed_frame == scene.frame_end and not settings.diagnostic, (
        settings.message, settings.diagnostic
    )
    assert command(args.solver, "replay", cache, "--frame", 12, "--substep", 0)[-1]["frame"] == 12
    assert bpy.ops.fluidity.attach() == {"FINISHED"}, (settings.message, settings.diagnostic)
    obj = bpy.data.objects["FluidiTy Fire Volume"]
    paths = [output / "bake-volumes" / f"fire_{frame:04d}.vdb" for frame in range(scene.frame_end + 1)]
    grids, metadata = openvdb.readAll(str(paths[-1]))
    by_name = {grid.name: grid for grid in grids}
    assert set(by_name) == set(GAS_GRID_NAMES)
    assert metadata["fluidity_frame"] == 12
    first_density = openvdb.read(str(paths[0]), "density")
    assert by_name["density"].evalMinMax()[0] < first_density.evalMinMax()[0]
    assert by_name["temperature"].evalMinMax()[1] > 300.0
    for name in ("fuel", "flame", "soot"):
        assert by_name[name].activeVoxelCount() > 0 and by_name[name].evalMinMax()[1] > 0.0, name
    assert by_name["density"].transform.indexToWorld((0, 0, 0)) == (0.0625, -0.0625, 0.0625)

    scene.frame_set(12)
    assert bpy.ops.fluidity.inspect() == {"FINISHED"}, (settings.message, settings.diagnostic)
    assert all(name in settings.point_detail for name in ("Density", "Temperature", "Fuel", "Flame"))
    obj.data.grids.load()
    assert {grid.name for grid in obj.data.grids} == set(GAS_GRID_NAMES)
    tracked = paths + [path for path in cache.rglob("*") if path.is_file()]
    before = snapshot(tracked)
    assert obj.asset_data and obj.data.materials[0].asset_data
    obj.data.display.density *= 0.5
    shader = next(node for node in obj.data.materials[0].node_tree.nodes if node.type == "PRINCIPLED_VOLUME")
    shader.inputs["Density"].default_value *= 1.5
    assert snapshot(tracked) == before
    camera_data = bpy.data.cameras.new("Plume Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.2
    camera = bpy.data.objects.new("Plume Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (2.3, 1.9, 1.2)
    target = Vector((0.48, -0.49, 0.40))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 2.0
    obj.data.filepath = "//bake-volumes/fire_0000.vdb"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "Fire in a draft.blend"), relative_remap=False)
    report = {name: {"voxels": grid.activeVoxelCount(), "min_max": grid.evalMinMax()} for name, grid in by_name.items()}
    (output / "gas-volume-receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS Fire in a draft, four independent fields, Volume asset and material edits preserve cache")


if __name__ == "__main__":
    main()
