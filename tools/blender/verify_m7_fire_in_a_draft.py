"""Build and qualify the Blender 5.2 Fire in a draft acceptance scene."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import bpy
from mathutils import Vector
import openvdb


def command(executable, *args):
    run = subprocess.run([str(executable), *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    return [json.loads(line) for line in run.stdout.splitlines()]


def solver_vector(vector):
    return dict(x=float(vector.x), y=float(vector.z), z=float(-vector.y))


def snapshot(paths):
    return {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--flatc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "blender"))
    from fluidity.gas_volume import GAS_GRID_NAMES, attach_gas_volume, export_gas_volume_frame

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "bake"
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene.render.fps = 24
    scene.frame_start = 0
    scene.frame_end = 12

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

    keys = []
    draft = []
    for frame in range(scene.frame_end + 1):
        scene.frame_set(frame)
        evaluated = bpy.context.evaluated_depsgraph_get()
        emitter = source.evaluated_get(evaluated)
        wind = field.evaluated_get(evaluated)
        direction = wind.matrix_world.to_quaternion() @ Vector((0, 0, 1))
        keys.append(dict(time_s=frame / 24, fuel_kg_per_m3_s=emitter["Fuel Rate"], heat_w_per_m3=emitter["Heat Rate"]))
        draft.append(dict(time_s=frame / 24, acceleration_m_per_s2=solver_vector(direction * wind.field.strength)))
    bounds = [solver_vector(source.matrix_world @ Vector(corner)) for corner in source.bound_box]
    lower = {axis: min(corner[axis] for corner in bounds) for axis in ("x", "y", "z")}
    upper = {axis: max(corner[axis] for corner in bounds) for axis in ("x", "y", "z")}
    identity = command(args.solver, "schema")[-1]
    zero = dict(x=0.0, y=0.0, z=0.0)
    packet = dict(
        build_digest=dict(bytes=list(bytes.fromhex(identity["build_digest"]))),
        world_origin_m=zero,
        extent=dict(x=8, y=12, z=8),
        cell_width_m=0.125,
        open_boundaries=59,
        initial_velocity_m_per_s=zero,
        gravity_m_per_s2=dict(x=0.0, y=-9.81, z=0.0),
        substeps_per_frame=4,
        emitters=[dict(lower_m=lower, upper_m=upper, keys=keys)],
        draft=draft,
        reaction=dict(ignition_temperature_k=299.0, burn_rate_per_s=7.0, heat_per_fuel_j_per_kg=4000000.0, soot_yield=0.2),
    )
    scene_input = output / "fire-in-a-draft.json"
    scene_input.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    subprocess.run([str(args.flatc), "-b", "-o", str(output),
                    str(root / "schemas/simulate/gas_scene.fbs"), str(scene_input)], check=True)
    resolved = scene_input.with_suffix(".flgs")
    assert command(args.solver, "validate", resolved)[-1]["model"] == "gas"
    assert command(args.solver, "simulate", resolved, "--cache", cache, "--frame-end", scene.frame_end)[-1]["frame"] == 12
    assert command(args.solver, "replay", cache, "--frame", 12, "--substep", 0)[-1]["frame"] == 12

    paths = [output / f"fire_{frame:04d}.vdb" for frame in range(scene.frame_end + 1)]
    for frame, path in enumerate(paths):
        export_gas_volume_frame(args.solver, cache, frame, path)
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

    obj = attach_gas_volume(scene, paths[0], 0, len(paths))
    scene.frame_set(12)
    obj.data.grids.load()
    assert {grid.name for grid in obj.data.grids} == set(GAS_GRID_NAMES)
    tracked = paths + [path for path in cache.rglob("*") if path.is_file()]
    before = snapshot(tracked)
    assert obj.asset_data and obj.data.materials[0].asset_data
    obj.data.display.density *= 0.5
    shader = next(node for node in obj.data.materials[0].node_tree.nodes if node.type == "PRINCIPLED_VOLUME")
    shader.inputs["Density"].default_value *= 1.5
    assert snapshot(tracked) == before
    obj.data.filepath = "//fire_0000.vdb"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "Fire in a draft.blend"), relative_remap=False)
    report = {name: {"voxels": grid.activeVoxelCount(), "min_max": grid.evalMinMax()} for name, grid in by_name.items()}
    (output / "gas-volume-receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS Fire in a draft, four independent fields, Volume asset and material edits preserve cache")


if __name__ == "__main__":
    main()
