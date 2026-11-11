"""Prepare or inspect a relocated Blender 5.3 grid asset qualification scene."""

import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import bpy


def create_witness(playback, channel):
    name = f"{channel} witness"
    mesh = bpy.data.meshes.new(name)
    witness = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(witness)
    witness.hide_render = True
    tree = bpy.data.node_groups.new("Read named volume grids", "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    node = tree.nodes.new
    link = tree.links.new
    info = node("GeometryNodeObjectInfo")
    info.inputs["Object"].default_value = playback
    density = node("GeometryNodeGetNamedGrid")
    density.inputs["Name"].default_value = "density"
    velocity = node("GeometryNodeGetNamedGrid")
    velocity.data_type = "VECTOR"
    velocity.inputs["Name"].default_value = "velocity"
    for grid in (density, velocity):
        link(info.outputs["Geometry"], grid.inputs["Volume"])
    points = node("GeometryNodeGridToPoints")
    if channel == "velocity":
        points.data_type = "VECTOR"
        link(velocity.outputs["Grid"], points.inputs["Grid"])
        vertices = node("GeometryNodePointsToVertices")
        link(points.outputs["Points"], vertices.inputs["Points"])
        link(vertices.outputs["Mesh"], node("NodeGroupOutput").inputs["Geometry"])
        witness.modifiers.new("Read grids", "NODES").node_group = tree
        return
    link(density.outputs["Grid"], points.inputs["Grid"])
    sample = node("GeometryNodeSampleGrid")
    sample.data_type = "VECTOR"
    sample.inputs["Interpolation"].default_value = "Nearest Neighbor"
    link(velocity.outputs["Grid"], sample.inputs["Grid"])

    geometry = points.outputs["Points"]
    for name, data_type, value in (
        ("density_value", "FLOAT", points.outputs["Value"]),
        ("velocity_value", "FLOAT_VECTOR", sample.outputs["Value"]),
    ):
        store = node("GeometryNodeStoreNamedAttribute")
        store.data_type = data_type
        store.domain = "POINT"
        store.inputs["Name"].default_value = name
        link(geometry, store.inputs["Geometry"])
        link(value, store.inputs["Value"])
        geometry = store.outputs["Geometry"]
    vertices = node("GeometryNodePointsToVertices")
    link(geometry, vertices.inputs["Points"])
    link(vertices.outputs["Mesh"], node("NodeGroupOutput").inputs["Geometry"])
    witness.modifiers.new("Read grids", "NODES").node_group = tree


def prepare(directory, cache, bridge, last):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
    from fluidity.grid_asset import VOXEL_SIZE, create_grid_asset

    directory.mkdir()
    archive = directory / "Points.abc"
    subprocess.run([bridge, "export", cache, str(last), archive], check=True)
    frames = [
        json.loads(subprocess.check_output([bridge, "inspect", cache, str(frame)], encoding="utf-8"))["points"]
        for frame in range(last + 1)
    ]
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_start = 0
    scene.frame_end = last
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "Grids.blend"))
    bpy.ops.wm.alembic_import(filepath=str(archive), as_background_job=False, always_add_cache_reader=True)
    playback = next(obj for obj in scene.objects if obj.type == "POINTCLOUD")
    cache_file = next(mod.cache_file for mod in playback.modifiers if mod.type == "MESH_SEQUENCE_CACHE")
    cache_file.filepath = "//Points.abc"
    cache_file.velocity_name = ".velocities"
    cache_file.velocity_unit = "SECOND"
    playback.modifiers.new("FluidiTy grids", "NODES").node_group = create_grid_asset()
    create_witness(playback, "density")
    create_witness(playback, "velocity")
    scene.frame_set(last)
    (directory / "expected.json").write_text(json.dumps({"voxel_size": VOXEL_SIZE, "frames": frames}), encoding="utf-8")
    bpy.ops.wm.save_mainfile()


def verify(directory):
    bpy.ops.wm.open_mainfile(filepath=str(directory / "Grids.blend"), use_scripts=False)
    assert not hasattr(bpy.types.Scene, "fluidity")
    receipt = json.loads((directory / "expected.json").read_text(encoding="utf-8"))
    voxel = receipt["voxel_size"]
    frames = receipt["frames"]
    witness = bpy.data.objects["density witness"]
    velocity_witness = bpy.data.objects["velocity witness"]
    scene = bpy.context.scene
    for frame in [*range(len(frames)), 0, len(frames) - 1]:
        expected = defaultdict(list)
        for point in frames[frame]:
            # The constant kernel assigns exact half-voxel ties to the lower index.
            key = tuple(math.ceil(position / voxel - 0.5) for position in point["position"])
            expected[key].append(point["velocity"])
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        mesh = witness.evaluated_get(graph).data
        velocity_mesh = velocity_witness.evaluated_get(graph).data
        velocity_keys = {tuple(round(position / voxel) for position in vertex.co) for vertex in velocity_mesh.vertices}
        assert velocity_keys == expected.keys(), (frame, len(velocity_keys), len(expected))
        assert len(mesh.vertices) == len(expected), (frame, len(mesh.vertices), len(expected))
        for vertex in mesh.vertices:
            key = tuple(round(position / voxel) for position in vertex.co)
            velocities = expected.pop(key)
            density = mesh.attributes["density_value"].data[vertex.index].value
            velocity = mesh.attributes["velocity_value"].data[vertex.index].vector
            assert math.isclose(density * voxel**3, len(velocities), abs_tol=1e-5), (frame, key, density)
            for actual, components in zip(velocity, zip(*velocities, strict=True), strict=True):
                assert math.isclose(actual, sum(components) / len(velocities), abs_tol=1e-5), (frame, key, velocity)
        assert not expected
    print(f"PASS: {len(frames)} frames, including {sum(not frame for frame in frames)} empty, plus reverse seek")


args = sys.argv[sys.argv.index("--") + 1 :]
if args[0] == "prepare":
    prepare(*(Path(arg).resolve() for arg in args[1:4]), int(args[4]))
else:
    verify(Path(args[1]).resolve())
