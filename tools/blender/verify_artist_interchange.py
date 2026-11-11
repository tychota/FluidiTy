"""Verify interchange against a real committed M1 bake, preserving artist state."""

import json
import math
import sys
from pathlib import Path
from uuid import uuid4

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
import fluidity

root, binaries, *additional_caches = map(Path, sys.argv[sys.argv.index("--") + 1 :])
fluidity.register()
fluidity.jobs.BIN_DIR = binaries.resolve()
bpy.ops.wm.open_mainfile(filepath=str((root / "First bake – fluide.blend").resolve()))
scene = bpy.context.scene
obj = scene.fluidity.playback_object
assert obj
original = (scene.frame_current, scene.frame_start, scene.frame_end, set(bpy.context.selected_objects), obj.data)


def verify_external_fields():
    modifier = next(item for item in obj.modifiers if item.type == "NODES")
    asset = modifier.node_group
    wrapper = bpy.data.node_groups.new("Field verification", "GeometryNodeTree")
    for direction in ("INPUT", "OUTPUT"):
        wrapper.interface.new_socket(name="Geometry", in_out=direction, socket_type="NodeSocketGeometry")
    source = wrapper.nodes.new("NodeGroupInput")
    output = wrapper.nodes.new("NodeGroupOutput")
    group = wrapper.nodes.new("GeometryNodeGroup")
    group.node_tree = asset
    wrapper.links.new(source.outputs["Geometry"], group.inputs["Geometry"])
    geometry = group.outputs["Points"]
    for name, kind in (("Velocity", "FLOAT_VECTOR"), ("Speed", "FLOAT"), ("Radius", "FLOAT")):
        store = wrapper.nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type = kind
        store.domain = "POINT"
        store.inputs["Name"].default_value = "test_" + name
        wrapper.links.new(geometry, store.inputs["Geometry"])
        wrapper.links.new(group.outputs[name], store.inputs["Value"])
        geometry = store.outputs["Geometry"]
    wrapper.links.new(geometry, output.inputs["Geometry"])
    try:
        modifier.node_group = wrapper
        for frame in (0, 3, 12):
            scene.frame_set(frame)
            data = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
            for index, point in enumerate(data.points):
                velocity = data.attributes["velocity"].data[index].vector
                field = data.attributes["test_Velocity"].data[index].vector
                assert (velocity - field).length < 1e-6
                assert math.isclose(data.attributes["test_Speed"].data[index].value, velocity.length, abs_tol=1e-6)
                assert math.isclose(data.attributes["test_Radius"].data[index].value, point.radius, abs_tol=1e-6)
    finally:
        modifier.node_group = asset
        scene.frame_set(original[0])
        bpy.data.node_groups.remove(wrapper)


verify_external_fields()
scene.render.fps = 30
try:
    fluidity.interchange.export_usd(scene, root / "Invalid timing.usda")
except ValueError:
    pass
else:
    raise AssertionError("USD must reject a changed sample clock")
assert not (root / "Invalid timing.usda").exists()
scene.render.fps = 24
scene.unit_settings.scale_length = 0.01
try:
    fluidity.interchange.export_usd(scene, root / "Invalid units.usda")
except ValueError:
    pass
else:
    raise AssertionError("USD must reject a changed length scale")
assert not (root / "Invalid units.usda").exists()
scene.unit_settings.scale_length = 1
attached = scene.fluidity.attached_cache
for cache_path in [attached, *[str(path.resolve()) for path in additional_caches]]:
    scene.fluidity.attached_cache = cache_path
    for extension, export in (("abc", fluidity.interchange.export_alembic), ("usda", fluidity.interchange.export_usd)):
        destination = root / f"Artist export-{uuid4().hex}.{extension}"
        cache = Path(bpy.path.abspath(cache_path))
        last = fluidity.playback.committed_last(cache)
        has_empty = any(not fluidity.playback.inspection(cache, frame)["points"] for frame in range(last + 1))
        if extension == "usda" and has_empty:
            try:
                export(scene, destination)
            except ValueError as error:
                assert "empty frames" in str(error)
            else:
                raise AssertionError("USD must reject Blender's lossy empty-frame export")
            assert not destination.exists()
            assert (
                scene.frame_current,
                scene.frame_start,
                scene.frame_end,
                set(bpy.context.selected_objects),
                obj.data,
            ) == original
            continue
        export(scene, destination)
        assert destination.stat().st_size > 0
        assert (
            scene.frame_current,
            scene.frame_start,
            scene.frame_end,
            set(bpy.context.selected_objects),
            obj.data,
        ) == original
        digest = destination.read_bytes()
        try:
            export(scene, destination)
        except ValueError:
            pass
        else:
            raise AssertionError("Existing export overwritten")
        assert destination.read_bytes() == digest
        before = set(bpy.data.objects)
        assert bpy.ops.fluidity.import_cache(filepath=str(destination.resolve())) == {"FINISHED"}
        created = set(bpy.data.objects) - before
        points = next(item for item in created if item.type == "POINTCLOUD")
        cache = Path(bpy.path.abspath(scene.fluidity.attached_cache))
        last = fluidity.playback.committed_last(cache)
        for frame in [*range(last + 1), last, 0, last]:
            scene.frame_set(frame)
            evaluated = points.evaluated_get(bpy.context.evaluated_depsgraph_get())
            expected = json.loads(fluidity.jobs.checked("blender_bridge", "inspect", cache, frame))
            assert len(evaluated.data.points) == len(expected["points"]), (extension, frame)
            for actual, expected_point in zip(evaluated.data.points, expected["points"], strict=True):
                world = evaluated.matrix_world @ actual.co
                assert max(abs(a - b) for a, b in zip(world, expected_point["position"], strict=True)) < 1e-5
                assert abs(actual.radius - expected_point["radius"]) < 1e-5
        for item in created:
            bpy.data.objects.remove(item, do_unlink=True)
        scene.frame_set(original[0])
scene.fluidity.attached_cache = attached
print("PASS: committed Alembic/USD export/import, point values, no overwrite, preserved scene state")
