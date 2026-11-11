"""Headless Blender control for native Mesh and Whitewater foam playback."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import bpy
from mathutils import Vector


MASK_NAME = "fluidity_foam_mask"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {str(path.relative_to(root)): (sha256(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def import_carrier(path, expected_type, object_path):
    before = set(bpy.data.objects)
    result = bpy.ops.wm.alembic_import(filepath=str(path), set_frame_range=False, as_background_job=False,
                                       always_add_cache_reader=True)
    added = list(set(bpy.data.objects) - before)
    assert result == {"FINISHED"} and len(added) == 1 and added[0].type == expected_type, added
    obj = added[0]
    reader = next(mod for mod in obj.modifiers if mod.type == "MESH_SEQUENCE_CACHE")
    assert reader.object_path == object_path
    reader.cache_file.velocity_name = ".velocities"
    reader.cache_file.velocity_unit = "SECOND"
    return obj, reader.cache_file


def sample(surface, foam, frame, radius=None, strength=None):
    bpy.context.scene.frame_set(frame)
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated_surface = surface.evaluated_get(graph)
    evaluated_foam = foam.evaluated_get(graph)
    mesh = evaluated_surface.data
    assert len(mesh.vertices) > 0 and len(mesh.polygons) > 0
    geometry = hashlib.sha256()
    local_origin = mesh.vertices[0].co
    six_times_volume = 0.0
    for vertex in mesh.vertices:
        geometry.update(struct.pack("<3f", *vertex.co))
    for face in mesh.polygons:
        assert len(face.vertices) == 3
        geometry.update(struct.pack("<3I", *face.vertices))
        a, b, c = (mesh.vertices[index].co - local_origin for index in face.vertices)
        six_times_volume += a.dot(b.cross(c))
    assert six_times_volume > 0, six_times_volume / 6
    local_foam = evaluated_surface.matrix_world.inverted() @ evaluated_foam.matrix_world
    points = [local_foam @ point.co for point in evaluated_foam.data.points]
    record = {"geometry_sha256": geometry.hexdigest(), "vertices": len(mesh.vertices),
              "triangles": len(mesh.polygons), "foam_points": len(points),
              "signed_volume_m3": six_times_volume / 6}
    if radius is not None:
        attribute = mesh.attributes.get(MASK_NAME)
        assert attribute is not None and attribute.domain == "POINT" and attribute.data_type == "FLOAT"
        values = [item.value for item in attribute.data]
        expected = [strength * max(0.0, 1 - min((vertex.co - point).length for point in points) / radius)
                    if points else 0.0 for vertex in mesh.vertices]
        assert len(values) == len(expected)
        record["maximum_mask_error"] = max(abs(actual - target) for actual, target in zip(values, expected, strict=True))
        record["nonzero_mask_vertices"] = sum(value > 0 for value in values)
        record["mask_sha256"] = hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()
        assert record["maximum_mask_error"] <= 2e-5, record
        assert bool(record["nonzero_mask_vertices"]) == bool(points), record
    return record, points


def render_mask_response(surface, foam, modifier, material, output, set_projection_inputs):
    scene = bpy.context.scene
    scene.frame_set(2)
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = surface.evaluated_get(graph)
    center = sum((evaluated.matrix_world @ vertex.co for vertex in evaluated.data.vertices), Vector())
    center /= len(evaluated.data.vertices)
    camera_data = bpy.data.cameras.new("Foam control camera")
    camera_data.type, camera_data.ortho_scale = "ORTHO", 2.3
    camera = bpy.data.objects.new("Foam control camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((3, -1, 1.4))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    light_data = bpy.data.lights.new("Foam control key", "AREA")
    light_data.energy, light_data.size = 380, 3
    light = bpy.data.objects.new("Foam control key", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((2, -2, 3))
    light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.engine = "CYCLES"
    scene.cycles.device, scene.cycles.samples = "CPU", 8
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    foam.hide_render = True
    buffers = []
    for name, strength in (("mask-off", 0), ("mask-on", 0.6)):
        set_projection_inputs(modifier, foam, material, radius=0.85, strength=strength)
        surface.update_tag()
        path = output / f"{name}.png"
        scene.render.filepath = str(path)
        assert bpy.ops.render.render(write_still=True) == {"FINISHED"}
        image = bpy.data.images.load(str(path), check_existing=False)
        buffers.append(list(image.pixels))
        bpy.data.images.remove(image)
    maximum_change = max(abs(a - b) for a, b in zip(*buffers, strict=True))
    assert maximum_change > 0.02, maximum_change
    return {"maximum_pixel_channel_change": maximum_change, "resolution": [64, 64]}


def prepare(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
    from fluidity.foam_projection import attach_foam_projection, create_foam_material, set_projection_inputs

    args.output.mkdir(parents=True, exist_ok=False)
    before = [snapshot(args.mesh_cache), snapshot(args.whitewater_cache)]
    archives = [args.output / "surface.abc", args.output / "foam.abc"]
    identities = []
    for command, cache, archive in zip(("export-surface", "export-foam"),
                                       (args.mesh_cache, args.whitewater_cache), archives, strict=True):
        result = subprocess.run([str(args.bridge), command, str(cache), "2", str(archive)],
                                check=True, capture_output=True, text=True, timeout=90)
        assert not result.stderr
        identities.append(json.loads(result.stdout))
    assert identities[0]["source_execution_id"] == identities[1]["source_execution_id"]
    archive_hashes = [sha256(path) for path in archives]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 24, 1
    scene.frame_set(2)
    surface, surface_cache = import_carrier(archives[0], "MESH", "/FluidiTySurface")
    foam, foam_cache = import_carrier(archives[1], "POINTCLOUD", "/FluidiTyFoam")
    for obj, identity in zip((surface, foam), identities, strict=True):
        obj["fluidity_execution"] = identity["execution_id"]
        obj["fluidity_source_execution"] = identity["source_execution_id"]
    baseline = [sample(surface, foam, frame)[0] for frame in range(3)]
    assert [record["foam_points"] for record in baseline] == [0, 16, 40], baseline
    material = create_foam_material("Control water", (0.01, 0.1, 0.3, 1), (0.85, 0.9, 1, 1))
    modifier = attach_foam_projection(surface, foam, material, radius=0.6, strength=1)
    initial = [sample(surface, foam, frame, 0.6, 1) for frame in range(3)]
    assert any((a - b).length > 1e-5 for a, b in zip(initial[1][1], initial[2][1])), "foam did not move"
    replacement = create_foam_material("Edited water", (0.02, 0.22, 0.4, 1), (1, 0.8, 0.5, 1))
    set_projection_inputs(modifier, foam, replacement, radius=0.85, strength=0.6)
    surface.update_tag()
    edited = [sample(surface, foam, frame, 0.85, 0.6)[0] for frame in range(3)]
    for old, projected, new in zip(baseline, initial, edited, strict=True):
        assert old["geometry_sha256"] == projected[0]["geometry_sha256"] == new["geometry_sha256"]
    assert edited[2]["mask_sha256"] != initial[2][0]["mask_sha256"]
    assert surface.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.materials[0].name == replacement.name
    render_evidence = render_mask_response(surface, foam, modifier, replacement, args.output, set_projection_inputs)
    assert sample(surface, foam, 2, 0.85, 0.6)[0] == edited[2]
    assert surface["fluidity_execution"] == identities[0]["execution_id"]
    assert foam["fluidity_execution"] == identities[1]["execution_id"]
    assert before == [snapshot(args.mesh_cache), snapshot(args.whitewater_cache)]
    assert archive_hashes == [sha256(path) for path in archives]
    assert not bpy.app.handlers.frame_change_pre and not bpy.app.handlers.frame_change_post
    bpy.context.preferences.filepaths.save_version = 0
    blend = args.output / "foam-control.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), relative_remap=False)
    surface_cache.filepath, foam_cache.filepath = "//surface.abc", "//foam.abc"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), relative_remap=False)
    receipt = {"blender_build": bpy.app.build_hash.decode(), "surface": surface.name, "foam": foam.name,
               "identities": identities, "archive_sha256": archive_hashes, "baseline": baseline,
               "initial": [row[0] for row in initial], "edited": edited, "render": render_evidence}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("PASS foam projection, motion, empty frame and geometry/cache invariance")


def reopen(path):
    receipt = json.loads(path.read_text(encoding="utf-8"))
    surface, foam = (bpy.data.objects[receipt[key]] for key in ("surface", "foam"))
    for obj, identity in zip((surface, foam), receipt["identities"], strict=True):
        assert obj["fluidity_execution"] == identity["execution_id"]
    assert not bpy.app.handlers.frame_change_pre and not bpy.app.handlers.frame_change_post
    assert [sample(surface, foam, frame, 0.85, 0.6)[0] for frame in range(3)] == receipt["edited"]
    assert [sha256(path.parent / name) for name in ("surface.abc", "foam.abc")] == receipt["archive_sha256"]
    print("PASS cache-only reopen without extension or solver")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--mesh-cache", type=Path)
    parser.add_argument("--whitewater-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reopen", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    assert bpy.app.build_hash == b"e13b728c6880", bpy.app.build_hash
    if args.reopen:
        reopen(args.reopen)
    else:
        assert all((args.bridge, args.mesh_cache, args.whitewater_cache, args.output))
        prepare(args)
