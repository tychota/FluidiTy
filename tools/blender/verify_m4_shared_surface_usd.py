"""Qualify the shared M4 surface exported by the production Blender bridge."""

from collections import Counter, defaultdict
from pathlib import Path
import shutil
import sys
import time

import bpy


WATER = "fluidity_face_phase_4"
OIL = "fluidity_face_phase_5"
INTERFACE = "fluidity_interface_4_5"
CLEAR = "fluidity_fraction_8"
DYE = "fluidity_fraction_14"
OIL_FRACTION = "fluidity_fraction_25"


def mesh_object():
    return next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")


def verify_surface(require_interface_composition=False):
    obj = mesh_object()
    assert any(mod.type == "MESH_SEQUENCE_CACHE" for mod in obj.modifiers)
    for frame in (0, 1):
        bpy.context.scene.frame_set(frame)
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        masks = [mesh.attributes[name] for name in (WATER, OIL, INTERFACE)]
        assert all(mask.domain == "FACE" and mask.data_type == "FLOAT" for mask in masks)
        counts = Counter()
        vertex_sides = defaultdict(set)
        edge_sides = defaultdict(list)
        triangle_keys = set()
        for polygon in mesh.polygons:
            values = [mask.data[polygon.index].value for mask in masks]
            assert sorted(values) == [0, 0, 1], values
            side = values.index(1)
            counts[side] += 1
            vertices = tuple(polygon.vertices)
            assert len(vertices) == 3
            key = tuple(sorted(vertices))
            assert key not in triangle_keys
            triangle_keys.add(key)
            for vertex in vertices:
                vertex_sides[vertex].add(side)
            for edge in polygon.edge_keys:
                edge_sides[edge].append(side)
        assert set(counts) == {0, 1, 2}, counts
        seam_edges = 0
        for sides in edge_sides.values():
            if len(sides) == 3:
                assert sorted(sides) == [0, 1, 2]
                seam_edges += 1
            else:
                assert len(sides) == 2, sides
        assert seam_edges > 0
        fractions = [mesh.attributes[name] for name in (CLEAR, DYE, OIL_FRACTION)]
        assert all(field.domain == "POINT" and field.data_type == "FLOAT" for field in fractions)
        for side, expected in ((0, (0.25, 0.75, 0)), (1, (0, 0, 1))):
            vertices = [vertex for vertex, sides in vertex_sides.items() if sides == {side}]
            assert vertices
            for vertex in vertices:
                values = tuple(field.data[vertex].value for field in fractions)
                assert all(abs(a - b) < 1e-6 for a, b in zip(values, expected, strict=True)), values
        if require_interface_composition:
            fields = [mesh.attributes[f"fluidity_interface_fraction_{substance}"] for substance in (8, 14, 25)]
            assert all(field.domain == "POINT" and field.data_type == "FLOAT" for field in fields)
            for vertex, sides in vertex_sides.items():
                expected = (0.25, 0.75, 1) if 2 in sides else (0, 0, 0)
                values = tuple(field.data[vertex].value for field in fields)
                assert all(abs(a - b) < 1e-6 for a, b in zip(values, expected, strict=True)), values
            assert any(sides == {2} for sides in vertex_sides.values())
            print("FRAME", frame, "INTERFACE COMPOSITION", "25/75 aqueous and pure oil")
        print("FRAME", frame, "FACES", dict(counts), "SEAM_EDGES", seam_edges)
    assert not bpy.app.handlers.frame_change_pre and not bpy.app.handlers.frame_change_post
    return obj


def prepare(archive, blend):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_set(0)
    assert bpy.ops.wm.usd_import(filepath=str(archive)) == {"FINISHED"}
    prepare_material(blend)


def prepare_material(blend):
    scene = bpy.context.scene
    obj = verify_surface()
    material = bpy.data.materials.new("Water dye and oil phase")
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    color = tree.nodes.new("ShaderNodeCombineColor")
    color.name = "Liquid color"
    dye_water = tree.nodes.new("ShaderNodeMath")
    dye_water.operation = "MULTIPLY"
    for name, target in ((DYE, dye_water.inputs[0]), (WATER, dye_water.inputs[1]), (OIL, color.inputs["Green"])):
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_name = name
        tree.links.new(attribute.outputs["Fac"], target)
    tree.links.new(dye_water.outputs[0], color.inputs["Red"])
    tree.links.new(color.outputs[0], emission.inputs["Color"])
    tree.links.new(emission.outputs[0], output.inputs["Surface"])
    obj.data.materials.append(material)
    camera_data = bpy.data.cameras.new("Qualification camera")
    camera = bpy.data.objects.new("Qualification camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (2.7, -2.55, 6)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.5
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.render.resolution_x = scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))


def attach(cache, binaries, blend, build=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
    import fluidity

    bpy.ops.wm.read_factory_settings(use_empty=True)
    fluidity.register()
    fluidity.jobs.BIN_DIR = binaries
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1
    scene.frame_start, scene.frame_end = 0, 2
    scene.frame_set(0)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    if build:
        settings = scene.fluidity
        settings.bake_cache = str(cache)
        source = cache
        inputs = {path: path.read_bytes() for path in source.rglob("*") if path.suffix in {".flmf", ".flrs", ".flef"}}
        cache = blend.with_name("built-mesh")
        settings.mesh_cache = str(cache)
        settings.mesh_spacing = settings.mesh_sigma = 0.25
        for resume in (False, True):
            assert bpy.ops.fluidity.build_surface(resume=resume) == {"FINISHED"}, settings.message
            deadline = time.monotonic() + 90
            while fluidity.jobs.is_running(scene) and time.monotonic() < deadline:
                fluidity.jobs.tick()
                time.sleep(0.01)
            assert not fluidity.jobs.is_running(scene), settings.message
            assert settings.message == "Surface build complete. Attach the committed surface to render.", (
                settings.message, settings.diagnostic
            )
            assert settings.committed_frame == 1
        assert {path: path.read_bytes() for path in inputs} == inputs
        assert bpy.ops.fluidity.build_surface() == {"CANCELLED"}
        assert (cache / "frames" / "00000000000000000001.flem").is_file()
        print("PASS Build Surface, Resume Surface and read-only Bake inputs")
    relocated = blend.with_name("relocated-mesh")
    shutil.copytree(cache, relocated)
    assert bpy.ops.fluidity.attach_surface(directory=str(cache)) == {"FINISHED"}
    obj = scene.fluidity.surface_object
    assert obj == mesh_object()
    assert scene.frame_current == 0
    archive = obj["fluidity_surface_archive"]
    data = obj.data
    assert archive.startswith("//")
    assert bpy.ops.fluidity.attach_surface(directory=str(cache)) == {"FINISHED"}
    assert scene.fluidity.surface_object == obj and obj.data == data
    assert obj["fluidity_surface_archive"] == archive

    prepare_material(blend)
    material = obj.data.materials[0]
    unrelated = bpy.data.objects.new("Artist control", None)
    scene.collection.objects.link(unrelated)
    unrelated.select_set(True)
    bpy.context.view_layer.objects.active = unrelated
    saved_frame = scene.frame_current
    assert bpy.ops.fluidity.attach_surface(directory=str(relocated)) == {"FINISHED"}
    assert scene.fluidity.surface_object == obj and obj.data.materials[0] == material
    assert obj["fluidity_surface_archive"] != archive and obj["fluidity_surface_archive"].startswith("//")
    assert scene.frame_current == saved_frame and bpy.context.view_layer.objects.active == unrelated
    assert unrelated.select_get()

    broken = blend.with_name("broken-mesh")
    shutil.copytree(cache, broken)
    next((broken / "blobs").glob("*.flef")).write_bytes(b"invalid")
    old_carriers = set(broken.glob("FluidiTySurface-*.usda"))
    bound = obj["fluidity_surface_archive"]
    try:
        result = bpy.ops.fluidity.attach_surface(directory=str(broken))
    except RuntimeError as error:
        assert "DATA_LOSS" in str(error).upper() or "digest" in str(error).lower(), error
    else:
        assert result == {"CANCELLED"}
    assert scene.fluidity.surface_object == obj and obj.data.materials[0] == material
    assert obj["fluidity_surface_archive"] == bound
    assert set(broken.glob("FluidiTySurface-*.usda")) == old_carriers
    seed_only = blend.with_name("seed-mesh")
    shutil.copytree(cache, seed_only)
    (seed_only / "frames" / "00000000000000000001.flem").unlink()
    assert bpy.ops.fluidity.attach_surface(directory=str(seed_only)) == {"FINISHED"}
    assert scene.fluidity.surface_object == obj and obj.data.materials[0] == material
    assert obj["fluidity_surface_last"] == 0
    assert bpy.ops.fluidity.attach_surface(directory=str(relocated)) == {"FINISHED"}
    assert scene.fluidity.surface_object == obj and obj.data.materials[0] == material
    assert obj["fluidity_surface_last"] == 1
    verify_surface()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    print("PASS attach, idempotence, relocation and failed replacement")

def reopen(blend):
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    obj = verify_surface()
    scene = bpy.context.scene
    tree = obj.data.materials[0].node_tree
    dye_link = tree.nodes["Liquid color"].inputs["Red"].links[0]
    for label in ("cached-material", "zero-dye-control"):
        if label == "zero-dye-control":
            tree.links.remove(dye_link)
            tree.nodes["Liquid color"].inputs["Red"].default_value = 0.0
        path = blend.with_name(label + ".png")
        scene.render.filepath = str(path)
        assert bpy.ops.render.render(write_still=True) == {"FINISHED"}
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = list(image.pixels)
        bpy.data.images.remove(image)
        channels = list(zip(pixels[::4], pixels[1::4], pixels[2::4], pixels[3::4], strict=True))
        red = sum(r > 0.3 and g < 0.05 and a > 0.99 for r, g, _, a in channels)
        green = sum(g > 0.3 and r < 0.05 and a > 0.99 for r, g, _, a in channels)
        assert green > 25, (label, green)
        assert (red > 25) if label == "cached-material" else red == 0, (label, red)
        print("RENDER", label, "DYE_PIXELS", red, "OIL_PIXELS", green)


def verify_saved_interface_composition(blend):
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    verify_surface(require_interface_composition=True)

if __name__ == "__main__":
    mode, *paths = sys.argv[sys.argv.index("--") + 1:]
    arguments = [Path(path).resolve() for path in paths]
    if mode == "build-attach":
        attach(*arguments, build=True)
    else:
        {"prepare": prepare, "attach": attach, "reopen": reopen, "sides": verify_saved_interface_composition}[mode](*arguments)
