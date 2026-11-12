"""Qualify the shared M4 surface exported by the production Blender bridge."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import bpy


WATER = "fluidity_face_phase_4"
OIL = "fluidity_face_phase_5"
INTERFACE = "fluidity_interface_4_5"
CLEAR = "fluidity_fraction_8"
DYE = "fluidity_fraction_14"
OIL_FRACTION = "fluidity_fraction_25"


def mesh_object():
    return next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")


def verify_surface():
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
        print("FRAME", frame, "FACES", dict(counts), "SEAM_EDGES", seam_edges)
    assert not bpy.app.handlers.frame_change_pre and not bpy.app.handlers.frame_change_post
    return obj


def prepare(archive, blend):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_set(0)
    assert bpy.ops.wm.usd_import(filepath=str(archive)) == {"FINISHED"}
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


if __name__ == "__main__":
    mode, *paths = sys.argv[sys.argv.index("--") + 1:]
    {"prepare": prepare, "reopen": reopen}[mode](*(Path(path).resolve() for path in paths))
