"""Verify a production mixed-surface USD export with Blender's native reader.

Run in Blender factory startup with --python-exit-code 1 and pass
`prepare SURFACE.usda OUTPUT.blend` or `reopen OUTPUT.blend` after --.
"""

import sys
from pathlib import Path

import bpy


EXPECTED = {"fluidity_phase_4": 1.0, "fluidity_fraction_8": 0.25, "fluidity_fraction_14": 0.75}


def surface():
    return next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")


def verify():
    obj = surface()
    assert any(mod.type == "MESH_SEQUENCE_CACHE" for mod in obj.modifiers)
    for frame in (0, 1):
        bpy.context.scene.frame_set(frame)
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        assert len(mesh.vertices) > 0 and len(mesh.polygons) > 0
        for name, expected in EXPECTED.items():
            attribute = mesh.attributes[name]
            assert attribute.domain == "POINT" and attribute.data_type == "FLOAT"
            assert all(abs(item.value - expected) < 1e-6 for item in attribute.data), (frame, name)
    assert not bpy.app.handlers.frame_change_pre and not bpy.app.handlers.frame_change_post
    return obj


def prepare(archive, blend):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 24
    bpy.context.scene.frame_set(0)
    assert bpy.ops.wm.usd_import(filepath=str(archive)) == {"FINISHED"}
    obj = verify()
    material = bpy.data.materials.new("Cached dye and phase")
    material.use_nodes = True
    tree = material.node_tree
    dye = tree.nodes.new("ShaderNodeAttribute")
    dye.attribute_name = "fluidity_fraction_14"
    phase = tree.nodes.new("ShaderNodeAttribute")
    phase.attribute_name = "fluidity_phase_4"
    shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    tree.links.new(dye.outputs["Fac"], shader.inputs["Base Color"])
    tree.links.new(phase.outputs["Fac"], shader.inputs["Roughness"])
    obj.data.materials.append(material)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))


def reopen(blend):
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    obj = verify()
    material = obj.data.materials[0]
    tree = material.node_tree
    shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    assert shader.inputs["Base Color"].links[0].from_node.attribute_name == "fluidity_fraction_14"
    assert shader.inputs["Roughness"].links[0].from_node.attribute_name == "fluidity_phase_4"


if __name__ == "__main__":
    mode, *paths = sys.argv[sys.argv.index("--") + 1:]
    paths = [Path(path).resolve() for path in paths]
    {"prepare": prepare, "reopen": reopen}[mode](*paths)
    print(f"PASS {mode} production mixed-surface USD")
