"""Native Geometry Nodes projection from a verified baked Foam carrier."""

import math

import bpy


MASK_NAME = "fluidity_foam_mask"
VERSION_KEY = "fluidity_foam_projection_version"
ASSET_VERSION = 1


def create_foam_material(name, water_color, foam_color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_type = "GEOMETRY"
    attribute.attribute_name = MASK_NAME
    attribute.location = (-480, 60)
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.inputs[1].default_value = water_color
    mix.inputs[2].default_value = foam_color
    mix.location = (-240, 60)
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (0, 60)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (280, 60)
    tree.links.new(attribute.outputs["Fac"], mix.inputs[0])
    tree.links.new(mix.outputs["Color"], shader.inputs["Base Color"])
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_foam_projection_asset():
    tree = bpy.data.node_groups.new("FluidiTy Foam Projection v1", "GeometryNodeTree")
    tree.asset_mark()
    tree[VERSION_KEY] = ASSET_VERSION
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for key, name, socket_type in (
        ("foam", "Foam Cache", "NodeSocketObject"),
        ("radius", "Projection Radius", "NodeSocketFloat"),
        ("strength", "Foam Strength", "NodeSocketFloat"),
        ("material", "Water Material", "NodeSocketMaterial"),
    ):
        socket = tree.interface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
        tree[f"fluidity_{key}_socket"] = socket.identifier
        if key == "radius":
            socket.default_value, socket.min_value = 0.1, 1e-6
            socket.description = "Distance from baked Foam points in surface-local metres"
        elif key == "strength":
            socket.default_value, socket.min_value, socket.max_value = 1, 0, 1

    def node(kind, x, y):
        item = tree.nodes.new(kind)
        item.location = (x, y)
        return item

    inputs = node("NodeGroupInput", -760, 180)
    foam = node("GeometryNodeObjectInfo", -560, -120)
    foam.transform_space = "RELATIVE"
    foam.inputs["As Instance"].default_value = False
    position = node("GeometryNodeInputPosition", -560, -340)
    proximity = node("GeometryNodeProximity", -340, -120)
    proximity.target_element = "POINTS"
    normalize = node("ShaderNodeMath", -120, -120)
    normalize.operation = "DIVIDE"
    falloff = node("ShaderNodeMath", 80, -120)
    falloff.operation = "SUBTRACT"
    falloff.inputs[0].default_value = 1
    falloff.use_clamp = True
    strength = node("ShaderNodeMath", 280, -120)
    strength.operation = "MULTIPLY"
    strength.use_clamp = True
    valid_mask = node("GeometryNodeSwitch", 480, -120)
    valid_mask.input_type = "FLOAT"
    valid_mask.inputs["False"].default_value = 0
    store = node("GeometryNodeStoreNamedAttribute", 700, 180)
    store.data_type, store.domain = "FLOAT", "POINT"
    store.inputs["Name"].default_value = MASK_NAME
    material = node("GeometryNodeSetMaterial", 920, 180)
    output = node("NodeGroupOutput", 1140, 180)
    for source, destination in (
        (inputs.outputs["Foam Cache"], foam.inputs["Object"]),
        (foam.outputs["Geometry"], proximity.inputs["Geometry"]),
        (position.outputs["Position"], proximity.inputs["Sample Position"]),
        (proximity.outputs["Distance"], normalize.inputs[0]),
        (inputs.outputs["Projection Radius"], normalize.inputs[1]),
        (normalize.outputs[0], falloff.inputs[1]),
        (falloff.outputs[0], strength.inputs[0]),
        (inputs.outputs["Foam Strength"], strength.inputs[1]),
        (proximity.outputs["Is Valid"], valid_mask.inputs["Switch"]),
        (strength.outputs[0], valid_mask.inputs["True"]),
        (inputs.outputs["Geometry"], store.inputs["Geometry"]),
        (valid_mask.outputs[0], store.inputs["Value"]),
        (store.outputs["Geometry"], material.inputs["Geometry"]),
        (inputs.outputs["Water Material"], material.inputs["Material"]),
        (material.outputs["Geometry"], output.inputs["Geometry"]),
    ):
        tree.links.new(source, destination)
    return tree


def set_projection_inputs(modifier, foam, material, *, radius, strength):
    """Change only rendering inputs on an existing Foam Projection modifier."""
    if not math.isfinite(radius) or radius < 1e-6 or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Foam projection requires a positive radius and strength in [0,1].")
    tree = modifier.node_group
    if tree is None or tree.get(VERSION_KEY) != ASSET_VERSION:
        raise ValueError("Expected a FluidiTy Foam Projection modifier.")
    inputs = modifier.properties.inputs
    for key, value in (("foam", foam), ("radius", radius), ("strength", strength), ("material", material)):
        getattr(inputs, tree[f"fluidity_{key}_socket"]).value = value


def attach_foam_projection(surface, foam, material, *, radius=0.1, strength=1):
    """Bind existing native carriers from the same accepted Simulate source."""
    source_execution = surface.get("fluidity_source_execution")
    if (surface.type != "MESH" or foam.type != "POINTCLOUD" or not source_execution
            or source_execution != foam.get("fluidity_source_execution")):
        raise ValueError("Surface and Foam must be native carriers from the same accepted source.")
    if not all(any(mod.type == "MESH_SEQUENCE_CACHE" for mod in obj.modifiers) for obj in (surface, foam)):
        raise ValueError("Surface and Foam must retain their native cache readers.")
    matches = [mod for mod in surface.modifiers if mod.type == "NODES" and mod.node_group
               and mod.node_group.get(VERSION_KEY) == ASSET_VERSION]
    if len(matches) > 1:
        raise ValueError("Expected at most one FluidiTy Foam Projection modifier.")
    if matches:
        modifier = matches[0]
    else:
        tree = next((item for item in bpy.data.node_groups if item.get(VERSION_KEY) == ASSET_VERSION), None)
        modifier = surface.modifiers.new("FluidiTy Foam Projection v1", "NODES")
        modifier.node_group = tree or create_foam_projection_asset()
    set_projection_inputs(modifier, foam, material, radius=radius, strength=strength)
    surface.update_tag()
    return modifier
