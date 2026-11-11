"""Attach verified M1 Points to saved Blender-native datablocks."""

import filecmp
import json
from pathlib import Path
from uuid import uuid4

import bpy

from . import jobs
from .scene_inputs import require_cadence

ASSET_VERSION = 2


def create_points_asset():
    tree = bpy.data.node_groups.new("FluidiTy Points v2", "GeometryNodeTree")
    tree.asset_mark()
    tree["fluidity_asset_version"] = ASSET_VERSION
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    for name, socket_type, description in (
        ("Points", "NodeSocketGeometry", "Uninstanced committed points; connect to a custom display or reconstruction"),
        ("Velocity", "NodeSocketVector", "Diagnostic Blender-space velocity in meters per second on the point domain"),
        ("Speed", "NodeSocketFloat", "Velocity magnitude in meters per second on the point domain"),
        ("Radius", "NodeSocketFloat", "Unscaled committed display radius in meters on the point domain"),
    ):
        socket = tree.interface.new_socket(name=name, in_out="OUTPUT", socket_type=socket_type)
        socket.description = description
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    size = tree.interface.new_socket(name="Size", in_out="INPUT", socket_type="NodeSocketFloat")
    size.default_value, size.min_value, size.max_value = 1.0, 0.01, 10.0
    size.description = "Display radius multiplier; does not change the committed simulation"
    material_socket = tree.interface.new_socket(name="Material", in_out="INPUT", socket_type="NodeSocketMaterial")
    tree["fluidity_size_socket"] = size.identifier
    tree["fluidity_material_socket"] = material_socket.identifier

    material = bpy.data.materials.new("FluidiTy Water Points")
    material.diffuse_color = (0.015, 0.42, 0.8, 1)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    emission = material.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = material.diffuse_color
    surface = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs[0], surface.inputs["Surface"])
    material_socket.default_value = material

    def node(kind, x, y):
        item = tree.nodes.new(kind)
        item.location = (x, y)
        return item

    inputs = node("NodeGroupInput", -560, 120)
    output = node("NodeGroupOutput", 410, 120)
    ico = node("GeometryNodeMeshIcoSphere", -560, -180)
    ico.inputs["Radius"].default_value = 1
    ico.inputs["Subdivisions"].default_value = 2
    set_material = node("GeometryNodeSetMaterial", -300, -180)
    radius = node("GeometryNodeInputRadius", -560, -420)
    scale = node("ShaderNodeMath", -300, -420)
    scale.operation = "MULTIPLY"
    instance = node("GeometryNodeInstanceOnPoints", 50, 120)
    velocity = node("GeometryNodeInputNamedAttribute", -560, -650)
    velocity.data_type = "FLOAT_VECTOR"
    velocity.inputs["Name"].default_value = "velocity"
    speed = node("ShaderNodeVectorMath", -280, -650)
    speed.operation = "LENGTH"
    for source, destination in (
        (inputs.outputs["Geometry"], instance.inputs["Points"]),
        (inputs.outputs["Size"], scale.inputs[1]),
        (inputs.outputs["Material"], set_material.inputs["Material"]),
        (radius.outputs["Radius"], scale.inputs[0]),
        (scale.outputs[0], instance.inputs["Scale"]),
        (ico.outputs["Mesh"], set_material.inputs["Geometry"]),
        (set_material.outputs["Geometry"], instance.inputs["Instance"]),
        (instance.outputs["Instances"], output.inputs["Geometry"]),
        (inputs.outputs["Geometry"], output.inputs["Points"]),
        (velocity.outputs["Attribute"], output.inputs["Velocity"]),
        (velocity.outputs["Attribute"], speed.inputs[0]),
        (speed.outputs["Value"], output.inputs["Speed"]),
        (radius.outputs["Radius"], output.inputs["Radius"]),
    ):
        tree.links.new(source, destination)
    return tree


def points_asset():
    return (
        next(
            (tree for tree in bpy.data.node_groups if tree.get("fluidity_asset_version") == ASSET_VERSION),
            None,
        )
        or create_points_asset()
    )


def relative(path):
    value = bpy.path.relpath(str(path))
    if not value.startswith("//"):
        raise jobs.ToolError("Move the cache to the same drive as the Blender project.")
    return value


def inspection(cache, frame):
    return json.loads(jobs.checked("blender_bridge", "inspect", cache, frame))


def committed_last(cache):
    """Find the last publication record; the bridge subsequently verifies its contents."""
    if not cache.is_dir():
        raise jobs.ToolError("Choose a directory containing a committed FluidiTy cache.")
    frames = [
        int(p.stem)
        for p in (cache / "frames").glob("*.flmf")
        if len(p.stem) == 20 and p.stem.isascii() and p.stem.isdecimal()
    ]
    if not frames:
        raise jobs.ToolError("This cache has no committed frames.")
    return max(frames)


def attach(scene, cache):
    require_cadence(scene)
    relative_cache = relative(cache)
    last = committed_last(cache)
    info = inspection(cache, last)
    identity = info["execution_id"]
    existing = scene.fluidity.playback_object
    if existing and not existing.get("fluidity_points"):
        raise jobs.ToolError("The saved Points binding is not a FluidiTy object.")
    if existing:
        reader = next(
            (
                mod
                for mod in existing.modifiers
                if mod.type == "MESH_SEQUENCE_CACHE" and mod.object_path == "/FluidiTyPoints"
            ),
            None,
        )
        if reader is None:
            raise jobs.ToolError("Restore the FluidiTy Points cache reader before relinking.")

    archive = cache / f"FluidiTyPoints-{uuid4().hex}.abc"
    archive_path = relative(archive)
    jobs.checked("blender_bridge", "export", cache, last, archive, timeout=60)
    if existing and scene.fluidity.attached_cache == relative_cache and existing.get("fluidity_execution") == identity:
        old_file = Path(bpy.path.abspath(reader.cache_file.filepath)) if reader.cache_file else None
        # Export revalidated all commits. Reuse the existing datablocks when the
        # carrier is byte-identical; a replaced carrier goes through import below.
        if (
            existing.get("fluidity_last") == last
            and old_file
            and old_file.is_file()
            and filecmp.cmp(archive, old_file, shallow=False)
        ):
            archive.unlink()
            return last

    old_objects = set(bpy.data.objects)
    old_cache_files = set(bpy.data.cache_files)
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    keep = None
    complete = False
    try:
        result = bpy.ops.wm.alembic_import(
            filepath=str(archive),
            set_frame_range=False,
            as_background_job=False,
            always_add_cache_reader=True,
        )
        imported = list(set(bpy.data.objects) - old_objects)
        if result != {"FINISHED"} or len(imported) != 1 or imported[0].type != "POINTCLOUD":
            raise jobs.ToolError("Blender could not open the verified Points archive.")
        candidate = imported[0]
        source = next((mod for mod in candidate.modifiers if mod.type == "MESH_SEQUENCE_CACHE"), None)
        if source is None or source.object_path != "/FluidiTyPoints":
            raise jobs.ToolError("Blender could not find the native Points reader.")
        source.cache_file.velocity_name = ".velocities"
        source.cache_file.velocity_unit = "SECOND"
        source.cache_file.filepath = archive_path
        scene.frame_set(last)
        evaluated = candidate.evaluated_get(bpy.context.evaluated_depsgraph_get())
        if len(evaluated.data.points) != len(info["points"]):
            raise jobs.ToolError("Blender could not read the last committed Points sample.")
        if existing:
            # Keep the object, downstream modifier and any user edits in place.
            previous_cache = reader.cache_file
            reader.cache_file = source.cache_file
            for constraint in existing.constraints:
                if constraint.type == "TRANSFORM_CACHE" and constraint.cache_file == previous_cache:
                    constraint.cache_file = source.cache_file
        else:
            candidate.name = "FluidiTy Points"
            modifier = candidate.modifiers.new("FluidiTy Points v1", "NODES")
            modifier.node_group = points_asset()
            tree = modifier.node_group
            inputs = modifier.properties.inputs
            getattr(inputs, tree["fluidity_size_socket"]).value = 1.0
            getattr(inputs, tree["fluidity_material_socket"]).value = next(
                item.default_value
                for item in tree.interface.items_tree
                if item.identifier == tree["fluidity_material_socket"]
            )
            collection = bpy.data.collections.new("FluidiTy Playback")
            scene.collection.children.link(collection)
            for previous in list(candidate.users_collection):
                previous.objects.unlink(candidate)
            collection.objects.link(candidate)
            keep = existing = candidate
            existing["fluidity_points"] = True
            scene.fluidity.playback_object = existing
        existing["fluidity_execution"] = identity
        existing["fluidity_last"] = last
        scene.fluidity.attached_cache = relative_cache
        complete = True
    finally:
        scene.frame_set(saved_frame, subframe=saved_subframe)
        for obj in set(bpy.data.objects) - old_objects:
            if obj != keep:
                bpy.data.objects.remove(obj, do_unlink=True)
        unused = tuple(
            cache_file for cache_file in set(bpy.data.cache_files) - old_cache_files if cache_file.users == 0
        )
        if unused:
            bpy.data.batch_remove(ids=unused)
        if not complete:
            try:
                archive.unlink(missing_ok=True)
            except PermissionError:
                # Blender can hold a native read handle until its dependency graph updates.
                pass
    return last
