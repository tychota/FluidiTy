"""Export verified gas frames as native Blender OpenVDB volumes."""

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from .volume_playback import _float_voxel


GAS_GRID_NAMES = ("density", "temperature", "fuel", "flame", "soot")


def export_gas_volume_frame(solver, cache, frame, destination):
    """Write one immutable VDB carrier from a solver-verified gas checkpoint."""
    import openvdb

    completed = subprocess.run(
        [os.fspath(solver), "check", "--cache", os.fspath(cache), "--frame", str(frame)],
        capture_output=True, encoding="utf-8", check=True, timeout=90,
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    if result.get("model") != "gas" or result.get("frame") != frame:
        raise ValueError("Solver did not return the requested gas frame")
    nx, ny, nz = result["extent"]
    fields = result["gas_fields"]
    if min(nx, ny, nz) <= 0 or len(fields) != nx * ny * nz:
        raise ValueError("Gas field extent differs from the checkpoint")
    spacing = result["spacing_m"]
    x0, y0, z0 = result["origin_m"]
    if spacing <= 0 or not all(map(math.isfinite, (spacing, x0, y0, z0))):
        raise ValueError("Invalid gas grid transform")
    transform = openvdb.createLinearTransform(spacing)
    half = 0.5 * spacing
    transform.postTranslate((x0 + half, -z0 - half, y0 + half))
    grids = [openvdb.FloatGrid() for _ in GAS_GRID_NAMES]
    for name, grid in zip(GAS_GRID_NAMES, grids, strict=True):
        grid.name = name
        grid.transform = transform
    accessors = [grid.getAccessor() for grid in grids]
    for index, values in enumerate(fields):
        z, remainder = divmod(index, nx * ny)
        y, x = divmod(remainder, nx)
        for accessor, value in zip(accessors, values, strict=True):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid gas field value")
            # Values at or below half the smallest binary32 subnormal round to zero.
            if value > 2.0 ** -150:
                accessor.setValueOn((x, -z, y), _float_voxel(value))
    metadata = {"fluidity_frame": frame, "fluidity_scene": result["scene_digest"]}
    destination = Path(destination)
    if destination.suffix != ".vdb":
        raise ValueError("Gas Volume carrier needs a .vdb extension")
    with tempfile.TemporaryDirectory(prefix=".fluidity-gas-", dir=destination.parent) as staging:
        temporary = Path(staging) / "frame.vdb"
        openvdb.write(str(temporary), grids, metadata)
        os.link(temporary, destination)
    return metadata


def create_gas_volume_material():
    """Create an editable native material asset using soot and flame for shading."""
    import bpy

    material = bpy.data.materials.new("FluidiTy Fire Volume")
    material.use_nodes = True
    material.asset_mark()
    material.asset_data.description = "Shade baked FluidiTy gas from soot, flame and temperature grids."
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (420, 40)
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    volume.location = (100, 40)
    volume.inputs["Density"].default_value = 30.0
    volume.inputs["Density Attribute"].default_value = "soot"
    volume.inputs["Temperature Attribute"].default_value = "temperature"
    volume.inputs["Blackbody Intensity"].default_value = 0.0
    volume.inputs["Temperature"].default_value = 1.0
    volume.inputs["Emission Color"].default_value = (1.0, 0.25, 0.02, 1.0)
    flame = tree.nodes.new("ShaderNodeAttribute")
    flame.attribute_name = "flame"
    flame.location = (-160, -110)
    tree.links.new(flame.outputs["Fac"], volume.inputs["Emission Strength"])
    tree.links.new(volume.outputs[0], output.inputs["Volume"])
    return material


def attach_gas_volume(scene, first_path, frame_start, frame_count):
    """Attach a Blender Volume sequence asset with a native volume material."""
    import bpy

    if frame_count < 1:
        raise ValueError("Gas playback requires at least one frame")
    volume = bpy.data.volumes.new("FluidiTy Gas Fields")
    volume.filepath = os.fspath(first_path)
    volume.is_sequence = True
    volume.frame_start = frame_start
    volume.frame_offset = -1
    volume.frame_duration = frame_count
    volume.sequence_mode = "CLIP"
    volume.materials.append(create_gas_volume_material())
    obj = bpy.data.objects.new("FluidiTy Fire Volume", volume)
    obj.asset_mark()
    obj.asset_data.description = "Native Volume sequence for committed FluidiTy gas fields."
    scene.collection.objects.link(obj)
    return obj
