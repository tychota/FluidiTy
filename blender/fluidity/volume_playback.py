"""Convert one authenticated FluidiTy volume frame into a native Blender carrier."""

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


DENSITY_GRID_NAMES = ("foam_count_per_m3", "bubble_count_per_m3", "gas_fraction")
KNOWN_GRID_NAME = "fluidity_known"


def _index_to_blender(index, extent):
    x, remainder = divmod(index, extent[1] * extent[2])
    y, z = divmod(remainder, extent[2])
    return x, -z, y


def _float_voxel(value):
    rounded, = struct.unpack("<f", struct.pack("<f", value))
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError("Positive volume density cannot be represented in OpenVDB FloatGrid")
    return rounded


def write_volume_frame(record, destination):
    """Write four named grids without overwriting an existing carrier.

    The record comes from the native bridge after cache and schema validation.
    Grid centers are in Blender world meters, with the same Y-up to Z-up
    rotation as Alembic playback. Active known voxels are distinct from density.
    """
    import openvdb

    destination = Path(destination)
    if destination.suffix != ".vdb":
        raise ValueError("Volume carrier needs a .vdb extension")
    extent = tuple(record["extent"])
    voxel_count = math.prod(extent)
    spacing_m = record["spacing_m"]
    x, y, z = record["origin_m"]
    transform = openvdb.createLinearTransform(spacing_m)
    transform.postTranslate((x, -z, y))
    names = (*DENSITY_GRID_NAMES, KNOWN_GRID_NAME)
    grids = [openvdb.FloatGrid() for _ in names]
    for name, grid in zip(names, grids, strict=True):
        grid.name = name
        grid.transform = transform
    known = grids[-1]
    runs = record["known_runs"]
    if runs == [[0, voxel_count]]:
        known.fill((0, 1 - extent[2], 0), (extent[0] - 1, 0, extent[1] - 1), 1.0, True)
    else:
        # A flat run can cross multiple z rows. Each fill is one row segment.
        for first, count in runs:
            stop = first + count
            while first < stop:
                start_coord = _index_to_blender(first, extent)
                end = min(stop, first + extent[2] - first % extent[2])
                known.fill(_index_to_blender(end - 1, extent), start_coord, 1.0, True)
                first = end
    accessors = [grid.getAccessor() for grid in grids[:-1]]
    for index, *values in record["density"]:
        coordinate = _index_to_blender(index, extent)
        for accessor, value in zip(accessors, values, strict=True):
            if value:
                accessor.setValueOn(coordinate, _float_voxel(value))
    metadata = {
        "fluidity_frame": record["frame"],
        "fluidity_execution": record["execution_id"],
        "fluidity_source_execution": record["source_execution_id"],
        "fluidity_payload": record["payload_digest"],
    }
    with tempfile.TemporaryDirectory(prefix=".fluidity-volume-", dir=destination.parent) as staging:
        temporary = Path(staging) / "frame.vdb"
        openvdb.write(str(temporary), grids, metadata)
        os.link(temporary, destination)
    return metadata


def export_volume_frame(bridge, cache, frame, destination):
    """Read a committed Volumes frame through the native bridge and export it."""
    completed = subprocess.run(
        [os.fspath(bridge), "inspect-volume", os.fspath(cache), str(frame)],
        capture_output=True, encoding="utf-8", check=True, timeout=90,
    )
    record = json.loads(completed.stdout)
    if record["frame"] != frame:
        raise ValueError("Volume bridge returned the wrong frame")
    return write_volume_frame(record, destination)
