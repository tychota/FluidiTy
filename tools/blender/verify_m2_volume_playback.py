"""Verify native Volume sequence playback on the pinned Blender beach control."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import openvdb


GRID_NAMES = ("foam_count_per_m3", "bubble_count_per_m3", "gas_fraction", "fluidity_known")
OBJECT_NAME = "FluidiTy Whitewater Volumes"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path, expected):
    grids, metadata = openvdb.readAll(str(path))
    by_name = {grid.name: grid for grid in grids}
    assert set(by_name) == set(GRID_NAMES)
    assert all(metadata[key] == value for key, value in expected.items())
    for grid in grids:
        assert grid.transform.indexToWorld((0, 0, 0)) == (-1, 1, -1)
        assert grid.transform.indexToWorld((1, -1, 1)) == (-0.75, 0.75, -0.75)
    known = by_name["fluidity_known"]
    assert known.activeVoxelCount() == 24 ** 3
    assert known.getAccessor().getValue((0, 0, 0)) == 1
    assert known.getAccessor().getValue((0, 1, 0)) == 0
    inventory = [
        math.fsum(item.value * item.count for item in by_name[name].citerOnValues()) * 0.25 ** 3
        for name in GRID_NAMES[:-1]
    ]
    return {"metadata": expected, "inventory": inventory, "sha256": sha256(path)}


def inspect_sequence():
    scene = bpy.context.scene
    for frame in range(3):
        scene.frame_set(frame)
        obj = bpy.data.objects[OBJECT_NAME].evaluated_get(bpy.context.evaluated_depsgraph_get())
        assert obj.data.grids.load(), obj.data.grids.error_message
        assert {grid.name for grid in obj.data.grids} == set(GRID_NAMES)
        assert Path(obj.data.grids.frame_filepath).name == f"beach-volume_{frame:04d}.vdb"


def prepare(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
    from fluidity.volume_playback import export_volume_frame

    output = args.output.resolve()
    paths = [output / f"beach-volume_{frame:04d}.vdb" for frame in range(3)]
    before = {path: (sha256(path), path.stat().st_mtime_ns) for path in args.cache.rglob("*") if path.is_file()}
    reports = []
    for frame, path in enumerate(paths):
        expected = export_volume_frame(args.bridge, args.cache, frame, path)
        reports.append(inspect(path, expected))
    assert reports[0]["inventory"] == [0, 0, 0]
    for actual, target in zip(reports[2]["inventory"], (16, 48, 48 * 4 * math.pi / 3 * 0.05 ** 3), strict=True):
        assert math.isclose(actual, target, rel_tol=1e-7), (actual, target)
    try:
        export_volume_frame(args.bridge, args.cache, 2, paths[2])
    except FileExistsError:
        pass
    else:
        raise AssertionError("Volume export overwrote a carrier")
    assert sha256(paths[2]) == reports[2]["sha256"]
    assert before == {path: (sha256(path), path.stat().st_mtime_ns) for path in args.cache.rglob("*") if path.is_file()}

    volume = bpy.data.volumes.new("Whitewater density")
    volume.filepath = str(paths[0])
    volume.is_sequence = True
    volume.frame_start = 0
    # Blender uses one-based sequence indices. The cache filenames use bake ordinals.
    volume.frame_offset = -1
    volume.frame_duration = 3
    volume.sequence_mode = "CLIP"
    obj = bpy.data.objects.new(OBJECT_NAME, volume)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 2
    inspect_sequence()
    volume.filepath = "//beach-volume_0000.vdb"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "beach-effects.blend"), relative_remap=False)
    (output / "volume-receipt.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print("PASS native Blender Volume sequence, inventory, transform, empty validity and immutable export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reopen", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    assert bpy.app.build_hash == b"e13b728c6880", bpy.app.build_hash
    if args.reopen:
        inspect_sequence()
        reports = json.loads(args.reopen.read_text(encoding="utf-8"))
        for frame, report in enumerate(reports):
            path = args.reopen.parent / f"beach-volume_{frame:04d}.vdb"
            assert inspect(path, report["metadata"]) == report
        print("PASS relocated native Volume playback")
    else:
        assert all((args.bridge, args.cache, args.output))
        prepare(args)
