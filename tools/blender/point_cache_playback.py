"""Prepare and evaluate the native Alembic Points fixture inside Blender.

The producer writes one sample at each time from 1/24 through 4/24 seconds.
At 24 fps Blender selects those samples at timeline frames 1 through 4.
"""

import argparse
import json
import math
import platform
import struct
import sys
from pathlib import Path

import bpy

OBJECT_NAME = "FluidiTyPoints"
ARCHIVE_NAME = f"{OBJECT_NAME}.abc"
RELATIVE_BINDING = f"//{ARCHIVE_NAME}"
FRAME_COUNTS = {1: 3, 2: 5, 3: 0, 4: 4}
FRAME_ORDER = (1, 2, 3, 4, 3, 1)
FPS = 24
TOLERANCE = 1e-6
DIAGNOSTIC_VELOCITY = (1.0, -2.0, 3.0)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_build(report):
    build = {
        "version": bpy.app.version_string,
        "hash": bpy.app.build_hash.decode("ascii"),
        "type": bpy.app.build_type.decode("ascii"),
        "platform": bpy.app.build_platform.decode("ascii"),
        "pointer_bits": struct.calcsize("P") * 8,
        "machine": platform.machine(),
    }
    report["blender"] = build
    require(
        build
        == {
            "version": "5.2.3 LTS",
            "hash": "e13b728c6880",
            "type": "Release",
            "platform": "Windows",
            "pointer_bits": 64,
            "machine": "AMD64",
        },
        f"Blender build does not match the pinned Windows x64 build: {build}",
    )


def native_cache(scene):
    point_cloud = scene.objects.get(OBJECT_NAME)
    require(point_cloud is not None and point_cloud.type == "POINTCLOUD", f"Expected point cloud {OBJECT_NAME}")
    modifiers = [modifier for modifier in point_cloud.modifiers if modifier.type == "MESH_SEQUENCE_CACHE"]
    require(len(modifiers) == 1, f"Expected one native cache modifier, got {len(modifiers)}")
    modifier = modifiers[0]
    cache = modifier.cache_file
    require(cache is not None, "The native cache modifier has no cache file")
    return point_cloud, modifier, cache


def check_cache_binding(scene, scene_path, report):
    point_cloud, modifier, cache = native_cache(scene)
    archive = Path(bpy.path.abspath(cache.filepath)).resolve()
    report["binding"] = {
        "object_type": point_cloud.type,
        "modifier_type": modifier.type,
        "alembic_object_path": modifier.object_path,
        "cache_path": cache.filepath,
        "resolved_cache_path": str(archive),
        "cache_present": archive.is_file(),
        "velocity_name": cache.velocity_name,
        "velocity_unit": cache.velocity_unit,
    }
    require(modifier.show_viewport, "Native cache modifier is disabled in the viewport")
    require(modifier.object_path == f"/{OBJECT_NAME}", f"Unexpected Alembic object path: {modifier.object_path!r}")
    require(cache.filepath == RELATIVE_BINDING, f"Expected relative cache binding, got {cache.filepath!r}")
    require(archive == scene_path.with_name(ARCHIVE_NAME), f"Cache does not resolve beside the scene: {archive}")
    return point_cloud


def check_scene(scene, report):
    settings = {
        "python_autoexec": bpy.context.preferences.filepaths.use_scripts_auto_execute,
        "fps": scene.render.fps,
        "fps_base": scene.render.fps_base,
        "unit_system": scene.unit_settings.system,
        "unit_scale": scene.unit_settings.scale_length,
        "frame_change_handlers": len(bpy.app.handlers.frame_change_pre) + len(bpy.app.handlers.frame_change_post),
    }
    report["scene_settings"] = settings
    require(
        settings
        == {
            "python_autoexec": False,
            "fps": FPS,
            "fps_base": 1.0,
            "unit_system": "METRIC",
            "unit_scale": 1.0,
            "frame_change_handlers": 0,
        },
        f"Unexpected scene settings: {settings}",
    )


def prepare(scene_path, report):
    scene_path = scene_path.resolve()
    report["scene_path"] = str(scene_path)
    archive = scene_path.with_name(ARCHIVE_NAME)
    require(archive.is_file(), f"Missing input archive: {archive}")

    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.frame_start = min(FRAME_COUNTS)
    scene.frame_end = max(FRAME_COUNTS)
    scene.frame_set(scene.frame_start)
    check_scene(scene, report)

    # Give Blender a base directory before assigning a relative path to the live modifier.
    bpy.context.preferences.filepaths.save_version = 0
    result = bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))
    require(result == {"FINISHED"}, f"Could not create scene: {result}")

    result = bpy.ops.wm.alembic_import(
        filepath=str(archive),
        set_frame_range=False,
        as_background_job=False,
        always_add_cache_reader=True,
    )
    require(result == {"FINISHED"}, f"Native Alembic import failed: {result}")
    _, _, cache = native_cache(scene)
    cache.velocity_name = ".velocities"
    cache.velocity_unit = "SECOND"
    cache.filepath = RELATIVE_BINDING
    result = bpy.ops.wm.save_mainfile()
    require(result == {"FINISHED"}, f"Could not save scene: {result}")
    check_cache_binding(scene, scene_path, report)
    require(report["binding"]["cache_present"], "Saved scene lost the external Alembic archive")


def close_enough(value, expected):
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=TOLERANCE)


def vector_matches(vector, expected):
    return all(close_enough(value, target) for value, target in zip(vector, expected, strict=True))


def sample_geometry(scene, point_cloud, frame):
    scene.frame_set(frame)
    evaluated = point_cloud.evaluated_get(bpy.context.evaluated_depsgraph_get())
    geometry = evaluated.data
    points = [
        {"position": tuple(evaluated.matrix_world @ point.co), "radius": point.radius} for point in geometry.points
    ]
    attribute = geometry.attributes.get("velocity")
    velocities = None
    if attribute is not None and attribute.domain == "POINT" and attribute.data_type == "FLOAT_VECTOR":
        velocities = [tuple(value.vector) for value in attribute.data]

    # The empty sample has no vectors to check. Record whether Blender retained the attribute.
    velocity_matches = None
    if points:
        velocity_matches = (
            velocities is not None
            and len(velocities) == len(points)
            and all(vector_matches(velocity, DIAGNOSTIC_VELOCITY) for velocity in velocities)
        )
    return {
        "frame": frame,
        "expected_count": FRAME_COUNTS[frame],
        "count": len(points),
        "points": points,
        "velocity": {
            "domain": attribute.domain if attribute is not None else None,
            "data_type": attribute.data_type if attribute is not None else None,
            "values": velocities,
            "matches_fixture": velocity_matches,
        },
    }


def verify(report):
    scene = bpy.context.scene
    require(bpy.data.filepath, "Verify must open the saved scene")
    scene_path = Path(bpy.data.filepath).resolve()
    report["scene_path"] = str(scene_path)
    point_cloud = check_cache_binding(scene, scene_path, report)
    check_scene(scene, report)
    errors = report["errors"]
    if not report["binding"]["cache_present"]:
        errors.append("External Alembic archive is missing")
    samples = report["samples"] = []
    for frame in FRAME_ORDER:
        sample = sample_geometry(scene, point_cloud, frame)
        samples.append(sample)
        if sample["count"] != sample["expected_count"]:
            errors.append(f"Frame {frame}: expected {sample['expected_count']} points, got {sample['count']}")
        # The index is local to each sample. The writer checks particle IDs in the archive.
        for index, point in enumerate(sample["points"]):
            expected_position = (frame, index, -(frame + index))
            expected_radius = (index + 1) / 16
            if not vector_matches(point["position"], expected_position):
                errors.append(
                    f"Frame {frame} point {index}: expected position {expected_position}, got {point['position']}"
                )
            if not close_enough(point["radius"], expected_radius):
                errors.append(f"Frame {frame} point {index}: expected radius {expected_radius}, got {point['radius']}")
    require(not errors, "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--scene", type=Path, help="output scene for prepare mode")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if (args.mode == "prepare") != (args.scene is not None):
        parser.error("--scene is required for prepare mode only")
    report = {"mode": args.mode, "status": "failed", "errors": []}
    try:
        check_build(report)
        if args.mode == "prepare":
            prepare(args.scene, report)
        else:
            verify(report)
        report["status"] = "passed"
    except Exception as error:
        if not report["errors"]:
            report["errors"].append(str(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
