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

POINT_CLOUD_OBJECT_NAME = "FluidiTyPoints"
ARCHIVE_NAME = f"{POINT_CLOUD_OBJECT_NAME}.abc"
RELATIVE_ARCHIVE_PATH = f"//{ARCHIVE_NAME}"
EXPECTED_POINT_COUNTS_BY_FRAME = {1: 3, 2: 5, 3: 0, 4: 4}
TIMELINE_SEEK_ORDER = (1, 2, 3, 4, 3, 1)
TIMELINE_FRAMES_PER_SECOND = 24
GEOMETRY_ABS_TOLERANCE = 1e-6
EXPECTED_DIAGNOSTIC_VELOCITY_M_PER_S = (1.0, -2.0, 3.0)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_pinned_blender_build(report):
    blender_build = {
        "version": bpy.app.version_string,
        "hash": bpy.app.build_hash.decode("ascii"),
        "type": bpy.app.build_type.decode("ascii"),
        "platform": bpy.app.build_platform.decode("ascii"),
        "pointer_bits": struct.calcsize("P") * 8,
        "machine": platform.machine(),
    }
    report["blender"] = blender_build
    require(
        blender_build
        == {
            "version": "5.2.3 LTS",
            "hash": "e13b728c6880",
            "type": "Release",
            "platform": "Windows",
            "pointer_bits": 64,
            "machine": "AMD64",
        },
        f"Blender build does not match the pinned Windows x64 build: {blender_build}",
    )


def get_native_point_cache_binding(scene):
    point_cloud_object = scene.objects.get(POINT_CLOUD_OBJECT_NAME)
    require(
        point_cloud_object is not None and point_cloud_object.type == "POINTCLOUD",
        f"Expected point cloud {POINT_CLOUD_OBJECT_NAME}",
    )
    modifiers = [
        cache_modifier
        for cache_modifier in point_cloud_object.modifiers
        if cache_modifier.type == "MESH_SEQUENCE_CACHE"
    ]
    require(len(modifiers) == 1, f"Expected one native cache modifier, got {len(modifiers)}")
    cache_modifier = modifiers[0]
    cache_file = cache_modifier.cache_file
    require(cache_file is not None, "The native cache modifier has no cache file")
    return point_cloud_object, cache_modifier, cache_file


def verify_relative_cache_binding(scene, scene_path, report):
    point_cloud_object, cache_modifier, cache_file = get_native_point_cache_binding(scene)
    archive = Path(bpy.path.abspath(cache_file.filepath)).resolve()
    report["binding"] = {
        "object_type": point_cloud_object.type,
        "modifier_type": cache_modifier.type,
        "alembic_object_path": cache_modifier.object_path,
        "cache_path": cache_file.filepath,
        "resolved_cache_path": str(archive),
        "cache_present": archive.is_file(),
        "velocity_name": cache_file.velocity_name,
        "velocity_unit": cache_file.velocity_unit,
    }
    require(cache_modifier.show_viewport, "Native cache modifier is disabled in the viewport")
    require(
        cache_modifier.object_path == f"/{POINT_CLOUD_OBJECT_NAME}",
        f"Unexpected Alembic object path: {cache_modifier.object_path!r}",
    )
    require(
        cache_file.filepath == RELATIVE_ARCHIVE_PATH, f"Expected relative cache binding, got {cache_file.filepath!r}"
    )
    require(archive == scene_path.with_name(ARCHIVE_NAME), f"Cache does not resolve beside the scene: {archive}")
    return point_cloud_object


def verify_scene_playback_settings(scene, report):
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
            "fps": TIMELINE_FRAMES_PER_SECOND,
            "fps_base": 1.0,
            "unit_system": "METRIC",
            "unit_scale": 1.0,
            "frame_change_handlers": 0,
        },
        f"Unexpected scene settings: {settings}",
    )


def save_point_cache_scene(scene_path, report):
    scene_path = scene_path.resolve()
    report["scene_path"] = str(scene_path)
    archive = scene_path.with_name(ARCHIVE_NAME)
    require(archive.is_file(), f"Missing input archive: {archive}")

    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.fps = TIMELINE_FRAMES_PER_SECOND
    scene.render.fps_base = 1.0
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.frame_start = min(EXPECTED_POINT_COUNTS_BY_FRAME)
    scene.frame_end = max(EXPECTED_POINT_COUNTS_BY_FRAME)
    scene.frame_set(scene.frame_start)
    verify_scene_playback_settings(scene, report)

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
    _, _, cache_file = get_native_point_cache_binding(scene)
    cache_file.velocity_name = ".velocities"
    cache_file.velocity_unit = "SECOND"
    cache_file.filepath = RELATIVE_ARCHIVE_PATH
    result = bpy.ops.wm.save_mainfile()
    require(result == {"FINISHED"}, f"Could not save scene: {result}")
    verify_relative_cache_binding(scene, scene_path, report)
    require(report["binding"]["cache_present"], "Saved scene lost the external Alembic archive")


def fixture_value_matches(value, expected):
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=GEOMETRY_ABS_TOLERANCE)


def fixture_vector_matches(vector, expected):
    return all(fixture_value_matches(value, target) for value, target in zip(vector, expected, strict=True))


def read_timeline_point_sample(scene, point_cloud_object, timeline_frame):
    scene.frame_set(timeline_frame)
    evaluated_point_cloud_object = point_cloud_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_point_cloud_data = evaluated_point_cloud_object.data
    points = [
        {"position": tuple(evaluated_point_cloud_object.matrix_world @ point.co), "radius": point.radius}
        for point in evaluated_point_cloud_data.points
    ]
    velocity_attribute = evaluated_point_cloud_data.attributes.get("velocity")
    diagnostic_velocities_m_per_s = None
    if (
        velocity_attribute is not None
        and velocity_attribute.domain == "POINT"
        and velocity_attribute.data_type == "FLOAT_VECTOR"
    ):
        diagnostic_velocities_m_per_s = [tuple(value.vector) for value in velocity_attribute.data]

    # The empty sample has no vectors to check. Record whether Blender retained the attribute.
    diagnostic_velocity_matches_fixture = None
    if points:
        diagnostic_velocity_matches_fixture = (
            diagnostic_velocities_m_per_s is not None
            and len(diagnostic_velocities_m_per_s) == len(points)
            and all(
                fixture_vector_matches(velocity, EXPECTED_DIAGNOSTIC_VELOCITY_M_PER_S)
                for velocity in diagnostic_velocities_m_per_s
            )
        )
    return {
        "frame": timeline_frame,
        "expected_count": EXPECTED_POINT_COUNTS_BY_FRAME[timeline_frame],
        "count": len(points),
        "points": points,
        "velocity": {
            "domain": velocity_attribute.domain if velocity_attribute is not None else None,
            "data_type": velocity_attribute.data_type if velocity_attribute is not None else None,
            "values": diagnostic_velocities_m_per_s,
            "matches_fixture": diagnostic_velocity_matches_fixture,
        },
    }


def verify_point_cache_playback(report):
    scene = bpy.context.scene
    require(bpy.data.filepath, "Verify must open the saved scene")
    scene_path = Path(bpy.data.filepath).resolve()
    report["scene_path"] = str(scene_path)
    point_cloud_object = verify_relative_cache_binding(scene, scene_path, report)
    verify_scene_playback_settings(scene, report)
    errors = report["errors"]
    if not report["binding"]["cache_present"]:
        errors.append("External Alembic archive is missing")
    samples = report["samples"] = []
    for timeline_frame in TIMELINE_SEEK_ORDER:
        sample = read_timeline_point_sample(scene, point_cloud_object, timeline_frame)
        samples.append(sample)
        if sample["count"] != sample["expected_count"]:
            errors.append(f"Frame {timeline_frame}: expected {sample['expected_count']} points, got {sample['count']}")
        # The index is local to each sample. The writer checks particle IDs in the archive.
        for sample_point_index, point in enumerate(sample["points"]):
            expected_world_position_m = (timeline_frame, sample_point_index, -(timeline_frame + sample_point_index))
            expected_display_radius_m = (sample_point_index + 1) / 16
            if not fixture_vector_matches(point["position"], expected_world_position_m):
                errors.append(
                    f"Frame {timeline_frame} point {sample_point_index}: expected position {expected_world_position_m}, got {point['position']}"
                )
            if not fixture_value_matches(point["radius"], expected_display_radius_m):
                errors.append(
                    f"Frame {timeline_frame} point {sample_point_index}: expected radius {expected_display_radius_m}, got {point['radius']}"
                )
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
        verify_pinned_blender_build(report)
        if args.mode == "prepare":
            save_point_cache_scene(args.scene, report)
        else:
            verify_point_cache_playback(report)
        report["status"] = "passed"
    except Exception as error:
        if not report["errors"]:
            report["errors"].append(str(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
