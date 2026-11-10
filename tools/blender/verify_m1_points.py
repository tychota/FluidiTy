"""Run in the pinned Blender against real, previously committed solver caches."""

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from point_cache_playback import verify_pinned_blender_build


def run(bridge, *args, ok=True):
    result = subprocess.run(
        [str(bridge), *map(str, args)], capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert (result.returncode == 0) == ok, result.stderr
    return result.stdout


def check_playback(bridge, cache, last, project):
    archive = project / "Points échantillons.abc"
    scene_path = project / "FluidiTy Test.blend"
    project.mkdir()
    run(bridge, "export", cache, last, archive)
    expected = [json.loads(run(bridge, "inspect", cache, frame)) for frame in range(last + 1)]
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))
    assert bpy.ops.wm.alembic_import(
        filepath=str(archive), as_background_job=False, set_frame_range=False, always_add_cache_reader=True
    ) == {"FINISHED"}
    obj = next(obj for obj in scene.objects if obj.type == "POINTCLOUD")
    modifier = next(mod for mod in obj.modifiers if mod.type == "MESH_SEQUENCE_CACHE")
    assert modifier.object_path == "/FluidiTyPoints"
    modifier.cache_file.filepath = bpy.path.relpath(str(archive))
    assert modifier.cache_file.filepath == "//" + archive.name
    bpy.ops.wm.save_mainfile()
    relocated = project.with_name(project.name + " déplacé")
    shutil.copytree(project, relocated)
    bpy.ops.wm.open_mainfile(filepath=str(relocated / scene_path.name))
    scene = bpy.context.scene
    obj = next(obj for obj in scene.objects if obj.type == "POINTCLOUD")
    assert Path(bpy.path.abspath(obj.modifiers[0].cache_file.filepath)).resolve() == relocated / archive.name
    for frame in [*range(last + 1), 0, last]:
        scene.frame_set(frame)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        sample = evaluated.data.points
        points = expected[frame]["points"]
        assert expected[frame]["time_seconds"] == frame / 24
        assert len(sample) == len(points), (frame, len(sample), len(points))
        for actual, target in zip(sample, points, strict=True):
            for x, y in zip(evaluated.matrix_world @ actual.co, target["position"], strict=True):
                assert math.isclose(x, y, abs_tol=1e-6), (frame, actual, target)
            assert math.isclose(actual.radius, target["radius"], abs_tol=1e-6)
    bpy.data.objects.remove(obj, do_unlink=True)
    print("M1 point playback counts:", [len(frame["points"]) for frame in expected])
    return [len(frame["points"]) for frame in expected]


def main():
    bridge, real_cache, empty_cache, output = (Path(p).resolve() for p in sys.argv[sys.argv.index("--") + 1 :])
    verify_pinned_blender_build({})
    root = Path(tempfile.mkdtemp(dir=output))
    assert any(check_playback(bridge, real_cache, 3, root / "Solver output"))
    assert check_playback(bridge, empty_cache, 3, root / "Empty start")[0] == 0
    corrupt = root / "corrupt"
    shutil.copytree(empty_cache, corrupt)
    manifest = corrupt / "frames" / "00000000000000000000.flmf"
    saved_manifest = manifest.read_bytes()
    output_path = root / "failed.abc"
    manifest.unlink()
    run(bridge, "export", corrupt, 3, output_path, ok=False)
    assert not output_path.exists()
    manifest.write_bytes(saved_manifest)
    for points in (corrupt / "blobs").glob("*.flpt"):
        points.write_bytes(b"corrupt")
    run(bridge, "export", corrupt, 3, output_path, ok=False)
    assert not output_path.exists()
    print("Verified committed solver points in native Blender playback")


main()
