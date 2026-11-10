"""Exercise the public extension operators against the actual packaged solver."""

import hashlib
import importlib
import math
import shutil
import sys
import time
from pathlib import Path

import bpy

tools = Path(__file__).resolve().parent
sys.path[:0] = [str(tools), str(tools.parents[1] / "blender")]
from make_m1_example import create_example


def committed(root):
    return {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ("frames", "blobs", "checkpoints")
        for p in (root / directory).glob("*")
        if p.is_file() and p.suffix in {".flmf", ".flpt", ".json", ".flcr"}
    }


def drain(fluidity, scene, cancel_after=None):
    deadline = time.monotonic() + 60
    cancelled = False
    while fluidity.is_running(scene):
        assert time.monotonic() < deadline, scene.fluidity.message
        fluidity.tick()
        if cancel_after is not None and scene.fluidity.committed_frame >= cancel_after and not cancelled:
            assert bpy.ops.fluidity.cancel() == {"FINISHED"}
            cancelled = True
        time.sleep(0.005)
    assert cancelled or cancel_after is None, scene.fluidity.message
    return scene.fluidity.message


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    root = Path(arguments[0]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if arguments[1] == "--installed":
        import addon_utils

        module_name = "bl_ext.user_default.fluidity"
        assert addon_utils.check(module_name) == (True, True)
        fluidity = importlib.import_module(module_name)
        bpy.ops.wm.open_mainfile(filepath=str(Path(fluidity.__file__).parent / "examples" / "First Bake.blend"))
    else:
        fluidity = importlib.import_module("fluidity")
        fluidity.jobs.BIN_DIR = Path(arguments[1]).resolve()
        fluidity.register()
        create_example()
    scene = bpy.context.scene
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "First bake – fluide.blend"))
    settings = scene.fluidity
    bpy.context.scene.render.fps = 25
    assert bpy.ops.fluidity.check_scene() == {"CANCELLED"}
    assert "fps" in settings.message
    bpy.context.scene.render.fps = 24
    scene.keyframe_insert(data_path="gravity", index=2, frame=0)
    assert bpy.ops.fluidity.check_scene() == {"CANCELLED"}
    assert "animation" in settings.message
    scene.animation_data_clear()
    assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, settings.message
    assert bpy.ops.fluidity.preview() == {"FINISHED"}, settings.message
    drain(fluidity, scene)
    preview = root / "FluidiTy Preview"
    assert committed(preview), settings.message
    assert (preview / ".simulate-job-status").read_text().strip() == "v1 complete", settings.message
    assert len(list((preview / "frames").glob("*.flmf"))) == 4
    assert not (root / "FluidiTy Bake").exists()
    assert bpy.ops.fluidity.bake() == {"FINISHED"}, settings.message
    drain(fluidity, scene, cancel_after=1)
    bake = root / "FluidiTy Bake"
    assert (bake / ".simulate-job-status").read_text().find("cancelled") >= 0, settings.message
    before = committed(bake)
    assert before
    assert bpy.ops.fluidity.resume() == {"FINISHED"}, settings.message
    drain(fluidity, scene)
    after = committed(bake)
    assert len(after) > len(before), settings.message
    assert all(after.get(path) == digest for path, digest in before.items())
    assert (preview / "resolved.flrs").read_bytes() != (bake / "resolved.flrs").read_bytes()
    from bpy_extras.anim_utils import action_get_channelbag_for_slot

    data = settings.paddle.animation_data
    x_curve = next(
        curve
        for curve in action_get_channelbag_for_slot(data.action, data.action_slot).fcurves
        if curve.array_index == 0
    )
    x_curve.keyframe_points[-1].co.y += 0.02
    x_curve.update()
    assert bpy.ops.fluidity.resume() == {"CANCELLED"}
    assert committed(bake) == after
    scene.render.fps = 30
    assert bpy.ops.fluidity.attach() == {"CANCELLED"}
    assert settings.playback_object is None and "24 fps" in settings.message
    scene.render.fps = 24
    assert bpy.ops.fluidity.attach() == {"FINISHED"}, settings.message
    point_object = settings.playback_object
    reader, display = point_object.modifiers
    first_archive = reader.cache_file.filepath
    assert first_archive.startswith("//")
    size_socket = display.node_group["fluidity_size_socket"]
    getattr(display.properties.inputs, size_socket).value = 1.35
    assert bpy.ops.fluidity.attach() == {"FINISHED"}, settings.message
    assert settings.playback_object == point_object
    assert math.isclose(getattr(display.properties.inputs, size_socket).value, 1.35, abs_tol=1e-6)
    assert reader.cache_file.filepath == first_archive
    scene.frame_set(3)
    assert bpy.ops.fluidity.inspect() == {"FINISHED"}, settings.message
    assert settings.inspect_frame == 3 and settings.point_count == 8 and settings.point_detail.startswith("ID ")
    # Copy the whole project, then move the cache alone and relink it explicitly.
    bpy.ops.wm.save_mainfile()
    relocated = root.with_name(root.name + " – déplacé")
    shutil.copytree(root, relocated)
    project = relocated / "First bake – fluide.blend"
    moved = relocated / "Cache indépendant"
    (relocated / bake.name).rename(moved)
    bpy.ops.wm.open_mainfile(filepath=str(project))
    scene = bpy.context.scene
    settings = scene.fluidity
    assert not Path(bpy.path.abspath(settings.playback_object.modifiers[0].cache_file.filepath)).exists()
    assert bpy.ops.fluidity.relink(directory=str(moved)) == {"FINISHED"}, settings.message
    point_object = settings.playback_object
    reader, display = point_object.modifiers
    assert math.isclose(getattr(display.properties.inputs, size_socket).value, 1.35, abs_tol=1e-6)
    for constraint in point_object.constraints:
        if constraint.type == "TRANSFORM_CACHE":
            assert constraint.cache_file == reader.cache_file
    prior_archive = reader.cache_file.filepath
    corrupt = relocated / "Bad cache"
    shutil.copytree(moved, corrupt)
    (corrupt / "frames" / "00000000000000000002.flmf").unlink()
    assert bpy.ops.fluidity.relink(directory=str(corrupt)) == {"CANCELLED"}
    assert settings.playback_object == point_object and reader.cache_file.filepath == prior_archive
    assert committed(moved) == after
    bpy.ops.wm.save_mainfile()
    view = bpy.context.preferences.view
    view.language = "fr_FR"
    view.use_translate_interface = True
    assert bpy.app.translations.pgettext_iface("Check Scene", "FluidiTy") == "Vérifier la scène"
    assert bpy.app.translations.pgettext_iface("Bake", "FluidiTy") == "Simuler"
    assert (
        bpy.app.translations.pgettext_iface("The bake settings changed. Choose a fresh Bake cache.")
        == "Les réglages ont changé. Choisissez un nouveau cache de simulation."
    )
    view.language = "en_US"
    assert bpy.app.translations.pgettext_iface("Resume", "FluidiTy") == "Resume"
    print("Verified Check Scene, independent Preview, Bake, cancel and Resume on", settings.backend)
    print("Verified Inspect, repeat attachment, relocation, independent relink, failed replacement and translations")


main()
