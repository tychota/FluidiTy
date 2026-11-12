"""Explicit artist actions, support diagnostics and native interchange."""

import json
import os
import tempfile
from pathlib import Path
from typing import ClassVar

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import jobs, playback, surface_playback
from .preferences import get_preferences
from .scene_inputs import require_cadence


class IdleOperator:
    bl_translation_context = "FluidiTy"

    @classmethod
    def poll(cls, context):
        if jobs.is_running(context.scene):
            cls.poll_message_set("Wait for the running job or cancel it first.")
            return False
        return True


def select_object(context, obj):
    if obj is None or obj.name not in context.view_layer.objects:
        return False
    if context.object and context.object.mode != "OBJECT":
        return False
    for selected in context.selected_objects:
        selected.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return True


class FLUIDITY_OT_select_role(bpy.types.Operator):
    bl_idname = "fluidity.select_role"
    bl_label = "Select Input"
    bl_description = "Select the assigned input object in the current view layer"
    role: EnumProperty(items=tuple((name, name.title(), "") for name in ("domain", "liquid", "paddle", "collider")))

    def execute(self, context):
        if not select_object(context, getattr(context.scene.fluidity, self.role)):
            self.report({"WARNING"}, "Assign a visible object and leave Edit Mode first.")
            return {"CANCELLED"}
        return {"FINISHED"}


class FLUIDITY_OT_select_playback(bpy.types.Operator):
    bl_idname = "fluidity.select_playback"
    bl_label = "Select Playback for Geometry Nodes"
    bl_description = "Select the playback object, then open the Geometry Nodes workspace to connect its node group"

    def execute(self, context):
        if not select_object(context, context.scene.fluidity.playback_object):
            self.report({"WARNING"}, "Attach a cache and leave Edit Mode first.")
            return {"CANCELLED"}
        return {"FINISHED"}


class FLUIDITY_OT_initialize_scene(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.initialize_scene"
    bl_label = "Initialize Scene Timing & Units"
    bl_description = (
        "Set 24 fps, meter units, frame start zero and the preferred backend; keep input objects and animation"
    )
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        scene.render.fps, scene.render.fps_base = 24, 1
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1
        scene.frame_start = 0
        scene.frame_end = max(2, min(scene.frame_end, 240))
        preferences = get_preferences(context)
        if preferences:
            scene.fluidity.backend = preferences.default_backend
        return {"FINISHED"}


def diagnostics_report(scene):
    """Return a local support report without launching a process."""
    settings = scene.fluidity
    return json.dumps(
        {
            "addon": "FluidiTy",
            "blender": bpy.app.version_string,
            "build": bpy.app.build_hash.decode("ascii", errors="replace"),
            "backend": settings.backend,
            "running": jobs.is_running(scene),
            "frame": scene.frame_current,
            "committed_frame": settings.committed_frame,
            "message": settings.message,
            "diagnostic": settings.diagnostic,
        },
        indent=2,
        ensure_ascii=False,
    )


class FLUIDITY_OT_copy_diagnostics(bpy.types.Operator):
    bl_idname = "fluidity.copy_diagnostics"
    bl_label = "Copy Support Report"
    bl_description = (
        "Copy Blender/build, backend, frame and solver diagnostic details to the clipboard; no files are sent"
    )

    def execute(self, context):
        context.window_manager.clipboard = diagnostics_report(context.scene)
        self.report({"INFO"}, "Support report copied.")
        return {"FINISHED"}


class FLUIDITY_OT_import_cache(IdleOperator, bpy.types.Operator, ImportHelper):
    bl_idname = "fluidity.import_cache"
    bl_label = "Import Alembic / USD / OpenVDB"
    bl_description = (
        "Import external playback using Blender's native readers; imported data is not a resumable solver cache"
    )
    filename_ext = ""
    filter_glob: StringProperty(default="*.abc;*.usd;*.usda;*.usdc;*.usdz;*.vdb", options={"HIDDEN"})

    def execute(self, context):
        path = Path(self.filepath)
        try:
            if not path.is_file():
                raise ValueError("Choose an existing interchange file.")
            suffix = path.suffix.lower()
            if suffix == ".abc":
                result = bpy.ops.wm.alembic_import(filepath=str(path), as_background_job=False, set_frame_range=False)
            elif suffix in {".usd", ".usda", ".usdc", ".usdz"}:
                result = bpy.ops.wm.usd_import(filepath=str(path), set_frame_range=False)
            elif suffix == ".vdb":
                result = bpy.ops.object.volume_import(filepath=str(path))
            else:
                raise ValueError("Choose Alembic, USD or OpenVDB.")
            return result
        except (OSError, ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}


def export_alembic(scene, destination):
    """Copy a verified committed archive to a fresh path; preserve all scene state."""
    settings = scene.fluidity
    raw = settings.attached_cache or settings.bake_cache
    if not bpy.data.filepath or not raw:
        raise ValueError("Save the project and choose a committed cache first.")
    cache = Path(bpy.path.abspath(raw)).resolve()
    destination = Path(destination).resolve()
    if destination.suffix.lower() != ".abc":
        raise ValueError("Choose an .abc destination.")
    if destination.exists():
        raise ValueError("Choose a new export path; existing files are preserved.")
    last = playback.committed_last(cache)
    if last < 0:
        raise ValueError("This cache has no committed frames.")
    # The bridge validates every committed frame and removes failed partial exports.
    jobs.checked("blender_bridge", "export", cache, last, destination, timeout=120)


def export_usd(scene, destination):
    """Export verified raw points through Blender, restoring frame/selection/range on failure.

    USD is an interchange derivative, not a resumable cache. Blender may omit
    stable IDs. No display node modifiers, materials or external textures are exported.
    """
    destination = Path(destination).resolve()
    if destination.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("Choose a .usd, .usda or .usdc destination.")
    if destination.exists():
        raise ValueError("Choose a new export path; existing files are preserved.")
    if scene != bpy.context.scene:
        raise ValueError("Export the active scene.")
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        raise ValueError("Leave Edit Mode before exporting.")
    if not bpy.data.filepath:
        raise ValueError("Save the project and choose a committed cache first.")
    require_cadence(scene)
    if scene.unit_settings.scale_length != 1:
        raise ValueError("Use one Blender unit per meter.")
    cache = Path(bpy.path.abspath(scene.fluidity.attached_cache or scene.fluidity.bake_cache)).resolve()
    last = playback.committed_last(cache)
    # Blender 5.2.3 creates separate USD Points prims when the Alembic reader
    # transitions through an empty sample. Reject this lossy derivative before
    # publishing anything; the checked Ogawa export preserves empty frames.
    if any(not playback.inspection(cache, frame)["points"] for frame in range(last + 1)):
        raise ValueError("This cache contains empty frames. Export Alembic to preserve them.")
    before_objects = set(bpy.data.objects)
    before_clouds = set(bpy.data.pointclouds)
    before_caches = set(bpy.data.cache_files)
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    frame, subframe = scene.frame_current, scene.frame_subframe
    first, end = scene.frame_start, scene.frame_end
    temporary = tempfile.TemporaryDirectory(prefix="fluidity-usd-")
    try:
        archive = Path(temporary.name) / "Verified Points.abc"
        export_alembic(scene, archive)
        result = bpy.ops.wm.alembic_import(filepath=str(archive), set_frame_range=False, as_background_job=False)
        if result != {"FINISHED"}:
            raise RuntimeError("Blender could not open the verified Points archive.")
        imported = set(bpy.data.objects) - before_objects
        if not any(obj.type == "POINTCLOUD" for obj in imported):
            raise RuntimeError("Blender could not find the native Points reader.")
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in imported:
            obj.select_set(True)
            for constraint in obj.constraints:
                if constraint.type == "TRANSFORM_CACHE":
                    constraint.cache_file.velocity_name = ".velocities"
                    constraint.cache_file.velocity_unit = "SECOND"
            for modifier in obj.modifiers:
                if modifier.type == "MESH_SEQUENCE_CACHE":
                    modifier.cache_file.velocity_name = ".velocities"
                    modifier.cache_file.velocity_unit = "SECOND"
        scene.frame_start, scene.frame_end = 0, last
        # Publish only a completed USD. The temporary file is on the same filesystem.
        with tempfile.TemporaryDirectory(prefix=".fluidity-usd-", dir=destination.parent) as staging:
            staged = Path(staging) / destination.name
            result = bpy.ops.wm.usd_export(
                filepath=str(staged),
                selected_objects_only=True,
                export_animation=True,
                export_materials=False,
                export_points=True,
                export_volumes=False,
                convert_world_material=False,
            )
            if result != {"FINISHED"} or not staged.is_file():
                raise RuntimeError("Blender could not export the verified points.")
            # Windows rename refuses to replace an existing path. On other hosts,
            # use an exclusive hard link for the same no-overwrite guarantee.
            if os.name == "nt":
                staged.rename(destination)
            else:
                os.link(staged, destination)
    finally:
        for obj in set(bpy.data.objects) - before_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        unused = [item for item in set(bpy.data.pointclouds) - before_clouds if item.users == 0]
        unused += [item for item in set(bpy.data.cache_files) - before_caches if item.users == 0]
        if unused:
            bpy.data.batch_remove(ids=unused)
        scene.frame_start, scene.frame_end = first, end
        scene.frame_set(frame, subframe=subframe)
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in selected:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active
        bpy.context.view_layer.update()
        temporary.cleanup()


class FLUIDITY_OT_export_alembic(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.export_alembic"
    bl_label = "Export Committed Alembic Points"
    bl_description = (
        "Export all verified committed samples with positions, radii, velocities and IDs to a new .abc file"
    )
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.abc", options={"HIDDEN"})

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            export_alembic(context.scene, self.filepath)
        except (OSError, ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, "Committed points exported.")
        return {"FINISHED"}


class FLUIDITY_OT_export_usd(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.export_usd"
    bl_label = "Export Committed USD Points"
    bl_description = (
        "Export verified raw point animation through Blender; preserves frame and selection, omits display materials"
    )
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.usd;*.usda;*.usdc", options={"HIDDEN"})

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            export_usd(context.scene, self.filepath)
        except (OSError, ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, "Committed points exported.")
        return {"FINISHED"}


class FLUIDITY_OT_attach_surface(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.attach_surface"
    bl_label = "Attach Surface Cache"
    bl_description = "Attach committed Mesh frames with their phase and composition attributes"
    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from . import project_path

        try:
            surface_playback.attach(context.scene, project_path(self.directory))
        except (OSError, ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, "Surface attached. Save the project to keep the binding.")
        return {"FINISHED"}

CLASSES = (
    FLUIDITY_OT_attach_surface,
    FLUIDITY_OT_select_role,
    FLUIDITY_OT_select_playback,
    FLUIDITY_OT_initialize_scene,
    FLUIDITY_OT_copy_diagnostics,
    FLUIDITY_OT_import_cache,
    FLUIDITY_OT_export_alembic,
    FLUIDITY_OT_export_usd,
)
