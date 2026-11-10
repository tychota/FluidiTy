"""FluidiTy's bounded M1 Blender workflow."""

import json
import subprocess
import tempfile
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, Operator, Panel, PropertyGroup

from . import jobs, playback
from .scene_inputs import SceneError, require_cadence, scene_inputs
from .translations import TRANSLATIONS

is_running = jobs.is_running
tick = jobs.tick


class FluidiTySettings(PropertyGroup):
    domain: PointerProperty(name="Domain", type=Object)
    liquid: PointerProperty(name="Initial Liquid", type=Object)
    paddle: PointerProperty(name="Paddle", type=Object)
    cell_width: FloatProperty(name="Cell Width", subtype="DISTANCE", default=0.125, min=0.001, max=10)
    backend: EnumProperty(
        name="Backend",
        items=(("cuda", "RTX", "Use the NVIDIA CUDA device"), ("cpu", "CPU", "Use the CPU backend")),
        default="cuda",
    )
    preview_cache: StringProperty(name="Preview Cache", default="//FluidiTy Preview")
    bake_cache: StringProperty(name="Bake Cache", default="//FluidiTy Bake")
    attached_cache: StringProperty(name="Attached Cache")
    playback_object: PointerProperty(type=Object)
    point_index: IntProperty(name="Point Index", min=0, max=1023)
    committed_frame: IntProperty(default=-1, options={"SKIP_SAVE"})
    elapsed: FloatProperty(options={"SKIP_SAVE"})
    heartbeat_age: FloatProperty(default=-1, options={"SKIP_SAVE"})
    message: StringProperty(options={"SKIP_SAVE"})
    diagnostic: StringProperty(options={"SKIP_SAVE"})
    inspect_frame: IntProperty(default=-1, options={"SKIP_SAVE"})
    point_count: IntProperty(options={"SKIP_SAVE"})
    point_detail: StringProperty(options={"SKIP_SAVE"})


def project_path(value):
    if not bpy.data.filepath:
        raise SceneError("project", "Save the Blender project before using a cache.")
    if not value:
        raise SceneError("cache", "Choose a cache directory.")
    if not value.startswith("//") and not Path(value).is_absolute():
        raise SceneError("cache", "Use a Blender project-relative or absolute cache path.")
    return Path(bpy.path.abspath(value)).resolve()


def cache_paths(settings):
    bake = project_path(settings.bake_cache)
    preview = project_path(settings.preview_cache)
    if bake == preview or bake in preview.parents or preview in bake.parents:
        raise SceneError("cache", "Choose separate, non-nested Preview and Bake directories.")
    return bake, preview


def check_device(backend):
    device = jobs.checked("simulate", "device-info", "--backend", backend)
    if backend == "cuda" and (
        device.get("device_name"),
        device.get("compiled_architecture"),
        device.get("driver_version"),
        device.get("runtime_version"),
    ) != ("NVIDIA GeForce RTX 5060 Ti", 1200, 13030, 13030):
        raise SceneError("backend", "Use the pinned RTX 5060 Ti device and driver, or select CPU.")


class SceneAction:
    bl_translation_context = "FluidiTy"

    def execute(self, context):
        scene = context.scene
        scene.fluidity.diagnostic = ""
        try:
            self.perform(scene)
        except SceneError as error:
            field, message = error.args
            scene.fluidity.message = f"{field}: {bpy.app.translations.pgettext_iface(message)}"
            return {"CANCELLED"}
        except jobs.ToolError as error:
            if error.record:
                scene.fluidity.diagnostic = str(error)
                scene.fluidity.message = "The solver rejected this request. Check the scene or cache."
            else:
                scene.fluidity.message = str(error)
            return {"CANCELLED"}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            scene.fluidity.message = bpy.app.translations.pgettext_iface(str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class FLUIDITY_OT_check_scene(SceneAction, Operator):
    bl_idname = "fluidity.check_scene"
    bl_label = "Check Scene"

    def perform(self, scene):
        packet = scene_inputs(scene)
        cache_paths(scene.fluidity)
        backend = scene.fluidity.backend
        check_device(backend)
        with tempfile.TemporaryDirectory(prefix="fluidity-check-") as directory:
            path, _ = jobs.prepare(packet, backend, directory)
            jobs.checked("simulate", "check", "--scene", path)
        scene.fluidity.message = "Scene is ready to bake."


class StartJob(SceneAction):
    preview = False
    resume = False

    @classmethod
    def poll(cls, context):
        return not is_running(context.scene)

    def perform(self, scene):
        settings = scene.fluidity
        packet = scene_inputs(scene, self.preview)
        bake, preview = cache_paths(settings)
        cache = preview if self.preview else bake
        if not self.resume and cache.exists():
            raise SceneError("cache", "Choose a fresh cache directory, or use Resume for an existing bake.")
        check_device(settings.backend)
        last_frame = min(scene.frame_end, 3) if self.preview else scene.frame_end
        try:
            jobs.start(scene, cache, packet, settings.backend, last_frame, self.resume)
        except jobs.ToolError as error:
            if self.resume and error.record.get("field") == "scene":
                raise SceneError("scene", "The bake settings changed. Choose a fresh Bake cache.") from error
            raise
        settings.committed_frame = -1
        settings.message = "Waiting for the first committed frame."
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick, first_interval=0.01)


class FLUIDITY_OT_preview(StartJob, Operator):
    bl_idname = "fluidity.preview"
    bl_label = "Preview"
    preview = True


class FLUIDITY_OT_bake(StartJob, Operator):
    bl_idname = "fluidity.bake"
    bl_label = "Bake"


class FLUIDITY_OT_resume(StartJob, Operator):
    bl_idname = "fluidity.resume"
    bl_label = "Resume"
    resume = True


class FLUIDITY_OT_cancel(SceneAction, Operator):
    bl_idname = "fluidity.cancel"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, context):
        return is_running(context.scene)

    def perform(self, scene):
        jobs.cancel(scene)
        scene.fluidity.message = "Cancelling at a safe boundary."


class FLUIDITY_OT_inspect(SceneAction, Operator):
    bl_idname = "fluidity.inspect"
    bl_label = "Inspect"

    def perform(self, scene):
        settings = scene.fluidity
        require_cadence(scene)
        cache = project_path(settings.attached_cache or settings.bake_cache)
        settings.inspect_frame = -1
        settings.point_detail = ""
        result = json.loads(jobs.checked("blender_bridge", "inspect", cache, scene.frame_current))
        points = result["points"]
        settings.inspect_frame, settings.point_count = result["frame"], len(points)
        if settings.point_index < len(points):
            point = points[settings.point_index]
            position = ", ".join(f"{value:.4g}" for value in point["position"])
            velocity = ", ".join(f"{value:.4g}" for value in point["velocity"])
            settings.point_detail = f"ID {point['id']} | ({position}) m | ({velocity}) m/s"
        settings.message = "Read the committed frame."


class FLUIDITY_OT_attach(SceneAction, Operator):
    bl_idname = "fluidity.attach"
    bl_label = "Attach Points"
    preview: bpy.props.BoolProperty(default=False)

    def perform(self, scene):
        bake, preview = cache_paths(scene.fluidity)
        playback.attach(scene, preview if self.preview else bake)
        scene.fluidity.message = "Points attached. Save the project to keep the binding."


class FLUIDITY_OT_relink(SceneAction, Operator):
    bl_idname = "fluidity.relink"
    bl_label = "Relink Cache"
    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def perform(self, scene):
        playback.attach(scene, project_path(self.directory))
        scene.fluidity.message = "Points attached. Save the project to keep the binding."


class FLUIDITY_PT_main(Panel):
    bl_label = "FluidiTy"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FluidiTy"

    def draw(self, context):
        settings = context.scene.fluidity
        layout = self.layout
        tr = bpy.app.translations.pgettext_iface

        def field(prop, label):
            row = layout.split(factor=0.44)
            row.label(text=tr(label, "FluidiTy"), translate=False)
            row.prop(settings, prop, text="")

        for prop, label in (
            ("domain", "Domain"),
            ("liquid", "Initial Liquid"),
            ("paddle", "Paddle"),
            ("cell_width", "Cell Width"),
            ("backend", "Solver"),
        ):
            field(prop, label)
        layout.operator("fluidity.check_scene")
        layout.separator()
        field("preview_cache", "Preview Cache")
        layout.operator("fluidity.preview")
        layout.operator("fluidity.attach", text="Attach Preview").preview = True
        field("bake_cache", "Bake Cache")
        row = layout.row()
        row.operator("fluidity.bake")
        row.operator("fluidity.resume")
        if is_running(context.scene):
            layout.operator("fluidity.cancel")
            layout.label(text=f"{settings.elapsed:.1f} s")
            if settings.heartbeat_age >= 0:
                layout.label(
                    text=f"{bpy.app.translations.pgettext_iface('Last heartbeat')}: {settings.heartbeat_age:.1f} s"
                )
        if settings.committed_frame >= 0:
            layout.label(text=f"{bpy.app.translations.pgettext_iface('Committed frame')}: {settings.committed_frame}")
        layout.separator()
        layout.operator("fluidity.attach")
        layout.operator("fluidity.relink")
        layout.prop(settings, "point_index")
        layout.operator("fluidity.inspect")
        if settings.inspect_frame >= 0:
            layout.label(text=f"{bpy.app.translations.pgettext_iface('Points')}: {settings.point_count}")
            layout.label(text="XYZ (m) • Vxyz (m/s)", translate=False)
            if settings.point_detail:
                for value in settings.point_detail.split(" | "):
                    layout.label(text=value, translate=False)
        if settings.message:
            # Keep guidance readable in a narrow sidebar.
            from textwrap import wrap

            for line in wrap(bpy.app.translations.pgettext_iface(settings.message), width=32):
                layout.label(text=line, translate=False)
            if settings.diagnostic:
                for line in wrap(settings.diagnostic, width=32):
                    layout.label(text=line, translate=False)


_CLASSES = (
    FluidiTySettings,
    FLUIDITY_OT_check_scene,
    FLUIDITY_OT_preview,
    FLUIDITY_OT_bake,
    FLUIDITY_OT_resume,
    FLUIDITY_OT_cancel,
    FLUIDITY_OT_inspect,
    FLUIDITY_OT_attach,
    FLUIDITY_OT_relink,
    FLUIDITY_PT_main,
)


@persistent
def _before_load(_filepath):
    jobs.close_all()


def register():
    bpy.app.translations.register(__name__, TRANSLATIONS)
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fluidity = PointerProperty(type=FluidiTySettings)
    bpy.app.handlers.load_pre.append(_before_load)


def unregister():
    jobs.close_all()
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    bpy.app.handlers.load_pre.remove(_before_load)
    del bpy.types.Scene.fluidity
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    bpy.app.translations.unregister(__name__)
