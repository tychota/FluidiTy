"""FluidiTy's bounded M1 Blender workflow."""

import json
import subprocess
import tempfile
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, Operator, PropertyGroup

from . import authoring, interchange, jobs, playback, ui
from .preferences import FLUIDITY_Preferences, preview_end
from .scene_inputs import SceneError, require_cadence, scene_inputs
from .translations import TRANSLATIONS

is_running = jobs.is_running
tick = jobs.tick


def invalidate_guidance(settings, _context):
    """Discard a previous preflight result when artist inputs change."""
    settings.message = ""
    settings.issue_field = ""
    settings.diagnostic = ""


def custom_material(settings, context):
    settings.material_preset = "CUSTOM"
    invalidate_guidance(settings, context)


class FluidiTyLiquidSource(PropertyGroup):
    geometry: PointerProperty(name="Source Box", type=Object)
    substance_id: IntProperty(name="Substance ID", min=1, default=1)
    phase_id: IntProperty(name="Miscibility Phase", min=1, default=1)
    density: FloatProperty(name="Density (kg/m³)", min=1, default=1000)
    emission_rate: FloatProperty(name="Particles per Second", min=0, default=0)
    emitted_volume: FloatProperty(name="Emitted Particle Volume (m³)", min=1e-9, default=0.001, precision=6)


class FLUIDITY_OT_liquid_source(Operator):
    bl_idname = "fluidity.liquid_source"
    bl_label = "Edit Liquid Sources"
    index: IntProperty(default=-1)

    def execute(self, context):
        settings = context.scene.fluidity
        if self.index < 0:
            if len(settings.liquid_sources) >= 64:
                return {"CANCELLED"}
            used_ids = {item.substance_id for item in settings.liquid_sources}
            source = settings.liquid_sources.add()
            source.substance_id = next(index for index in range(1, 65) if index not in used_ids)
        elif self.index < len(settings.liquid_sources):
            settings.liquid_sources.remove(self.index)
        else:
            return {"CANCELLED"}
        invalidate_guidance(settings, context)
        return {"FINISHED"}

class FluidiTySettings(PropertyGroup):
    liquid_sources: CollectionProperty(type=FluidiTyLiquidSource)
    surface_object: PointerProperty(type=Object)
    attached_surface_cache: StringProperty()
    mesh_cache: StringProperty(name="Surface Cache", default="//FluidiTy Surface", subtype="DIR_PATH")
    mesh_spacing: FloatProperty(name="Surface Spacing (m)", default=0.0125, min=0.00001, precision=5)
    mesh_sigma: FloatProperty(name="Surface Smoothing (m)", default=0.025, min=0.00001, precision=5)
    phase_tension: FloatProperty(
        name="Phase Interface Tension (N/m)", min=0, default=0, precision=4, update=invalidate_guidance
    )
    boundary_mode: EnumProperty(
        name="Boundaries",
        description="Choose the actual collision model used by the solver",
        items=(
            ("TANK", "Open Tank", "Domain floor and four walls; open top"),
            ("FREE", "Free Space", "No solid boundaries; domain is initial containment only"),
            ("PLANE", "Plane Collider", "Infinite floor at the top of the collider box"),
            ("SLAB", "Slab Collider", "Infinite X slab using the collider box thickness"),
            ("LEGACY_PADDLE", "Legacy Paddle", "Original first-bake packet for existing scenes"),
        ),
        default="LEGACY_PADDLE",
        update=invalidate_guidance,
    )
    collider: PointerProperty(
        name="Collider", type=Object, description="Box defining the analytic plane or slab", update=invalidate_guidance
    )
    material_preset: EnumProperty(name="Fluid Preset", items=authoring.PRESET_ITEMS, default="CUSTOM")
    density: FloatProperty(
        name="Density",
        description="Liquid mass per volume in kg/m³",
        default=1000,
        min=1,
        max=50000,
        update=custom_material,
    )
    surface_tension: FloatProperty(
        name="Surface Tension",
        description="Interface tension in N/m; zero disables capillary forces",
        default=0,
        min=0,
        max=10,
        precision=4,
        update=custom_material,
    )
    friction: FloatProperty(
        name="Wall Friction",
        description="Dimensionless Coulomb friction coefficient at solid contact",
        default=0,
        min=0,
        max=2,
        update=invalidate_guidance,
    )
    initial_velocity: FloatVectorProperty(
        name="Initial Velocity",
        description="World-space initial liquid velocity in m/s",
        size=3,
        default=(0, 0, 0),
        min=-100,
        max=100,
        update=invalidate_guidance,
    )
    energy_blend: FloatProperty(
        name="Velocity Retention",
        description="Particle energy blend in [0,1]; lower values damp the motion more",
        default=0.6,
        min=0,
        max=1,
        update=invalidate_guidance,
    )
    maximum_step: FloatProperty(
        name="Maximum Time Step",
        description="Maximum attempted solver step in seconds; stability can require smaller steps",
        default=0.01,
        min=0.00001,
        max=0.04,
        precision=5,
        update=invalidate_guidance,
    )
    pressure_iterations: IntProperty(
        name="Pressure Iterations",
        description="Pressure solve iteration budget; a failed convergence gate never publishes a frame",
        default=256,
        min=1,
        max=256,
        update=invalidate_guidance,
    )
    domain: PointerProperty(
        name="Domain",
        type=Object,
        description="Static closed box defining the validated scene extent",
        update=invalidate_guidance,
    )
    liquid: PointerProperty(
        name="Initial Liquid",
        type=Object,
        description="Static closed box seeded with liquid particles at frame zero",
        update=invalidate_guidance,
    )
    paddle: PointerProperty(
        name="Paddle",
        type=Object,
        description="Closed box spanning domain Y/Z, animated linearly along world X",
        update=invalidate_guidance,
    )
    cell_width: FloatProperty(
        name="Cell Width",
        subtype="DISTANCE",
        default=0.125,
        precision=4,
        min=0.001,
        max=10,
        description="Solver cell width in meters; smaller cells seed more particles (M1 limit: 1024)",
        update=invalidate_guidance,
    )
    backend: EnumProperty(
        name="Solver",
        translation_context="FluidiTy",
        items=(("cuda", "RTX", "Use the NVIDIA CUDA device"), ("cpu", "CPU", "Use the CPU backend")),
        default="cuda",
        update=invalidate_guidance,
    )
    preview_cache: StringProperty(
        name="Preview Cache",
        default="//FluidiTy Preview",
        subtype="DIR_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"},
        description="Separate directory for the shortened preview; existing committed data is never overwritten",
        update=invalidate_guidance,
    )
    bake_cache: StringProperty(
        name="Bake Cache",
        default="//FluidiTy Bake",
        subtype="DIR_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"},
        description="Committed simulation cache; choose a new directory for changed inputs",
        update=invalidate_guidance,
    )
    attached_cache: StringProperty(name="Attached Cache")
    playback_object: PointerProperty(type=Object)
    point_index: IntProperty(name="Point Index", min=0, max=1023)
    committed_frame: IntProperty(default=-1, options={"SKIP_SAVE"})
    target_frame: IntProperty(default=-1, options={"SKIP_SAVE"})
    elapsed: FloatProperty(options={"SKIP_SAVE"})
    heartbeat_age: FloatProperty(default=-1, options={"SKIP_SAVE"})
    message: StringProperty(options={"SKIP_SAVE"})
    diagnostic: StringProperty(options={"SKIP_SAVE"})
    inspect_frame: IntProperty(default=-1, options={"SKIP_SAVE"})
    point_count: IntProperty(options={"SKIP_SAVE"})
    point_detail: StringProperty(options={"SKIP_SAVE"})
    issue_field: StringProperty(options={"SKIP_SAVE"})


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

    @classmethod
    def poll(cls, context):
        if is_running(context.scene):
            cls.poll_message_set("Wait for the running job or cancel it first.")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        scene.fluidity.diagnostic = ""
        scene.fluidity.issue_field = ""
        try:
            self.perform(scene)
        except SceneError as error:
            field, message = error.args
            scene.fluidity.issue_field = field
            scene.fluidity.message = f"{field}: {bpy.app.translations.pgettext_iface(message, 'FluidiTy')}"
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
    bl_description = "Validate geometry, timing, cache paths and the selected backend before baking"
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

    def perform(self, scene):
        settings = scene.fluidity
        packet = scene_inputs(scene, self.preview)
        bake, preview = cache_paths(settings)
        cache = preview if self.preview else bake
        if not self.resume and cache.exists():
            raise SceneError("cache", "Choose a fresh cache directory, or use Resume for an existing bake.")
        check_device(settings.backend)
        last_frame = preview_end(scene) if self.preview else scene.frame_end
        try:
            jobs.start(scene, cache, packet, settings.backend, last_frame, self.resume)
        except jobs.ToolError as error:
            if self.resume and error.record.get("field") == "scene":
                raise SceneError("scene", "The bake settings changed. Choose a fresh Bake cache.") from error
            raise
        settings.committed_frame = -1
        settings.target_frame = last_frame
        settings.message = "Waiting for the first committed frame."
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick, first_interval=0.01)


class FLUIDITY_OT_preview(StartJob, Operator):
    bl_idname = "fluidity.preview"
    bl_description = (
        "Bake a shortened run in the separate Preview cache; the end frame is configured in Add-on Preferences"
    )
    bl_label = "Preview"
    preview = True


class FLUIDITY_OT_bake(StartJob, Operator):
    bl_idname = "fluidity.bake"
    bl_description = "Start a new simulation in a fresh Bake cache directory"
    bl_label = "Bake"


class FLUIDITY_OT_resume(StartJob, Operator):
    bl_idname = "fluidity.resume"
    bl_description = "Continue compatible committed Bake data; changed solver inputs require a fresh cache"
    bl_label = "Resume"
    resume = True


class FLUIDITY_OT_cancel(SceneAction, Operator):
    bl_idname = "fluidity.cancel"
    bl_description = "Request cancellation at a safe solver boundary; preserve committed frames for Resume"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, context):
        return is_running(context.scene)

    def perform(self, scene):
        jobs.cancel(scene)
        scene.fluidity.message = "Cancelling at a safe boundary."


class FLUIDITY_OT_inspect(SceneAction, Operator):
    bl_idname = "fluidity.inspect"
    bl_description = "Read verified point positions, velocities and stable IDs for the current committed frame"
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


class FLUIDITY_OT_build_surface(SceneAction, Operator):
    bl_idname = "fluidity.build_surface"
    bl_label = "Build Surface"
    bl_description = "Reconstruct native Mesh frames from the committed Bake cache"
    resume: BoolProperty(default=False, options={"SKIP_SAVE"})

    def perform(self, scene):
        require_cadence(scene)
        settings = scene.fluidity
        source = project_path(settings.bake_cache)
        output = project_path(settings.mesh_cache)
        if source == output or source in output.parents or output in source.parents:
            raise SceneError("cache", "Choose a separate, non-nested Surface cache directory.")
        if not self.resume and output.exists():
            raise SceneError("cache", "Choose a fresh Surface cache directory, or use Resume Surface.")
        last_frame = playback.committed_last(source)
        jobs.start_mesh(scene, source, output, last_frame, settings.mesh_spacing, settings.mesh_sigma, self.resume)
        settings.committed_frame = -1
        settings.target_frame = last_frame
        settings.message = "Waiting for the first committed Surface frame."
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick, first_interval=0.01)

class FLUIDITY_OT_attach(SceneAction, Operator):
    bl_idname = "fluidity.attach"
    bl_description = "Validate and attach native Alembic Points playback without replacing artist display edits"
    bl_label = "Attach Points"
    preview: bpy.props.BoolProperty(default=False)

    def perform(self, scene):
        bake, preview = cache_paths(scene.fluidity)
        playback.attach(scene, preview if self.preview else bake)
        scene.fluidity.message = "Points attached. Save the project to keep the binding."


class FLUIDITY_OT_relink(SceneAction, Operator):
    bl_idname = "fluidity.relink"
    bl_description = "Validate a relocated committed cache before replacing the existing playback binding"
    bl_label = "Relink Cache"
    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def perform(self, scene):
        playback.attach(scene, project_path(self.directory))
        scene.fluidity.message = "Points attached. Save the project to keep the binding."


_CLASSES = (
    FluidiTyLiquidSource,
    FLUIDITY_OT_liquid_source,
    FluidiTySettings,
    FLUIDITY_OT_check_scene,
    FLUIDITY_OT_preview,
    FLUIDITY_OT_bake,
    FLUIDITY_OT_resume,
    FLUIDITY_OT_cancel,
    FLUIDITY_OT_inspect,
    FLUIDITY_OT_attach,
    FLUIDITY_OT_build_surface,
    FLUIDITY_OT_relink,
    FLUIDITY_Preferences,
    *authoring.CLASSES,
    *interchange.CLASSES,
    *ui.CLASSES,
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
