"""Artist-facing Physics Properties workflow; drawing never launches tools."""

from textwrap import wrap
from typing import ClassVar

import bpy

from . import authoring, jobs
from .preferences import get_preferences


def message(layout, text, icon="NONE"):
    translated = bpy.app.translations.pgettext_iface(text, "FluidiTy")
    region = bpy.context.region
    scale = bpy.context.preferences.system.ui_scale
    width = max(20, min(80, int(((region.width if region else 360) - 68 * scale) / (7 * scale))))
    column = layout.column(align=True)
    column.scale_y = 0.85
    for index, line in enumerate(wrap(translated, width=width)):
        column.label(text=line, icon=icon if index == 0 else "NONE", translate=False)


class PhysicsPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"
    bl_translation_context = "FluidiTy"


class FLUIDITY_PT_main(PhysicsPanel, bpy.types.Panel):
    bl_label = "FluidiTy"

    def draw(self, context):
        settings = context.scene.fluidity
        layout = self.layout
        if not bpy.data.filepath:
            layout.operator("wm.save_as_mainfile", text="Save Project Before Baking...", icon="FILE_TICK")
        if jobs.is_running(context.scene):
            layout.label(text="Simulation Running", icon="TIME")
            if settings.target_frame >= 0:
                layout.progress(
                    factor=max(0, min(1, (settings.committed_frame + 1) / (settings.target_frame + 1))),
                    type="BAR",
                    text=bpy.app.translations.pgettext_iface("Committed Frames", "FluidiTy"),
                )
            layout.operator("fluidity.cancel", icon="CANCEL")
            layout.label(text=f"{settings.elapsed:.1f} s", translate=False)
            if settings.committed_frame >= 0:
                layout.label(
                    text=f"{bpy.app.translations.pgettext_iface('Committed frame')}: {settings.committed_frame}"
                )
            if settings.heartbeat_age >= 0:
                layout.label(
                    text=f"{bpy.app.translations.pgettext_iface('Last heartbeat')}: {settings.heartbeat_age:.1f} s"
                )
        if settings.message:
            message(layout, settings.message, "INFO")
        if not settings.domain and not settings.liquid:
            layout.operator("fluidity.create_setup", icon="ADD")
        if context.object and context.object.type == "MESH":
            row = layout.row(align=True)
            row.enabled = not jobs.is_running(context.scene)
            row.label(text="Assign Selected")
            for role, label in (("domain", "Domain"), ("liquid", "Liquid")):
                row.operator("fluidity.assign_role", text=label).role = role


class ChildPanel(PhysicsPanel):
    bl_parent_id = "FLUIDITY_PT_main"
    help_topic = ""

    def draw_header_preset(self, _context):
        if self.help_topic:
            self.layout.operator("fluidity.help", text="", icon="QUESTION", emboss=False).topic = self.help_topic


def input_object(layout, settings, name, label):
    row = layout.row(align=True)
    row.alert = settings.issue_field in {name, "roles"}
    row.prop(settings, name, text=label)
    if getattr(settings, name):
        row.operator("fluidity.select_role", text="", icon="RESTRICT_SELECT_OFF").role = name
    elif name != "paddle":
        row.operator("fluidity.add_input", text="", icon="ADD").role = name


def controls(layout, scene):
    layout.use_property_split = True
    layout.use_property_decorate = False
    column = layout.column(align=True)
    column.enabled = not jobs.is_running(scene)
    return column


class FLUIDITY_PT_inputs(ChildPanel, bpy.types.Panel):
    bl_label = "Domain"
    help_topic = "DOMAIN"

    def draw(self, context):
        scene = context.scene
        column = controls(self.layout, scene)
        settings = scene.fluidity
        input_object(column, settings, "domain", "Domain")
        column.prop(settings, "boundary_mode")
        if settings.boundary_mode == "TANK":
            column.label(text="Open Top · Five Solid Walls", icon="MESH_CUBE")
        elif settings.boundary_mode == "FREE":
            column.label(text="No Solid Boundaries", icon="INFO")
        column.prop(settings, "cell_width")
        column.operator("fluidity.initialize_scene", text="Set Timing & Units", icon="SCENE_DATA")


class FLUIDITY_PT_fluid(ChildPanel, bpy.types.Panel):
    bl_label = "Fluid"
    help_topic = "FLUID"

    def draw(self, context):
        scene = context.scene
        settings = scene.fluidity
        column = controls(self.layout, scene)
        if not settings.liquid_sources:
            input_object(column, settings, "liquid", "Initial Liquid")
        if settings.boundary_mode != "LEGACY_PADDLE":
            for index, source in enumerate(settings.liquid_sources):
                box = column.box()
                row = box.row(align=True)
                row.prop(source, "geometry")
                row.operator("fluidity.liquid_source", text="", icon="X").index = index
                for field in ("substance_id", "phase_id", "density", "emission_rate"):
                    box.prop(source, field)
                if source.emission_rate > 0:
                    box.prop(source, "emitted_volume")
            column.operator("fluidity.liquid_source", text="Add Liquid Source", icon="ADD")
            if len({source.phase_id for source in settings.liquid_sources}) > 1:
                interactions = column.box()
                interactions.label(text="Interactions")
                interactions.prop(settings, "phase_tension")
        material = column.column(align=True)
        material.enabled = settings.boundary_mode != "LEGACY_PADDLE"
        if not settings.liquid_sources:
            row = material.row(align=True)
            row.label(text="Fluid Preset")
            preset_name = next(
                label for key, label, _description in authoring.PRESET_ITEMS if key == settings.material_preset
            )
            row.operator_menu_enum("fluidity.apply_preset", "preset", text=preset_name)
            material.prop(settings, "density", text="Density (kg/m³)")
        material.prop(settings, "surface_tension", text="Surface Tension (N/m)")
        for index, label in enumerate(("Velocity X (m/s)", "Velocity Y (m/s)", "Velocity Z (m/s)")):
            material.prop(settings, "initial_velocity", index=index, text=label)


class FLUIDITY_PT_colliders(ChildPanel, bpy.types.Panel):
    bl_label = "Collision"
    help_topic = "COLLIDER"

    def draw(self, context):
        scene = context.scene
        settings = scene.fluidity
        column = controls(self.layout, scene)
        if settings.boundary_mode == "LEGACY_PADDLE":
            input_object(column, settings, "paddle", "Animated Paddle")
        elif settings.boundary_mode in {"PLANE", "SLAB"}:
            input_object(column, settings, "collider", "Collider")
            if context.object and context.object.type == "MESH":
                column.operator("fluidity.assign_role", text="Use Selected as Collider").role = "collider"
            column.prop(settings, "friction")
            column.label(text="Infinite Plane" if settings.boundary_mode == "PLANE" else "Infinite X Slab", icon="INFO")
        elif settings.boundary_mode == "TANK":
            column.label(text="Uses Domain Walls", icon="MESH_CUBE")
            column.prop(settings, "friction")
        else:
            column.label(text="Free Space", icon="INFO")


class FLUIDITY_PT_simulation(ChildPanel, bpy.types.Panel):
    bl_label = "Simulation"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        column = layout.column()
        column.enabled = not jobs.is_running(scene)
        column.prop(scene.fluidity, "backend")
        column.prop(scene, "frame_end", text="End Frame")
        advanced = column.column(align=True)
        advanced.enabled = scene.fluidity.boundary_mode != "LEGACY_PADDLE"
        advanced.prop(scene.fluidity, "maximum_step", text="Max Step (s)")
        advanced.prop(scene.fluidity, "energy_blend")
        advanced.prop(scene.fluidity, "pressure_iterations")
        column.operator("fluidity.check_scene", icon="CHECKMARK")


class FLUIDITY_PT_cache(ChildPanel, bpy.types.Panel):
    bl_label = "Cache & Bake"
    help_topic = "CACHE"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.use_property_split = True
        column = layout.column()
        column.enabled = not jobs.is_running(scene)
        column.prop(scene.fluidity, "preview_cache")
        row = column.row(align=True)
        row.operator("fluidity.preview", icon="PLAY")
        row.operator("fluidity.attach", text="Attach Preview", icon="POINTCLOUD_DATA").preview = True
        column.separator()
        column.prop(scene.fluidity, "bake_cache")
        row = column.row(align=True)
        row.operator("fluidity.bake", icon="RENDER_ANIMATION")
        row.operator("fluidity.resume", icon="RECOVER_LAST")
        row = column.row(align=True)
        row.operator("fluidity.attach", icon="POINTCLOUD_DATA")
        row.operator("fluidity.relink", text="Relink Cache...", icon="FILE_FOLDER")


class FLUIDITY_PT_display(ChildPanel, bpy.types.Panel):
    bl_label = "Display & Geometry Nodes"
    help_topic = "NODES"
    bl_options: ClassVar[set[str]] = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        obj = context.scene.fluidity.playback_object
        if not obj:
            message(layout, "Attach a cache to edit point size and material.")
            return
        layout.label(text=obj.name, icon="POINTCLOUD_DATA", translate=False)
        modifier = next((item for item in obj.modifiers if item.type == "NODES" and item.node_group), None)
        if modifier:
            tree = modifier.node_group
            for key, label in (("fluidity_size_socket", "Size"), ("fluidity_material_socket", "Material")):
                identifier = tree.get(key)
                if identifier and hasattr(modifier.properties.inputs, identifier):
                    layout.prop(getattr(modifier.properties.inputs, identifier), "value", text=label)
        layout.operator("fluidity.select_playback", icon="NODETREE")


class FLUIDITY_PT_interchange(ChildPanel, bpy.types.Panel):
    bl_label = "Import & Export"
    help_topic = "EXPORT"
    bl_options: ClassVar[set[str]] = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        column = layout.column()
        column.enabled = not jobs.is_running(context.scene)
        column.operator("fluidity.import_cache", text="Import Cache...", icon="IMPORT")
        column.operator("fluidity.export_alembic", text="Export Alembic...", icon="EXPORT")
        column.operator("fluidity.export_usd", text="Export USD...", icon="EXPORT")


class FLUIDITY_PT_diagnostics(ChildPanel, bpy.types.Panel):
    bl_label = "Diagnostics"
    bl_options: ClassVar[set[str]] = {"DEFAULT_CLOSED"}

    def draw(self, context):
        settings = context.scene.fluidity
        layout = self.layout
        layout.operator("fluidity.copy_diagnostics", icon="COPYDOWN")
        preferences = get_preferences(context)
        if preferences:
            layout.prop(preferences, "show_debug")
        if preferences and preferences.show_debug:
            layout.prop(settings, "point_index")
            row = layout.row()
            row.enabled = not jobs.is_running(context.scene)
            row.operator("fluidity.inspect", icon="VIEWZOOM")
            if settings.inspect_frame >= 0:
                layout.label(text=f"{settings.inspect_frame}: {settings.point_count}", translate=False)
                for value in settings.point_detail.split(" | "):
                    message(layout, value)
            if settings.diagnostic:
                message(layout, settings.diagnostic)


CLASSES = (
    FLUIDITY_PT_main,
    FLUIDITY_PT_inputs,
    FLUIDITY_PT_fluid,
    FLUIDITY_PT_colliders,
    FLUIDITY_PT_simulation,
    FLUIDITY_PT_cache,
    FLUIDITY_PT_display,
    FLUIDITY_PT_interchange,
    FLUIDITY_PT_diagnostics,
)
