"""Artist-facing Physics Properties workflow; drawing never launches tools."""

from textwrap import wrap
from typing import ClassVar

import bpy

from . import jobs
from .preferences import get_preferences


def message(layout, text, icon="NONE"):
    translated = bpy.app.translations.pgettext_iface(text, "FluidiTy")
    region = bpy.context.region
    scale = bpy.context.preferences.system.ui_scale
    width = max(20, min(80, int(((region.width if region else 360) - 68 * scale) / (7 * scale))))
    for index, line in enumerate(wrap(translated, width=width)):
        layout.label(text=line, icon=icon if index == 0 else "NONE", translate=False)


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
            message(layout, "Save the project before previewing or baking.", "FILE_BLEND")
            layout.operator("wm.save_as_mainfile", text="Save Project...", icon="FILE_TICK")
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
        else:
            message(layout, "Assign the three inputs, check the scene, then preview.", "INFO")


class ChildPanel(PhysicsPanel):
    bl_parent_id = "FLUIDITY_PT_main"


class FLUIDITY_PT_inputs(ChildPanel, bpy.types.Panel):
    bl_label = "Inputs"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        column = layout.column()
        column.enabled = not jobs.is_running(scene)
        for name in ("domain", "liquid", "paddle"):
            row = column.row(align=True)
            row.alert = scene.fluidity.issue_field in {name, "roles"}
            row.prop(scene.fluidity, name)
            row.operator("fluidity.select_role", text="", icon="RESTRICT_SELECT_OFF").role = name
        message(column, "M1 inputs must be closed, axis-aligned boxes. Paddle translates on world X.")
        column.operator("fluidity.initialize_scene", text="Set Timing & Units", icon="SCENE_DATA")


class FLUIDITY_PT_simulation(ChildPanel, bpy.types.Panel):
    bl_label = "Simulation"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        column = layout.column()
        column.enabled = not jobs.is_running(scene)
        column.prop(scene.fluidity, "cell_width")
        column.prop(scene.fluidity, "backend")
        column.prop(scene, "frame_end", text="End Frame")
        message(column, "Frame 0 is time zero. Use 24 fps and one Blender unit per meter.")
        column.operator("fluidity.check_scene", icon="CHECKMARK")


class FLUIDITY_PT_cache(ChildPanel, bpy.types.Panel):
    bl_label = "Cache & Bake"

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
        message(column, "Preview and Bake use separate caches. Resume preserves committed frames.")


class FLUIDITY_PT_display(ChildPanel, bpy.types.Panel):
    bl_label = "Display & Geometry Nodes"
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
        message(layout, "Connect Points to your own nodes. Velocity is in m/s; Radius is in meters.")
        message(layout, "Display edits do not change the simulation cache.")


class FLUIDITY_PT_interchange(ChildPanel, bpy.types.Panel):
    bl_label = "Import & Export"
    bl_options: ClassVar[set[str]] = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        column = layout.column()
        column.enabled = not jobs.is_running(context.scene)
        column.operator("fluidity.import_cache", text="Import Cache...", icon="IMPORT")
        column.operator("fluidity.export_alembic", text="Export Alembic...", icon="EXPORT")
        column.operator("fluidity.export_usd", text="Export USD...", icon="EXPORT")
        message(column, "Alembic exports verified committed points, including empty frames.")
        message(column, "USD export requires nonempty samples in Blender 5.2.3. Use Alembic for empty frames.")
        message(column, "USD and VDB import use native Blender readers. M1 does not generate volume grids.")


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
    FLUIDITY_PT_simulation,
    FLUIDITY_PT_cache,
    FLUIDITY_PT_display,
    FLUIDITY_PT_interchange,
    FLUIDITY_PT_diagnostics,
)
