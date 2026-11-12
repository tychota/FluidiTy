"""Scene creation, assignment and explicit physical parameter presets."""

from typing import ClassVar

import bpy
from bpy.props import EnumProperty

from .interchange import IdleOperator, select_object
from .preferences import get_preferences

PRESETS = {
    "WATER": ("Water", 1000.0, 0.072),
    "LOW_TENSION": ("Low Surface Tension", 1000.0, 0.025),
    "HIGH_TENSION": ("High Surface Tension", 1000.0, 0.15),
    "NO_TENSION": ("No Surface Tension", 1000.0, 0.0),
}
PRESET_ITEMS = tuple(
    (key, values[0], "Editable density and surface tension starting point") for key, values in PRESETS.items()
)
PRESET_ITEMS += (("CUSTOM", "Custom", "Use the current density and surface tension"),)


def apply_preset(settings, preset):
    """Change only the material quantities represented by a preset."""
    if preset != "CUSTOM":
        _, density, surface_tension = PRESETS[preset]
        settings.density = density
        settings.surface_tension = surface_tension
    settings.material_preset = preset


def box(name, center, dimensions):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


class FLUIDITY_OT_create_setup(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.create_setup"
    bl_label = "Create Liquid Setup"
    bl_description = (
        "Create a real open tank domain and initial liquid box with the Water preset; keep existing scene objects"
    )
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return super().poll(context) and not context.scene.fluidity.domain and not context.scene.fluidity.liquid

    def execute(self, context):
        if context.object and context.object.mode != "OBJECT":
            self.report({"WARNING"}, "Leave Edit Mode before creating simulation objects.")
            return {"CANCELLED"}
        settings = context.scene.fluidity
        settings.domain = box("FluidiTy Domain", (0, 0, 0.5), (1.5, 1.5, 1))
        settings.liquid = box("FluidiTy Liquid", (0, 0, 0.45), (0.5, 0.5, 0.5))
        settings.boundary_mode = "TANK"
        preferences = get_preferences(context)
        settings.cell_width = preferences.default_cell_width if preferences else 0.125
        apply_preset(settings, preferences.default_material if preferences else "WATER")
        bpy.ops.fluidity.initialize_scene()
        context.scene.frame_end = 48
        select_object(context, settings.domain)
        return {"FINISHED"}


class FLUIDITY_OT_assign_role(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.assign_role"
    bl_label = "Assign Selected Object"
    bl_description = "Assign the active mesh to a simulation role; geometry and materials are preserved"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}
    role: EnumProperty(
        items=(
            ("domain", "Domain", ""),
            ("liquid", "Liquid", ""),
            ("collider", "Collider", ""),
            ("cutter", "Cutter", ""),
            ("rigid_box", "Rigid Box", ""),
        )
    )

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != "MESH" or obj.mode != "OBJECT":
            self.report({"WARNING"}, "Select a mesh in Object Mode.")
            return {"CANCELLED"}
        settings = context.scene.fluidity
        if self.role == "collider" and settings.boundary_mode not in {"PLANE", "SLAB"}:
            self.report({"WARNING"}, "Choose Plane or Slab collision before assigning a collider.")
            return {"CANCELLED"}
        active_roles = ["domain", "liquid"]
        if settings.boundary_mode == "LEGACY_PADDLE":
            active_roles.append("paddle")
        elif settings.boundary_mode in {"PLANE", "SLAB"}:
            active_roles.append("collider")
        active_roles.extend(("cutter", "rigid_box", "elastic_solid"))
        for role in active_roles:
            if role != self.role and getattr(settings, role) == obj:
                self.report({"WARNING"}, "Assign a different object to each simulation role.")
                return {"CANCELLED"}
        setattr(settings, self.role, obj)
        return {"FINISHED"}


class FLUIDITY_OT_add_input(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.add_input"
    bl_label = "Add Simulation Object"
    bl_description = "Create and assign a box for an unassigned simulation role"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}
    role: EnumProperty(items=(("domain", "Domain", ""), ("liquid", "Liquid", ""), ("collider", "Collider", "")))

    def execute(self, context):
        settings = context.scene.fluidity
        if getattr(settings, self.role):
            self.report({"WARNING"}, "This role already has an object. Clear it before creating a replacement.")
            return {"CANCELLED"}
        if context.object and context.object.mode != "OBJECT":
            return {"CANCELLED"}
        if self.role == "collider" and settings.boundary_mode not in {"PLANE", "SLAB"}:
            self.report({"WARNING"}, "Choose Plane or Slab collision before assigning a collider.")
            return {"CANCELLED"}
        center, dimensions = {
            "domain": ((0, 0, 0.5), (1.5, 1.5, 1)),
            "liquid": ((0, 0, 0.45), (0.5, 0.5, 0.5)),
            "collider": ((-0.6, 0, 0.5), (0.1, 2, 2))
            if settings.boundary_mode == "SLAB"
            else ((0, 0, -0.05), (2, 2, 0.1)),
        }[self.role]
        setattr(settings, self.role, box("FluidiTy " + self.role.title(), center, dimensions))
        return {"FINISHED"}


class FLUIDITY_OT_apply_preset(IdleOperator, bpy.types.Operator):
    bl_idname = "fluidity.apply_preset"
    bl_label = "Apply Fluid Preset"
    bl_description = (
        "Apply only density and surface tension; preserve objects, velocities, solver settings and cache paths"
    )
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}
    preset: EnumProperty(items=PRESET_ITEMS)

    def execute(self, context):
        apply_preset(context.scene.fluidity, self.preset)
        return {"FINISHED"}


class FLUIDITY_OT_help(bpy.types.Operator):
    bl_idname = "fluidity.help"
    bl_label = "FluidiTy Help"
    bl_description = "Show supported geometry, units and workflow details"
    topic: EnumProperty(
        items=tuple((key, key.title(), "") for key in ("DOMAIN", "FLUID", "COLLIDER", "CACHE", "NODES", "EXPORT"))
    )

    def invoke(self, context, _event):
        return context.window_manager.invoke_popup(self, width=440)

    def draw(self, _context):
        from .ui import message

        topics = {
            "DOMAIN": "Open Tank uses the domain floor and four side walls; the top is open. Free Space uses the domain only for initial containment. Inputs must be closed axis-aligned boxes. Keep 24 fps and one unit per meter.",
            "FLUID": "Presets change density (kg/m³) and surface tension (N/m). They are editable starting points. Initial velocity is in world-space m/s. Viscosity is not implemented by this solver; these presets do not simulate honey or oil.",
            "COLLIDER": "The current engine supports one analytic collision model: tank walls, an infinite floor plane, or an infinite X slab. Plane uses the top of its box; Slab uses its X thickness. A collider may be static or translate linearly. Arbitrary meshes and multiple independent colliders are not supported.",
            "CACHE": "Preview and Bake have independent caches. Cancel preserves committed frames. Resume requires unchanged physical inputs. Relink validates a replacement before changing playback.",
            "NODES": "Connect the Points output to your own Geometry Nodes. Velocity and Speed use m/s; Radius uses meters. Size and material are display controls and never change the simulation.",
            "EXPORT": "Alembic preserves empty point samples. Blender 5.2.3 USD export requires nonempty samples. USD and VDB import use native Blender readers. The solver does not produce volume grids.",
        }
        message(self.layout, topics[self.topic])

    def execute(self, _context):
        return {"FINISHED"}


CLASSES = (
    FLUIDITY_OT_create_setup,
    FLUIDITY_OT_assign_role,
    FLUIDITY_OT_add_input,
    FLUIDITY_OT_apply_preset,
    FLUIDITY_OT_help,
)
