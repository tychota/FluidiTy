"""Application defaults; stored in preferences rather than project files."""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty


class FLUIDITY_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    default_backend: EnumProperty(
        name="Default Backend",
        description="Backend used by Initialize Scene; existing scenes keep their settings",
        items=(("cuda", "RTX", "Qualified NVIDIA CUDA backend"), ("cpu", "CPU", "CPU reference backend")),
        default="cuda",
    )
    preview_frames: IntProperty(
        name="Preview End Frame",
        description="Last frame of a shortened preview, capped by the scene end frame; frame zero is included",
        default=3,
        min=2,
        max=240,
    )
    show_debug: BoolProperty(
        name="Show Debug Details",
        description="Show raw solver diagnostics and committed-point inspection in Physics Properties",
        default=False,
    )

    def draw(self, _context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "default_backend")
        layout.prop(self, "preview_frames")
        layout.prop(self, "show_debug")
        layout.label(text="Simulation controls: Properties > Physics > FluidiTy", icon="PHYSICS")
        layout.label(text="M1: boxes, linear paddle motion, 24 fps, meters", icon="INFO")


def get_preferences(context=None):
    entry = (context or bpy.context).preferences.addons.get(__package__)
    return entry.preferences if entry else None


def preview_end(scene):
    preferences = get_preferences()
    return min(scene.frame_end, preferences.preview_frames if preferences else 3)
