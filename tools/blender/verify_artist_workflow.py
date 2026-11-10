"""Headless checks for Physics discovery, field connectivity and localization."""

import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
import fluidity

fluidity.register()
panels = [cls for cls in fluidity.ui.CLASSES if issubclass(cls, bpy.types.Panel)]
assert len(panels) >= 6
assert all(cls.bl_space_type == "PROPERTIES" and cls.bl_context == "physics" for cls in panels)
assert fluidity.FLUIDITY_Preferences.bl_rna.properties["preview_frames"].default == 3
assert fluidity.FluidiTySettings.bl_rna.properties["cell_width"].description
assert fluidity.FluidiTySettings.bl_rna.properties["preview_cache"].is_path_supports_blend_relative
assert fluidity.FluidiTySettings.bl_rna.properties["bake_cache"].is_path_supports_blend_relative
tree = fluidity.playback.create_points_asset()
outputs = {item.name for item in tree.interface.items_tree if item.in_out == "OUTPUT"}
assert {"Geometry", "Points", "Velocity", "Speed", "Radius"} <= outputs, outputs
assert fluidity.translations.TRANSLATIONS["fr_FR"][("FluidiTy", "Simulation")] == "Simulation"
assert bpy.ops.fluidity.copy_diagnostics() == {"FINISHED"}
assert json.loads(fluidity.interchange.diagnostics_report(bpy.context.scene))["addon"] == "FluidiTy"
assert bpy.ops.fluidity.select_role(role="domain") == {"CANCELLED"}
fluidity.unregister()
fluidity.register()
fluidity.unregister()
print("PASS: Physics panels, preferences, fields, catalog, diagnostics and registration")
