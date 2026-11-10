"""Capture the installed extension's actual sidebar from an isolated Blender UI."""

import sys
from pathlib import Path

import bpy

root = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
root.mkdir(parents=True, exist_ok=True)
sequence = iter((("en_US", False), ("en_US", True), ("fr_FR", False), ("fr_FR", True)))
window = bpy.context.window_manager.windows[0]
area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
area.spaces.active.show_region_ui = True
region = next(region for region in area.regions if region.type == "UI")
view = bpy.context.preferences.view
view.use_translate_interface = True
bpy.context.preferences.use_preferences_save = False
pending = None


def capture():
    global pending
    if pending:
        language, error = pending
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.screen.screenshot(
                filepath=str(root / f"{language}-{'recovery' if error else 'setup'}.png"), hide_props_region=False
            )
        print("Sidebar width in pixels:", region.width, flush=True)
    pending = next(sequence, None)
    if pending is None:
        bpy.ops.wm.quit_blender()
        return None
    language, error = pending
    view.language = language
    region.active_panel_category = "FluidiTy"
    with bpy.context.temp_override(window=window, area=area, region=region):
        scene = window.scene
        if error:
            scene.render.fps = 25
        bpy.ops.fluidity.check_scene()
        scene.render.fps = 24
    area.tag_redraw()
    return 0.75


bpy.app.timers.register(capture, first_interval=1.0)
