"""Reopen the native gas Volume without the extension and verify render-only edits."""

import hashlib
import json
from pathlib import Path
import sys

import bpy


def receipt(paths):
    return {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in paths}


def render(scene, destination):
    scene.render.filepath = str(destination)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(destination), check_existing=False)
    try:
        pixels = tuple(image.pixels)
        return tuple(sum(pixels[channel::4]) for channel in range(3))
    finally:
        bpy.data.images.remove(image)


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(bpy.data.filepath).parent
    cache = root / "bake"
    tracked = [*root.glob("fire_*.vdb"), *[path for path in cache.rglob("*") if path.is_file()]]
    before = receipt(tracked)
    assert len(list(root.glob("fire_*.vdb"))) == 13
    scene = bpy.context.scene
    obj = bpy.data.objects["FluidiTy Fire Volume"]
    assert obj.asset_data and obj.data.materials[0].asset_data
    for frame in (0, 12):
        scene.frame_set(frame)
        obj.data.grids.load()
        assert {grid.name for grid in obj.data.grids} == {"density", "temperature", "fuel", "flame", "soot"}

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.render.resolution_x = 96
    scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2
    scene.render.image_settings.file_format = "PNG"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.view_settings.view_transform = "Standard"

    baseline = render(scene, output / "original.png")
    assert sum(baseline) > 0.0, baseline
    shader = next(node for node in obj.data.materials[0].node_tree.nodes if node.type == "PRINCIPLED_VOLUME")
    obj.data.display.density *= 0.75
    shader.inputs["Emission Color"].default_value = (0.02, 0.25, 1.0, 1.0)
    edited = render(scene, output / "edited.png")
    assert baseline[0] > edited[0] and edited[2] > baseline[2], (baseline, edited)
    assert receipt(tracked) == before
    (output / "render-receipt.json").write_text(
        json.dumps({"original_rgb_sum": baseline, "edited_rgb_sum": edited, "unchanged_files": len(before)}, indent=2),
        encoding="utf-8",
    )
    print("PASS reopened Volume renders material changes without touching the bake")


if __name__ == "__main__":
    main()
