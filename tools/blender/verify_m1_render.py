"""Run in a fresh Blender process with extensions disabled and no native solver."""

import hashlib
import sys
from pathlib import Path

import bpy


def main():
    root = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    assert not hasattr(bpy.types.Scene, "fluidity")
    assert not bpy.app.handlers.frame_change_pre and not bpy.app.handlers.frame_change_post
    scene = bpy.context.scene
    obj = scene.objects["FluidiTy Points"]
    reader, display = obj.modifiers
    assert reader.type == "MESH_SEQUENCE_CACHE"
    assert display.type == "NODES" and display.node_group.get("fluidity_asset_version") in {1, 2}
    assert reader.cache_file.filepath.startswith("//")
    assert Path(bpy.path.abspath(reader.cache_file.filepath)).is_file()
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    hashes = []
    for frame in (0, 10):
        scene.frame_set(frame)
        path = root / f"points-{frame:04d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(path), check_existing=False)
        alpha = image.pixels[:][3::4]
        assert max(alpha) > 0.9 and min(alpha) == 0
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert hashes[0] != hashes[1]
    print("Extension-disabled PNG SHA-256:", *hashes)


main()
