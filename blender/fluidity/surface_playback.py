"""Attach a verified Mesh cache as a native Blender USD surface."""

import filecmp
import json
import subprocess
from pathlib import Path
from uuid import uuid4

import bpy

from . import jobs, playback
from .scene_inputs import require_cadence


def native_reader(obj):
    return next((mod for mod in obj.modifiers
                 if mod.type == "MESH_SEQUENCE_CACHE" and mod.object_path == "/FluidiTySurface"), None)


def attach(scene, cache):
    """Validate a complete Mesh publication before updating the saved binding."""
    require_cadence(scene)
    if not bpy.data.filepath:
        raise jobs.ToolError("Save the Blender project before using a cache.")
    relative_cache = playback.relative(cache)
    last = playback.committed_last(cache, "flem")
    existing = scene.fluidity.surface_object
    if existing and (existing.type != "MESH" or not existing.get("fluidity_surface")):
        raise jobs.ToolError("The saved Surface binding is not a FluidiTy object.")
    previous_reader = native_reader(existing) if existing else None
    if existing and existing.get("fluidity_surface_last", 0) > 0 and previous_reader is None:
        raise jobs.ToolError("Restore the FluidiTy Surface cache reader before relinking.")

    archive = cache / f"FluidiTySurface-{uuid4().hex}.usda"
    archive_path = playback.relative(archive)
    old_objects = set(bpy.data.objects)
    old_meshes = set(bpy.data.meshes)
    old_caches = set(bpy.data.cache_files)
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    view_layer = bpy.context.view_layer
    active = view_layer.objects.active
    selected = tuple(bpy.context.selected_objects)
    keep = None
    complete = False
    try:
        try:
            response = jobs.checked("blender_bridge", "export-surface-usd", cache, last, archive, timeout=120)
        except subprocess.TimeoutExpired as error:
            raise jobs.ToolError("Surface export timed out.") from error
        info = json.loads(response)
        if existing and scene.fluidity.attached_surface_cache == relative_cache:
            old_archive = existing.get("fluidity_surface_archive")
            if (old_archive and existing.get("fluidity_surface_execution") == info["execution_id"]
                    and existing.get("fluidity_surface_last") == last):
                old_path = Path(bpy.path.abspath(old_archive))
                if old_path.is_file() and filecmp.cmp(archive, old_path, shallow=False):
                    return last

        result = bpy.ops.wm.usd_import(
            filepath=str(archive), set_frame_range=False, relative_path=False,
            import_materials=False, read_mesh_attributes=True,
        )
        imported = tuple(set(bpy.data.objects) - old_objects)
        if result != {"FINISHED"} or len(imported) != 1 or imported[0].type != "MESH":
            raise jobs.ToolError("Blender could not open the verified Surface archive.")
        candidate = imported[0]
        reader = native_reader(candidate)
        if last > 0 and (reader is None or reader.cache_file is None):
            raise jobs.ToolError("Blender could not find the native Surface reader.")
        if reader:
            reader.cache_file.filepath = archive_path
        scene.frame_set(last)
        evaluated = candidate.evaluated_get(bpy.context.evaluated_depsgraph_get())
        if (len(evaluated.data.vertices), len(evaluated.data.polygons)) != (info["vertex_count"], info["triangle_count"]):
            raise jobs.ToolError("Blender could not read the last committed Surface sample.")

        if existing:
            for material in existing.data.materials:
                candidate.data.materials.append(material)
            if reader:
                target = previous_reader or existing.modifiers.new("FluidiTy Surface", "MESH_SEQUENCE_CACHE")
                existing.modifiers.move(existing.modifiers.find(target.name), 0)
                target.object_path = reader.object_path
                target.read_data = reader.read_data
                target.cache_file = reader.cache_file
            elif previous_reader:
                existing.modifiers.remove(previous_reader)
            existing.data = candidate.data
        else:
            candidate.name = "FluidiTy Surface"
            collection = bpy.data.collections.new("FluidiTy Surface Playback")
            scene.collection.children.link(collection)
            for owner in tuple(candidate.users_collection):
                owner.objects.unlink(candidate)
            collection.objects.link(candidate)
            keep = existing = candidate
            scene.fluidity.surface_object = existing
        existing["fluidity_surface"] = True
        existing["fluidity_surface_archive"] = archive_path
        existing["fluidity_surface_execution"] = info["execution_id"]
        existing["fluidity_surface_source_execution"] = info["source_execution_id"]
        existing["fluidity_surface_last"] = last
        scene.fluidity.attached_surface_cache = relative_cache
        complete = True
        return last
    finally:
        for obj in set(bpy.data.objects) - old_objects:
            if obj != keep:
                bpy.data.objects.remove(obj, do_unlink=True)
        unused = [item for item in set(bpy.data.meshes) - old_meshes if not item.users]
        unused += [item for item in set(bpy.data.cache_files) - old_caches if not item.users]
        if unused:
            bpy.data.batch_remove(ids=unused)
        scene.frame_set(saved_frame, subframe=saved_subframe)
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in selected:
            obj.select_set(True)
        view_layer.objects.active = active
        if not complete:
            try:
                archive.unlink(missing_ok=True)
            except PermissionError:
                # Blender may keep the file open until the dependency graph updates.
                pass
