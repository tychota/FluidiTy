"""Resolve native Blender gas authoring and attach committed Volume playback."""

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

from . import gas_volume, jobs
from .scene_inputs import SceneError, require_cadence


def _solver_vector(vector):
    return dict(x=float(vector.x), y=float(vector.z), z=float(-vector.y))


def _bounds(obj):
    points = [_solver_vector(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    axes = ("x", "y", "z")
    return ({axis: min(point[axis] for point in points) for axis in axes},
            {axis: max(point[axis] for point in points) for axis in axes})


def scene_packet(scene):
    """Sample gas input from the scene's evaluated frame-zero domain and animated keys."""
    require_cadence(scene)
    settings = scene.fluidity
    if scene.frame_start != 0 or not 1 <= scene.frame_end <= 240:
        raise SceneError("timing", "Gas requires frame zero and an end frame between 1 and 240.")
    if settings.domain is None or settings.domain.type != "MESH":
        raise SceneError("domain", "Choose a gas domain box.")
    if settings.gas_source is None or settings.gas_source.type != "MESH":
        raise SceneError("gas source", "Choose a fuel and heat source box.")
    wind = settings.gas_wind
    if wind is not None and (wind.field is None or wind.field.type != "WIND"):
        raise SceneError("gas wind", "Choose a Blender Wind field.")
    keys, draft = [], []
    original = scene.frame_current, scene.frame_subframe
    try:
        scene.frame_set(0)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        origin, end = _bounds(settings.domain.evaluated_get(depsgraph))
        source_bounds = _bounds(settings.gas_source.evaluated_get(depsgraph))
        extent = {}
        for axis in origin:
            count = (end[axis] - origin[axis]) / settings.cell_width
            extent[axis] = round(count)
            if extent[axis] < 1 or not math.isclose(count, extent[axis], abs_tol=1e-5):
                raise SceneError("domain", "Gas domain box dimensions must be whole cell widths.")
        if math.prod(extent.values()) > 4096:
            raise SceneError("domain", "Gas domain exceeds 4096 cells.")
        for frame in range(scene.frame_end + 1):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            emitter = settings.gas_source.evaluated_get(depsgraph)
            if _bounds(settings.domain.evaluated_get(depsgraph)) != (origin, end) or _bounds(emitter) != source_bounds:
                raise SceneError("gas source", "Gas domain and emitter boxes must stay fixed during the bake.")
            rates = [emitter.get("Fuel Rate", 0.0), emitter.get("Heat Rate", 0.0)]
            if not all(isinstance(rate, (int, float)) and math.isfinite(rate) and rate >= 0 for rate in rates):
                raise SceneError("gas source", "Fuel Rate and Heat Rate must be finite nonnegative numbers.")
            keys.append(dict(time_s=frame / 24, fuel_kg_per_m3_s=rates[0], heat_w_per_m3=rates[1]))
            if wind is not None:
                field = wind.evaluated_get(depsgraph)
                direction = field.matrix_world.to_quaternion() @ Vector((0, 0, 1))
                draft.append(dict(time_s=frame / 24,
                                  acceleration_m_per_s2=_solver_vector(direction * field.field.strength)))
    finally:
        scene.frame_set(original[0], subframe=original[1])
    return dict(
        backend=settings.backend.capitalize(),
        world_origin_m=origin,
        extent=extent,
        cell_width_m=settings.cell_width,
        open_boundaries=59,
        initial_velocity_m_per_s=dict(x=0.0, y=0.0, z=0.0),
        gravity_m_per_s2=_solver_vector(scene.gravity if scene.use_gravity else Vector()),
        substeps_per_frame=settings.gas_substeps,
        emitters=[dict(lower_m=source_bounds[0], upper_m=source_bounds[1], keys=keys)],
        draft=draft,
        reaction=dict(ignition_temperature_k=299.0, burn_rate_per_s=7.0,
                      heat_per_fuel_j_per_kg=4000000.0, soot_yield=0.2),
    )


def prepare(packet, directory):
    """Resolve through the packaged solver, binding its actual executable identity."""
    identity = jobs.checked("simulate", "schema")
    data = dict(packet, build_digest=dict(bytes=list(bytes.fromhex(identity["build_digest"]))))
    packet_path = Path(directory) / "gas.json"
    scene_path = Path(directory) / "gas.flgs"
    packet_path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    jobs.checked("simulate", "gas-resolve", packet_path, scene_path)
    return scene_path


def start(scene, cache, packet, last_frame, resume):
    def command(directory):
        resolved = prepare(packet, directory)
        args = [jobs.binary("simulate"), "run", "--stage", "simulate", "--scene", resolved, "--cache", cache,
                "--frame-end", str(last_frame)]
        if resume:
            args.append("--resume")
        return args

    return jobs.start_job(scene, "gas", command)


def attach(scene, cache):
    """Export and attach the contiguous published gas frame range."""
    checkpoints = cache / "checkpoints"
    frames = sorted(int(path.stem) for path in checkpoints.glob("*.flgc") if path.stem.isdecimal())
    if not frames or frames != list(range(frames[-1] + 1)):
        raise SceneError("cache", "Gas playback requires a contiguous cache starting at frame zero.")
    directory = cache.with_name(cache.name + "-volumes")
    directory.mkdir(exist_ok=True)
    solver = jobs.binary("simulate")
    for frame in frames:
        path = directory / f"fire_{frame:04d}.vdb"
        if not path.exists():
            gas_volume.export_gas_volume_frame(solver, cache, frame, path)
    return gas_volume.attach_gas_volume(scene, directory / "fire_0000.vdb", 0, len(frames))
