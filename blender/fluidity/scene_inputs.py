"""Resolve the bounded M1 Blender scene into the native author's input packet."""

import math
from itertools import product
from typing import NamedTuple

import bpy
from bpy_extras.anim_utils import action_get_channelbag_for_slot
from mathutils import Matrix


class SceneError(ValueError):
    pass


def require_cadence(scene):
    if scene.render.fps != 24 or scene.render.fps_base != 1:
        raise SceneError("fps", "Set the scene to 24 fps.")


def box_bounds(obj, field, depsgraph, *, local=False):
    if not obj or obj.type != "MESH":
        raise SceneError(field, "Select a closed mesh box.")
    if obj.parent or obj.constraints or obj.modifiers or obj.data.shape_keys:
        raise SceneError(field, "Apply modifiers and remove parents, constraints and shape keys.")
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.data
    if len(mesh.vertices) != 8 or len(mesh.polygons) != 6 or len(mesh.edges) != 12:
        raise SceneError(field, "Use a closed, world-axis-aligned box.")
    corners = [vertex.co if local else evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    low = tuple(min(point[axis] for point in corners) for axis in range(3))
    high = tuple(max(point[axis] for point in corners) for axis in range(3))
    if any(not math.isfinite(value) or abs(value) > 100 for value in (*low, *high)):
        raise SceneError(field, "Keep world coordinates within 100 meters of the origin.")
    if any(upper - lower < 1e-5 for lower, upper in zip(low, high, strict=True)):
        raise SceneError(field, "Give the box positive thickness on every axis.")
    corner_bits = [
        tuple(
            0 if abs(point[axis] - low[axis]) < 1e-6 else 1 if abs(point[axis] - high[axis]) < 1e-6 else -1
            for axis in range(3)
        )
        for point in corners
    ]
    if set(corner_bits) != set(product((0, 1), repeat=3)):
        raise SceneError(field, "Use a closed, world-axis-aligned box.")
    faces = set()
    for polygon in mesh.polygons:
        vertices = [corner_bits[index] for index in polygon.vertices]
        if len(vertices) != 4 or len(set(vertices)) != 4:
            raise SceneError(field, "Use a closed, world-axis-aligned box.")
        if any(
            sum(x != y for x, y in zip(a, b, strict=True)) != 1
            for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True)
        ):
            raise SceneError(field, "Use a closed, world-axis-aligned box.")
        faces.add(frozenset(vertices))
    if len(faces) != 6:
        raise SceneError(field, "Use a closed, world-axis-aligned box.")
    return low, high


def paddle_keys(paddle, frame_end):
    data = paddle.animation_data
    if not data or not data.action or not data.action_slot or data.drivers or data.nla_tracks:
        raise SceneError("paddle", "Animate X with two linear location keys that cover the bake.")
    bag = action_get_channelbag_for_slot(data.action, data.action_slot)
    curves = list(bag.fcurves) if bag else []
    if not 1 <= len(curves) <= 3:
        raise SceneError("paddle", "Animate X with two linear location keys that cover the bake.")
    moving = []
    for curve in curves:
        keys = list(curve.keyframe_points)
        if (
            curve.data_path != "location"
            or curve.modifiers
            or curve.mute
            or len(keys) != 2
            or keys[0].interpolation != "LINEAR"
            or keys[0].co.x > 0
            or keys[1].co.x < frame_end
        ):
            raise SceneError("paddle", "Animate X with two linear location keys that cover the bake.")
        if keys[0].co.y != keys[1].co.y:
            moving.append(curve.array_index)
    if moving != [0]:
        raise SceneError("paddle", "Translate the paddle along world X only.")


def legacy_scene_inputs(scene, preview=False):
    """Return a native-author packet using evaluated Blender meters and 24 fps time zero."""
    settings = scene.fluidity
    require_cadence(scene)
    if scene.unit_settings.scale_length != 1:
        raise SceneError("units", "Use one Blender unit per meter.")
    if scene.animation_data:
        raise SceneError("animation", "Remove animation and drivers from scene settings.")
    domain, liquid, paddle = settings.domain, settings.liquid, settings.paddle
    if not all((domain, liquid, paddle)) or len({domain, liquid, paddle}) != 3:
        raise SceneError("roles", "Assign distinct objects to Domain, Initial Liquid and Paddle.")
    if domain.animation_data or liquid.animation_data:
        raise SceneError("roles", "Keep the domain and initial liquid static.")
    frame_end = scene.frame_end
    if scene.frame_start != 0 or not 2 <= frame_end <= 240:
        raise SceneError("range", "Set a frame range starting at 0 and ending from 2 through 240.")
    paddle_keys(paddle, frame_end)
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    try:
        scene.frame_set(0)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        domain_low, domain_high = box_bounds(domain, "domain", depsgraph)
        liquid_low, liquid_high = box_bounds(liquid, "liquid", depsgraph)
        paddle_low, paddle_high = box_bounds(paddle, "paddle", depsgraph)
        scene.frame_set(frame_end)
        paddle_end_low, paddle_end_high = box_bounds(paddle, "paddle", depsgraph)
    finally:
        scene.frame_set(saved_frame, subframe=saved_subframe)
    if any(
        a < lo - 1e-6 or b > hi + 1e-6
        for a, b, lo, hi in zip(liquid_low, liquid_high, domain_low, domain_high, strict=True)
    ):
        raise SceneError("liquid", "Place the initial liquid inside the domain.")
    shift = paddle_end_low[0] - paddle_low[0]
    if abs(shift) < 1e-6 or any(
        abs(end[axis] - start[axis] - (shift if axis == 0 else 0)) > 1e-6
        for end, start in ((paddle_end_low, paddle_low), (paddle_end_high, paddle_high))
        for axis in range(3)
    ):
        raise SceneError("paddle", "Translate the paddle along world X only.")
    if liquid_low[0] <= paddle_high[0] and liquid_high[0] >= paddle_low[0]:
        raise SceneError("paddle", "Move the starting paddle clear of the initial liquid.")
    if any(
        paddle_low[axis] > domain_low[axis] + 1e-6 or paddle_high[axis] < domain_high[axis] - 1e-6 for axis in (1, 2)
    ):
        raise SceneError("paddle", "Extend the paddle across the domain on Y and Z.")
    cell = settings.cell_width
    if not 0.001 <= cell <= 10:
        raise SceneError("cell_width", "Choose a cell width from 0.001 to 10 meters.")
    # Solver lattice axes are Y-up. Visit cells in stable (z, y, x) order.
    low = (liquid_low[0], liquid_low[2], -liquid_high[1])
    high = (liquid_high[0], liquid_high[2], -liquid_low[1])
    ranges = [
        range(math.floor(lo / cell - 0.5) + 1, math.ceil(hi / cell - 0.5)) for lo, hi in zip(low, high, strict=True)
    ]
    count = math.prod(map(len, ranges))
    if not 1 <= count <= 1024:
        raise SceneError("cell_width", "Choose a cell width that seeds 1 to 1024 particles.")
    gx, gy, gz = scene.gravity if scene.use_gravity else (0, 0, 0)
    center = (paddle_low[0] + paddle_high[0]) / 2
    half_width = (paddle_high[0] - paddle_low[0]) / 2
    lines = [
        "FLUIDITY-BLENDER-1",
        f"{int(preview)} {cell!r}",
        f"{gx!r} {gz!r} {-gy!r}",
        f"{center!r} {half_width!r} {shift * 24 / frame_end!r}",
        str(count),
    ]
    for z, y, x in product(ranges[2], ranges[1], ranges[0]):
        lines.append(" ".join(repr((coordinate + 0.5) * cell) for coordinate in (x, y, z)))
    return "\n".join(lines) + "\n"


def linear_motion(obj, frame_end, field="collider"):
    """Require a static object or fully covered constant location motion."""
    data = obj.animation_data
    if not data:
        return
    if not data.action or not data.action_slot or data.drivers or data.nla_tracks:
        raise SceneError(field, "Use linear location animation or a static object.")
    bag = action_get_channelbag_for_slot(data.action, data.action_slot)
    for curve in bag.fcurves if bag else ():
        keys = list(curve.keyframe_points)
        if (
            curve.data_path != "location"
            or curve.modifiers
            or curve.mute
            or len(keys) != 2
            or keys[0].co.x > 0
            or keys[1].co.x < frame_end
            or keys[0].interpolation != "LINEAR"
        ):
            raise SceneError(field, "Use two linear location keys covering the simulation.")


class CutterSlab(NamedTuple):
    """Time-zero slab geometry in solver world coordinates, in meters and m/s."""

    unit_normal: tuple[float, float, float]
    start_center_coordinate: float
    half_width: float
    wall_velocity: tuple[float, float, float]


def resolve_cutter(scene):
    """Resolve a domain-spanning X slab from the assigned Cutter's native animation."""
    require_cadence(scene)
    if scene.unit_settings.scale_length != 1:
        raise SceneError("units", "Use one Blender unit per meter.")
    if scene.frame_start != 0 or not 2 <= scene.frame_end <= 240:
        raise SceneError("range", "Set a frame range starting at 0 and ending from 2 through 240.")
    settings = scene.fluidity
    cutter = settings.cutter
    domain = settings.domain
    if not cutter or cutter in {domain, settings.liquid, settings.collider, settings.paddle}:
        raise SceneError("cutter", "Assign a distinct mesh object to Cutter.")
    if not domain or domain.animation_data:
        raise SceneError("domain", "Assign a static domain box.")

    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    try:
        linear_motion(cutter, scene.frame_end, "cutter")
        scene.frame_set(0)
        graph = bpy.context.evaluated_depsgraph_get()
        domain_low, domain_high = box_bounds(domain, "domain", graph)
        low, high = box_bounds(cutter, "cutter", graph)
        scene.frame_set(scene.frame_end)
        graph = bpy.context.evaluated_depsgraph_get()
        end_low, end_high = box_bounds(cutter, "cutter", graph)
    except SceneError as error:
        field, remedy = error.args
        if field != "cutter":
            raise
        raise SceneError(field, f"{cutter.name}: {remedy}") from error
    finally:
        scene.frame_set(saved_frame, subframe=saved_subframe)

    shift = tuple(end - start for start, end in zip(low, end_low, strict=True))
    if any(abs(end - start - delta) > 1e-6 for start, end, delta in zip(high, end_high, shift, strict=True)):
        raise SceneError("cutter", f"{cutter.name}: Keep dimensions fixed during translation.")
    if any(
        start_low[axis] > domain_low[axis] + 1e-6 or start_high[axis] < domain_high[axis] - 1e-6
        for start_low, start_high in ((low, high), (end_low, end_high))
        for axis in (1, 2)
    ):
        raise SceneError("cutter", f"{cutter.name}: Extend the box across the domain on Y and Z.")
    seconds = scene.frame_end / 24
    return CutterSlab(
        (1, 0, 0),
        (low[0] + high[0]) / 2,
        (high[0] - low[0]) / 2,
        (shift[0] / seconds, shift[2] / seconds, -shift[1] / seconds),
    )


def cutter_frame_packet(scene):
    """Encode a dry elastic box and translating Cutter for the native frame runner."""
    settings = scene.fluidity
    slab = resolve_cutter(scene)
    solid = settings.elastic_solid
    if not solid or solid in {settings.domain, settings.cutter, settings.liquid, settings.collider, settings.paddle}:
        raise SceneError("elastic_solid", "Assign a distinct static box to Elastic Solid.")
    if solid.animation_data:
        raise SceneError("elastic_solid", f"{solid.name}: Remove animation from the initial elastic solid.")

    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    try:
        scene.frame_set(0)
        graph = bpy.context.evaluated_depsgraph_get()
        domain_low, domain_high = box_bounds(settings.domain, "domain", graph)
        solid_low, solid_high = box_bounds(solid, "elastic_solid", graph)
    finally:
        scene.frame_set(saved_frame, subframe=saved_subframe)
    if any(
        low < domain_min or high > domain_max
        for low, high, domain_min, domain_max in zip(solid_low, solid_high, domain_low, domain_high, strict=True)
    ):
        raise SceneError("elastic_solid", f"{solid.name}: Place the initial elastic solid inside the domain.")

    cell = settings.cell_width
    if not 0.001 <= cell <= 10:
        raise SceneError("cell_width", "Choose a cell width from 0.001 to 10 meters.")
    low = (solid_low[0], solid_low[2], -solid_high[1])
    high = (solid_high[0], solid_high[2], -solid_low[1])
    ranges = [
        range(math.floor(lo / cell - 0.5) + 1, math.ceil(hi / cell - 0.5)) for lo, hi in zip(low, high, strict=True)
    ]
    count = math.prod(map(len, ranges))
    if not 1 <= count <= 1024:
        raise SceneError("cell_width", "Choose a cell width that seeds 1 to 1024 particles.")

    def values(items):
        return " ".join(map(repr, items))

    lines = [
        "FLUIDITY-CUTTER-1",
        values(
            (
                cell,
                settings.elastic_density,
                settings.elastic_shear_modulus,
                settings.elastic_first_lame_parameter,
                settings.friction,
                settings.elastic_substeps,
            )
        ),
        values((domain_low[0], domain_low[2], -domain_high[1])),
        values((domain_high[0], domain_high[2], -domain_low[1])),
        values((slab.start_center_coordinate, slab.half_width, *slab.wall_velocity)),
        str(count),
    ]
    for z, y, x in product(ranges[2], ranges[1], ranges[0]):
        lines.append(values((coordinate + 0.5) * cell for coordinate in (x, y, z)))
    return "\n".join(lines) + "\n"


def extended_liquid_rows(sources, bounds, cell, initial_velocity, phase_tension, writes_viscosity):
    """Build canonical registry, interaction, emitter and seed rows."""
    materials = {}
    emitters = []
    seeds = []
    for ordinal, (source, (blender_low, blender_high)) in enumerate(zip(sources, bounds, strict=True), 1):
        material = (source.phase_id, source.density, source.viscosity if source.override_viscosity else -1.0)
        if materials.setdefault(source.substance_id, material) != material:
            raise SceneError("liquid", "Give each substance ID one phase, density and viscosity.")
        low = (blender_low[0], blender_low[2], -blender_high[1])
        high = (blender_high[0], blender_high[2], -blender_low[1])
        widths = tuple(b - a for a, b in zip(low, high, strict=True))
        cells = tuple(max(1, math.ceil(width / cell)) for width in widths)
        count = math.prod(cells)
        if len(seeds) + count > 1024:
            raise SceneError("cell_width", "Choose a cell width that seeds 1 to 1024 particles.")
        volume = math.prod(widths) / count
        if not math.isfinite(volume) or volume <= 0:
            raise SceneError("liquid", "Give each source box positive volume.")
        for indices in product(*(range(size) for size in cells)):
            position = tuple(
                start + (index + 0.5) * width / size
                for start, index, width, size in zip(low, indices, widths, cells, strict=True)
            )
            seeds.append((*position, source.substance_id, volume))
        if source.emission_rate > 0:
            center = tuple((a + b) * 0.5 for a, b in zip(low, high, strict=True))
            emitters.append((
                ordinal, source.substance_id, source.emission_rate, source.emitted_volume,
                *center, 0, 0, 0, *initial_velocity,
            ))
    phases = sorted({material[0] for material in materials.values()})
    if len(phases) > 2:
        raise SceneError("liquid", "Assign sources to at most two miscibility phases.")
    pairs = [(*phases, phase_tension)] if len(phases) == 2 else []
    rows = [(len(materials), len(pairs), len(emitters))]
    rows.extend((substance, *material[:3 if writes_viscosity else 2])
                for substance, material in sorted(materials.items()))
    rows.extend(pairs)
    rows.extend(emitters)
    rows.append((len(seeds),))
    rows.extend(seeds)
    return [" ".join(map(repr, row)) for row in rows]

def rigid_scene_packets(scene, preview=False):
    """Resolve the planar tank and the rigid box from its native frame-zero transform."""
    settings = scene.fluidity
    box = settings.rigid_box
    sources = settings.liquid_sources
    if not box or settings.cutter or settings.boundary_mode != "TANK" or not settings.planar_fill:
        raise SceneError("rigid_box", "Assign a Rigid Box and use a planar tank fill without a Cutter.")
    if settings.backend != "cpu" or settings.friction or settings.surface_tension:
        raise SceneError("rigid_box", "Use CPU, zero wall friction, and zero surface tension for a rigid bake.")
    if not sources or any(source.emission_rate or source.override_viscosity for source in sources):
        raise SceneError("rigid_box", "Use initial liquid sources without emission or viscosity overrides.")
    if len({(source.substance_id, source.phase_id, source.density) for source in sources}) != 1:
        raise SceneError("rigid_box", "Use one liquid substance for a rigid bake.")
    if box in {settings.domain, settings.liquid, settings.collider, settings.elastic_solid,
               *(source.geometry for source in sources)}:
        raise SceneError("rigid_box", "Assign a distinct mesh to Rigid Box.")
    if box.animation_data or box.rigid_body:
        raise SceneError("rigid_box", "Clear transform animation and Blender rigid-body physics from the driven box.")

    liquid_packet = scene_inputs(scene, preview, rigid=True)
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    try:
        scene.frame_set(0)
        graph = bpy.context.evaluated_depsgraph_get()
        low, high = box_bounds(box, "rigid_box", graph, local=True)
        transform = box.evaluated_get(graph).matrix_world
        center, rotation, scale = transform.decompose()
        if any(abs(a + b) > 1e-6 for a, b in zip(low, high, strict=True)):
            raise SceneError("rigid_box", "Center the box mesh on its object origin.")
        if any(not math.isfinite(value) or value <= 0 for value in (*scale, *box.scale)):
            raise SceneError("rigid_box", "Use positive scale on every box axis.")
        if any(not math.isfinite(value) or abs(value) > 100 for value in center):
            raise SceneError("rigid_box", "Keep the box center within 100 meters of the origin.")
        extent = tuple((b - a) * size * 0.5 for a, b, size in zip(low, high, scale, strict=True))
        if any(not math.isfinite(value) or value <= 0 for value in extent):
            raise SceneError("rigid_box", "Give the box finite positive dimensions.")
        axes = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
        orientation = tuple((axes @ rotation.to_matrix()).to_quaternion())
        orientation_norm = math.hypot(*orientation)
        orientation = tuple(value / orientation_norm for value in orientation)
    finally:
        scene.frame_set(saved_frame, subframe=saved_subframe)

    mass = settings.rigid_mass
    inertia = tuple(mass * (extent[(axis + 1) % 3] ** 2 + extent[(axis + 2) % 3] ** 2) / 3 for axis in range(3))
    packet = "FLUIDITY-RIGID-1 " + " ".join(map(repr, (
        settings.rigid_substeps, mass, *inertia, *extent, center.x, center.z, -center.y, *orientation,
        0, 0, 0, 0, 0, 0,
    ))) + "\n"
    return liquid_packet, packet

def snow_seed_rows(settings, bounds, cell):
    low, high = bounds
    center_x, center_y = ((low[axis] + high[axis]) / 2 for axis in (0, 1))
    radius_x, radius_y = ((high[axis] - low[axis]) / 2 for axis in (0, 1))
    height = high[2] - low[2]
    spacing = 0.5 * cell
    ranges = [range(math.floor(a / spacing - 0.5) + 1, math.ceil(b / spacing - 0.5))
              for a, b in zip(low, high, strict=True)]
    if math.prod(map(len, ranges)) > 8192:
        raise SceneError("snow_mound", "Increase cell width or reduce the snow mound size.")
    seeds = []
    for z, y, x in product(ranges[2], ranges[1], ranges[0]):
        px, py, pz = ((index + 0.5) * spacing for index in (x, y, z))
        ellipsoid = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2
        ellipsoid += ((pz - low[2]) / height) ** 2
        if ellipsoid < 1:
            seeds.append((px, pz, -py))
    if not 1 <= len(seeds) <= 1024:
        raise SceneError("snow_mound", "Choose a cell width that seeds 1 to 1024 snow particles.")
    parameters = (
        1, 1, settings.snow_lambda, settings.snow_shear, settings.snow_compression,
        settings.snow_stretch, settings.snow_hardening, settings.snow_density, spacing ** 3, len(seeds),
    )
    return [" ".join(map(repr, row)) for row in (parameters, *seeds)]


def scene_inputs(scene, preview=False, *, rigid=False):
    """Resolve supported domain/material/collider authoring; retain v1 scenes exactly."""
    settings = scene.fluidity
    if settings.cutter:
        raise SceneError("cutter", "Cutter requires an elastic bake. Clear Cutter to bake the liquid scene.")
    if settings.rigid_box and not rigid:
        raise SceneError("rigid_box", "Use a coupled rigid bake for the assigned box.")
    sources = settings.liquid_sources
    if settings.boundary_mode == "LEGACY_PADDLE":
        if sources:
            raise SceneError("liquid", "Choose a boundary model for the liquid sources.")
        return legacy_scene_inputs(scene, preview)
    require_cadence(scene)
    if scene.unit_settings.scale_length != 1:
        raise SceneError("units", "Use one Blender unit per meter.")
    if scene.animation_data:
        raise SceneError("animation", "Remove animation and drivers from scene settings.")
    if scene.frame_start != 0 or not 2 <= scene.frame_end <= 240:
        raise SceneError("range", "Set a frame range starting at 0 and ending from 2 through 240.")
    liquids = [source.geometry for source in sources] if sources else [settings.liquid]
    if not settings.domain or any(not obj or obj == settings.domain for obj in liquids):
        raise SceneError("roles", "Assign distinct Domain and Initial Liquid objects.")
    if len(set(liquids)) != len(liquids):
        raise SceneError("liquid", "Assign a distinct box to each liquid source.")
    if settings.domain.animation_data or any(obj.animation_data for obj in liquids):
        raise SceneError("roles", "Keep the domain and initial liquid static.")
    snow_platen = settings.boundary_mode == "SNOW_PLATEN"
    if snow_platen:
        if not settings.snow_mound or settings.snow_mound in {settings.domain, settings.collider, *liquids}:
            raise SceneError("snow_mound", "Assign a distinct Snow Mound bounds box.")
        if settings.snow_mound.animation_data:
            raise SceneError("snow_mound", "Keep the Snow Mound bounds static.")
        if not any(source.emission_rate > 0 for source in sources) or any(source.override_viscosity for source in sources):
            raise SceneError("liquid", "Use a water emitter with positive emission and no viscosity override.")
    using_collider = settings.boundary_mode in {"PLANE", "SLAB", "SNOW_PLATEN"}
    collider = settings.collider if using_collider else None
    if using_collider and not collider:
        raise SceneError("collider", "Assign a collider for the selected boundary model.")
    if collider and collider in {settings.domain, *liquids}:
        raise SceneError("roles", "Assign a different object to each simulation role.")
    if collider:
        linear_motion(collider, scene.frame_end)
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    try:
        scene.frame_set(0)
        graph = bpy.context.evaluated_depsgraph_get()
        domain_low, domain_high = box_bounds(settings.domain, "domain", graph)
        liquid_bounds = [box_bounds(obj, "liquid", graph) for obj in liquids]
        if snow_platen:
            snow_bounds = box_bounds(settings.snow_mound, "snow_mound", graph)
        liquid_low = tuple(min(low[axis] for low, _ in liquid_bounds) for axis in range(3))
        liquid_high = tuple(max(high[axis] for _, high in liquid_bounds) for axis in range(3))
        if collider:
            collider_low, collider_high = box_bounds(collider, "collider", graph)
            scene.frame_set(scene.frame_end)
            end_low, end_high = box_bounds(collider, "collider", graph)
            shift = tuple(end_low[i] - collider_low[i] for i in range(3))
            if any(abs(end_high[i] - collider_high[i] - shift[i]) > 1e-6 for i in range(3)):
                raise SceneError("collider", "Keep collider dimensions fixed during translation.")
            wall_velocity = tuple(value * 24 / scene.frame_end for value in shift)
    finally:
        scene.frame_set(saved_frame, subframe=saved_subframe)
    if any(
        a < lo - 1e-6 or b > hi + 1e-6
        for a, b, lo, hi in zip(liquid_low, liquid_high, domain_low, domain_high, strict=True)
    ):
        raise SceneError("liquid", "Place the initial liquid inside the domain.")
    cell = settings.cell_width
    low = (liquid_low[0], liquid_low[2], -liquid_high[1])
    high = (liquid_high[0], liquid_high[2], -liquid_low[1])
    ranges = [
        range(math.floor(lo / cell - 0.5) + 1, math.ceil(hi / cell - 0.5)) for lo, hi in zip(low, high, strict=True)
    ]
    count = math.prod(map(len, ranges))
    if not sources and not 1 <= count <= 1024:
        raise SceneError("cell_width", "Choose a cell width that seeds 1 to 1024 particles.")
    gx, gy, gz = scene.gravity if scene.use_gravity else (0, 0, 0)
    vx, vy, vz = settings.initial_velocity
    planar_fill = bool(sources and settings.boundary_mode == "TANK" and settings.planar_fill)
    viscosity_authoring = planar_fill or any(source.override_viscosity for source in sources)
    if planar_fill and not domain_low[2] < settings.fill_height <= domain_high[2]:
        raise SceneError("liquid", "Place the fill height above the tank floor and inside the domain.")
    if planar_fill and liquid_high[2] > settings.fill_height:
        raise SceneError("liquid", "Place each source box below the planar fill height.")
    version = "FLUIDITY-BLENDER-2"
    if sources:
        version = "FLUIDITY-BLENDER-4" if viscosity_authoring else "FLUIDITY-BLENDER-3"
    if snow_platen:
        version = "FLUIDITY-BLENDER-5"
    lines = [
        version,
        f"{int(preview)} {cell!r}",
        f"{gx!r} {gz!r} {-gy!r}",
        f"{settings.density!r} {settings.surface_tension!r} {settings.friction!r}",
        f"{settings.energy_blend!r} {settings.maximum_step!r} {settings.pressure_iterations}",
        f"{vx!r} {vz!r} {-vy!r}",
    ]
    if settings.boundary_mode == "TANK":
        lines.extend(
            ["3", f"{domain_low[0]!r} {domain_high[0]!r} {domain_low[2]!r} {-domain_high[1]!r} {-domain_low[1]!r} 0"]
        )
    elif settings.boundary_mode == "FREE":
        lines.append("0")
    else:
        wx, wy, wz = wall_velocity
        if snow_platen:
            if abs(wx) > 1e-6 or abs(wy) > 1e-6 or wz >= 0:
                raise SceneError("collider", "Animate the overhead platen down along Blender Z.")
            if max(liquid_high[2], snow_bounds[1][2]) > collider_low[2] or snow_bounds[0][2] < 0:
                raise SceneError("collider", "Place water and the snow mound below the platen and snow above Z = 0.")
            lines.extend(["1", f"0 -1 0 {-collider_low[2]!r} 0 0 {wz!r} 0"])
        elif settings.boundary_mode == "PLANE":
            if liquid_low[2] < collider_high[2] - 1e-6:
                raise SceneError("collider", "Place liquid above the plane collider.")
            lines.extend(["1", f"0 1 0 {collider_high[2]!r} 0 {wx!r} {wz!r} {-wy!r}"])
        else:
            if liquid_low[0] <= collider_high[0] and liquid_high[0] >= collider_low[0]:
                raise SceneError("collider", "Move the slab clear of the initial liquid.")
            center = (collider_low[0] + collider_high[0]) / 2
            half_width = (collider_high[0] - collider_low[0]) / 2
            lines.extend(["2", f"1 0 0 {center!r} {half_width!r} {wx!r} {wz!r} {-wy!r}"])
    if sources:
        if viscosity_authoring:
            lines.append(f"1 {settings.fill_height!r}" if planar_fill else "0")
        lines.extend(extended_liquid_rows(
            sources, liquid_bounds, cell, (vx, vz, -vy), settings.phase_tension, viscosity_authoring
        ))
        if snow_platen:
            lines.extend(snow_seed_rows(settings, snow_bounds, cell))
        return "\n".join(lines) + "\n"
    lines.append(str(count))
    for z, y, x in product(ranges[2], ranges[1], ranges[0]):
        lines.append(" ".join(repr((coordinate + 0.5) * cell) for coordinate in (x, y, z)))
    return "\n".join(lines) + "\n"
