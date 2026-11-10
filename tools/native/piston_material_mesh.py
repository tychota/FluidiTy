"""Piston geometry from a transported, piecewise-affine material mesh."""

import itertools
import math

H = 0.0625
SHAPE = (16, 8, 8)
LOW = (-0.5, 0.0, -0.25)
HIGH_X = 0.5
HIGH_Z = 0.25
REFERENCE_Y = 0.5
REFERENCE_TETRA_VOLUME = H ** 3 / 6
NEAR_DEGENERATE_RELATIVE_J = 1e-10
INJECTIVITY_MARGIN = 1e-12

CELLS = tuple(itertools.product(*(range(n) for n in SHAPE)))
VERTICES = tuple(itertools.product(*(range(n + 1) for n in SHAPE)))
TETRA_AXES = tuple(itertools.permutations(range(3)))


def _stencil(vertex, extent):
    if vertex == 0:
        return ((0, 1.5), (1, -0.5))
    if vertex == extent:
        return ((extent - 2, -0.5), (extent - 1, 1.5))
    return ((vertex - 1, 0.5), (vertex, 0.5))


def reconstruct(centers):
    """Map reference-lattice center indices to transported vertex positions in m."""
    if centers.keys() != set(CELLS):
        raise ValueError("centers must cover the entire reference lattice")
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point)
           for point in centers.values()):
        raise ValueError("each center must have three finite coordinates")
    vertices = {}
    for vertex in VERTICES:
        stencils = (_stencil(index, extent) for index, extent in zip(vertex, SHAPE))
        weighted_points = [
            (math.prod(weight for _, weight in stencil), centers[tuple(index for index, _ in stencil)])
            for stencil in itertools.product(*stencils)
        ]
        vertices[vertex] = tuple(
            math.fsum(weight * point[axis] for weight, point in weighted_points)
            for axis in range(3)
        )
    return vertices


def signed_volume(tetra):
    """Compute the oriented volume in m³ for four ordered points."""
    p, q, r, s = tetra
    a, b, c = (tuple(u - v for u, v in zip(point, p)) for point in (q, r, s))
    return math.fsum((
        a[0] * (b[1] * c[2] - b[2] * c[1]),
        a[1] * (b[2] * c[0] - b[0] * c[2]),
        a[2] * (b[0] * c[1] - b[1] * c[0]),
    )) / 6.0


def _edge_on_plane(positive, negative, y):
    fraction = (positive[1] - y) / (positive[1] - negative[1])
    return (
        positive[0] + fraction * (negative[0] - positive[0]),
        y,
        positive[2] + fraction * (negative[2] - positive[2]),
    )


def _one_vertex_cap(apex, other, y):
    intersections = [_edge_on_plane(apex, point, y) for point in other]
    return abs(signed_volume((apex, *intersections)))


def _two_vertex_prism(inside, outside, y):
    a, b = inside
    c, d = outside
    p, q = _edge_on_plane(a, c, y), _edge_on_plane(a, d, y)
    r, s = _edge_on_plane(b, c, y), _edge_on_plane(b, d, y)
    # Triangular end faces (a, p, q) and (b, r, s) share three rails.
    return math.fsum(abs(signed_volume(part)) for part in (
        (a, b, p, q),
        (b, p, q, s),
        (b, p, r, s),
    ))


def clipped_signed_volumes(tetra, y, volume=None):
    """Return oriented volumes (below, above) split by the world plane y in m."""
    volume = signed_volume(tetra) if volume is None else volume
    above = [point for point in tetra if point[1] > y]
    below = [point for point in tetra if point[1] <= y]
    if not above:
        return volume, 0.0
    if not below:
        return 0.0, volume
    if len(above) == 1:
        cap = math.copysign(_one_vertex_cap(above[0], below, y), volume)
        return volume - cap, cap
    if len(below) == 1:
        cap = math.copysign(_one_vertex_cap(below[0], above, y), volume)
        return cap, volume - cap
    below_volume = math.copysign(_two_vertex_prism(below, above, y), volume)
    above_volume = math.copysign(_two_vertex_prism(above, below, y), volume)
    return below_volume, above_volume


def _clip_prism(a, b, c, d, e, f):
    # First triangle (a, b, c), second (d, e, f), rails a-d, b-e, c-f.
    return ((a, b, c, d), (b, c, d, f), (b, d, e, f))


def _clip_tetra(tetra, axis, plane, inward):
    # The returned pieces partition the kept convex region by geometric volume.
    inside = [point for point in tetra if inward * (point[axis] - plane) >= 0]
    if not inside:
        return ()
    if len(inside) == 4:
        return (tetra,)
    outside = [point for point in tetra if inward * (point[axis] - plane) < 0]

    def intersection(p, q):
        fraction = (plane - p[axis]) / (q[axis] - p[axis])
        return tuple(
            plane if component == axis else left + fraction * (right - left)
            for component, (left, right) in enumerate(zip(p, q))
        )

    if len(inside) == 1:
        a, = inside
        return ((a, *(intersection(a, point) for point in outside)),)
    if len(inside) == 2:
        a, b = inside
        c, d = outside
        return _clip_prism(a, intersection(a, c), intersection(a, d),
                           b, intersection(b, c), intersection(b, d))
    a, b, c = inside
    d, = outside
    return _clip_prism(a, b, c, intersection(a, d), intersection(b, d), intersection(c, d))


def _tank_intersection(tetra, clip_planes):
    fragments = (tetra,)
    for axis, plane, inward in clip_planes:
        fragments = tuple(
            piece for fragment in fragments for piece in _clip_tetra(fragment, axis, plane, inward)
        )
        if not fragments:
            break
    return fragments


def _tetra_chain(vertices, cell, axes):
    index = list(cell)
    points = [vertices[cell]]
    for axis in axes:
        index[axis] += 1
        points.append(vertices[tuple(index)])
    return points


def _displacement_gradient_norm(chain, axes):
    # Consecutive reference edges advance by H in exactly one axis each.
    return math.sqrt(math.fsum(
        ((chain[i + 1][component] - chain[i][component]) / H - float(component == axis)) ** 2
        for i, axis in enumerate(axes)
        for component in range(3)
    ))


def _boundary_offsets(vertices, axis, high, plane):
    boundary_index = SHAPE[axis] if high else 0
    sign = -1 if high else 1
    all_offsets = {index: sign * (point[axis] - plane) for index, point in vertices.items()}
    face_offsets = [offset for index, offset in all_offsets.items() if index[axis] == boundary_index]
    return {
        "wall_plane_m": plane,
        "face_vertex_count": len(face_offsets),
        "positive_gap_vertex_count": sum(offset > 0 for offset in face_offsets),
        "outside_vertex_count": sum(offset < 0 for offset in face_offsets),
        "on_plane_vertex_count": sum(offset == 0 for offset in face_offsets),
        "min_offset_m": min(face_offsets),
        "max_offset_m": max(face_offsets),
        "mean_offset_m": math.fsum(face_offsets) / len(face_offsets),
        "rms_offset_m": math.sqrt(math.fsum(offset ** 2 for offset in face_offsets) / len(face_offsets)),
        "max_absolute_offset_m": max(abs(offset) for offset in face_offsets),
        "min_offset_all_mesh_vertices_m": min(all_offsets.values()),
    }


def measure(centers, piston_x):
    """Measure the unmodified transported mesh and compare its faces to the wall planes."""
    vertices = reconstruct(centers)
    volumes, below_volumes, above_volumes, gradient_norms = [], [], [], []
    split_errors, tank_below_volumes, tank_above_volumes = [], [], []
    clip_planes = (
        (0, piston_x, 1),
        (0, HIGH_X, -1),
        (1, LOW[1], 1),
        (2, LOW[2], 1),
        (2, HIGH_Z, -1),
    )
    for cell in CELLS:
        for axes in TETRA_AXES:
            chain = _tetra_chain(vertices, cell, axes)
            gradient_norms.append(_displacement_gradient_norm(chain, axes))
            odd_permutation = sum(a > b for a, b in itertools.combinations(axes, 2)) % 2
            if odd_permutation:
                chain[1], chain[2] = chain[2], chain[1]
            volume = signed_volume(chain)
            below, above = clipped_signed_volumes(chain, REFERENCE_Y, volume)
            volumes.append(volume)
            below_volumes.append(below)
            above_volumes.append(above)
            split_errors.append(abs(math.fsum((below, above, -volume))))
            if volume != 0.0:
                for fragment in _tank_intersection(chain, clip_planes):
                    # The fragment order can differ from the parent order.
                    # Count the kept region with its parent's material orientation.
                    fragment_volume = math.copysign(abs(signed_volume(fragment)), volume)
                    tank_below, tank_above = clipped_signed_volumes(fragment, REFERENCE_Y, fragment_volume)
                    tank_below_volumes.append(tank_below)
                    tank_above_volumes.append(tank_above)

    walls = {
        "piston": _boundary_offsets(vertices, 0, False, piston_x),
        "right": _boundary_offsets(vertices, 0, True, HIGH_X),
        "floor": _boundary_offsets(vertices, 1, False, LOW[1]),
        "low_z": _boundary_offsets(vertices, 2, False, LOW[2]),
        "high_z": _boundary_offsets(vertices, 2, True, HIGH_Z),
    }
    max_gradient = max(gradient_norms)
    min_relative_j = min(volumes) / REFERENCE_TETRA_VOLUME
    below_capacity = (HIGH_X - piston_x) * (REFERENCE_Y - LOW[1]) * (HIGH_Z - LOW[2])
    unsigned_below = math.fsum(abs(volume) for volume in below_volumes)
    tank_below = math.fsum(tank_below_volumes)
    tank_above = math.fsum(tank_above_volumes)
    injective = max_gradient + INJECTIVITY_MARGIN < 1.0
    top_y = [point[1] for index, point in vertices.items() if index[1] == SHAPE[1]]
    return {
        "tetra_count": len(volumes),
        "signed_total_m3": math.fsum(volumes),
        "unsigned_total_m3": math.fsum(abs(volume) for volume in volumes),
        "signed_below_m3": math.fsum(below_volumes),
        "signed_above_m3": math.fsum(above_volumes),
        "unsigned_below_m3": unsigned_below,
        "tank_below_plane_capacity_m3": below_capacity,
        "tank_below_plane_void_lower_bound_m3": max(0.0, below_capacity - unsigned_below),
        "tank_clipped_below_m3": tank_below,
        "tank_clipped_above_m3": tank_above,
        "tank_clipped_below_plane_void_m3": max(0.0, below_capacity - tank_below) if injective else None,
        "split_absolute_volume_residual_sum_m3": math.fsum(split_errors),
        "inversion_count": sum(volume < 0 for volume in volumes),
        "near_degenerate_count": sum(
            abs(volume) <= NEAR_DEGENERATE_RELATIVE_J * REFERENCE_TETRA_VOLUME for volume in volumes
        ),
        "min_relative_tetra_determinant": min_relative_j,
        "max_relative_tetra_determinant": max(volumes) / REFERENCE_TETRA_VOLUME,
        "max_displacement_gradient_frobenius": max_gradient,
        "injectivity_certified": injective,
        "top_boundary_y_min_m": min(top_y),
        "top_boundary_y_max_m": max(top_y),
        "wall_fit": walls,
    }


FLOOR_AND_SIDES = ((1, LOW[1], 1), (2, LOW[2], 1), (2, HIGH_Z, -1))
BELOW_AREA = (REFERENCE_Y - LOW[1]) * (HIGH_Z - LOW[2])
EDGES = tuple(itertools.combinations(range(4), 2))
IDS = tuple(
    (ix, iy, iz)
    for iz in range(SHAPE[2]) for iy in range(SHAPE[1]) for ix in range(SHAPE[0])
)


def snapshot(particles, position_key):
    """Recover the complete reference-index map from stable particle IDs."""
    positions = {}
    for particle in particles:
        identifier = particle["id"]
        if type(identifier) is not int or identifier in positions or not 1 <= identifier <= len(IDS):
            raise ValueError("duplicate or unexpected material IDs")
        positions[identifier] = tuple(particle[position_key])
    if len(positions) != len(IDS):
        raise ValueError("missing material IDs")
    return {index: positions[identifier] for identifier, index in enumerate(IDS, 1)}


def _clip_polygon(polygon, axis, plane, inward):
    kept = []
    if not polygon:
        return kept
    for left, right in zip(polygon[-1:] + polygon[:-1], polygon, strict=True):
        left_in = inward * (left[axis] - plane) >= 0
        right_in = inward * (right[axis] - plane) >= 0
        if left_in != right_in:
            fraction = (plane - left[axis]) / (right[axis] - left[axis])
            point = [a + fraction * (b - a) for a, b in zip(left, right)]
            point[axis] = plane
            kept.append(tuple(point))
        if right_in:
            kept.append(right)
    return kept


def _polygon_area(polygon):
    return abs(math.fsum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(polygon[-1:] + polygon[:-1], polygon, strict=True)
    )) / 2


def _wet_section(tetra, x):
    """Intersect a tetrahedron with x, taking the inward trace at coplanar faces."""
    distances = [point[0] - x for point in tetra]
    if min(distances) > 0 or max(distances) <= 0:
        return 0.0, 0.0
    polygon = [point[1:] for point, distance in zip(tetra, distances, strict=True)
               if distance == 0]
    for i, j in EDGES:
        if distances[i] * distances[j] < 0:
            fraction = distances[i] / (distances[i] - distances[j])
            polygon.append(tuple(a + fraction * (b - a) for a, b in zip(tetra[i][1:], tetra[j][1:])))
    if len(polygon) < 3:
        return 0.0, 0.0
    cy, cz = (math.fsum(p[i] for p in polygon) / len(polygon) for i in (0, 1))
    polygon.sort(key=lambda p: math.atan2(p[1] - cz, p[0] - cy))
    for axis, plane, inward in FLOOR_AND_SIDES:
        polygon = _clip_polygon(polygon, axis - 1, plane, inward)
    total = _polygon_area(polygon)
    below = _polygon_area(_clip_polygon(polygon, 0, REFERENCE_Y, -1))
    return below, total - below


def sweep(positions, x0, x1=None, *, offsets=()):
    """Measure the inward wall section and a frozen start-of-step stroke strip.

    The strip is clipped to the stationary floor and side walls. Additional
    nonnegative inward section offsets probe sensitivity without moving any
    vertices. An uncertified material map cannot supply a wet occupancy.
    """
    if not LOW[0] <= x0 < HIGH_X or (x1 is not None and not x0 < x1 <= HIGH_X):
        raise ValueError("invalid piston stroke")
    if any(not 0 <= offset < HIGH_X - x0 for offset in offsets):
        raise ValueError("invalid inward section offset")
    vertices = reconstruct(positions)
    section_parts = [[] for _ in range(len(offsets) + 1)]
    section_planes = (x0, *(x0 + offset for offset in offsets))
    below_volumes, above_volumes = [], []
    max_gradient = 0.0
    strip_planes = ((0, x0, 1), (0, x1, -1), *FLOOR_AND_SIDES)
    for cell in CELLS:
        for axes in TETRA_AXES:
            tetra = _tetra_chain(vertices, cell, axes)
            max_gradient = max(max_gradient, _displacement_gradient_norm(tetra, axes))
            minimum_x = min(point[0] for point in tetra)
            maximum_x = max(point[0] for point in tetra)
            for plane, parts in zip(section_planes, section_parts, strict=True):
                if minimum_x <= plane < maximum_x:
                    parts.append(_wet_section(tetra, plane))
            if x1 is not None and minimum_x < x1 and maximum_x > x0:
                for fragment in _tank_intersection(tetra, strip_planes):
                    below, above = clipped_signed_volumes(fragment, REFERENCE_Y, abs(signed_volume(fragment)))
                    below_volumes.append(below)
                    above_volumes.append(above)
    if max_gradient + INJECTIVITY_MARGIN >= 1.0:
        raise ValueError("material interpolant lacks an injectivity certificate")
    areas = tuple(tuple(math.fsum(part[i] for part in parts) for i in (0, 1)) for parts in section_parts)
    return {
        "area_below_m2": areas[0][0],
        "area_above_m2": areas[0][1],
        "strip_below_m3": math.fsum(below_volumes),
        "strip_above_m3": math.fsum(above_volumes),
        "offset_area_below_above_m2": areas[1:],
        "max_displacement_gradient_frobenius": max_gradient,
    }
