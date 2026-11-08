"""Experimental transfer stencils for an aligned 2:1 interface on a 2D MAC lattice.

Face-normal anchors use centered cubic splines at integer lattice coordinates.
Transverse anchors use centered quadratic splines at half-integer coordinates.
The fine half-plane starts at zero on the interface axis in coordinates relative
to the interface origin. Active fine splines have their full closed support in
that half-plane. Coarse splines with support extending outside it retain their
coarse anchors and only the dyadic child terms that extend outside it.

Weights are dimensionless. Gradients differentiate with respect to the particle
world position and have units of inverse length. The stencil describes numerical
support for one face component. Face ownership, masked support, other interface
topologies, more levels, and 3D junctions are outside this experiment.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

# The experiment only needs a local patch. This keeps lattice indices and their
# dyadic shifts exactly representable at both supported precisions.
_MAX_NORMALIZED_COORDINATE = 64


@dataclass(frozen=True, slots=True)
class BasisStencil:
    """Positive-weight rows ordered by level, then ascending world x and y.

    For n rows, anchors and physical gradients have shape (n, 2) in (x, y)
    order. Weights and levels have shape (n,). The numeric arrays use the
    requested floating dtype, and integer levels are 0 for coarse and 1 for fine.
    All arrays are read-only.
    """

    anchor_world_positions: np.ndarray
    weights: np.ndarray
    physical_gradients: np.ndarray
    levels: np.ndarray


def _validate_axis(axis, name):
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral) or axis not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")


def _real_vector(value):
    vector = np.asarray(value)
    if vector.shape != (2,) or vector.dtype.kind not in "iuf":
        raise ValueError("world positions and origins must be real 2D vectors")
    return vector


def _normalize_query(particle_world_position, origin, cell_width, dtype, finest_cell_ratio):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    if isinstance(cell_width, (bool, np.bool_)) or not isinstance(cell_width, Real):
        raise TypeError("cell width must be a real scalar")

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        position = _real_vector(particle_world_position).astype(dtype)
        origin = _real_vector(origin).astype(dtype)
        cell_width = dtype.type(cell_width)
        finest_cell_width = cell_width * finest_cell_ratio
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(origin)):
            raise ValueError("world position and origin must be finite in the requested dtype")
        if not np.isfinite(cell_width) or cell_width <= 0 or not np.isfinite(1 / finest_cell_width):
            raise ValueError("cell width and inverse finest cell width must be finite and positive")

        normalized_position = (position - origin) / cell_width
        if not np.all(np.abs(normalized_position) <= _MAX_NORMALIZED_COORDINATE):
            raise ValueError(
                f"normalized position must be in [-{_MAX_NORMALIZED_COORDINATE}, {_MAX_NORMALIZED_COORDINATE}]"
            )

    return position, normalized_position, origin, cell_width


def _support_radius(is_face_normal):
    return 2 if is_face_normal else 1.5


def _axis_anchors(normalized_coordinate, is_face_normal):
    first_index = int(np.floor(normalized_coordinate)) - 1
    if is_face_normal:
        return np.arange(first_index, first_index + 4, dtype=normalized_coordinate.dtype)
    return np.arange(first_index, first_index + 3, dtype=normalized_coordinate.dtype) + 0.5


def _centered_cardinal(displacement, is_face_normal):
    """Evaluate a cardinal spline and its derivative in lattice units."""
    absolute_displacement = np.abs(displacement)
    distance_to_support_edge = np.maximum(_support_radius(is_face_normal) - absolute_displacement, 0)
    if is_face_normal:
        interior = absolute_displacement < 1
        interior_weight = (4 - absolute_displacement**2 * (6 - 3 * absolute_displacement)) / 6
        weight = np.where(interior, interior_weight, distance_to_support_edge**3 / 6)
        interior_gradient = absolute_displacement * (1.5 * absolute_displacement - 2)
        gradient = np.where(interior, interior_gradient, -0.5 * distance_to_support_edge**2)
        return weight, gradient * np.sign(displacement)

    interior = absolute_displacement < 0.5
    weight = np.where(interior, 0.75 - displacement**2, 0.5 * distance_to_support_edge**2)
    gradient = np.where(interior, -2 * displacement, -distance_to_support_edge * np.sign(displacement))
    return weight, gradient


def _uniform_axis(normalized_coordinate, is_face_normal):
    anchors = _axis_anchors(normalized_coordinate, is_face_normal)
    weights, gradients = _centered_cardinal(normalized_coordinate - anchors, is_face_normal)
    return anchors, weights, gradients


def _truncated_coarse_axis(normalized_coordinate, is_face_normal):
    radius = _support_radius(is_face_normal)
    coarse_anchors = _axis_anchors(normalized_coordinate, is_face_normal)
    coarse_anchors = coarse_anchors[coarse_anchors < radius]
    dtype = normalized_coordinate.dtype
    if is_face_normal:
        child_offsets = np.array([-2, -1, 0, 1, 2], dtype=dtype)
        refinement_mask = np.array([1, 4, 6, 4, 1], dtype=dtype) / 8
    else:
        child_offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=dtype)
        refinement_mask = np.array([1, 3, 3, 1], dtype=dtype) / 4

    # Child centers are in fine lattice units. Only children whose lower support
    # endpoint is negative remain. Keep their positive contributions directly.
    child_anchors = 2 * coarse_anchors[:, None] + child_offsets
    retained_children = child_anchors < radius
    child_weights, child_gradients = _centered_cardinal(2 * normalized_coordinate - child_anchors, is_face_normal)
    weights = (child_weights * retained_children) @ refinement_mask
    coarse_unit_gradients = 2 * ((child_gradients * retained_children) @ refinement_mask)
    return coarse_anchors, weights, coarse_unit_gradients


def _active_fine_axis(normalized_coordinate, is_face_normal):
    anchors, weights, gradients = _uniform_axis(normalized_coordinate, is_face_normal)
    contained = anchors >= _support_radius(is_face_normal)
    return anchors[contained], weights[contained], gradients[contained]


def _tensor_stencil(axis_stencils, position, normalized_position, origin, cell_width, level):
    (x_anchors, x_weights, x_gradients), (y_anchors, y_weights, y_gradients) = axis_stencils
    weights = (x_weights[:, None] * y_weights).ravel()
    positive = weights > 0
    normalized_anchors = np.column_stack((np.repeat(x_anchors, y_anchors.size), np.tile(y_anchors, x_anchors.size)))
    normalized_gradients = np.column_stack(
        ((x_gradients[:, None] * y_weights).ravel(), (x_weights[:, None] * y_gradients).ravel())
    )
    normalized_anchors = normalized_anchors[positive]
    with np.errstate(over="ignore", invalid="ignore"):
        anchor_world_positions = origin + cell_width * normalized_anchors
        round_trip_offsets = (anchor_world_positions - position) / cell_width
        offset_error = np.abs(round_trip_offsets - (normalized_anchors - normalized_position))
    if not np.all(np.isfinite(anchor_world_positions)):
        raise ValueError("world anchor positions must be finite in the requested dtype")

    # Compare geometry in level cell units. A two-epsilon budget keeps
    # coordinate rounding small enough for the FP32 moment measurements.
    if not np.all(offset_error <= 2 * np.finfo(position.dtype).eps):
        raise ValueError("world anchor offsets lose lattice precision in the requested dtype")

    physical_gradients = normalized_gradients[positive] / cell_width
    levels = np.full(anchor_world_positions.shape[0], level, dtype=np.int8)
    return anchor_world_positions, weights[positive], physical_gradients, levels


def _immutable_stencil(*arrays):
    # Bytes backing prevents callers from re-enabling writes through setflags.
    return BasisStencil(
        *(np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape) for values in arrays)
    )


def evaluate_coarse_fine_basis(
    particle_world_position, *, interface_origin, coarse_cell_width, face_normal_axis, interface_axis, dtype
):
    """Evaluate one MAC component at a 2D world position on the two-level patch.

    The fine half-plane has world coordinate >= interface_origin[interface_axis].
    The face-normal and interface axes are 0 (x) or 1 (y). The finite input
    vectors are in world length units, with the particle position in
    [-64, 64] coarse cells of the origin along each axis. Coordinate subtraction
    uses the selected float32 or float64 dtype. With a positive coarse cell width,
    each returned world anchor must be finite, and its offset from the particle
    must agree with the lattice offset to within two dtype epsilons in that
    anchor's level cell units. Invalid inputs raise ValueError or TypeError.
    """
    _validate_axis(face_normal_axis, "face_normal_axis")
    _validate_axis(interface_axis, "interface_axis")
    position, normalized_position, origin, coarse_cell_width = _normalize_query(
        particle_world_position, interface_origin, coarse_cell_width, dtype, finest_cell_ratio=0.5
    )

    # The tensor child mask factors along the two axes. Half-plane containment
    # depends only on the interface axis, so the intact sum along the other axis
    # is exactly its unrefined cardinal spline by the dyadic identity.
    coarse_axes = [
        _truncated_coarse_axis(coordinate, axis == face_normal_axis)
        if axis == interface_axis
        else _uniform_axis(coordinate, axis == face_normal_axis)
        for axis, coordinate in enumerate(normalized_position)
    ]
    fine_normalized_position = 2 * normalized_position
    fine_axes = [
        _active_fine_axis(coordinate, axis == face_normal_axis)
        if axis == interface_axis
        else _uniform_axis(coordinate, axis == face_normal_axis)
        for axis, coordinate in enumerate(fine_normalized_position)
    ]
    coarse = _tensor_stencil(coarse_axes, position, normalized_position, origin, coarse_cell_width, level=0)
    fine = _tensor_stencil(fine_axes, position, fine_normalized_position, origin, coarse_cell_width / 2, level=1)
    return _immutable_stencil(*(np.concatenate(parts) for parts in zip(coarse, fine, strict=True)))


def evaluate_uniform_cubic_normal_basis(
    particle_world_position, *, lattice_origin, cell_width, face_normal_axis, dtype
):
    """Evaluate the same-degree uniform comparator with all rows on level 0.

    The face-normal axis is 0 (x) or 1 (y). The finite input vectors are in world
    length units, with the particle position in [-64, 64] cells of lattice_origin
    along each axis. Coordinate subtraction uses the selected float32 or float64
    dtype. With a positive cell width, each returned world anchor must be finite,
    and its offset from the particle must agree with the lattice offset to within
    two dtype epsilons in cell units. Invalid inputs raise ValueError or TypeError.
    """
    _validate_axis(face_normal_axis, "face_normal_axis")
    position, normalized_position, origin, cell_width = _normalize_query(
        particle_world_position, lattice_origin, cell_width, dtype, finest_cell_ratio=1
    )
    axes = [_uniform_axis(coordinate, axis == face_normal_axis) for axis, coordinate in enumerate(normalized_position)]
    return _immutable_stencil(*_tensor_stencil(axes, position, normalized_position, origin, cell_width, level=0))
