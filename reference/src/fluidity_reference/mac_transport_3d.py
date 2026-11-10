"""Float64 transport reconstruction on a complete uniform 3D MAC lattice."""

from math import copysign, floor
from numbers import Real

import numpy as np

from .basis import quadratic_stencil


def _cubic_normal_basis(delta):
    """Evaluate centered B3 and its lattice derivative for |delta| <= 2."""
    distance = abs(delta)
    if distance < 1.0:
        weight = 2.0 / 3.0 + delta**2 * (0.5 * distance - 1.0)
        derivative = delta * (1.5 * distance - 2.0)
        return weight, derivative
    remaining = 2.0 - distance
    return remaining**3 / 6.0, -copysign(0.5 * remaining**2, delta)


def _quadratic_row(position, h, axis, face_value):
    offset = [0.5, 0.5, 0.5]
    offset[axis] = 0.0
    nodes, weights, gradients = quadratic_stencil(position, h, offset)
    samples = np.array([face_value(axis, tuple(map(int, node))) for node in nodes], dtype=np.float64)
    return samples @ weights, samples @ gradients


def _mixed_row(position, lattice_position, h, axis, face_value):
    first, second = (axis + 1) % 3, (axis + 2) % 3
    nodes, weights, gradients = quadratic_stencil(position[[first, second]], h, [0.5, 0.5])
    velocity = 0.0
    gradient = np.zeros(3, dtype=np.float64)
    normal_base = floor(lattice_position[axis])
    for normal_index in range(normal_base - 1, normal_base + 3):
        normal_weight, normal_derivative = _cubic_normal_basis(lattice_position[axis] - normal_index)
        normal_gradient = normal_derivative / h
        for node, transverse_weight, transverse_gradient in zip(nodes, weights, gradients):
            index = [0, 0, 0]
            index[axis] = normal_index
            index[first], index[second] = map(int, node)
            sample = float(face_value(axis, tuple(index)))
            velocity += sample * normal_weight * transverse_weight
            gradient[axis] += sample * normal_gradient * transverse_weight
            gradient[first] += sample * normal_weight * transverse_gradient[0]
            gradient[second] += sample * normal_weight * transverse_gradient[1]
    return velocity, gradient


def sample_mac_transport(position, h, face_value, *, reconstruction):
    """Return float64 velocity (m/s) and its component-row gradient (1/s).

    Supply a finite three-vector position in metres relative to the grid origin
    and finite positive spacing h in metres. The lattice coordinates and their
    integer and half-integer stencil anchors must be representable in float64.
    The callback ``face_value(axis, index)`` supplies a finite velocity in m/s
    for axis 0, 1 or 2 and a tuple of signed (x, y, z) integer indices. Its
    physical anchor is h times the index with a half shift on transverse axes.
    A cell with index i spans [i*h, (i+1)*h) along each axis.

    ``quadratic`` uses B2 in all directions. ``mixed`` uses normal B3 and
    transverse B2. ``rt0`` interpolates between the containing cell's two
    normal faces for each component, with a constant transverse extension
    inside that cell. At a cell boundary RT0 uses the positive-side cell on
    each tied axis and returns that cell's gradient.

    Supply all stencil donors at the same time on an unmasked uniform grid,
    including selected zero-weight donors at knots. A missing donor propagates
    the callback's KeyError. Invalid position, spacing or reconstruction raises
    ValueError. The gradient differentiates the interpolated field rather than
    fitting an affine momentum field.
    """
    if reconstruction not in ("quadratic", "rt0", "mixed"):
        raise ValueError("reconstruction must be quadratic, rt0 or mixed")
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, Real):
        raise ValueError("spacing must be finite and positive")
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("spacing must be finite and positive")
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position must be a finite three-vector")

    velocity = np.empty(3, dtype=np.float64)
    gradient = np.zeros((3, 3), dtype=np.float64)
    if reconstruction == "quadratic":
        for axis in range(3):
            velocity[axis], gradient[axis] = _quadratic_row(position, h, axis, face_value)
        return velocity, gradient

    lattice_position = position / h
    if reconstruction == "mixed":
        for axis in range(3):
            velocity[axis], gradient[axis] = _mixed_row(position, lattice_position, h, axis, face_value)
        return velocity, gradient

    cell = tuple(map(floor, lattice_position))
    for axis in range(3):
        upper_face = list(cell)
        upper_face[axis] += 1
        lower_value = float(face_value(axis, cell))
        upper_value = float(face_value(axis, tuple(upper_face)))
        fraction = lattice_position[axis] - cell[axis]
        normal_difference = upper_value - lower_value
        velocity[axis] = lower_value + fraction * normal_difference
        gradient[axis, axis] = normal_difference / h
    return velocity, gradient
