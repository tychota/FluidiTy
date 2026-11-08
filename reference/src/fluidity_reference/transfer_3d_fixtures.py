"""Exact manufactured transfer data on a staggered lattice across a block corner."""

from fractions import Fraction
from itertools import product
from math import prod

_HALF = Fraction(1, 2)
_RADIUS = Fraction(3, 2)


def _centered_quadratic(particle_offset_from_face_in_cells):
    """Return the centered weight and derivative in normalized coordinates."""
    distance = abs(particle_offset_from_face_in_cells)
    if distance <= _HALF:
        return Fraction(3, 4) - distance**2, -2 * particle_offset_from_face_in_cells
    if distance < _RADIUS:
        distance_to_support_edge = _RADIUS - distance
        derivative = distance_to_support_edge if particle_offset_from_face_in_cells < 0 else -distance_to_support_edge
        return _HALF * distance_to_support_edge**2, derivative
    return Fraction(0), Fraction(0)


def uniform_mac_transfer_3d_fixture():
    """Evaluate the per-face affine deposit and basis derivatives exactly before export."""
    cell_width = Fraction(1)
    world_origin = (Fraction(0),) * 3
    particle_world_position = (Fraction(15, 4), Fraction(-1, 4), Fraction(15, 4))
    particle_mass = 2
    particle_velocity = (1, -2, 4)
    particle_velocity_slopes = ((2, 4, -1), (-5, 1, 5), (7, -2, 3))

    fixture_arrays = {
        "cell_width": [float(cell_width)],
        "world_origin": [float(value) for value in world_origin],
        "position": [float(value) for value in particle_world_position],
        "particle_mass": [float(particle_mass)],
        "velocity": [float(value) for value in particle_velocity],
        "affine_velocity_fit": [[float(value) for value in row] for row in particle_velocity_slopes],
        "active_blocks": list(product((0, 1), (-1, 0), (0, 1))),
        "face_indices": [],
        "face_displacements": [],
        "weights": [],
        "weight_gradients": [],
        "expected_mass": [],
        "expected_pure_momentum": [],
    }
    for axis, particle_velocity_slope_row in enumerate(particle_velocity_slopes):
        # Scan the signed lattice of the eight blocks and retain anchors in the
        # open support. This case has three positive weights on every axis.
        for lattice_index in product(range(8), range(-4, 4), range(8)):
            face_world_anchor = [
                world_origin[coordinate] + cell_width * (index + (0 if coordinate == axis else _HALF))
                for coordinate, index in enumerate(lattice_index)
            ]
            face_offset_from_particle = [
                face_coordinate - particle_coordinate
                for face_coordinate, particle_coordinate in zip(face_world_anchor, particle_world_position)
            ]
            axis_weights_and_derivatives = [
                _centered_quadratic(-axis_offset / cell_width)
                for axis_offset in face_offset_from_particle
            ]
            weight = prod(axis_weight for axis_weight, _ in axis_weights_and_derivatives)
            if weight == 0:
                continue

            weight_gradient = [
                derivative * prod(
                    other_weight for coordinate, (other_weight, _) in enumerate(axis_weights_and_derivatives)
                    if coordinate != derivative_coordinate
                ) / cell_width
                for derivative_coordinate, (_, derivative) in enumerate(axis_weights_and_derivatives)
            ]
            mass_contribution = weight * particle_mass
            # The weighted face offsets sum to zero on each staggered lattice.
            # This lets the particle's affine velocity term preserve row momentum.
            velocity_at_face = particle_velocity[axis] + sum(
                slope * axis_offset
                for slope, axis_offset in zip(particle_velocity_slope_row, face_offset_from_particle)
            )
            pure_momentum_contribution = mass_contribution * velocity_at_face

            fixture_arrays["face_indices"].append((axis, *lattice_index))
            fixture_arrays["face_displacements"].append([float(axis_offset) for axis_offset in face_offset_from_particle])
            fixture_arrays["weights"].append(float(weight))
            fixture_arrays["weight_gradients"].append([float(derivative) for derivative in weight_gradient])
            fixture_arrays["expected_mass"].append(float(mass_contribution))
            fixture_arrays["expected_pure_momentum"].append(float(pure_momentum_contribution))

    metadata = {
        "unit_system": "SI",
        "units": {
            "cell_width": "m",
            "world_origin": "m",
            "position": "m",
            "particle_mass": "kg",
            "velocity": "m/s",
            "affine_velocity_fit": "s^-1",
            "active_blocks": "1",
            "face_indices": "1",
            "face_displacements": "m",
            "weights": "1",
            "weight_gradients": "m^-1",
            "expected_mass": "kg",
            "expected_pure_momentum": "kg*m/s",
        },
        "provenance": "Exact rational tensor products of centered piecewise quadratics at physical MAC face anchors.",
        "default_policy": "Complete support on the eight listed blocks. Face indices are (normal axis, x, y, z).",
        "tolerances": {"atol": 2e-12, "rtol": 2e-12},
    }
    return fixture_arrays, metadata
