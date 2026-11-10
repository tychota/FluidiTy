"""Exact-property fixtures for transport on complete uniform MAC support."""

from itertools import product
from math import inf, nextafter

import numpy as np
import pytest

from fluidity_reference.mac_transport_3d import sample_mac_transport


SPACINGS = (1 / 8, 1 / 16, 1 / 32, 1 / 64)
VALUE_ATOL = 2e-13
GRADIENT_ATOL = 2e-11
LATTICE_POSITIONS = np.array(
    [
        (-2.0, -1.0, 1.0),
        (-1.5, 0.5, -0.5),
        (0.5, -1.0, -0.5),
        (-0.5, 1.5, -1.0),
        (-1.25, -0.125, 0.75),
        (1.125, -1.75, -0.625),
        (0.375, 0.25, 1.875),
    ],
    dtype=np.float64,
)
INDICES = tuple(product(range(-5, 6), repeat=3))


def _affine_face_values(h, intercept, gradient):
    def face_value(axis, index):
        assert type(axis) is int and 0 <= axis < 3
        assert isinstance(index, tuple) and all(type(i) is int for i in index)
        anchor = h * np.array([i if d == axis else i + 0.5 for d, i in enumerate(index)])
        value = intercept[axis] + gradient[axis] @ anchor
        assert abs(value) <= 8.0
        return value

    return face_value


@pytest.mark.parametrize("h", SPACINGS)
@pytest.mark.parametrize("reconstruction", ("quadratic", "rt0", "mixed"))
def test_reconstructs_constant_velocity(h, reconstruction):
    expected = np.array([2.0, -1.0, 0.25])
    donors = _affine_face_values(h, expected, np.zeros((3, 3)))
    for lattice_position in LATTICE_POSITIONS:
        velocity, gradient = sample_mac_transport(h * lattice_position, h, donors, reconstruction=reconstruction)
        assert velocity.dtype == gradient.dtype == np.float64
        np.testing.assert_allclose(velocity, expected, atol=VALUE_ATOL, rtol=0.0)
        np.testing.assert_allclose(gradient, np.zeros((3, 3)), atol=GRADIENT_ATOL, rtol=0.0)


@pytest.mark.parametrize("h", SPACINGS)
@pytest.mark.parametrize("reconstruction", ("quadratic", "mixed", "rt0"))
def test_reconstructs_affine_velocity(h, reconstruction):
    intercept = np.array([1.5, -0.75, 0.25])
    expected_gradient = np.array([[2.0, 0.5, -1.5], [-1.25, -0.5, 0.75], [3.0, -2.0, 1.0]])
    if reconstruction == "rt0":
        expected_gradient = np.diag(np.diag(expected_gradient))
    donors = _affine_face_values(h, intercept, expected_gradient)
    for lattice_position in LATTICE_POSITIONS:
        position = h * lattice_position
        velocity, gradient = sample_mac_transport(position, h, donors, reconstruction=reconstruction)
        np.testing.assert_allclose(velocity, intercept + expected_gradient @ position, atol=VALUE_ATOL, rtol=0.0)
        np.testing.assert_allclose(gradient, expected_gradient, atol=GRADIENT_ATOL, rtol=0.0)


@pytest.mark.parametrize("h", SPACINGS)
def test_rt0_uses_positive_side_cell_at_transverse_face(h):
    shear = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    donors = _affine_face_values(h, np.zeros(3), shear)
    position = h * np.array([0.25, -1.0, -0.5])
    velocity, gradient = sample_mac_transport(position, h, donors, reconstruction="rt0")
    np.testing.assert_allclose(velocity, [-h, 0.0, 0.0], atol=VALUE_ATOL, rtol=0.0)
    np.testing.assert_allclose(gradient, np.zeros((3, 3)), atol=GRADIENT_ATOL, rtol=0.0)
    assert abs(velocity[0] - (shear @ position)[0]) == pytest.approx(h, abs=VALUE_ATOL, rel=0.0)


@pytest.mark.parametrize(("h", "face_index"), ((0.1, -3), (0.3, -3), (0.1, 17), (0.1, 2**40 + 1)))
def test_rt0_distinguishes_rounded_face_from_adjacent_positions(h, face_index):
    def face_value(axis, index):
        if axis == 0:
            return index[1] - face_index
        if axis == 1:
            return (index[1] - face_index) ** 2
        return 0.0

    face_position = face_index * h
    positions = (nextafter(face_position, -inf), face_position, nextafter(face_position, inf))
    transverse_velocities = []
    normal_velocities = []
    normal_gradients = []
    for coordinate in positions:
        velocity, gradient = sample_mac_transport((0.25 * h, coordinate, 0.25 * h), h, face_value, reconstruction="rt0")
        transverse_velocities.append(velocity[0])
        normal_velocities.append(velocity[1])
        normal_gradients.append(gradient[1, 1])

    assert transverse_velocities == [-1.0, 0.0, 0.0]
    lower_width = face_position - (face_index - 1) * h
    upper_width = (face_index + 1) * h - face_position
    expected_velocities = [(face_position - positions[0]) / lower_width, 0.0, (positions[2] - face_position) / upper_width]
    expected_gradients = [-1 / lower_width, 1 / upper_width, 1 / upper_width]
    assert normal_velocities == pytest.approx(expected_velocities, abs=VALUE_ATOL, rel=0.0)
    assert normal_gradients == pytest.approx(expected_gradients, abs=GRADIENT_ATOL, rel=0.0)


def _shift(index, axis):
    return tuple(i + (axis == d) for d, i in enumerate(index))


def _discrete_curl_faces(h):
    # Each edge is indexed by its low endpoint and has a dyadic potential in m²/s.
    rng = np.random.default_rng(521)
    potential = dict(zip(product(range(3), INDICES), h * rng.integers(-16, 17, size=3 * len(INDICES)) / 16))
    faces = {}
    for axis in range(3):
        first, second = (axis + 1) % 3, (axis + 2) % 3
        for index in INDICES:
            if index[first] == 5 or index[second] == 5:
                continue
            first_difference = potential[second, _shift(index, first)] - potential[second, index]
            second_difference = potential[first, _shift(index, second)] - potential[first, index]
            faces[axis, index] = (first_difference - second_difference) / h
    return faces


def _cell_divergence(faces, cell, h):
    x, y, z = cell
    return (
        faces[0, (x + 1, y, z)] - faces[0, (x, y, z)]
        + faces[1, (x, y + 1, z)] - faces[1, (x, y, z)]
        + faces[2, (x, y, z + 1)] - faces[2, (x, y, z)]
    ) / h


def _cell_quadratic_convolution(lattice_position, cell_divergence):
    # Evaluate the centered cardinal quadratic via third differences of positive-part squares.
    # The finite table also includes cells outside the support, where this polynomial is zero.
    total = 0.0
    for cell, divergence in cell_divergence.items():
        weight = 1.0
        for coordinate, index in zip(lattice_position, cell):
            delta = coordinate - (index + 0.5)
            weight *= 0.5 * sum(
                coefficient * max(delta + knot, 0.0) ** 2
                for knot, coefficient in ((1.5, 1), (0.5, -3), (-0.5, 3), (-1.5, -1))
            )
        total += weight * divergence
    return total


@pytest.mark.parametrize("h", SPACINGS)
def test_mixed_and_rt0_traces_vanish_on_discrete_curl(h):
    faces = _discrete_curl_faces(h)
    assert max(abs(value) for value in faces.values()) <= 8.0
    assert all(
        _cell_divergence(faces, cell, h) == 0.0 for cell in product(range(-4, 4), repeat=3)
    )
    for lattice_position in LATTICE_POSITIONS:
        for reconstruction in ("mixed", "rt0"):
            _, gradient = sample_mac_transport(
                h * lattice_position, h, lambda axis, index: faces[axis, index], reconstruction=reconstruction
            )
            assert abs(np.trace(gradient)) <= GRADIENT_ATOL


@pytest.mark.parametrize("h", SPACINGS)
def test_quadratic_trace_does_not_commute_on_discrete_curl(h):
    faces = _discrete_curl_faces(h)
    traces = []
    for lattice_position in LATTICE_POSITIONS:
        _, gradient = sample_mac_transport(
            h * lattice_position, h, lambda axis, index: faces[axis, index], reconstruction="quadratic"
        )
        traces.append(abs(np.trace(gradient)))
    assert max(traces) > 1e-6


@pytest.mark.parametrize("h", SPACINGS)
@pytest.mark.parametrize("reconstruction", ("mixed", "rt0"))
def test_trace_matches_varying_nonzero_cell_divergence(h, reconstruction):
    faces = {
        (axis, (x, y, z)): ((3 * x**2 - 5 * y * z + 7 * axis * z + 11 * y**2 + 17 * axis) % 67 - 33) / 8
        for axis in range(3)
        for x, y, z in INDICES
    }
    assert max(abs(value) for value in faces.values()) <= 8.0
    cell_divergence = {cell: _cell_divergence(faces, cell, h) for cell in product(range(-4, 4), repeat=3)}
    assert min(cell_divergence.values()) < 0.0 < max(cell_divergence.values())
    for lattice_position in LATTICE_POSITIONS:
        if reconstruction == "mixed":
            expected_trace = _cell_quadratic_convolution(lattice_position, cell_divergence)
        else:
            cell = tuple(np.floor(lattice_position).astype(int))
            expected_trace = cell_divergence[cell]
        _, gradient = sample_mac_transport(
            h * lattice_position, h, lambda axis, index: faces[axis, index], reconstruction=reconstruction
        )
        assert np.trace(gradient) == pytest.approx(expected_trace, abs=GRADIENT_ATOL, rel=0.0)


@pytest.mark.parametrize("reconstruction", ("quadratic", "rt0", "mixed"))
def test_missing_donor_raises_key_error(reconstruction):
    missing = (1, (-1, 0, -1))
    faces = dict.fromkeys(product(range(3), INDICES), 1.0)
    del faces[missing]
    with pytest.raises(KeyError) as error:
        sample_mac_transport(
            (-0.75, 0.25, -0.75), 1.0, lambda axis, index: faces[axis, index], reconstruction=reconstruction
        )
    assert error.value.args == (missing,)


@pytest.mark.parametrize(
    ("position", "h", "reconstruction"),
    (
        ((0.0, 0.0), 1.0, "quadratic"),
        ((0.0, np.nan, 0.0), 1.0, "quadratic"),
        ((0.0, 0.0, 0.0), 0.0, "rt0"),
        ((1.6e308, 0.0, 0.0), 1e308, "rt0"),
        ((1.0, 0.0, 0.0), nextafter(0.0, inf), "rt0"),
        ((0.0, 0.0, 0.0), np.inf, "mixed"),
        ((0.0, 0.0, 0.0), 1.0, "cubic"),
    ),
)
def test_rejects_invalid_sampling_inputs(position, h, reconstruction):
    def face_value(axis, index):
        pytest.fail("invalid input read a face")

    with pytest.raises(ValueError):
        sample_mac_transport(position, h, face_value, reconstruction=reconstruction)
