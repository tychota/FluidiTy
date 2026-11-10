"""Exact and manufactured fixtures for transport on complete uniform MAC support."""

import json
from functools import cache
from itertools import product
from math import cos, exp, inf, log2, nextafter, pi, sin, sqrt

import numpy as np
import pytest

from fluidity_reference.mac_transport_3d import sample_mac_transport


SPACINGS = (1 / 8, 1 / 16, 1 / 32, 1 / 64)
VALUE_ATOL = 2e-13
GRADIENT_ATOL = 2e-11
FLUX_ATOL = 1e-15
RECONSTRUCTIONS = ("quadratic", "rt0", "mixed")
SUBCELL_PHASES = (0.0, 0.25, 0.5, 0.75)
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


@pytest.mark.parametrize("h", SPACINGS)
@pytest.mark.parametrize("normal_axis", range(3))
def test_single_face_flux_with_zero_discrete_divergence(h, normal_axis):
    face = (-2, 1, -1)
    first_axis, second_axis = (normal_axis + 1) % 3, (normal_axis + 2) % 3
    local_axes = [normal_axis, first_axis, second_axis]
    face_center = h * np.array([i if axis == normal_axis else i + 0.5 for axis, i in enumerate(face)])
    face_area = h**2

    def face_value(axis, index):
        anchor = h * np.array([i if d == axis else i + 0.5 for d, i in enumerate(index)])
        normal, first, second = (anchor - face_center)[local_axes]
        # m/s with linear coefficients in s^-1 and quadratic coefficients in (m s)^-1.
        if axis == normal_axis:
            return -0.75 + normal - first + 0.5 * second + normal**2 + 2 * first**2 - 0.5 * second**2
        if axis == first_axis:
            return -(1 + 2 * normal) * first
        return 0.0

    stored_speed = face_value(normal_axis, face)
    for direction in range(3):
        neighbors = [
            tuple(i + (offset if axis == direction else 0) for axis, i in enumerate(face))
            for offset in (-1, 0, 1)
        ]
        assert len({face_value(normal_axis, index) for index in neighbors}) == 3

    # The 27 cells cover the quadratic cell convolution on the face interior.
    for offset in product((-1, 0, 1), repeat=3):
        cell = tuple(i + delta for i, delta in zip(face, offset))
        outward_difference = sum(
            face_value(axis, _shift(cell, axis)) - face_value(axis, cell) for axis in range(3)
        )
        assert outward_difference == 0.0

    # Transverse B2 knots lie at integer lattice coordinates. The face edges
    # are consecutive knots in each tangent, so each split has one interval.
    # Two Gauss points per interval integrate degree <= 3 with weight h/2.
    offsets = (-h / (2 * sqrt(3)), h / (2 * sqrt(3)))
    mixed_normal_speeds = []
    rt0_lower_normal_speeds = []
    rt0_upper_normal_speeds = []
    mixed_traces = []
    for first_offset, second_offset in product(offsets, repeat=2):
        point = face_center.copy()
        point[first_axis] += first_offset
        point[second_axis] += second_offset
        velocity, gradient = sample_mac_transport(point, h, face_value, reconstruction="mixed")
        mixed_normal_speeds.append(velocity[normal_axis])
        mixed_traces.append(np.trace(gradient))
        velocity, _ = sample_mac_transport(point, h, face_value, reconstruction="rt0")
        rt0_upper_normal_speeds.append(velocity[normal_axis])
        point[normal_axis] = nextafter(point[normal_axis], -inf)
        velocity, _ = sample_mac_transport(point, h, face_value, reconstruction="rt0")
        rt0_lower_normal_speeds.append(velocity[normal_axis])

    trace_max_abs = max(map(abs, mixed_traces))
    assert trace_max_abs <= GRADIENT_ATOL
    rt0_lower_flux = face_area * np.mean(rt0_lower_normal_speeds)
    rt0_upper_flux = face_area * np.mean(rt0_upper_normal_speeds)
    mixed_flux = face_area * np.mean(mixed_normal_speeds)
    stored_flux = face_area * stored_speed
    # The continuum face average adds the tangential quadratic moment h²/12.
    # The B3 normal and B2 transverse second moments add h²/3 and h²/4.
    continuum_flux = stored_flux + h**4 / 8
    expected_mixed_bias = (17 / 24) * h**4
    assert rt0_lower_flux == pytest.approx(stored_flux, abs=FLUX_ATOL, rel=0.0)
    assert rt0_upper_flux == pytest.approx(stored_flux, abs=FLUX_ATOL, rel=0.0)
    assert mixed_flux - continuum_flux == pytest.approx(expected_mixed_bias, abs=FLUX_ATOL, rel=0.0)
    assert mixed_flux - stored_flux > FLUX_ATOL

    print(json.dumps({
        "fixture": "single_face_flux", "h_m": h, "normal_axis": normal_axis, "face_index": face,
        "stored_flux_m3_per_s": stored_flux, "continuum_flux_m3_per_s": continuum_flux,
        "rt0_lower_flux_m3_per_s": rt0_lower_flux, "rt0_upper_flux_m3_per_s": rt0_upper_flux,
        "mixed_flux_m3_per_s": mixed_flux, "mixed_minus_stored_flux_m3_per_s": mixed_flux - stored_flux,
        "mixed_trace_max_abs_per_s": trace_max_abs,
    }, allow_nan=False))


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


def _stratified_positions(h):
    # One sample per 1/8 m cube in [-1/2, 1/2)^3, with the same phase in each
    # stratum at every h. Select the middle cell (positive-side on a tie).
    cells_per_stratum = int(SPACINGS[0] / h)
    positions = []
    for x, y, z in product(range(-4, 4), repeat=3):
        phase_indices = (x + y, y + z, z + x)
        cell = np.array((x, y, z)) * cells_per_stratum + cells_per_stratum // 2
        phase = np.array([SUBCELL_PHASES[index % len(SUBCELL_PHASES)] for index in phase_indices])
        positions.append(h * (cell + phase))
    return np.array(positions)


def _curved_face_values(h, wave_number):
    @cache
    def face_value(axis, index):
        x, y, _ = index
        # Signed x and y face anchors. The field is independent of z.
        if axis == 0:
            return sin(wave_number * h * x) * cos(wave_number * h * (y + 0.5))
        if axis == 1:
            return -cos(wave_number * h * (x + 0.5)) * sin(wave_number * h * y)
        return 0.0

    return face_value


def _analytic_curved_field(positions, wave_number):
    x, y, _ = wave_number * positions.T
    velocity = np.zeros_like(positions)
    velocity[:, 0] = np.sin(x) * np.cos(y)
    velocity[:, 1] = -np.cos(x) * np.sin(y)
    gradient = np.zeros((len(positions), 3, 3))
    diagonal = wave_number * np.cos(x) * np.cos(y)
    shear = wave_number * np.sin(x) * np.sin(y)
    gradient[:, 0, 0] = diagonal
    gradient[:, 0, 1] = -shear
    gradient[:, 1, 0] = shear
    gradient[:, 1, 1] = -diagonal
    return velocity, gradient


def _error_stats(errors, positions):
    errors = np.abs(errors)
    assert np.all(np.isfinite(errors))
    location, *component = np.unravel_index(np.argmax(errors), errors.shape)
    component_maxima = np.max(errors, axis=0)
    return {
        "rms": float(np.sqrt(np.mean(errors**2))),
        "rms_by_component": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        "max_abs_by_component": component_maxima.tolist(),
        "max_abs_position_m_by_component": positions[np.argmax(errors, axis=0)].tolist(),
        "max_abs": float(errors[(location, *component)]),
        "max_abs_position_m": positions[location].tolist(),
        "max_abs_component": [int(axis) for axis in component],
    }


@pytest.mark.parametrize("field", ("smooth", "nyquist"))
@pytest.mark.parametrize("reconstruction", RECONSTRUCTIONS)
def test_curved_spatial_measurement(field, reconstruction):
    previous_errors = {}
    for h in SPACINGS:
        wave_number = 2 * pi if field == "smooth" else pi / h
        positions = _stratified_positions(h)
        donors = _curved_face_values(h, wave_number)
        exact_velocity, exact_gradient = _analytic_curved_field(positions, wave_number)
        samples = [
            sample_mac_transport(position, h, donors, reconstruction=reconstruction) for position in positions
        ]
        velocity, gradient = map(np.array, zip(*samples))
        value_error = _error_stats(velocity - exact_velocity, positions)
        gradient_error = _error_stats(gradient - exact_gradient, positions)
        if field == "smooth" and reconstruction != "rt0":
            bound = wave_number**2 * h**2 / 3 + VALUE_ATOL
            assert value_error["max_abs"] <= bound, f"smooth spline value bound at h={h} m"
        if field == "nyquist":
            # sin(pi*i) vanishes at every normal anchor. The remaining samples
            # are roundoff; between the faces the exact field still has O(1) amplitude.
            assert np.max(np.abs(velocity)) <= VALUE_ATOL
            assert value_error["max_abs"] > 0.5

        for metric, stats in (("value_m_per_s", value_error), ("gradient_per_s", gradient_error)):
            if metric in previous_errors:
                stats["dyadic_rms_rate"] = log2(previous_errors[metric]["rms"] / stats["rms"])
                stats["dyadic_max_rate"] = log2(previous_errors[metric]["max_abs"] / stats["max_abs"])
            previous_errors[metric] = stats
        print(json.dumps({
            "fixture": field, "reconstruction": reconstruction, "h_m": h, "wavelength_m": 2 * pi / wave_number,
            "sample_count": len(positions), "value_m_per_s": value_error, "gradient_per_s": gradient_error,
        }, allow_nan=False))


@pytest.mark.parametrize("reconstruction", RECONSTRUCTIONS)
def test_affine_euler_position_and_volume(reconstruction):
    # Units are metres and seconds. Unit strain over 1 s keeps the affine
    # reference independent of both the reconstruction and the Euler loop.
    h = SPACINGS[-1]
    initial = np.array([(-0.5, 0.25, -0.125), (0.375, -0.5, 0.25), (0.25, 0.125, -0.375)])
    donors = _affine_face_values(h, np.zeros(3), np.diag([1.0, -1.0, 0.0]))
    exact_flow_position = initial * [exp(1.0), exp(-1.0), 1.0]
    exact_flow_ratio = exp(1.0) * exp(-1.0)
    previous_errors = {}
    for dt in SPACINGS:
        steps = int(1 / dt)
        positions = initial.copy()
        euler_volume_ratio = np.ones(len(positions))
        material_j = np.ones(len(positions))
        for _ in range(steps):
            for particle, position in enumerate(positions):
                velocity, gradient = sample_mac_transport(position, h, donors, reconstruction=reconstruction)
                positions[particle] = position + dt * velocity
                euler_volume_ratio[particle] *= np.linalg.det(np.eye(3) + dt * gradient)
                material_j[particle] *= exp(dt * np.trace(gradient))

        discrete_position = initial * [(1 + dt)**steps, (1 - dt)**steps, 1.0]
        discrete_volume_ratio = (1 - dt**2)**steps
        np.testing.assert_allclose(positions, discrete_position, atol=VALUE_ATOL, rtol=0.0)
        np.testing.assert_allclose(
            euler_volume_ratio, discrete_volume_ratio, atol=2e-13, rtol=0.0,
            err_msg="Euler map must use its discrete determinant",
        )
        np.testing.assert_allclose(material_j, 1.0, atol=2e-13, rtol=0.0)
        assert exact_flow_ratio == pytest.approx(1.0, abs=2e-13, rel=0.0)

        position_error = _error_stats(positions - exact_flow_position, initial)
        volume_defect = float(np.max(np.abs(euler_volume_ratio - exact_flow_ratio)))
        errors = {"position": position_error["rms"], "euler_volume": volume_defect}
        rates = {metric: log2(previous_errors[metric] / error) for metric, error in errors.items()
                 if metric in previous_errors}
        previous_errors = errors
        print(json.dumps({
            "fixture": "affine_time", "reconstruction": reconstruction, "h_m": h, "dt_s": dt, "steps": steps,
            "initial_positions_m": initial.tolist(), "euler_positions_m": positions.tolist(),
            "analytic_discrete_positions_m": discrete_position.tolist(), "exact_flow_positions_m": exact_flow_position.tolist(),
            "position_error_m_by_initial_position": position_error, "exact_flow_volume_ratio": exact_flow_ratio,
            "analytic_discrete_volume_ratio": discrete_volume_ratio, "euler_volume_ratio": euler_volume_ratio.tolist(),
            "euler_volume_defect": volume_defect, "material_j": material_j.tolist(),
            "discrete_position_max_abs_error_m": float(np.max(np.abs(positions - discrete_position))),
            "discrete_volume_max_abs_error": float(np.max(np.abs(euler_volume_ratio - discrete_volume_ratio))),
            "material_j_max_abs_error": float(np.max(np.abs(material_j - 1.0))), "dyadic_rates": rates,
        }, allow_nan=False))
