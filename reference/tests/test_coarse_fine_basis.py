from dataclasses import FrozenInstanceError
from itertools import product
from math import comb, factorial

import numpy as np
import pytest
from fluidity_reference.basis import quadratic_stencil
from fluidity_reference.experiments.coarse_fine_basis import (
    evaluate_coarse_fine_basis,
    evaluate_uniform_cubic_normal_basis,
)


def _cardinal(degree, delta):
    """Independent centered spline from its truncated-power representation."""
    radius = (degree + 1) / 2
    if abs(delta) >= radius:
        return 0.0, 0.0
    positive_parts = [max(0.0, delta + radius - knot) for knot in range(degree + 2)]
    coefficients = [(-1) ** knot * comb(degree + 1, knot) for knot in range(degree + 2)]
    weight = sum(c * v**degree for c, v in zip(coefficients, positive_parts, strict=True)) / factorial(degree)
    gradient = sum(c * v ** (degree - 1) for c, v in zip(coefficients, positive_parts, strict=True))
    return weight, gradient / factorial(degree - 1)


def _tensor_cardinal(degrees, position, anchor, width):
    (weight_x, gradient_x), (weight_y, gradient_y) = [
        _cardinal(degree, delta) for degree, delta in zip(degrees, (position - anchor) / width, strict=True)
    ]
    return np.array([weight_x * weight_y, gradient_x * weight_y / width, weight_x * gradient_y / width])


def _dyadic_children(degrees, coarse_anchor):
    for child in product(*(range(degree + 2) for degree in degrees)):
        offset = np.array([(index - (degree + 1) / 2) / 2 for index, degree in zip(child, degrees, strict=True)])
        coefficient = np.prod(
            [comb(degree + 1, index) / 2**degree for index, degree in zip(child, degrees, strict=True)]
        )
        yield coarse_anchor + offset, coefficient


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("face_normal_axis", [0, 1])
def test_dyadic_child_identity_matches_independent_coarse_cardinal(dtype, face_normal_axis):
    degrees = tuple(3 if axis == face_normal_axis else 2 for axis in range(2))
    coarse_anchor = np.array([0 if axis == face_normal_axis else 0.5 for axis in range(2)])
    child_coefficients = {
        tuple(anchor): coefficient for anchor, coefficient in _dyadic_children(degrees, coarse_anchor)
    }
    for position in [np.array([0.0, 0.5]), np.array([-0.4375, 0.3125]), np.array([0.875, -0.3125])]:
        fine = evaluate_uniform_cubic_normal_basis(
            position, lattice_origin=(0, 0), cell_width=0.5, face_normal_axis=face_normal_axis, dtype=dtype
        )
        coefficients = np.array(
            [child_coefficients.get(tuple(anchor), 0) for anchor in fine.anchor_world_positions], dtype
        )
        refined = np.concatenate(([coefficients @ fine.weights], coefficients @ fine.physical_gradients))
        assert refined.dtype == dtype
        expected = _tensor_cardinal(degrees, position, coarse_anchor, 1.0)
        np.testing.assert_allclose(refined, expected, rtol=0, atol=3e-7 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("face_normal_axis,interface_axis", product(range(2), repeat=2))
def test_transition_matches_direct_tensor_truncation(face_normal_axis, interface_axis):
    degrees = tuple(3 if axis == face_normal_axis else 2 for axis in range(2))
    shift = np.array([0 if degree == 3 else 0.5 for degree in degrees])
    radius = (degrees[interface_axis] + 1) / 2
    for position in [np.array([-0.4375, -0.3125]), np.array([0.0, 0.0]), np.array([0.375, 0.3125])]:
        expected = {}
        for lattice_index in product(range(-4, 5), repeat=2):
            coarse_anchor = lattice_index + shift
            fine_anchor = coarse_anchor / 2
            if fine_anchor[interface_axis] >= radius / 2:
                fine = _tensor_cardinal(degrees, position, fine_anchor, 0.5)
                if fine[0] > 0:
                    expected[(1, *fine_anchor)] = fine
            if coarse_anchor[interface_axis] >= radius:
                continue
            truncated = np.zeros(3)
            for anchor, coefficient in _dyadic_children(degrees, coarse_anchor):
                if anchor[interface_axis] < radius / 2:
                    truncated += coefficient * _tensor_cardinal(degrees, position, anchor, 0.5)
            if truncated[0] > 0:
                expected[(0, *coarse_anchor)] = truncated
        stencil = evaluate_coarse_fine_basis(
            position,
            interface_origin=(0, 0),
            coarse_cell_width=1.0,
            face_normal_axis=face_normal_axis,
            interface_axis=interface_axis,
            dtype=np.float64,
        )
        keys = [
            (int(level), *anchor) for level, anchor in zip(stencil.levels, stencil.anchor_world_positions, strict=True)
        ]
        assert keys == sorted(expected)
        np.testing.assert_allclose(
            np.column_stack((stencil.weights, stencil.physical_gradients)),
            [expected[key] for key in keys],
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("face_normal_axis", [0, 1])
@pytest.mark.parametrize("coarse_cell_width", [0.5, 2.0])
def test_transition_preserves_affine_moments_and_rotates(dtype, face_normal_axis, coarse_cell_width):
    moment_tolerance = 1e-6 if dtype == np.float32 else 1e-12
    derivative_tolerance = 2e-6 if dtype == np.float32 else 1e-12
    rotation = np.array([[0, -1], [1, 0]], dtype=dtype)
    interface_origin = np.array([-3.25, -7.5], dtype=dtype)
    rotated_interface_origin = np.array([-8.5, -4.0], dtype=dtype)
    cases = [
        ((-4.0, -0.375), {0}),
        ((-0.4375, -0.3125), {0}),
        ((0.0, 0.5), {0}),
        ((0.375, 0.3125), {0, 1}),
        ((4.0, 0.25), {1}),
    ]
    for point, expected_levels in cases:
        local_position = np.array(point, dtype=dtype)
        position = interface_origin + coarse_cell_width * local_position
        rotated_position = rotated_interface_origin + coarse_cell_width * (rotation @ local_position)
        stencil = evaluate_coarse_fine_basis(
            position,
            interface_origin=interface_origin,
            coarse_cell_width=coarse_cell_width,
            face_normal_axis=face_normal_axis,
            interface_axis=0,
            dtype=dtype,
        )
        rotated = evaluate_coarse_fine_basis(
            rotated_position,
            interface_origin=rotated_interface_origin,
            coarse_cell_width=coarse_cell_width,
            face_normal_axis=1 - face_normal_axis,
            interface_axis=1,
            dtype=dtype,
        )
        for result, query in [(stencil, position), (rotated, rotated_position)]:
            assert (
                result.anchor_world_positions.dtype == result.weights.dtype == result.physical_gradients.dtype == dtype
            )
            assert np.issubdtype(result.levels.dtype, np.integer)
            assert set(result.levels) == expected_levels
            assert np.all(result.weights > 0)
            delta_over_width = (result.anchor_world_positions - query) / coarse_cell_width
            dimensionless_gradients = result.physical_gradients * coarse_cell_width
            assert result.weights.sum().dtype == (result.weights @ delta_over_width).dtype == dtype
            np.testing.assert_allclose(result.weights.sum(), 1, rtol=0, atol=moment_tolerance)
            np.testing.assert_allclose(result.weights @ delta_over_width, 0, rtol=0, atol=moment_tolerance)
            np.testing.assert_allclose(dimensionless_gradients.sum(axis=0), 0, rtol=0, atol=derivative_tolerance)
            np.testing.assert_allclose(
                delta_over_width.T @ dimensionless_gradients, np.eye(2), rtol=0, atol=derivative_tolerance
            )

        expected_rotated_anchors = (
            stencil.anchor_world_positions - interface_origin
        ) @ rotation.T + rotated_interface_origin
        rotated_rows = {tuple(anchor): row for row, anchor in enumerate(rotated.anchor_world_positions)}
        corresponding_rows = [rotated_rows[tuple(anchor)] for anchor in expected_rotated_anchors]
        np.testing.assert_array_equal(rotated.levels[corresponding_rows], stencil.levels)
        np.testing.assert_allclose(rotated.weights[corresponding_rows], stencil.weights, rtol=0, atol=moment_tolerance)
        np.testing.assert_allclose(
            rotated.physical_gradients[corresponding_rows] * coarse_cell_width,
            (stencil.physical_gradients @ rotation.T) * coarse_cell_width,
            rtol=0,
            atol=derivative_tolerance,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("face_normal_axis", [0, 1])
def test_uniform_comparator_second_moment_matches_cardinal_values(dtype, face_normal_axis):
    origin = np.array([-3.25, -4.5], dtype=dtype)
    cell_width = dtype(0.5)
    expected = np.eye(2) / 4
    expected[face_normal_axis, face_normal_axis] = 1 / 3
    for local_position in [(0, 0.5), (-0.4375, 0.3125), (0.875, -0.3125)]:
        position = origin + cell_width * np.array(local_position, dtype=dtype)
        stencil = evaluate_uniform_cubic_normal_basis(
            position, lattice_origin=origin, cell_width=cell_width, face_normal_axis=face_normal_axis, dtype=dtype
        )
        assert set(stencil.levels) == {0}
        deltas = (stencil.anchor_world_positions - position) / cell_width
        second_moment = (deltas.T * stencil.weights) @ deltas
        assert second_moment.dtype == dtype
        np.testing.assert_allclose(second_moment, expected, rtol=0, atol=3e-7 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("face_normal_axis", [0, 1])
def test_uniform_quadratic_oracle_has_nonnested_mac_normal_knot(face_normal_axis):
    offset = np.full(2, 0.5)
    offset[face_normal_axis] = 0
    normal_coordinates = [0.375, 0.5, 0.625]
    for width, normal_index, expected_curvatures in [(1.0, 0, [-2, 1]), (0.5, 1, [-8, -8])]:
        marginal_derivatives = []
        for normal_coordinate in normal_coordinates:
            position = np.full(2, -0.3125)
            position[face_normal_axis] = normal_coordinate
            indices, weights, gradients = quadratic_stencil(position, width, offset)
            deltas = (indices + offset) - position / width
            np.testing.assert_allclose(weights @ deltas, 0, rtol=0, atol=1e-14)
            np.testing.assert_allclose((deltas.T * weights) @ deltas, np.eye(2) / 4, rtol=0, atol=1e-14)
            marginal_derivatives.append(gradients[indices[:, face_normal_axis] == normal_index, face_normal_axis].sum())
        # The coarse lattice breaks at x/h = 1/2. The fine lattice breaks only at odd quarters of h.
        np.testing.assert_allclose(np.diff(marginal_derivatives) * 8, expected_curvatures, rtol=0, atol=1e-14)


def test_stencil_is_immutable():
    stencil = evaluate_coarse_fine_basis(
        (0.375, 0.3125),
        interface_origin=(0, 0),
        coarse_cell_width=1,
        face_normal_axis=0,
        interface_axis=0,
        dtype=np.float32,
    )
    with pytest.raises(FrozenInstanceError):
        stencil.weights = np.empty(0)
    for values in (stencil.anchor_world_positions, stencil.weights, stencil.physical_gradients, stencil.levels):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)


@pytest.mark.parametrize(
    "position,overrides",
    [
        ((0,), {}),
        ((0, np.nan), {}),
        ((0, np.inf), {}),
        ((0, 0), {"interface_origin": (0, np.inf)}),
        ((0, 0), {"coarse_cell_width": 0}),
        ((0, 0), {"coarse_cell_width": np.inf}),
        ((0, 0), {"coarse_cell_width": np.nextafter(np.float32(0), np.float32(1))}),
        ((0, 0), {"face_normal_axis": 2}),
        ((0, 0), {"interface_axis": -1}),
        ((0, 0), {"interface_axis": True}),
        ((0, 0), {"dtype": np.float16}),
        ((1e20, 0), {}),
        ((1e20, 1e20), {"interface_origin": (1e20, 1e20)}),
    ],
)
def test_unusable_transition_inputs_are_rejected(position, overrides):
    options = {
        "interface_origin": (0, 0),
        "coarse_cell_width": 1.0,
        "face_normal_axis": 0,
        "interface_axis": 0,
        "dtype": np.float32,
    }
    with pytest.raises((TypeError, ValueError)):
        evaluate_coarse_fine_basis(position, **(options | overrides))


@pytest.fixture(params=["transition", "uniform"])
def evaluate_geometry(request):
    def evaluate(position, origin, width, dtype):
        if request.param == "transition":
            return evaluate_coarse_fine_basis(
                position,
                interface_origin=origin,
                coarse_cell_width=width,
                face_normal_axis=0,
                interface_axis=0,
                dtype=dtype,
            )
        return evaluate_uniform_cubic_normal_basis(
            position, lattice_origin=origin, cell_width=width, face_normal_axis=0, dtype=dtype
        )

    return evaluate


@pytest.mark.parametrize(
    "dtype,origin,offset",
    [
        (np.float32, (1e6, -1e6), (0.4375, 0.3125)),
        (np.float64, (1e12, -1e12), (0.4375, 0.3125)),
        (np.float32, (0, 0), (63.9, -57.3125)),
    ],
)
def test_rejects_inaccurate_world_anchor_offsets(evaluate_geometry, dtype, origin, offset):
    origin = np.array(origin, dtype=dtype)
    query = origin + np.array(offset, dtype=dtype)
    with pytest.raises(ValueError):
        evaluate_geometry(query, origin, dtype(1.1), dtype)


@pytest.mark.parametrize("dtype,origin_x,width", [(np.float32, 3.4e38, 6e36), (np.float64, 1.79e308, 3e306)])
def test_rejects_overflowing_world_anchor_reconstruction(evaluate_geometry, dtype, origin_x, width):
    with pytest.raises(ValueError):
        evaluate_geometry((0, 0), (origin_x, 0), dtype(width), dtype)


@pytest.mark.parametrize("dtype,origin_x", [(np.float32, 1e6), (np.float64, 1e12)])
def test_accepts_exact_world_anchor_geometry_near_patch_limit(evaluate_geometry, dtype, origin_x):
    origin = np.array((origin_x, -origin_x), dtype=dtype)
    query = origin + np.array((63.875, -57.3125), dtype=dtype)
    stencil = evaluate_geometry(query, origin, dtype(1), dtype)
    offsets = stencil.anchor_world_positions - query
    tolerance = 2 * np.finfo(dtype).eps
    np.testing.assert_allclose(stencil.weights @ offsets, 0, rtol=0, atol=tolerance)
    np.testing.assert_allclose(offsets.T @ stencil.physical_gradients, np.eye(2), rtol=0, atol=tolerance)
