"""Manufactured SI targets for independent complete-support MAC fit degree comparisons."""

from itertools import product
import json

import numpy as np
import pytest

from fluidity_reference.experiments.mac_fit_degrees import compare_mac_fit_degrees as compare


SPACINGS_M = (1 / 8, 1 / 16, 1 / 32, 1 / 64)
VELOCITY_ATOL_M_PER_S = 1e-12
GRADIENT_ATOL_PER_S = 1e-10
# Coefficients multiply dimensionless particle-centered coordinates.
COEFFICIENT_ATOL_M_PER_S = 1e-12
PHASES = ((0, 0.5, 0.25), (0.25, 0.75, 0.5), (0.75, 0, 0.75), (0.125, 0.375, 0.875))
CELL_INDICES = ((-3, 1, -1), (2, -2, 1), (-1, -3, 2), (1, 2, -3))
BASE_VELOCITY = np.array((1.25, -0.75, 0.625))
LINEAR_GRADIENT = np.array(((1, -0.5, 0.25), (0.5, -0.75, 0.25), (-0.25, 0.625, 0.75)))
HESSIAN = np.array((
    ((1.25, 0.5, -0.25), (0.5, -0.5, 0.375), (-0.25, 0.375, 0.75)),
    ((-0.5, 0.375, 0.5), (0.375, -0.75, -0.25), (0.5, -0.25, -0.25)),
    ((0.25, -0.375, 0.125), (-0.375, 1.25, 0.25), (0.125, 0.25, 0.5)),
))


def _polynomial(position_m, degree):
    gradient = LINEAR_GRADIENT if degree >= 1 else np.zeros((3, 3))
    hessian = HESSIAN if degree == 2 else np.zeros((3, 3, 3))
    velocity = BASE_VELOCITY + gradient @ position_m + 0.5 * np.einsum(
        "i,aij,j->a", position_m, hessian, position_m
    )
    return velocity, gradient + hessian @ position_m, hessian


def _face_sample(field, h_m, component, index):
    # Spell out the MAC face anchors separately from the fitter.
    i, j, k = index
    if component == 0:
        anchor = (i, j + 0.5, k + 0.5)
    elif component == 1:
        anchor = (i + 0.5, j, k + 0.5)
    else:
        anchor = (i + 0.5, j + 0.5, k)
    return field(h_m * np.array(anchor))[0][component]


def _record(fit):
    return {
        "rank_by_component": fit.rank.tolist(),
        "singular_values_by_component": fit.singular_values.tolist(),
        "weighted_residual_l2_m_per_s_by_component": fit.weighted_residual_l2_m_per_s.tolist(),
    }


def test_cubic_fits_share_one_complete_signed_mac_sample():
    calls = []

    def sample(component, index):
        calls.append((component, index))
        return 1.0

    h_m = 1 / 8
    result = compare(h_m * np.array((-2.5, 1.25, -0.125)), h_m, sample)
    expected = {
        (component, index)
        for component, y_start in enumerate((-1, 0, -1))
        for index in product(range(-4, 0), range(y_start, y_start + 4), range(-2, 2))
    }
    assert len(calls) == len(expected) == 192
    assert set(calls) == expected
    np.testing.assert_array_equal(result.positive_weight_counts, (64, 48, 48))
    np.testing.assert_array_equal(result.affine.rank, (4, 4, 4))
    np.testing.assert_array_equal(result.quadratic.rank, (10, 10, 10))


@pytest.mark.parametrize("degree", (0, 1, 2))
def test_analytic_polynomial_patches(degree):
    for h_m in SPACINGS_M:
        observed = []
        for cell_index, phase in zip(CELL_INDICES, PHASES):
            position_m = h_m * (np.array(cell_index) + phase)
            target_value, target_gradient, target_hessian = _polynomial(position_m, degree)

            def sample(component, index):
                velocity = _face_sample(lambda point: _polynomial(point, degree), h_m, component, index)
                assert abs(velocity) <= 8.0
                return velocity

            result = compare(position_m, h_m, sample)
            assert np.max(np.abs(target_gradient)) <= 4.0
            second_order = np.column_stack((
                0.5 * target_hessian[:, 0, 0], target_hessian[:, 0, 1], target_hessian[:, 0, 2],
                0.5 * target_hessian[:, 1, 1], target_hessian[:, 1, 2], 0.5 * target_hessian[:, 2, 2],
            ))
            scaled_coefficients = np.column_stack((target_value, h_m * target_gradient, h_m**2 * second_order))
            row = {"cell_index": cell_index, "phase": phase}
            for fit_degree, fit in ((1, result.affine), (2, result.quadratic)):
                columns = 4 if fit_degree == 1 else 10
                np.testing.assert_array_equal(fit.rank, (columns,) * 3)
                value_error = float(np.max(np.abs(fit.velocity_m_per_s - target_value)))
                gradient_error = float(np.max(np.abs(fit.gradient_per_s - target_gradient)))
                coefficient_error = float(np.max(np.abs(fit.coefficients_m_per_s - scaled_coefficients[:, :columns])))
                assert gradient_error <= GRADIENT_ATOL_PER_S
                if fit_degree >= degree:
                    assert value_error <= VELOCITY_ATOL_M_PER_S
                    assert coefficient_error <= COEFFICIENT_ATOL_M_PER_S
                    assert np.max(fit.weighted_residual_l2_m_per_s) <= VELOCITY_ATOL_M_PER_S
                else:
                    # B3 has second central moment 1/3 per axis and zero odd
                    # moments through degree three. Only the affine value is biased.
                    bias = h_m**2 / 6 * np.trace(target_hessian, axis1=1, axis2=2)
                    np.testing.assert_allclose(fit.velocity_m_per_s - target_value, bias, rtol=0, atol=VELOCITY_ATOL_M_PER_S)
                    assert value_error > 1e-5
                    assert np.min(fit.weighted_residual_l2_m_per_s) > 1e-5
                row[f"fit_degree_{fit_degree}"] = {
                    **_record(fit),
                    "max_value_error_m_per_s": value_error,
                    "max_gradient_error_per_s": gradient_error,
                    "max_scaled_coefficient_error_m_per_s": coefficient_error,
                }
            bias = h_m**2 / 6 * np.trace(target_hessian, axis1=1, axis2=2)
            np.testing.assert_allclose(
                result.spline_velocity_m_per_s, target_value + bias, rtol=0, atol=VELOCITY_ATOL_M_PER_S
            )
            np.testing.assert_allclose(
                result.spline_gradient_per_s, target_gradient, rtol=0, atol=GRADIENT_ATOL_PER_S
            )
            observed.append(row)
        print(json.dumps({"fixture": "polynomial", "field_degree": degree, "h_m": h_m, "phases": observed},
                         allow_nan=False))


@pytest.mark.parametrize("h_m", SPACINGS_M)
def test_quadratic_spline_knot_tie_loses_one_quadratic_mode(h_m):
    # For the x row, the normal offsets +/-1/2 give xi_x**2 = 1/4.
    position_m = h_m * np.array((-2.5, 1.25, -0.125))
    result = compare(position_m, h_m, lambda _axis, _index: 1.0, spline_degree=2)
    np.testing.assert_array_equal(result.positive_weight_counts, (18, 27, 27))
    np.testing.assert_array_equal(result.affine.rank, (4, 4, 4))
    np.testing.assert_array_equal(result.quadratic.rank, (9, 10, 10))
    assert result.quadratic.singular_values[0, -1] < 1e-14
    assert np.min(result.quadratic.singular_values[1:, -1]) > 1e-3
    assert result.quadratic.coefficients_m_per_s is None
    assert result.quadratic.velocity_m_per_s is None
    assert result.quadratic.gradient_per_s is None
    print(json.dumps({"fixture": "quadratic_spline_tie", "h_m": h_m, "quadratic": _record(result.quadratic)},
                     allow_nan=False))


def test_asymmetric_cubic_patch_has_third_order_quadratic_intercept_bias():
    # At quarter-cell phase the x-row's centered nodes are
    # (-5, -1, 3, 7)/4 with weights (27, 235, 121, 1)/384.
    # The odd moments m1 and m3 vanish, but m5 = -25/256.
    # Removing the quadratic column yields intercept
    # -m2*m5/(m4 - m2**2) = 75/431 for the lattice-scale cubic.
    h_m = 1 / 8
    position_m = h_m * np.array((-2.75, 1.25, -0.125))

    def sample(component, index):
        if component == 0:
            return (h_m * index[0] - position_m[0]) ** 3
        return 0.0

    result = compare(position_m, h_m, sample)
    assert abs(result.affine.velocity_m_per_s[0]) <= VELOCITY_ATOL_M_PER_S
    assert abs(result.spline_velocity_m_per_s[0]) <= VELOCITY_ATOL_M_PER_S
    expected_bias_m_per_s = 75 * h_m**3 / 431
    np.testing.assert_allclose(
        result.quadratic.velocity_m_per_s[0], expected_bias_m_per_s, rtol=0, atol=VELOCITY_ATOL_M_PER_S
    )


def test_smooth_spatial_refinement():
    # Each component is an oblique physical wave with a distinct phase and
    # amplitude. Keep the same 512 equal-volume strata and phases at every h.
    amplitude = np.array((0.75, -0.625, 0.5))
    wave_number_per_m = np.pi * np.array(((1, 0.5, -0.25), (0.25, -1, 0.5), (-0.5, 0.25, 1)))
    wave_phase = np.array((0.125, 0.375, 0.625))

    def field(position_m):
        angle = wave_number_per_m @ position_m + wave_phase
        velocity = amplitude * np.sin(angle)
        gradient = (amplitude * np.cos(angle))[:, None] * wave_number_per_m
        return velocity, gradient

    previous = {}
    for h_m in SPACINGS_M:
        cells_per_stratum = int(SPACINGS_M[0] / h_m)
        errors = {name: [] for name in ("affine", "quadratic", "cubic_spline")}
        max_fit_gradient_delta = 0.0
        max_spline_gradient_delta = 0.0

        def sample(component, index):
            return _face_sample(field, h_m, component, index)

        for i, j, k in product(range(-4, 4), repeat=3):
            cell_index = cells_per_stratum * np.array((i, j, k)) + cells_per_stratum // 2
            phase = (np.array((i + j, j + k, k + i)) % 4) / 4
            position_m = h_m * (cell_index + phase)
            target_value, target_gradient = field(position_m)
            result = compare(position_m, h_m, sample)
            affine, quadratic = result.affine, result.quadratic
            max_fit_gradient_delta = max(
                max_fit_gradient_delta, float(np.max(np.abs(affine.gradient_per_s - quadratic.gradient_per_s)))
            )
            max_spline_gradient_delta = max(
                max_spline_gradient_delta,
                float(np.max(np.abs(affine.gradient_per_s - result.spline_gradient_per_s))),
            )
            np.testing.assert_allclose(
                affine.velocity_m_per_s, result.spline_velocity_m_per_s, rtol=0, atol=VELOCITY_ATOL_M_PER_S
            )
            for name, velocity, gradient in (
                ("affine", affine.velocity_m_per_s, affine.gradient_per_s),
                ("quadratic", quadratic.velocity_m_per_s, quadratic.gradient_per_s),
                ("cubic_spline", result.spline_velocity_m_per_s, result.spline_gradient_per_s),
            ):
                errors[name].append((velocity - target_value, gradient - target_gradient))

        observations = {}
        for name, pairs in errors.items():
            observations[name] = {}
            for quantity, values in zip(("value_m_per_s", "gradient_per_s"), zip(*pairs)):
                errors_array = np.asarray(values)
                assert np.all(np.isfinite(errors_array))
                rms = float(np.sqrt(np.mean(errors_array**2)))
                maximum = float(np.max(np.abs(errors_array)))
                stats = {"rms": rms, "max_abs": maximum}
                if (name, quantity) in previous:
                    coarse_rms, coarse_maximum = previous[name, quantity]
                    stats["dyadic_rms_rate"] = float(np.log2(coarse_rms / rms))
                    stats["dyadic_max_rate"] = float(np.log2(coarse_maximum / maximum))
                    if h_m == SPACINGS_M[-1]:
                        # A cubic can bias the quadratic intercept at asymmetric phases.
                        minimum_rate = 2.75 if name == "quadratic" and quantity == "value_m_per_s" else 1.75
                        assert stats["dyadic_rms_rate"] > minimum_rate
                previous[name, quantity] = rms, maximum
                observations[name][quantity] = stats

        assert observations["quadratic"]["value_m_per_s"]["rms"] < observations["affine"]["value_m_per_s"]["rms"]
        assert max_fit_gradient_delta <= GRADIENT_ATOL_PER_S
        assert max_spline_gradient_delta > 1e-6
        print(json.dumps({
            "fixture": "smooth", "h_m": h_m, "sample_count": 512, "methods": observations,
            "max_affine_quadratic_gradient_delta_per_s": max_fit_gradient_delta,
            "max_cubic_spline_affine_gradient_delta_per_s": max_spline_gradient_delta,
        }, allow_nan=False))
