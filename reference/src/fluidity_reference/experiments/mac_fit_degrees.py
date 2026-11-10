"""Independent weighted polynomial fits on complete, uniform 3D MAC supports."""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class MacPolynomialFit:
    """One fit per velocity component, with coefficients in m/s.

    Columns are 1, x, y, z and, for degree two, xx, xy, xz, yy, yz, zz.
    The coordinates are relative to the query and scaled by 1/h. Ranks,
    descending singular values and weighted residual norms are retained
    when any component loses rank. The coefficient array and its derived
    velocity and physical gradient are then all absent.
    """

    coefficients_m_per_s: np.ndarray | None
    velocity_m_per_s: np.ndarray | None
    gradient_per_s: np.ndarray | None
    rank: np.ndarray
    singular_values: np.ndarray
    weighted_residual_l2_m_per_s: np.ndarray


@dataclass(frozen=True)
class MacFitComparison:
    """Two fits and the ordinary spline interpolant on identical face samples."""

    affine: MacPolynomialFit
    quadratic: MacPolynomialFit
    spline_velocity_m_per_s: np.ndarray
    spline_gradient_per_s: np.ndarray
    positive_weight_counts: np.ndarray


def _basis(delta, degree):
    """Centered B-spline weights and query derivatives in lattice coordinates."""
    distance = np.abs(delta)
    if degree == 3:
        remaining = np.maximum(2.0 - distance, 0.0)
        inner = distance < 1.0
        weight = np.where(inner, (4.0 - 6.0 * distance**2 + 3.0 * distance**3) / 6.0,
                          remaining**3 / 6.0)
        radial_derivative = np.where(inner, -2.0 * distance + 1.5 * distance**2,
                                     -0.5 * remaining**2)
    else:
        remaining = np.maximum(1.5 - distance, 0.0)
        inner = distance < 0.5
        weight = np.where(inner, 0.75 - distance**2, 0.5 * remaining**2)
        radial_derivative = np.where(inner, -2.0 * distance, -remaining)
    return weight, radial_derivative * np.sign(delta)


def _summarize_fit(rows, column_count, h_m):
    coefficients, rank, singular_values, residual = map(np.array, zip(*rows))
    if np.any(rank < column_count):
        return MacPolynomialFit(None, None, None, rank, singular_values, residual)
    return MacPolynomialFit(
        coefficients, coefficients[:, 0], coefficients[:, 1:4] / h_m, rank, singular_values, residual
    )


def compare_mac_fit_degrees(position_m, h_m, face_value, *, spline_degree=3):
    """Return affine and quadratic weighted fits and a separate spline interpolation.

    The position is a finite three-vector in metres and h is finite and positive
    in metres. Their lattice coordinates must resolve signed int64 anchors and
    half-cell shifts in float64. The callback supplies a finite m/s value for
    each component and signed (x, y, z) face index, including selected donors
    with zero weight. A donor is anchored at h times its index plus half-cell
    shifts along the two transverse axes. A missing donor propagates KeyError.

    Both fits use the same samples and tensor weights. Degree three B-splines
    are the experiment kernel. Degree two is the rank-loss control. Each
    fit uses float64 least squares on sqrt(weight) times the monomial design,
    with NumPy's default numerical rank cutoff. The spline gradient separately
    differentiates the weighted interpolation in physical coordinates.
    """
    position_m = np.asarray(position_m, dtype=np.float64)
    if position_m.shape != (3,) or np.any(~np.isfinite(position_m)):
        raise ValueError("position must be a finite three-vector")
    if not np.isfinite(h_m) or h_m <= 0:
        raise ValueError("spacing must be finite and positive")
    if spline_degree not in (2, 3):
        raise ValueError("spline degree must be two or three")

    with np.errstate(over="ignore"):
        lattice_position = position_m / h_m
    # The largest lattice magnitude must leave room for every donor and its
    # half-cell offset below the binary64 unit-spacing boundary.
    maximum_lattice_magnitude = 2.0**52 - 4
    if np.any(~np.isfinite(lattice_position)) or np.any(np.abs(lattice_position) > maximum_lattice_magnitude):
        raise ValueError("position and spacing cannot resolve MAC half-cell support")
    local_indices = np.array(list(product(range(spline_degree + 1), repeat=3)), dtype=np.int64)
    affine_rows = []
    quadratic_rows = []
    spline_velocity = np.empty(3, dtype=np.float64)
    spline_gradient = np.empty((3, 3), dtype=np.float64)
    positive_weight_counts = np.empty(3, dtype=np.int64)
    for component in range(3):
        mac_shift = np.full(3, 0.5)
        mac_shift[component] = 0.0
        shifted_query = lattice_position - mac_shift
        first_index = np.floor(shifted_query - (spline_degree - 1) / 2).astype(np.int64)
        face_indices = first_index + local_indices
        delta = shifted_query - face_indices
        axis_weights, axis_derivatives = _basis(delta, spline_degree)
        weights = np.prod(axis_weights, axis=1)
        positive_weight_counts[component] = np.count_nonzero(weights)
        samples = np.array(
            [face_value(component, tuple(map(int, index))) for index in face_indices], dtype=np.float64
        )
        if np.any(~np.isfinite(samples)):
            raise ValueError("face samples must be finite")

        spline_velocity[component] = samples @ weights
        for axis in range(3):
            other_weights = axis_weights[:, (axis + 1) % 3] * axis_weights[:, (axis + 2) % 3]
            physical_derivative = axis_derivatives[:, axis] * other_weights / h_m
            spline_gradient[component, axis] = samples @ physical_derivative

        # Center and scale before forming the monomials so their magnitude
        # stays O(1) across resolution, including the second-order columns.
        x, y, z = -delta.T
        monomials = np.column_stack((np.ones_like(x), x, y, z, x*x, x*y, x*z, y*y, y*z, z*z))
        sqrt_weights = np.sqrt(weights)
        weighted_samples = sqrt_weights * samples
        weighted_design = sqrt_weights[:, None] * monomials
        for column_count, rows in ((4, affine_rows), (10, quadratic_rows)):
            design = weighted_design[:, :column_count]
            coefficients, _, rank, singular_values = np.linalg.lstsq(design, weighted_samples, rcond=None)
            residual = np.linalg.norm(design @ coefficients - weighted_samples)
            rows.append((coefficients, rank, singular_values, residual))

    return MacFitComparison(
        _summarize_fit(affine_rows, 4, h_m),
        _summarize_fit(quadratic_rows, 10, h_m),
        spline_velocity,
        spline_gradient,
        positive_weight_counts,
    )
