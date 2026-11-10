"""Covariance ellipsoids for manufactured isolated liquid drops in SI units.

The fixed 1.04h particle graph is a component hypothesis for these drop clouds.
A connected graph alone cannot certify convexity, a filled interior, or isolation
from solid geometry. The reference leaves these preconditions to the caller.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class DropEllipsoid:
    """Fitted world-space center, ascending semiaxes (m), and column principal axes."""

    center: np.ndarray
    semiaxes: np.ndarray
    frame: np.ndarray

    def level_set(self, world_points):
        """Sample the volume-radius-scaled implicit field, negative inside, in meters.

        Accepts a point or an array with a final dimension of three. Its zero
        surface is exact for the fit. The magnitude is suitable for geometry
        comparisons and generally differs from the Euclidean signed distance.
        """
        principal_points = (np.asarray(world_points) - self.center) @ self.frame
        volume_radius = np.exp(np.log(self.semiaxes).mean())
        return volume_radius * (np.linalg.norm(principal_points / self.semiaxes, axis=-1) - 1)

    def signed_distance(self, world_points):
        """Return the Euclidean signed distance in meters, negative inside."""
        principal_points = (np.asarray(world_points) - self.center) @ self.frame
        scale = self.semiaxes[-1]
        axes = self.semiaxes / scale
        flat_points = np.abs(principal_points.reshape(-1, 3)) / scale
        distances = [_principal_signed_distance(point, axes) for point in flat_points]
        return scale * np.asarray(distances).reshape(principal_points.shape[:-1])

    def air_link_curvature(self, wet_center, air_center):
        """Return outward curvature (1/m) at the zero crossing of a wet-to-air segment.

        The computed start level must be strictly negative and the computed
        end level must be nonnegative. Raises ValueError for an unbracketed
        segment, including one whose analytic endpoint rounds inside.
        """
        wet = (np.asarray(wet_center) - self.center) @ self.frame
        air = (np.asarray(air_center) - self.center) @ self.frame
        direction = air - wet

        def level(fraction):
            return np.linalg.norm((wet + fraction * direction) / self.semiaxes) - 1

        if not (level(0) < 0 <= level(1)):
            raise ValueError("air link must bracket the ellipsoid from wet to air")
        fraction = brentq(level, 0, 1)
        inverse_squared_axes = (self.semiaxes[-1] / self.semiaxes) ** 2
        gradient = inverse_squared_axes * ((wet + fraction * direction) / self.semiaxes[-1])
        length = np.linalg.norm(gradient)
        normal = gradient / length
        tangential_trace = inverse_squared_axes @ (1 - normal**2)
        return tangential_trace / (length * self.semiaxes[-1])


def _principal_signed_distance(point, axes):
    # For a closest point y, y_i = a_i² x_i / (a_i² + lambda).
    # Shift lambda by the smallest squared axis to isolate the medial-axis pole.
    axes_squared = axes**2
    offsets = axes_squared - axes_squared[0]
    numerator = axes * point

    def surface_residual(shifted_multiplier):
        denominator = offsets + shifted_multiplier
        normalized_surface = np.divide(
            numerator, denominator, out=np.zeros(3), where=denominator > 0
        )
        return np.linalg.norm(normalized_surface) - 1

    inside = np.linalg.norm(point / axes) < 1
    lower = max(0, float(np.max(numerator - offsets))) if inside else axes_squared[0]
    upper = axes_squared[0] if inside else axes_squared[0] + 2 * np.linalg.norm(numerator)
    # Each normalized surface coordinate has magnitude at most one. A lower
    # bound with zero residual already gives the root, including axial queries.
    if surface_residual(lower) <= 0:
        shifted_multiplier = lower
    else:
        shifted_multiplier = brentq(surface_residual, lower, upper, xtol=np.finfo(float).tiny)

    denominator = offsets + shifted_multiplier
    normalized_surface = np.divide(
        numerator, denominator, out=np.zeros(3), where=denominator > 0
    )
    if shifted_multiplier == 0:
        # Coordinates in the shortest-axis subspace are zero in the query.
        # Fill any one of them to put the nearest point on the ellipsoid.
        normalized_surface[0] = np.sqrt(max(0, 1 - normalized_surface @ normalized_surface))
    distance = np.linalg.norm(axes * normalized_surface - point)
    return -distance if inside else distance


def _fit_component(points, volumes):
    material_volume = volumes.sum()
    weights = volumes / material_volume
    origin = points[0]
    offsets = points - origin
    center_offset = weights @ offsets
    centered = offsets - center_offset
    covariance = (centered.T * weights) @ centered
    variances, frame = np.linalg.eigh(covariance)
    if variances[0] <= np.finfo(float).eps * variances[-1]:
        raise ValueError("drop component must have full-rank covariance")

    axis_ratios = np.sqrt(variances / variances[-1])
    volume_radius = np.cbrt(material_volume) * np.cbrt(3 / (4 * np.pi))
    semiaxes = axis_ratios * (volume_radius / np.cbrt(axis_ratios.prod()))
    center = origin + center_offset
    for array in (center, semiaxes, frame):
        array.setflags(write=False)
    return DropEllipsoid(center, semiaxes, frame)


def fit_isolated_drops(particle_positions, material_volumes, cell_width):
    """Fit one volume-matched ellipsoid per connected component, in input order.

    Accepts finite world positions with shape (N, 3), one positive material
    volume (m³) per particle, and positive cell width (m). Returns an immutable
    tuple ordered by the first input particle of each component. Raises
    ValueError for invalid samples, degenerate fits, or fitted components
    whose separation cannot be certified along their center-to-center axis.
    """
    points = np.asarray(particle_positions, dtype=float)
    volumes = np.asarray(material_volumes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("expected nonempty particle positions with shape (N, 3)")
    if volumes.shape != (len(points),) or not np.all(np.isfinite(volumes) & (volumes > 0)):
        raise ValueError("expected one finite positive material volume per particle")
    if not np.all(np.isfinite(points)) or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("particle positions must be finite and distinct")
    if not np.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("cell width must be finite and positive")

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            pairs = cKDTree(points).query_pairs(1.04 * cell_width, output_type="ndarray")
            adjacency = csr_matrix((np.ones(len(pairs)), pairs.T), shape=(len(points), len(points)))
            count, labels = connected_components(adjacency, directed=False)
            drops = tuple(_fit_component(points[labels == i], volumes[labels == i]) for i in range(count))
            for left, right in combinations(drops, 2):
                separation = right.center - left.center
                center_distance = np.linalg.norm(separation)
                if center_distance == 0:
                    raise ValueError("fitted drop separation is ambiguous")
                direction = separation / center_distance
                left_support = np.linalg.norm(left.semiaxes * (left.frame.T @ direction))
                right_support = np.linalg.norm(right.semiaxes * (right.frame.T @ direction))
                if center_distance <= left_support + right_support:
                    raise ValueError("fitted drop separation is ambiguous")
        except FloatingPointError as error:
            raise ValueError("particle data exceed the numerical range of the drop fit") from error
    return drops
