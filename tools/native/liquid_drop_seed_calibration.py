"""Independent support calibration for the frozen transported l=2 drop seed."""

from dataclasses import dataclass
import itertools
import math
import re

import numpy as np

from liquid_acceptance_metrics import SphereUnionSections, _adaptive_sections, finite_number, vector3


CELL_WIDTH_M = 0.025
PHYSICAL_RADIUS_M = 0.125
REFERENCE_VOLUME_M3 = 4 * math.pi * PHYSICAL_RADIUS_M**3 / 3
STRETCH_Y = 1.03
STRETCH_XZ = 1 / math.sqrt(STRETCH_Y)
SEED_COUNT = 552
REFERENCE_QUADRUPOLE_M2 = PHYSICAL_RADIUS_M**2 * (STRETCH_Y**2 - STRETCH_XZ**2) / 5


@dataclass(frozen=True)
class UnionAxisMeasure:
    volume_m3: float
    second_moment_m5: float
    estimated_volume_error_m3: float
    estimated_second_moment_error_m5: float


@dataclass(frozen=True)
class SupportTrial:
    support_radius_m: float
    x_axis_measure: UnionAxisMeasure


@dataclass(frozen=True)
class DropSeedCalibration:
    seed_digest: str
    support_radius_m: float
    bracket: tuple[SupportTrial, SupportTrial]
    trials: tuple[SupportTrial, ...]
    x_axis_measure: UnionAxisMeasure
    y_axis_measure: UnionAxisMeasure
    refined_x_axis_measure: UnionAxisMeasure
    refined_y_axis_measure: UnionAxisMeasure
    quadrupole_m2: float
    refined_quadrupole_m2: float
    estimated_quadrupole_error_m2: float
    refined_estimated_quadrupole_error_m2: float


def authenticated_drop_positions(seed, expected_digest):
    """Check a native read-seed receipt and return ordered, transformed centers in meters."""
    if (
        not isinstance(expected_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        or seed.get("state_digest") != expected_digest
        or seed.get("event") != "seed"
        or seed.get("fixture_version") != 1
    ):
        raise ValueError("drop seed receipt identity mismatch")
    if any(seed.get(key) != 0 for key in ("random_seed", "frame", "time_seconds", "substep", "accepted_step", "epoch")):
        raise ValueError("drop seed must start at zero clock and random seed")
    particles = seed.get("particles")
    if not isinstance(particles, list) or len(particles) != SEED_COUNT or seed.get("next_particle_id") != SEED_COUNT + 1:
        raise ValueError("drop seed must have 552 ordered particles")

    lattice = [(index + 0.5) * CELL_WIDTH_M for index in range(-5, 5)]
    expected_positions = [
        (x * STRETCH_XZ, y * STRETCH_Y, z * STRETCH_XZ)
        for z, y, x in itertools.product(lattice, repeat=3)
        if x*x + y*y + z*z < PHYSICAL_RADIUS_M**2
    ]
    positions = []
    reference_volume_m3 = REFERENCE_VOLUME_M3 / SEED_COUNT
    for expected_id, (particle, expected_position) in enumerate(zip(particles, expected_positions, strict=True), 1):
        if type(particle.get("id")) is not int or particle["id"] != expected_id:
            raise ValueError("drop seed particle IDs must be 1 through 552 in order")
        position = tuple(vector3(particle.get("world_position")))
        if any(abs(actual - expected) > math.ulp(expected) for actual, expected in zip(position, expected_position)):
            raise ValueError("drop seed position differs from the signed stretched lattice")
        if (
            particle.get("volume_ratio") != 1
            or particle.get("affine_enabled") is not True
            or particle.get("pic_initialized") is not False
            or particle.get("tag") != 0
            or particle.get("side_tag") != 0
            or particle.get("world_velocity") != [0, 0, 0]
            or particle.get("velocity_slopes") != [[0, 0, 0]] * 3
        ):
            raise ValueError("drop seed initial particle state mismatch")
        for key, expected in (("reference_volume", reference_volume_m3), ("mass", 1000 * reference_volume_m3)):
            value = particle.get(key)
            if not finite_number(value) or abs(value - expected) > math.ulp(expected):
                raise ValueError(f"drop seed particle {key} mismatch")
        positions.append(position)
    return positions


def _validate_symmetry(positions):
    centers = set(positions)
    if not centers or len(centers) != len(positions):
        raise ValueError("expected nonempty unique sphere centers")
    if any(
        tuple(-value if axis == reflected else value for axis, value in enumerate(point)) not in centers
        for point in centers for reflected in range(3)
    ):
        raise ValueError("sphere centers lack the three exact reflections")
    if any((z, y, x) not in centers for x, y, z in centers):
        raise ValueError("sphere centers lack exact x/z exchange")


def measure_symmetric_union(centers, support_radius_m, moment_scale_m, volume_tolerance_m3, *, axis=0):
    """Integrate volume and the chosen Cartesian second moment of a symmetric fixed-radius union.

    Exact symmetry centers the union at the origin and permits eight times
    the positive-octant integral. The existing quadrature operates on volume
    and normalized moment so both error estimates have volume units until
    the final scale conversion.
    """
    positions = [tuple(vector3(center)) for center in centers]
    _validate_symmetry(positions)
    if axis == 1:
        positions = [(y, x, z) for x, y, z in positions]
    elif axis != 0:
        raise ValueError("second moment axis must be x or y")
    if not finite_number(support_radius_m) or support_radius_m <= 0:
        raise ValueError("invalid sphere support radius")
    if not finite_number(moment_scale_m) or moment_scale_m <= 0:
        raise ValueError("invalid second moment scale")
    if not finite_number(volume_tolerance_m3) or volume_tolerance_m3 <= 0:
        raise ValueError("invalid quadrature tolerance")
    upper_x, upper_y, upper_z = [max(point[axis] for point in positions) + support_radius_m for axis in range(3)]
    scale_squared = moment_scale_m * moment_scale_m
    if not all(finite_number(bound) for bound in (upper_x, upper_y, upper_z, scale_squared)) or scale_squared == 0:
        raise ValueError("invalid finite integration bounds or second moment scale")
    octant_tolerance = volume_tolerance_m3 / 8
    section_tolerance = octant_tolerance / (3 * upper_x)
    if section_tolerance < math.ulp(support_radius_m * support_radius_m):
        raise ValueError("quadrature tolerance is below binary64 section resolution")

    geometry = SphereUnionSections(positions, support_radius_m, (0.0, upper_z), 0.0, upper_y)

    def sections(coordinates):
        areas = np.asarray([geometry.section(x, section_tolerance) for x in coordinates])
        weights = (coordinates / moment_scale_m)**2
        return tuple(np.column_stack((column, column * weights)) for column in areas[:, :, 0].T)

    integral, errors = _adaptive_sections(
        sections, 0.0, upper_x, geometry.x_breakpoints(), octant_tolerance, orders=(2, 4),
    )
    if (
        not np.all(np.isfinite(integral))
        or not np.all(np.isfinite(errors))
        or integral[0] <= 0
        or integral[1] < 0
        or np.any(errors < 0)
        or sum(errors) > octant_tolerance
    ):
        raise ValueError("sphere union quadrature did not meet the estimated error tolerance")
    return UnionAxisMeasure(
        volume_m3=float(8 * integral[0]),
        second_moment_m5=float(8 * scale_squared * integral[1]),
        estimated_volume_error_m3=float(8 * errors[0]),
        estimated_second_moment_error_m5=float(8 * scale_squared * errors[1]),
    )


def _bisect_support(positions, cell_width_m, physical_radius_m, target_volume_m3, *, max_iterations):
    if type(max_iterations) is not int or not 1 <= max_iterations <= 14:
        raise ValueError("support calibration allows 1 through 14 iterations")

    def trial(radius_m):
        measure = measure_symmetric_union(positions, radius_m, physical_radius_m, target_volume_m3 * 0.00125)
        return SupportTrial(radius_m, measure)

    lower_radius_m, upper_radius_m = 0.5 * cell_width_m, 0.9 * cell_width_m
    bracket = trial(lower_radius_m), trial(upper_radius_m)
    lower_measure, upper_measure = (endpoint.x_axis_measure for endpoint in bracket)
    if not (
        lower_measure.volume_m3 + lower_measure.estimated_volume_error_m3 < target_volume_m3
        < upper_measure.volume_m3 - upper_measure.estimated_volume_error_m3
    ):
        raise ValueError("support interval does not bracket the target volume")
    trials = []
    for _ in range(max_iterations):
        sample = trial((lower_radius_m + upper_radius_m) / 2)
        trials.append(sample)
        estimate = sample.x_axis_measure
        volume_difference_m3 = estimate.volume_m3 - target_volume_m3
        if abs(volume_difference_m3) + estimate.estimated_volume_error_m3 < target_volume_m3 * 0.001:
            return bracket, tuple(trials)
        if abs(volume_difference_m3) <= estimate.estimated_volume_error_m3:
            raise ValueError("support bisection cannot resolve the volume discrepancy")
        if volume_difference_m3 < 0:
            lower_radius_m = sample.support_radius_m
        else:
            upper_radius_m = sample.support_radius_m
    raise ValueError("support bisection did not converge within the iteration budget")


def _check_drop_moments(x, y, volume_fraction):
    for measurement in (x, y):
        volume_error = abs(measurement.volume_m3 - REFERENCE_VOLUME_M3) + measurement.estimated_volume_error_m3
        if volume_error > REFERENCE_VOLUME_M3 * volume_fraction:
            raise ValueError("drop volume differs from the physical sphere")
    if abs(x.volume_m3 - y.volume_m3) > x.estimated_volume_error_m3 + y.estimated_volume_error_m3:
        raise ValueError("drop volume estimates disagree after transposition")

    quadrupole_m2 = (y.second_moment_m5 - x.second_moment_m5) / x.volume_m3
    # Reflection gives zero centroid. X/Z exchange cancels the transverse
    # moments into I_x. Propagate the measured discrepancies through Q.
    estimated_error_m2 = (
        x.estimated_second_moment_error_m5 + y.estimated_second_moment_error_m5
        + abs(quadrupole_m2) * x.estimated_volume_error_m3
    ) / (x.volume_m3 - x.estimated_volume_error_m3)
    if abs(quadrupole_m2 - REFERENCE_QUADRUPOLE_M2) + estimated_error_m2 > 0.10 * REFERENCE_QUADRUPOLE_M2:
        raise ValueError("drop quadrupole differs from the analytic ellipsoid")
    return quadrupole_m2, estimated_error_m2


def calibrate_drop_seed(seed, expected_digest, *, max_iterations=14):
    """Fit support to the native seed volume and remeasure the selected binary64 radius."""
    positions = authenticated_drop_positions(seed, expected_digest)
    bracket, trials = _bisect_support(
        positions, CELL_WIDTH_M, PHYSICAL_RADIUS_M, REFERENCE_VOLUME_M3, max_iterations=max_iterations,
    )
    final_trial = trials[-1]
    radius_m = final_trial.support_radius_m
    x = final_trial.x_axis_measure
    y = measure_symmetric_union(positions, radius_m, PHYSICAL_RADIUS_M, REFERENCE_VOLUME_M3 * 0.00125, axis=1)
    quadrupole, quadrupole_error = _check_drop_moments(x, y, 0.005)
    refined_x = measure_symmetric_union(positions, radius_m, PHYSICAL_RADIUS_M, REFERENCE_VOLUME_M3 * 0.0003125)
    refined_y = measure_symmetric_union(
        positions, radius_m, PHYSICAL_RADIUS_M, REFERENCE_VOLUME_M3 * 0.0003125, axis=1,
    )
    refined_quadrupole, refined_quadrupole_error = _check_drop_moments(refined_x, refined_y, 0.0025)
    if any(
        abs(initial.volume_m3 - refined.volume_m3)
        + initial.estimated_volume_error_m3 + refined.estimated_volume_error_m3 > REFERENCE_VOLUME_M3 * 0.0025
        for initial, refined in ((x, refined_x), (y, refined_y))
    ):
        raise ValueError("drop volume changed beyond tolerance on refinement")
    return DropSeedCalibration(
        seed_digest=expected_digest,
        support_radius_m=radius_m,
        bracket=bracket,
        trials=trials,
        x_axis_measure=x,
        y_axis_measure=y,
        refined_x_axis_measure=refined_x,
        refined_y_axis_measure=refined_y,
        quadrupole_m2=quadrupole,
        refined_quadrupole_m2=refined_quadrupole,
        estimated_quadrupole_error_m2=quadrupole_error,
        refined_estimated_quadrupole_error_m2=refined_quadrupole_error,
    )
