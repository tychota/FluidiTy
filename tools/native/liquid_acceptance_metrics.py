"""Physical measurements from verified, frame-ordered free-fall points."""

from functools import lru_cache
import math
import sys

FREE_FALL_CASES = {"free-fall-pic": 0.0, "free-fall-default": 0.6, "free-fall-retained": 1.0}
FREE_FALL_LAST_FRAME = 6
FRAMES_PER_SECOND = 24
GRAVITY_M_PER_S2 = 9.81
MAX_RELATIVE_ACCELERATION_ERROR = 0.001
MAX_TRANSVERSE_SPEED_M_PER_S = 1e-10


def finite_number(value):
    """JSON booleans are not numeric observations."""
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


def vector3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(map(finite_number, value)):
        raise ValueError("expected a finite three-component vector")
    return value


def uniform_history_velocities(energy, stored_velocity, sampled_old_velocity, sampled_new_velocity):
    """Evaluate the scalar §5.1 equation with maximum liquid separation.

    In uniform free fall, partition of unity gives sampled_old_velocity equal to
    stored_velocity. Then the residual is zero, so the affine-in-energy stored
    path has equal endpoint values. The advecting path also has zero correction
    for any separation weight, including the maximum beta = energy.
    """
    if not finite_number(sampled_old_velocity):
        raise ValueError("consistent old history requires a finite old sample")
    if not finite_number(energy) or not 0 <= energy <= 1:
        raise ValueError("energy must be in [0, 1]")
    if not all(map(finite_number, (stored_velocity, sampled_new_velocity))):
        raise ValueError("expected finite stored and new velocities")
    residual = stored_velocity - sampled_old_velocity
    retained_velocity = 0.99 * energy * residual
    return sampled_new_velocity + retained_velocity, sampled_new_velocity + energy * retained_velocity


def evaluate_free_fall(frame_samples):
    """Measure each stored-velocity interval and report position change separately.

    Samples must cover exactly frames 0 through 6, with the single stable ID 1
    at the frozen seed in frame zero. The input's accepted clocks are used for
    velocity differences, after checking them against the 24 fps schedule.
    Malformed observations raise ValueError. Valid observations outside the
    physical tolerances return their measurements with passed=False.
    """
    if len(frame_samples) != FREE_FALL_LAST_FRAME + 1:
        raise ValueError("free fall requires all seven committed frames")
    positions = []
    velocities = []
    times = []
    for ordinal, sample in enumerate(frame_samples):
        if not isinstance(sample, dict) or type(sample.get("frame")) is not int or sample["frame"] != ordinal:
            raise ValueError(f"unexpected frame ordinal at sample {ordinal}")
        time = sample.get("time_seconds")
        if not finite_number(time) or time != ordinal / FRAMES_PER_SECOND:
            raise ValueError(f"wrong accepted time at frame {ordinal}")
        points = sample.get("points")
        if not isinstance(points, list) or len(points) != 1 or not isinstance(points[0], dict):
            raise ValueError(f"expected exactly one point at frame {ordinal}")
        point = points[0]
        if type(point.get("id")) is not int or point["id"] != 1:
            raise ValueError(f"wrong particle ID at frame {ordinal}")
        times.append(time)
        positions.append(vector3(point.get("world_position")))
        velocities.append(vector3(point.get("world_velocity")))
    if tuple(positions[0]) != (2.5, 2.5, 2.5) or tuple(velocities[0]) != (0, 0, 0):
        raise ValueError("frame zero does not match the frozen free-fall seed")

    intervals = []
    for end_frame in range(1, len(times)):
        start_frame = end_frame - 1
        duration = times[end_frame] - times[start_frame]
        acceleration = (velocities[end_frame][1] - velocities[start_frame][1]) / duration
        relative_error = abs(acceleration + GRAVITY_M_PER_S2) / GRAVITY_M_PER_S2
        interval_displacement = [end - start for start, end in zip(positions[start_frame], positions[end_frame])]
        if not all(map(finite_number, (acceleration, relative_error, *interval_displacement))):
            raise ValueError(f"nonfinite measurement at frame {end_frame}")
        intervals.append({
            "end_frame": end_frame,
            "duration_s": duration,
            "stored_acceleration_y_m_per_s2": acceleration,
            "relative_acceleration_error": relative_error,
            "world_displacement_m": interval_displacement,
        })

    displacement = [end - start for start, end in zip(positions[0], positions[-1])]
    transverse_speed = max(math.hypot(velocity[0], velocity[2]) for velocity in velocities)
    if not all(map(finite_number, (transverse_speed, *displacement))):
        raise ValueError("nonfinite transverse or displacement measurement")
    maximum_acceleration_error = max(interval["relative_acceleration_error"] for interval in intervals)
    return {
        "passed": (
            maximum_acceleration_error <= MAX_RELATIVE_ACCELERATION_ERROR
            and transverse_speed <= MAX_TRANSVERSE_SPEED_M_PER_S
        ),
        "gravity_m_per_s2": GRAVITY_M_PER_S2,
        "maximum_relative_acceleration_error": maximum_acceleration_error,
        "maximum_transverse_speed_m_per_s": transverse_speed,
        "world_displacement_m": displacement,
        "intervals": intervals,
    }


HYDROSTATIC_CASE = "hydrostatic-column"
PISTON_CASE = "open-tank-piston"
PISTON_REFINED_CASE = "open-tank-piston-refined"
TANK_CASES = {
    HYDROSTATIC_CASE: (240, 128, 0.01, (230, 240)),
    PISTON_CASE: (48, 1024, 0.01, (30, 40)),
    PISTON_REFINED_CASE: (48, 1024, 0.005, (30, 40)),
}
VOLUME_DRIFT_LIMIT = 0.01
PISTON_MISMATCH_LIMIT = 0.01
PISTON_REFINEMENT_LIMIT = 0.0025
PISTON_DENOMINATOR_FLOOR_M3 = 1e-8
HYDROSTATIC_SPEED_LIMIT_M_PER_S = 1e-3


def check_frame_time(sample, ordinal):
    time_seconds = sample.get("time_seconds")
    if type(sample.get("frame")) is not int or sample["frame"] != ordinal:
        raise ValueError(f"wrong frame ordinal at sample {ordinal}")
    if not finite_number(time_seconds) or time_seconds != ordinal / FRAMES_PER_SECOND:
        raise ValueError(f"wrong accepted time at frame {ordinal}")


def check_ids(records, particle_count, description):
    # Accepted particles use spatial order; stable IDs identify the material.
    # Sorting the inventory for comparison never changes the raw record order.
    if not isinstance(records, list) or len(records) != particle_count:
        raise ValueError(f"wrong complete particle ID inventory in {description}")
    ids = [record.get("id") for record in records if isinstance(record, dict)]
    if not all(type(value) is int for value in ids) or sorted(ids) != list(range(1, particle_count + 1)):
        raise ValueError(f"wrong complete particle ID inventory in {description}")


def evaluate_volume_budget(frame_samples, expected_flux_samples):
    """Compare every frame's point-derived material volume with its accounted target.

    Expected flux entries are signed material volume entering the tank during
    each frame interval, with zero at frame zero. Moving solid displacement is
    a separate quantity. The declared liquid has zero sources and compression.
    """
    if not frame_samples or len(expected_flux_samples) != len(frame_samples):
        raise ValueError("volume budget requires one flux observation per frame")
    if expected_flux_samples[0] != 0:
        raise ValueError("material flux before the initial frame must be zero")
    particle_count = len(frame_samples[0]["points"])
    if not particle_count:
        raise ValueError("volume budget requires seed particles")
    volumes = []
    boundary_volumes = []
    for ordinal, (sample, boundary_volume) in enumerate(zip(frame_samples, expected_flux_samples)):
        check_frame_time(sample, ordinal)
        if not finite_number(boundary_volume):
            raise ValueError(f"nonfinite material boundary volume at frame {ordinal}")
        for field in ("source_volume_m3", "compression_volume_m3"):
            if not finite_number(sample.get(field)) or sample[field] != 0:
                raise ValueError(f"undeclared {field} at frame {ordinal}")
        points = sample.get("points")
        check_ids(points, particle_count, f"frame {ordinal}")
        radii = [point.get("radius") for point in points]
        if any(not finite_number(radius) or radius <= 0 for radius in radii):
            raise ValueError(f"invalid material radius at frame {ordinal}")
        ratios = []
        for point in points:
            vector3(point.get("world_position"))
            vector3(point.get("world_velocity"))
            ratio = point.get("volume_ratio")
            if not finite_number(ratio) or ratio <= 0:
                raise ValueError(f"invalid volume ratio at frame {ordinal}, ID {point['id']}")
            ratios.append(ratio)
        material_volume = math.fsum(4 * math.pi / 3 * radius ** 3 for radius in radii)
        if not finite_number(material_volume) or material_volume <= 0:
            raise ValueError(f"invalid material volume at frame {ordinal}")
        boundary_volumes.append(boundary_volume)
        cumulative_boundary_volume = math.fsum(boundary_volumes)
        target_volume = (volumes[0]["point_material_volume_m3"] if volumes else material_volume)
        target_volume += cumulative_boundary_volume
        if not finite_number(target_volume) or target_volume <= 0:
            raise ValueError(f"invalid accounted target volume at frame {ordinal}")
        volumes.append({
            "frame": ordinal,
            "point_material_volume_m3": material_volume,
            "minimum_numerical_volume_ratio": min(ratios),
            "maximum_numerical_volume_ratio": max(ratios),
            "cumulative_material_boundary_volume_m3": cumulative_boundary_volume,
            "source_volume_m3": sample["source_volume_m3"],
            "compression_volume_m3": sample["compression_volume_m3"],
            "target_volume_m3": target_volume,
            "relative_drift": abs(material_volume - target_volume) / target_volume,
        })
    worst = max(volumes, key=lambda frame: frame["relative_drift"])
    return {
        "passed": worst["relative_drift"] <= VOLUME_DRIFT_LIMIT,
        "particle_count": particle_count,
        "maximum_relative_drift": worst["relative_drift"],
        "maximum_drift_frame": worst["frame"],
        "frames": volumes,
    }


def hydrostatic_pressure(checkpoint):
    # Signed (z,y,x) lattice centers. The fixed plane closes the pressure
    # domain at y=1 m; pressure-cell centers terminate at 0.9375 m.
    expected = [(x, y, z) for z in range(-2, 2) for y in range(8) for x in range(-2, 2)]
    cells = checkpoint.get("pressure_cells")
    if not isinstance(cells, list) or len(cells) != len(expected):
        raise ValueError("wrong hydrostatic pressure-cell inventory")
    bands = [[] for _ in range(8)]
    for cell, index in zip(cells, expected):
        pressure = cell.get("pressure_pa")
        if cell.get("cell_index") != list(index) or cell.get("occupancy") != 1 or not finite_number(pressure):
            raise ValueError("invalid hydrostatic pressure cell")
        y = index[1]
        depth_m = 1 - (y + 0.5) / 8
        expected_pressure = 1000 * GRAVITY_M_PER_S2 * depth_m
        bands[y].append(pressure - expected_pressure)
    return {
        "frame": checkpoint["frame"],
        "ambient_gauge_pa": 0,
        "inferred_surface_gauge_from_top_cells_pa": [
            min(bands[-1]), max(bands[-1]),
        ],
        "depth_bands": [{
            "depth_m": 1 - (y + 0.5) / 8,
            "expected_pressure_pa": 1000 * GRAVITY_M_PER_S2 * (1 - (y + 0.5) / 8),
            "mean_signed_error_pa": math.fsum(errors) / len(errors),
            "max_absolute_error_pa": max(map(abs, errors)),
        } for y, errors in enumerate(bands)],
    }


def evaluate_hydrostatic_column(frame_samples, checkpoint_samples):
    """Evaluate the 10 s column and report both retained pressure profiles."""
    last_frame, particle_count, _, retained_frames = TANK_CASES[HYDROSTATIC_CASE]
    if len(frame_samples) != last_frame + 1 or len(checkpoint_samples) != len(retained_frames):
        raise ValueError("hydrostatic column requires frames 0–240 and checkpoints 230 and 240")
    check_ids(frame_samples[0].get("points"), particle_count, "hydrostatic seed")
    volume = evaluate_volume_budget(frame_samples, [0] * len(frame_samples))
    speed = max(math.hypot(*point["world_velocity"]) for point in frame_samples[48]["points"])
    pressures = []
    for ordinal, checkpoint in zip(retained_frames, checkpoint_samples):
        check_frame_time(checkpoint, ordinal)
        pressures.append(hydrostatic_pressure(checkpoint))
    speed_passed = speed < HYDROSTATIC_SPEED_LIMIT_M_PER_S
    return {
        "passed": speed_passed and volume["passed"],
        "frame_48_time_seconds": frame_samples[48]["time_seconds"],
        "frame_48_maximum_speed_m_per_s": speed,
        "speed_passed": speed_passed,
        "volume_budget": volume,
        "pressure": pressures,
    }


def _merged_interval_lengths(lower, upper, floor_y, reference_y):
    import numpy as np

    lower = np.maximum(floor_y, lower)
    upper = np.maximum(floor_y, upper)
    order = np.argsort(lower, axis=1, kind="stable")
    lower = np.take_along_axis(lower, order, axis=1)
    upper = np.take_along_axis(upper, order, axis=1)
    merged_end = np.maximum.accumulate(upper, axis=1)
    lower[:, 1:] = np.maximum(lower[:, 1:], merged_end[:, :-1])
    below = np.maximum(0, np.minimum(upper, reference_y) - lower)
    above = np.maximum(0, upper - np.maximum(lower, reference_y))
    return np.column_stack((np.sum(below, axis=1), np.sum(above, axis=1)))


def split_interval_lengths(intervals, floor_y, reference_y):
    """Return union lengths in [floor, reference] and above reference."""
    import numpy as np

    bounds = np.asarray(intervals, dtype=float).reshape(-1, 2)
    result = _merged_interval_lengths(bounds[None, :, 0], bounds[None, :, 1], floor_y, reference_y)
    return tuple(map(float, result[0]))


def _adaptive_sections(evaluate, lower, upper, breakpoints, tolerance, *, orders=(4, 8)):
    """Integrate two independent lengths and retain summed local error estimates.

    The Gauss rules share no nodes. Refit intervals contributing the largest
    discrepancies and combine both quadrature and nested-section errors.
    """
    import numpy as np

    if lower == upper:
        return np.zeros(2), np.zeros(2)
    boundaries = np.unique([lower, *(value for value in breakpoints if lower < value < upper), upper])
    intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    quadrature = _gauss_rules(orders)

    def estimates(intervals):
        midpoints = np.mean(intervals, axis=1)
        half_widths = (intervals[:, 1] - intervals[:, 0]) / 2
        estimates_by_order = []
        errors_by_order = []
        for nodes, weights in quadrature:
            positions = midpoints[:, None] + half_widths[:, None] * nodes
            values, nested_errors = evaluate(positions.ravel())
            integral = np.einsum("j,ijk->ik", weights, values.reshape(len(intervals), len(nodes), 2))
            estimates_by_order.append(half_widths[:, None] * integral)
            nested_errors = nested_errors.reshape(len(intervals), len(nodes), 2)
            errors_by_order.append(2 * half_widths[:, None] * np.max(nested_errors, axis=1))
        coarse, fine = estimates_by_order
        nested = np.maximum(*errors_by_order)
        return fine, np.abs(fine - coarse), nested

    values, local_errors, nested = estimates(intervals)
    for _ in range(20):
        total_error = np.sum(local_errors + nested, axis=0)
        if np.sum(total_error) <= tolerance or len(intervals) >= 32768:
            break
        interval_error = np.sum(local_errors, axis=1)
        order = np.argsort(-interval_error, kind="stable")
        cumulative_error = np.cumsum(interval_error[order])
        if cumulative_error[-1] == 0:
            break
        refine_count = np.searchsorted(cumulative_error, cumulative_error[-1] * 0.7) + 1
        refine = np.zeros(len(intervals), dtype=bool)
        refine[order[:refine_count]] = True
        selected = intervals[refine]
        midpoints = np.mean(selected, axis=1)
        split = np.concatenate((
            np.column_stack((selected[:, 0], midpoints)),
            np.column_stack((midpoints, selected[:, 1])),
        ))
        new_values, new_errors, new_nested = estimates(split)
        intervals = np.concatenate((intervals[~refine], split))
        values = np.concatenate((values[~refine], new_values))
        local_errors = np.concatenate((local_errors[~refine], new_errors))
        nested = np.concatenate((nested[~refine], new_nested))
    return np.sum(values, axis=0), np.sum(local_errors + nested, axis=0)


# Cache the deterministic rules, shared by the two spatial passes and all steps.
@lru_cache(maxsize=2)
def _gauss_rules(orders):
    from numpy.polynomial.legendre import leggauss

    return tuple(leggauss(order) for order in orders)


class SphereUnionSections:
    """Independent tank-clipped cross-sections of a fixed-radius sphere union."""

    def __init__(self, positions, radius, z_bounds, floor_y, reference_y):
        import numpy as np
        from scipy.spatial import cKDTree

        if not finite_number(radius) or radius <= 0 or not finite_number(radius * radius) or radius * radius == 0:
            raise ValueError("invalid sphere-section radius")
        if not all(map(finite_number, (*z_bounds, floor_y, reference_y))):
            raise ValueError("nonfinite sphere-section bounds")
        if not z_bounds[0] < z_bounds[1] or not floor_y < reference_y:
            raise ValueError("invalid sphere-section bounds")
        coordinates = np.asarray([vector3(position) for position in positions], dtype=float).reshape(-1, 3)
        # Coincident fixed-radius spheres are one geometric support.
        self.positions = np.unique(coordinates, axis=0)
        self.radius = radius
        self.floor_y = floor_y
        self.reference_y = reference_y
        self.z_bounds = z_bounds

        # Only overlapping neighbors can contribute an intersection circle.
        diameter = 2 * radius
        first, second = cKDTree(self.positions).query_pairs(diameter, output_type="ndarray").T
        offsets = self.positions[second] - self.positions[first]
        transverse = np.hypot(offsets[:, 1], offsets[:, 2])
        distance = np.hypot(offsets[:, 0], transverse)
        overlaps = (distance > 0) & (distance < diameter)
        self._pair_first = first[overlaps]
        self._pair_second = second[overlaps]
        self._pair_offset = offsets[overlaps]
        self._pair_transverse = transverse[overlaps]
        distance = distance[overlaps]
        circle_radius = np.sqrt((radius - distance / 2) * (radius + distance / 2))
        x_center = self.positions[self._pair_first, 0] + self._pair_offset[:, 0] / 2
        x_span = circle_radius * (self._pair_transverse / distance)
        self._pair_low_x = x_center - x_span
        self._pair_high_x = x_center + x_span

    def x_breakpoints(self):
        # Project tangencies with the sphere, each clipping plane, and
        # each y/z clipping edge onto x. A clipped cap can fit entirely
        # between the quadrature nodes on the sphere's full support.
        import numpy as np

        center_x, center_y, center_z = self.positions.T
        radius_sq = np.full_like(center_x, self.radius ** 2)
        y_distance_sq = (0, (center_y - self.floor_y) ** 2, (center_y - self.reference_y) ** 2)
        z_low, z_high = self.z_bounds
        z_distance_sq = (0, (center_z - z_low) ** 2, (center_z - z_high) ** 2)
        for dy_sq in y_distance_sq:
            for dz_sq in z_distance_sq:
                support_sq = radius_sq - dy_sq - dz_sq
                intersects = support_sq > 0
                span = np.sqrt(support_sq[intersects])
                yield from center_x[intersects] - span
                yield from center_x[intersects] + span

        yield from self._pair_low_x
        yield from self._pair_high_x

    def _pair_crossings_z(self, x, cut_radius_sq):
        # A fixed x plane can cross a pair intersection only inside its
        # projected x interval. Concentric cuts have no isolated crossings.
        import numpy as np

        active = (self._pair_low_x < x) & (x < self._pair_high_x) & (self._pair_transverse > 0)
        first = self._pair_first[active]
        second = self._pair_second[active]
        radii_first = np.sqrt(np.maximum(0, cut_radius_sq[first]))
        radii_second = np.sqrt(np.maximum(0, cut_radius_sq[second]))
        distance = self._pair_transverse[active]
        radius_difference = radii_first - radii_second
        radius_sum = radii_first + radii_second
        crosses = (np.abs(radius_difference) < distance) & (distance < radius_sum)
        distance = distance[crosses]
        if not len(distance):
            return ()

        # The circle-circle chord is perpendicular to the center offset.
        # Factoring the squared radius difference avoids cancellation when
        # the cross-sections have similar radii.
        along = (distance + radius_difference[crosses] / distance * radius_sum[crosses]) / 2
        radius = radii_first[crosses]
        half_chord = np.sqrt(np.maximum(0, (radius - along) * (radius + along)))
        center_yz = self.positions[first[crosses], 1:]
        direction_yz = self._pair_offset[active, 1:][crosses] / distance[:, None]
        chord_center = center_yz + along[:, None] * direction_yz
        chord_offset = half_chord[:, None] * direction_yz[:, [1, 0]] * [-1, 1]
        crossings = []
        for endpoints in (chord_center - chord_offset, chord_center + chord_offset):
            y, z = endpoints.T
            visible = (y >= self.floor_y) & (self.z_bounds[0] < z) & (z < self.z_bounds[1])
            crossings.extend(z[visible])
        return crossings

    def section(self, x, tolerance):
        import numpy as np

        x_cut_radius_sq = self.radius ** 2 - (x - self.positions[:, 0]) ** 2
        active = x_cut_radius_sq > 0
        center_y = self.positions[active, 1]
        center_z = self.positions[active, 2]
        disk_radius_sq = x_cut_radius_sq[active]
        if not len(center_y):
            return (0.0, 0.0), (0.0, 0.0)
        z_min, z_max = self.z_bounds
        disk_radius = np.sqrt(disk_radius_sq)
        pair_crossings = self._pair_crossings_z(x, x_cut_radius_sq)
        edges = np.linspace(z_min, z_max, math.ceil((z_max - z_min) / self.radius) + 1)
        area = np.zeros(2)
        error = np.zeros(2)
        for low_z, high_z in zip(edges[:-1], edges[1:]):
            candidates = (center_z + disk_radius > low_z) & (center_z - disk_radius < high_z)
            if not np.any(candidates):
                continue
            ys = center_y[candidates]
            zs = center_z[candidates]
            radii_sq = disk_radius_sq[candidates]
            # A cut support starts/ends either at the disk boundary or where a
            # sphere becomes tangent to the floor or the reference plane.
            # Pair crossings change which vertical intervals overlap.
            breakpoints = [z for z in pair_crossings if low_z < z < high_z]
            for height_sq in (0, (ys - self.floor_y) ** 2, (ys - self.reference_y) ** 2):
                support_sq = radii_sq - height_sq
                intersects = support_sq > 0
                span = np.sqrt(support_sq[intersects])
                breakpoints.extend(zs[intersects] - span)
                breakpoints.extend(zs[intersects] + span)

            def evaluate(z):
                depth_sq = radii_sq[None, :] - (z[:, None] - zs[None, :]) ** 2
                height = np.sqrt(np.maximum(0, depth_sq))
                lengths = _merged_interval_lengths(ys[None, :] - height, ys[None, :] + height,
                                                   self.floor_y, self.reference_y)
                return lengths, np.zeros_like(lengths)

            tile_tolerance = tolerance * (high_z - low_z) / (z_max - z_min)
            result, tile_error = _adaptive_sections(evaluate, low_z, high_z, breakpoints, tile_tolerance)
            area += result
            error += tile_error
        return tuple(map(float, area)), tuple(map(float, error))


def integrate_piston_sweep(sections, start_x, end_x, breakpoints, tolerance):
    """Integrate signed volume below and above reference along a swept x interval."""
    import numpy as np

    if not finite_number(tolerance) or tolerance <= 0 or not all(map(finite_number, (start_x, end_x))):
        raise ValueError("invalid piston sweep extent or integration tolerance")
    lower, upper = sorted((start_x, end_x))
    sign = 1 if end_x >= start_x else -1

    def evaluate(x):
        values, errors = zip(*(sections(value) for value in x))
        return np.asarray(values), np.asarray(errors)

    volumes, errors = _adaptive_sections(evaluate, lower, upper, breakpoints, tolerance, orders=(2, 4))
    return {
        "swept_below_m3": sign * float(volumes[0]),
        "swept_above_m3": sign * float(volumes[1]),
        "swept_below_estimated_error_m3": float(errors[0]),
        "swept_above_estimated_error_m3": float(errors[1]),
    }


def compare_piston_estimates(receiving_m3, swept_wet_m3, wall_flux_m3, solver_flux_m3):
    """Compare all pairs relative to the independently measured wet wall flux."""
    denominator = max(abs(wall_flux_m3), PISTON_DENOMINATOR_FLOOR_M3)
    comparisons = {
        "receiving_to_wall_relative_mismatch": abs(receiving_m3 - wall_flux_m3) / denominator,
        "swept_to_wall_relative_mismatch": abs(swept_wet_m3 - wall_flux_m3) / denominator,
        "solver_to_wall_relative_mismatch": abs(solver_flux_m3 - wall_flux_m3) / denominator,
        "receiving_to_solver_relative_mismatch": abs(receiving_m3 - solver_flux_m3) / denominator,
        "swept_to_solver_relative_mismatch": abs(swept_wet_m3 - solver_flux_m3) / denominator,
        "receiving_to_swept_relative_mismatch": abs(receiving_m3 - swept_wet_m3) / denominator,
    }
    return {
        "passed": max(comparisons.values()) <= PISTON_MISMATCH_LIMIT,
        "denominator_m3": denominator,
        **comparisons,
    }


PISTON_INITIAL_X_M = -0.5
PISTON_TANK_HIGH_X_M = 0.5
PISTON_TANK_Z_M = (-0.25, 0.25)
PISTON_REFERENCE_Y_M = 0.5
PISTON_VELOCITY_M_PER_S = 0.015625
PISTON_PARTICLE_RADIUS_M = 0.9 * 0.0625
PISTON_STROKE_M = 0.03125
PISTON_REFERENCE_SECTION_M2 = 0.25


def verify_piston_step(record, previous, maximum_dt):
    step = record.get("accepted_step")
    dt = record.get("accepted_dt_seconds")
    if (type(step) is not int or step != previous["accepted_step"] + 1
            or not finite_number(dt) or not 0 < dt <= maximum_dt):
        raise ValueError("wrong piston step ordinal or accepted duration")
    end_time = record.get("time_seconds")
    start_time = previous["time_seconds"]
    if not finite_number(end_time) or end_time <= start_time or not math.isclose(
            start_time + dt, end_time, rel_tol=0, abs_tol=math.ulp(end_time)):
        raise ValueError(f"wrong accepted piston time at step {step}")
    expected_start = math.fma(PISTON_VELOCITY_M_PER_S, start_time, PISTON_INITIAL_X_M)
    expected_end = math.fma(PISTON_VELOCITY_M_PER_S, dt, expected_start)
    if record.get("step_start_piston_low_x_m") != expected_start or record.get("step_end_piston_low_x_m") != expected_end:
        raise ValueError(f"wrong piston pose at step {step}")
    if not finite_number(record.get("step_start_inward_piston_flux_m3_per_s")):
        raise ValueError(f"missing signed solver piston flux at step {step}")


def _piston_union(positions):
    return SphereUnionSections(
        positions, PISTON_PARTICLE_RADIUS_M, PISTON_TANK_Z_M, 0, PISTON_REFERENCE_Y_M,
    )


def _piston_volume(positions, start_x, end_x, tolerance):
    geometry = _piston_union(positions)
    section_tolerance = tolerance / (2 * abs(end_x - start_x))
    return integrate_piston_sweep(
        lambda x: geometry.section(x, section_tolerance),
        start_x, end_x, geometry.x_breakpoints(), tolerance,
    )


def _piston_spatial_pass(frame_samples, steps, tolerance, *, progress):
    # Reserve half the error allowance for endpoint reconstructions and half
    # for the accumulated wall and swept volumes. The tighter pass is separate.
    volume_tolerance = tolerance / 4
    area_tolerance = tolerance / (4 * PISTON_STROKE_M)
    first_x = PISTON_INITIAL_X_M
    final_x = math.fma(PISTON_VELOCITY_M_PER_S, frame_samples[-1]["time_seconds"], PISTON_INITIAL_X_M)
    initial = _piston_volume(steps[0]["start_positions"], first_x, PISTON_TANK_HIGH_X_M, volume_tolerance)
    if progress:
        progress({"event": "endpoint", "endpoint": "initial", "tolerance_m3": tolerance, **initial})
    final_positions = [point["world_position"] for point in frame_samples[-1]["points"]]
    final = _piston_volume(final_positions, final_x, PISTON_TANK_HIGH_X_M, volume_tolerance)
    if progress:
        progress({"event": "endpoint", "endpoint": "final", "tolerance_m3": tolerance, **final})

    step_results = []
    for step in steps:
        geometry = _piston_union(step["start_positions"])
        start_x = step["step_start_piston_low_x_m"]
        end_x = step["step_end_piston_low_x_m"]
        stroke = PISTON_VELOCITY_M_PER_S * step["accepted_dt_seconds"]
        (below, above), area_error = geometry.section(start_x, area_tolerance)
        sweep = integrate_piston_sweep(
            lambda x: geometry.section(x, area_tolerance / 2),
            start_x, end_x, geometry.x_breakpoints(), stroke * area_tolerance,
        )
        result = {
            "accepted_step": step["accepted_step"],
            "time_seconds": step["time_seconds"],
            "accepted_dt_seconds": step["accepted_dt_seconds"],
            "step_start_piston_low_x_m": start_x,
            "step_end_piston_low_x_m": end_x,
            "wet_area_below_m2": below,
            "wet_area_above_m2": above,
            "wall_below_m3": below * stroke,
            "wall_above_m3": above * stroke,
            "wall_estimated_error_m3": math.fsum(area_error) * stroke,
            "solver_inward_flux_m3": step["step_start_inward_piston_flux_m3_per_s"] * step["accepted_dt_seconds"],
            **sweep,
        }
        step_results.append(result)
        if progress:
            progress({"event": "step", "tolerance_m3": tolerance, **result})

    totals = {
        key: math.fsum(step[key] for step in step_results)
        for key in ("wall_below_m3", "wall_above_m3", "wall_estimated_error_m3", "solver_inward_flux_m3",
                    "swept_below_m3", "swept_above_m3",
                    "swept_below_estimated_error_m3", "swept_above_estimated_error_m3")
    }
    wall_m3 = totals["wall_below_m3"] + totals["wall_above_m3"]
    swept_wet_m3 = totals["swept_below_m3"] + totals["swept_above_m3"]
    receiving_m3 = final["swept_above_m3"] - initial["swept_above_m3"] + totals["swept_above_m3"]
    initial_void_m3 = (PISTON_TANK_HIGH_X_M - first_x) * PISTON_REFERENCE_SECTION_M2 - initial["swept_below_m3"]
    final_void_m3 = (PISTON_TANK_HIGH_X_M - final_x) * PISTON_REFERENCE_SECTION_M2 - final["swept_below_m3"]
    observed_stroke_m = math.fsum(step["step_end_piston_low_x_m"] - step["step_start_piston_low_x_m"] for step in steps)
    full_below_sweep_m3 = observed_stroke_m * PISTON_REFERENCE_SECTION_M2
    swept_dry_m3 = full_below_sweep_m3 - totals["swept_below_m3"]
    below_contribution_m3 = math.fsum(map(abs, (initial_void_m3, final_void_m3, swept_dry_m3)))
    spatial_error_m3 = math.fsum(
        section[key] for section in (initial, final, totals)
        for key in ("swept_below_estimated_error_m3", "swept_above_estimated_error_m3")
    ) + totals["wall_estimated_error_m3"]
    comparison = compare_piston_estimates(receiving_m3, swept_wet_m3, wall_m3, totals["solver_inward_flux_m3"])
    return {
        **totals,
        **comparison,
        "initial_reconstruction": initial,
        "final_reconstruction": final,
        "full_wall_flux_m3": wall_m3,
        "full_wetted_sweep_m3": swept_wet_m3,
        "receiving_m3": receiving_m3,
        "initial_void_below_reference_m3": initial_void_m3,
        "final_void_below_reference_m3": final_void_m3,
        "swept_dry_below_reference_m3": swept_dry_m3,
        "full_below_reference_sweep_m3": full_below_sweep_m3,
        "below_reference_contribution_m3": below_contribution_m3,
        "below_reference_contribution_relative": below_contribution_m3 / comparison["denominator_m3"],
        "spatial_estimated_error_m3": spatial_error_m3,
        "requested_spatial_error_m3": tolerance,
        "steps": step_results,
    }


PISTON_COMPARISON_VOLUMES = (
    "receiving_m3", "full_wall_flux_m3", "full_wetted_sweep_m3", "solver_inward_flux_m3",
)
PISTON_SPATIAL_VOLUMES = (
    "wall_below_m3", "wall_above_m3", "swept_below_m3", "swept_above_m3", "receiving_m3",
    "initial_void_below_reference_m3", "final_void_below_reference_m3", "swept_dry_below_reference_m3",
)


def evaluate_piston_displacement(frame_samples, accepted_step_samples, *, maximum_dt=0.01, progress=None):
    """Evaluate the independently reconstructed 2 s piston at two spatial tolerances.

    Each step sample contains the exact step-start centers recovered from
    the verified initial state and previous accepted trace, plus the current
    signed solver flux, clock, duration and start/end wall poses.
    """
    last_frame, particle_count, _, _ = TANK_CASES[PISTON_CASE]
    if len(frame_samples) != last_frame + 1 or not accepted_step_samples:
        raise ValueError("piston requires frames 0–48 and the accepted step inventory")
    if maximum_dt not in (0.01, 0.005):
        raise ValueError("wrong piston maximum dt")
    check_ids(frame_samples[0].get("points"), particle_count, "piston seed")
    volume = evaluate_volume_budget(frame_samples, [0] * len(frame_samples))
    previous = frame_samples[0]
    if previous.get("accepted_step") != 0:
        raise ValueError("wrong initial piston clock")
    for step in accepted_step_samples:
        verify_piston_step(step, previous, maximum_dt)
        if len(step["start_positions"]) != particle_count:
            raise ValueError("wrong step-start piston particle count")
        previous = step
    if previous["accepted_step"] != frame_samples[-1]["accepted_step"] or previous["time_seconds"] != 2:
        raise ValueError("piston steps do not reach the last verified frame")
    tolerance = PISTON_STROKE_M * PISTON_REFERENCE_SECTION_M2 * 0.0005
    coarse = _piston_spatial_pass(frame_samples, accepted_step_samples, tolerance, progress=progress)
    fine = _piston_spatial_pass(frame_samples, accepted_step_samples, tolerance / 4, progress=progress)
    spatial_change_m3 = math.fsum(abs(fine[key] - coarse[key]) for key in PISTON_SPATIAL_VOLUMES)
    denominator = fine["denominator_m3"]
    spatial_error_m3 = max(coarse["spatial_estimated_error_m3"], fine["spatial_estimated_error_m3"])
    spatial_passed = max(spatial_change_m3, spatial_error_m3) <= PISTON_REFINEMENT_LIMIT * denominator
    saturation_passed = fine["below_reference_contribution_relative"] <= PISTON_REFINEMENT_LIMIT
    return {
        "passed": fine["passed"] and spatial_passed and saturation_passed and volume["passed"],
        "maximum_dt_seconds": maximum_dt,
        "spatial_passed": spatial_passed,
        "saturation_passed": saturation_passed,
        "spatial_repeat_absolute_change_m3": spatial_change_m3,
        "spatial_relative_change": spatial_change_m3 / denominator,
        "spatial_relative_estimated_error": spatial_error_m3 / denominator,
        "primary_spatial": coarse,
        "tight_spatial": fine,
        "volume_budget": volume,
    }


def compare_piston_time_refinement(primary, refined):
    """Compare the two declared maximum-dt configurations using primary flux."""
    if (primary["maximum_dt_seconds"], refined["maximum_dt_seconds"]) != (0.01, 0.005):
        raise ValueError("wrong piston temporal refinement pair")
    base = primary["tight_spatial"]
    fine = refined["tight_spatial"]
    relative_changes = {key: abs(fine[key] - base[key]) / base["denominator_m3"] for key in PISTON_COMPARISON_VOLUMES}
    temporal_passed = max(relative_changes.values()) <= PISTON_REFINEMENT_LIMIT
    return {
        "passed": temporal_passed and primary["passed"] and refined["passed"],
        "temporal_passed": temporal_passed,
        "denominator_m3": base["denominator_m3"],
        "relative_changes": relative_changes,
    }


def evaluate_transport_components(frame_samples, support_radius, h):
    """Track sphere-connected material components by exact stable-ID overlap."""
    from collections import defaultdict
    from scipy.spatial import cKDTree

    previous = {}
    previous_volumes = {}
    threshold = h ** 3 / 8
    inventories = []
    for frame in frame_samples:
        points = frame["points"]
        parents = list(range(len(points)))

        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        if points:
            tree = cKDTree([point["world_position"] for point in points])
            for i, j in tree.query_pairs(2 * support_radius, output_type="ndarray"):
                first = root(i)
                second = root(j)
                parents[max(first, second)] = min(first, second)
        groups = defaultdict(list)
        volumes = {point["id"]: 4 * math.pi / 3 * point["radius"] ** 3 for point in points}
        for i, point in enumerate(points):
            groups[root(i)].append(point["id"])
        current = {}
        components = []
        for ids in sorted((sorted(ids) for ids in groups.values()), key=lambda ids: ids[0]):
            label = ids[0]
            overlap = defaultdict(int)
            for particle_id in ids:
                current[particle_id] = label
                if particle_id in previous:
                    overlap[previous[particle_id]] += 1
            components.append({
                "minimum_id": label,
                "particle_ids": ids,
                "material_volume_m3": math.fsum(volumes[particle_id] for particle_id in ids),
                "previous_component_overlap_counts": dict(sorted(overlap.items())),
            })
        lost_by_component = defaultdict(list)
        for particle_id in sorted(previous.keys() - current.keys()):
            lost_by_component[previous[particle_id]].append(particle_id)
        lost = [{
            "previous_component_minimum_id": label,
            "particle_ids": ids,
            "material_volume_m3": math.fsum(previous_volumes[particle_id] for particle_id in ids),
        } for label, ids in sorted(lost_by_component.items())]
        inventories.append({
            "frame": frame["frame"],
            "components": components,
            "lost_below_threshold": [entry for entry in lost if entry["material_volume_m3"] < threshold],
            "lost_at_or_above_threshold": [entry for entry in lost if entry["material_volume_m3"] >= threshold],
        })
        previous = current
        previous_volumes = volumes
    return {"loss_threshold_m3": threshold, "frames": inventories}
