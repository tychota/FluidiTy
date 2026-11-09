"""Independent acceptance metrics for verified Task 5 liquid records."""

from itertools import combinations, product
import math
from statistics import median
from typing import NamedTuple

from liquid_acceptance_metrics import (
    FRAMES_PER_SECOND, GRAVITY_M_PER_S2, check_frame_time, check_ids, finite_number, vector3,
)

SPHERE_RADIUS_M = 0.125
LIQUID_DENSITY_KG_PER_M3 = 1000
SURFACE_TENSION_N_PER_M = 1
SPHERE_VOLUME_M3 = 4 * math.pi / 3 * SPHERE_RADIUS_M**3
STATIC_COARSE_CASE = "static-sphere-coarse"
STATIC_FINE_CASE = "static-sphere-fine"
DROP_CASE = "oscillating-drop-l2"
CAPILLARY_CASES = {STATIC_COARSE_CASE: (3, 24), STATIC_FINE_CASE: (5, 24), DROP_CASE: (5, 216)}
STATIC_CHECKPOINTS = (10, 20)
DROP_FIT_WINDOW_SECONDS = (0.25, 9)
MAXIMUM_DT_SECONDS = 0.01
DAM_BLEND_VALUES = (0, 0.6, 1)
DAM_LAST_FRAME = 48
DAM_REPORT_FRAMES = (0, 12, 24, 48)
DAM_CELL_WIDTH_M = 0.0625
DAM_PARTICLE_REFERENCE_VOLUME_M3 = DAM_CELL_WIDTH_M**3
DAM_PARTICLE_MASS_KG = LIQUID_DENSITY_KG_PER_M3 * DAM_PARTICLE_REFERENCE_VOLUME_M3


class _Point(NamedTuple):
    position_m: tuple
    velocity_m_per_s: tuple
    material_volume_m3: float
    volume_ratio: float


def _sphere_indices(radius_in_cells, bound_in_cells):
    # Signed integer coordinates about the world grid origin. The point
    # coordinates use the center of each cell, including at negative indices.
    extent = range(-radius_in_cells, radius_in_cells)
    return {
        index for index in product(extent, repeat=3)
        if math.fsum((coordinate + 0.5)**2 for coordinate in index) < bound_in_cells**2
    }


def _frame_points(samples, ordinals, particle_count, reference_volume_m3):
    if len(samples) != len(ordinals):
        raise ValueError(f"expected exactly frames {ordinals[0]} through {ordinals[-1]}")
    for frame, sample in zip(ordinals, samples):
        if not isinstance(sample, dict):
            raise ValueError(f"invalid point frame {frame}")
        check_frame_time(sample, frame)
        points = sample.get("points")
        check_ids(points, particle_count, f"frame {frame}")
        ordered = []
        for point in sorted(points, key=lambda record: record["id"]):
            radius = point.get("radius")
            ratio = point.get("volume_ratio")
            if not finite_number(radius) or radius <= 0 or not finite_number(ratio) or ratio <= 0:
                raise ValueError(f"invalid point volume at frame {frame}, ID {point['id']}")
            try:
                material_volume_m3 = 4 * math.pi / 3 * radius**3
            except OverflowError as error:
                raise ValueError(f"invalid point radius at frame {frame}, ID {point['id']}") from error
            if material_volume_m3 <= 0 or not math.isclose(
                material_volume_m3, reference_volume_m3 * ratio, rel_tol=1e-14,
            ):
                raise ValueError(f"point radius disagrees with material volume at frame {frame}, ID {point['id']}")
            ordered.append(_Point(
                tuple(vector3(point.get("world_position"))),
                tuple(vector3(point.get("world_velocity"))),
                material_volume_m3,
                ratio,
            ))
        yield sample["time_seconds"], ordered


def _static_resolution(frames, checkpoints, case, pressure_limit):
    radius_in_cells, last_frame = CAPILLARY_CASES[case]
    particle_count = len(_sphere_indices(radius_in_cells, radius_in_cells))
    reference_volume_m3 = SPHERE_VOLUME_M3 / particle_count
    maximum_speed = -1
    maximum_speed_frame = None
    for frame, (_, points) in enumerate(
        _frame_points(frames, range(1, last_frame + 1), particle_count, reference_volume_m3), 1,
    ):
        frame_speed = max(math.hypot(*point.velocity_m_per_s) for point in points)
        if not finite_number(frame_speed):
            raise ValueError(f"nonfinite maximum particle speed at frame {frame}")
        if frame_speed > maximum_speed:
            maximum_speed = frame_speed
            maximum_speed_frame = frame

    if len(checkpoints) != len(STATIC_CHECKPOINTS):
        raise ValueError("static sphere requires retained checkpoints 10 and 20")
    target_pa = 2 * SURFACE_TENSION_N_PER_M / SPHERE_RADIUS_M
    core_indices = _sphere_indices(radius_in_cells, radius_in_cells / 2)
    pressure_checks = []
    for frame, checkpoint in zip(STATIC_CHECKPOINTS, checkpoints):
        if not isinstance(checkpoint, dict):
            raise ValueError(f"invalid pressure checkpoint {frame}")
        check_frame_time(checkpoint, frame)
        cells = checkpoint.get("pressure_cells")
        if not isinstance(cells, list):
            raise ValueError(f"missing pressure cells at frame {frame}")
        seen_indices = set()
        core_pressures = []
        for cell in cells:
            index = cell.get("cell_index") if isinstance(cell, dict) else None
            if (not isinstance(index, (list, tuple)) or len(index) != 3
                    or any(type(coordinate) is not int for coordinate in index)):
                raise ValueError(f"invalid signed pressure-cell coordinate at frame {frame}")
            index = tuple(index)
            pressure_pa = cell.get("pressure_pa")
            occupancy = cell.get("occupancy")
            if (index in seen_indices or not finite_number(pressure_pa)
                    or type(occupancy) is not int or occupancy not in (0, 1, 2)):
                raise ValueError(f"invalid pressure cell {index} at frame {frame}")
            seen_indices.add(index)
            if occupancy == 1 and index in core_indices:
                core_pressures.append(pressure_pa)
        if len(core_pressures) != len(core_indices):
            raise ValueError(f"incomplete liquid core pressure samples at frame {frame}")
        median_pa = median(core_pressures)
        error_pa = abs(median_pa - target_pa)
        if not finite_number(error_pa):
            raise ValueError(f"nonfinite pressure error at frame {frame}")
        pressure_checks.append({
            "frame": frame,
            "time_seconds": checkpoint["time_seconds"],
            "liquid_core_cell_count": len(core_pressures),
            "median_pressure_pa": median_pa,
            "absolute_pressure_error_pa": error_pa,
            "relative_pressure_error": error_pa / target_pa,
        })
    worst = max(pressure_checks, key=lambda check: check["relative_pressure_error"])
    return {
        "pressure_passed": worst["relative_pressure_error"] <= pressure_limit,
        "relative_pressure_error_limit": pressure_limit,
        "maximum_absolute_pressure_error_pa": worst["absolute_pressure_error_pa"],
        "maximum_relative_pressure_error": worst["relative_pressure_error"],
        "pressure_checkpoints": pressure_checks,
        "maximum_speed_m_per_s": maximum_speed,
        "maximum_speed_frame": maximum_speed_frame,
    }


def evaluate_static_sphere(coarse_frames, coarse_checkpoints, fine_frames, fine_checkpoints):
    """Evaluate frames 1–24 and checkpoint pressures at frames 10 and 20.

    Inputs are verified point and checkpoint records in frame order. Each
    pressure median uses the signed liquid cell centers inside R/2 at that
    checkpoint. Speed gates use the largest particle speed over all frames.
    Malformed observations raise ValueError. Physical misses return passed=False.
    """
    coarse = _static_resolution(coarse_frames, coarse_checkpoints, STATIC_COARSE_CASE, 0.10)
    fine = _static_resolution(fine_frames, fine_checkpoints, STATIC_FINE_CASE, 0.05)
    fine_speed_limit = 0.05 * math.sqrt(SURFACE_TENSION_N_PER_M / (LIQUID_DENSITY_KG_PER_M3 * SPHERE_RADIUS_M))
    refinement_allowance = 1e-8
    refinement_passed = fine["maximum_speed_m_per_s"] <= coarse["maximum_speed_m_per_s"] + refinement_allowance
    fine_speed_passed = fine["maximum_speed_m_per_s"] <= fine_speed_limit
    return {
        "passed": coarse["pressure_passed"] and fine["pressure_passed"] and refinement_passed and fine_speed_passed,
        "target_pressure_pa": 2 * SURFACE_TENSION_N_PER_M / SPHERE_RADIUS_M,
        "fine_speed_limit_m_per_s": fine_speed_limit,
        "refinement_allowance_m_per_s": refinement_allowance,
        "refinement_passed": refinement_passed,
        "fine_speed_passed": fine_speed_passed,
        "coarse": coarse,
        "fine": fine,
    }


def _quadrupole(points):
    total_volume = math.fsum(point.material_volume_m3 for point in points)
    center = [
        math.fsum(point.material_volume_m3 / total_volume * point.position_m[axis] for point in points)
        for axis in range(3)
    ]
    contributions = []
    for point in points:
        x, y, z = (coordinate - mean for coordinate, mean in zip(point.position_m, center))
        contributions.append(point.material_volume_m3 / total_volume * (y * y - (x * x + z * z) / 2))
    result = math.fsum(contributions)
    if not finite_number(result):
        raise ValueError("nonfinite volume-weighted quadrupole")
    return result


def _interpolated_extrema(samples):
    extrema = {"positive": [], "negative": []}
    for sign, orientation in (("positive", 1), ("negative", -1)):
        for left, middle, right in zip(samples, samples[1:], samples[2:]):
            before, peak, after = (orientation * sample["quadrupole_m2"] for sample in (left, middle, right))
            # The earlier member of a two-sample plateau owns its extremum.
            if not before < peak >= after:
                continue
            slope = (after - before) / 2
            curvature = (before - peak) + (after - peak)
            offset = -slope / curvature
            time = middle["time_seconds"] + offset / FRAMES_PER_SECOND
            amplitude = peak + slope * offset / 2
            if DROP_FIT_WINDOW_SECONDS[0] <= time <= DROP_FIT_WINDOW_SECONDS[1] and amplitude > 0:
                extrema[sign].append({"time_seconds": time, "amplitude_m2": amplitude})
    return extrema


def _rms_fit_residual(groups, slope):
    # Each extremum sign has a separate phase or initial log amplitude.
    residuals = []
    for observations in groups:
        intercepts = [observation - index * slope for index, observation in enumerate(observations)]
        intercept = math.fsum(intercepts) / len(intercepts)
        residuals.extend((value - intercept)**2 for value in intercepts)
    return math.sqrt(math.fsum(residuals) / len(residuals))


def evaluate_drop_frequency(frame_samples):
    """Measure the 0–216 frame drop from verified points, at exactly 24 fps.

    The normalized material-volume weighted central moment is in m². Local
    parabolas use adjacent accepted samples, and only extrema in [0.25, 9] s
    contribute to the fit. Each sign must contribute a complete period. The
    time residual is RMS around the median period with one phase per sign.
    The damping residual uses the same construction with log magnitudes.
    Malformed observations raise ValueError. Physical misses return passed=False.
    """
    radius_in_cells, last_frame = CAPILLARY_CASES[DROP_CASE]
    particle_count = len(_sphere_indices(radius_in_cells, radius_in_cells))
    samples = []
    for time, points in _frame_points(
        frame_samples, range(last_frame + 1), particle_count, SPHERE_VOLUME_M3 / particle_count,
    ):
        samples.append({"time_seconds": time, "quadrupole_m2": _quadrupole(points)})
    extrema = _interpolated_extrema(samples)
    period_count_by_sign = {sign: max(0, len(peaks) - 1) for sign, peaks in extrema.items()}
    target = math.sqrt(8 * SURFACE_TENSION_N_PER_M / (LIQUID_DENSITY_KG_PER_M3 * SPHERE_RADIUS_M**3))
    frequency_limit = 0.05
    damping_limit = 0.20
    result = {
        "passed": False,
        "frequency_passed": False,
        "damping_passed": False,
        "fit_window_seconds": DROP_FIT_WINDOW_SECONDS,
        "relative_frequency_error_limit": frequency_limit,
        "damping_ratio_limit": damping_limit,
        "target_angular_frequency_rad_per_s": target,
        "angular_frequency_rad_per_s": None,
        "damping_corrected_angular_frequency_rad_per_s": None,
        "relative_frequency_error": None,
        "median_period_seconds": None,
        "fit_time_rms_residual_seconds": None,
        "log_amplitude_decrement": None,
        "damping_ratio": None,
        "damping_log_amplitude_rms_residual": None,
        "period_count_by_sign": period_count_by_sign,
        "extrema": extrema,
        "quadrupole_samples": samples,
    }
    if any(count < 1 for count in period_count_by_sign.values()):
        result["failure"] = "insufficient_same_sign_periods"
        return result

    times = [[peak["time_seconds"] for peak in peaks] for peaks in extrema.values()]
    log_amplitudes = [[math.log(peak["amplitude_m2"]) for peak in peaks] for peaks in extrema.values()]
    period = median(end - start for group in times for start, end in zip(group, group[1:]))
    decrement = median(start - end for group in log_amplitudes for start, end in zip(group, group[1:]))
    angular_frequency = math.tau / period
    correction = math.hypot(math.tau, decrement)
    damping_ratio = decrement / correction
    frequency_error = abs(angular_frequency - target) / target
    frequency_passed = finite_number(angular_frequency) and angular_frequency > 0 and frequency_error <= frequency_limit
    damping_passed = damping_ratio <= damping_limit
    result.update(
        passed=frequency_passed and damping_passed,
        frequency_passed=frequency_passed,
        damping_passed=damping_passed,
        angular_frequency_rad_per_s=angular_frequency,
        damping_corrected_angular_frequency_rad_per_s=correction / period,
        relative_frequency_error=frequency_error,
        median_period_seconds=period,
        fit_time_rms_residual_seconds=_rms_fit_residual(times, period),
        log_amplitude_decrement=decrement,
        damping_ratio=damping_ratio,
        damping_log_amplitude_rms_residual=_rms_fit_residual(log_amplitudes, -decrement),
    )
    return result


def verify_capillary_steps(traces, *, case):
    """Check each accepted native trace from time zero through the case's final frame.

    The analytic limit uses frozen h, density and surface tension in SI units.
    The finite reported limit must match it to relative tolerance 1e-14 for
    floating-point evaluation and serialization. The accepted duration is
    compared directly to the analytic limit and the frozen maximum dt.
    Missing or inconsistent accepted clocks raise ValueError. Bound failures
    return passed=False with the first violating step for each check.
    """
    if case not in CAPILLARY_CASES:
        raise ValueError(f"no capillary fixture named {case}")
    radius_in_cells, last_frame = CAPILLARY_CASES[case]
    h = SPHERE_RADIUS_M / radius_in_cells
    bound = math.sqrt(LIQUID_DENSITY_KG_PER_M3 * h**3 / (4 * math.pi * SURFACE_TENSION_N_PER_M))
    capillary_violation = report_mismatch = maximum_dt_violation = None
    frame, substep, time = 0, 0, 0.0
    maximum_dt = 0.0
    reported_bounds = []
    for ordinal, trace in enumerate(traces, 1):
        measurement = trace.get("accepted_measurement") if isinstance(trace, dict) else None
        if not isinstance(measurement, dict):
            raise ValueError(f"missing accepted measurement at step {ordinal}")
        dt = measurement.get("accepted_dt_seconds")
        end_time = trace.get("time_seconds")
        target_time = (frame + 1) / FRAMES_PER_SECOND
        if (frame == last_frame or type(trace.get("accepted_step")) is not int or trace["accepted_step"] != ordinal
                or not finite_number(dt) or not 0 < dt <= target_time - time
                or not finite_number(end_time) or not time < end_time <= target_time):
            raise ValueError(f"invalid accepted duration or clock at step {ordinal}")
        accepted_time = time + dt
        if dt == target_time - time and math.nextafter(accepted_time, target_time) == target_time:
            accepted_time = target_time
        if accepted_time == target_time:
            frame, substep = frame + 1, 0
        else:
            substep += 1
        if (end_time != accepted_time or type(trace.get("frame")) is not int or trace["frame"] != frame
                or type(trace.get("substep")) is not int or trace["substep"] != substep):
            raise ValueError(f"inconsistent accepted frame clock at step {ordinal}")
        time = end_time
        maximum_dt = max(maximum_dt, dt)

        reported = measurement.get("step_start_capillary_bound_seconds")
        if finite_number(reported):
            reported_bounds.append(reported)
        if capillary_violation is None and dt > bound:
            capillary_violation = ordinal
        if maximum_dt_violation is None and dt > MAXIMUM_DT_SECONDS:
            maximum_dt_violation = ordinal
        if report_mismatch is None and (
            not finite_number(reported) or not math.isclose(reported, bound, rel_tol=1e-14) or dt > reported
        ):
            report_mismatch = ordinal
    if frame != last_frame:
        raise ValueError(f"missing accepted steps through frame {last_frame}")
    return {
        "passed": capillary_violation is None and maximum_dt_violation is None and report_mismatch is None,
        "analytic_capillary_bound_seconds": bound,
        "maximum_configured_dt_seconds": MAXIMUM_DT_SECONDS,
        "maximum_accepted_dt_seconds": maximum_dt,
        "reported_bound_range_seconds": [min(reported_bounds), max(reported_bounds)] if reported_bounds else None,
        "accepted_step_count": ordinal,
        "first_capillary_violation_step": capillary_violation,
        "first_maximum_dt_violation_step": maximum_dt_violation,
        "first_reported_bound_mismatch_step": report_mismatch,
    }


def _dam_seed_positions():
    h = DAM_CELL_WIDTH_M
    return [
        ((x + 0.5) * h, (y + 0.5) * h, (z + 0.5) * h)
        for z, y, x in product(range(4), range(8), range(8))
    ]


def _dam_summary(frame, points):
    front_positions = sorted(point.position_m[0] for point in points)
    # Linear interpolation at zero-based rank p(n-1).
    front_rank = 0.95 * (len(front_positions) - 1)
    lower_rank = math.floor(front_rank)
    lower_x = front_positions[lower_rank]
    front = lower_x + (front_positions[lower_rank + 1] - lower_x) * (front_rank - lower_rank)
    speeds = [math.hypot(*point.velocity_m_per_s) for point in points]
    energy_j = 0.5 * DAM_PARTICLE_MASS_KG * math.fsum(speed * speed for speed in speeds)
    volume_m3 = math.fsum(point.material_volume_m3 for point in points)
    if not all(map(finite_number, (front, energy_j, volume_m3, *speeds))):
        raise ValueError(f"nonfinite dam-break measurement at frame {frame}")
    return {
        "frame": frame,
        "time_seconds": frame / FRAMES_PER_SECOND,
        "material_volume_m3": volume_m3,
        "front_p95_x_m": front,
        "kinetic_energy_j": energy_j,
        "maximum_speed_m_per_s": max(speeds),
    }


def compare_dam_break_energy_blends(frame_samples_by_blend):
    """Compare verified point frames 0–48 for blends 0, 0.6 and 1.

    Match frozen initial positions, rest velocities and volumes by stable ID,
    then compare the paired positions and report front, energy, volume and
    speed at frames 0, 12, 24 and 48. Identical post-seed report data for any
    blend pair disqualifies the comparison. Report the measured differences
    and require a strict endpoint excess over 0.01 m or 1% of initial
    potential energy. Malformed samples or incompatible seeds raise ValueError.
    """
    if set(frame_samples_by_blend) != set(DAM_BLEND_VALUES) or not all(map(finite_number, frame_samples_by_blend)):
        raise ValueError("dam break requires exactly blends 0, 0.6 and 1")
    seed_positions = _dam_seed_positions()
    summaries = {}
    point_reports = {}
    for blend in DAM_BLEND_VALUES:
        frames = frame_samples_by_blend[blend]
        summaries[blend] = []
        point_reports[blend] = []
        for frame, (_, points) in enumerate(
            _frame_points(frames, range(DAM_LAST_FRAME + 1), len(seed_positions), DAM_PARTICLE_REFERENCE_VOLUME_M3),
        ):
            if frame == 0:
                for raw_point in frames[0]["points"]:
                    if "mass" in raw_point and (
                        not finite_number(raw_point["mass"]) or raw_point["mass"] != DAM_PARTICLE_MASS_KG
                    ):
                        raise ValueError(f"incompatible dam-break seed mass for blend {blend}")
                for point, position in zip(points, seed_positions):
                    if point.position_m != position or point.velocity_m_per_s != (0, 0, 0) or point.volume_ratio != 1:
                        raise ValueError(f"incompatible dam-break seed geometry for blend {blend}")
            if frame in DAM_REPORT_FRAMES:
                point_reports[blend].append(points)
                summaries[blend].append(_dam_summary(frame, points))

    initial_energy_j = DAM_PARTICLE_MASS_KG * GRAVITY_M_PER_S2 * math.fsum(
        point.position_m[1] for point in point_reports[0][0]
    )
    energy_limit_j = 0.01 * initial_energy_j
    front_limit_m = 0.01
    duplicated = []
    blend_differences = []
    for first, second in combinations(DAM_BLEND_VALUES, 2):
        if point_reports[first][1:] == point_reports[second][1:]:
            duplicated.append((first, second))
        differences = []
        for first_points, second_points, start, end in zip(
            point_reports[first], point_reports[second], summaries[first], summaries[second],
        ):
            position_difference = max(
                math.dist(p.position_m, q.position_m) for p, q in zip(first_points, second_points)
            )
            if not finite_number(position_difference):
                raise ValueError(f"nonfinite dam-break position difference at frame {start['frame']}")
            differences.append({
                "frame": start["frame"],
                "time_seconds": start["time_seconds"],
                "maximum_position_difference_m": position_difference,
                "front_difference_m": abs(start["front_p95_x_m"] - end["front_p95_x_m"]),
                "kinetic_energy_difference_j": abs(start["kinetic_energy_j"] - end["kinetic_energy_j"]),
            })
        blend_differences.append({"blend_pair": (first, second), "frames": differences})
    endpoint_differences = next(pair["frames"] for pair in blend_differences if pair["blend_pair"] == (0, 1))
    endpoint_difference_passed = any(
        difference["front_difference_m"] > front_limit_m or difference["kinetic_energy_difference_j"] > energy_limit_j
        for difference in endpoint_differences
    )
    return {
        "passed": not duplicated and endpoint_difference_passed,
        "endpoint_difference_passed": endpoint_difference_passed,
        "initial_potential_energy_j": initial_energy_j,
        "front_difference_limit_m": front_limit_m,
        "kinetic_energy_difference_limit_j": energy_limit_j,
        "blends": summaries,
        "duplicated_blend_pairs": duplicated,
        "blend_differences": blend_differences,
        "endpoint_differences": endpoint_differences,
    }
