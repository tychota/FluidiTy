"""Independent sampled controls for Task 5 capillary and dam-break gates."""

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquid_capillary_acceptance_metrics import (
    _interpolated_extrema,
    compare_dam_break_energy_blends,
    evaluate_drop_frequency,
    evaluate_static_sphere,
    verify_capillary_steps,
)


def sphere_centers(cells_per_radius):
    h = 0.125 / cells_per_radius
    return [
        ((x + 0.5) * h, (y + 0.5) * h, (z + 0.5) * h)
        for z in range(-cells_per_radius, cells_per_radius)
        for y in range(-cells_per_radius, cells_per_radius)
        for x in range(-cells_per_radius, cells_per_radius)
        if (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2 < cells_per_radius**2
    ]


def point(particle_id, position, volume, velocity=(0, 0, 0), ratio=1):
    return {
        "id": particle_id,
        "world_position": position,
        "world_velocity": velocity,
        "radius": math.cbrt(3 * volume / (4 * math.pi)),
        "volume_ratio": ratio,
    }


def frame(ordinal, points):
    return {"frame": ordinal, "time_seconds": ordinal / 24, "points": points}


def sphere_samples(cells_per_radius, speed=0):
    centers = sphere_centers(cells_per_radius)
    volume = 4 * math.pi / 3 * 0.125**3 / len(centers)
    frames = [
        frame(ordinal, [
            point(particle_id, position, volume, (speed, 0, 0))
            for particle_id, position in enumerate(centers, 1)
        ])
        for ordinal in range(1, 25)
    ]
    checkpoints = [
        {
            **frame(ordinal, []),
            "pressure_cells": [
                {
                    "cell_index": [x, y, z],
                    "occupancy": 1,
                    "pressure_pa": 16 if (
                        (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2 <= (cells_per_radius / 2)**2
                    ) else 40,
                }
                for z in range(-cells_per_radius, cells_per_radius)
                for y in range(-cells_per_radius, cells_per_radius)
                for x in range(-cells_per_radius, cells_per_radius)
                if (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2 < cells_per_radius**2
            ],
        }
        for ordinal in (10, 20)
    ]
    return frames, checkpoints


def test_static_sphere_uses_signed_core_and_separate_checkpoint_medians():
    coarse = sphere_samples(3, 0.002)
    fine = sphere_samples(5, 0.001)
    assert evaluate_static_sphere(*coarse, *fine)["passed"]
    for cell in fine[1][1]["pressure_cells"]:
        if cell["pressure_pa"] == 16:
            cell["pressure_pa"] = 17
    result = evaluate_static_sphere(*coarse, *fine)
    assert not result["passed"]
    assert [check["median_pressure_pa"] for check in result["fine"]["pressure_checkpoints"]] == [16, 17]
    assert result["fine"]["maximum_relative_pressure_error"] == 0.0625


@pytest.mark.parametrize("coarse_speed, spike_speed, refinement_passed, fine_speed_passed", [
    (0.001, 0.002, False, True),
    (0.010, 0.005, True, False),
])
def test_static_sphere_one_frame_spike(coarse_speed, spike_speed, refinement_passed, fine_speed_passed):
    coarse = sphere_samples(3, coarse_speed)
    fine = sphere_samples(5, 0.001)
    fine[0][6]["points"][0]["world_velocity"] = (0, 0, spike_speed)
    result = evaluate_static_sphere(*coarse, *fine)
    assert not result["passed"]
    assert result["refinement_passed"] is refinement_passed
    assert result["fine_speed_passed"] is fine_speed_passed
    assert result["fine"]["maximum_speed_frame"] == 7
    assert result["fine"]["maximum_speed_m_per_s"] == spike_speed


def drop_samples(omega, decay, *, amplitude=0.001, offset=0, phase=0):
    centers = sphere_centers(5)
    volume = 4 * math.pi / 3 * 0.125**3 / len(centers)
    # Two symmetric sheets at +/-sqrt(C + q(t)) yield an exact analytic
    # quadrupole when the transverse second moment is C. The perturbation
    # cancels only with material-volume weights. The center translates.
    n = len(centers)
    frames = []
    for ordinal in range(217):
        time = ordinal / 24
        q = amplitude * math.exp(-decay * time) * math.cos(omega * time + phase)
        perturbation = 0.0015 * math.cos(3.3 * time)
        points = []
        for particle_id in range(1, n + 1):
            direction = 1 if particle_id % 2 else -1
            ratio = 1.25 if (particle_id - 1) % 4 < 2 else 0.75
            group_quadrupole = q + perturbation * (1 if ratio > 1 else -5 / 3)
            y_scale = math.sqrt(0.01 + group_quadrupole)
            position = (0.1 * direction + offset * time, direction * y_scale - offset * time, 0.1 * direction)
            points.append(point(particle_id, position, volume * ratio, ratio=ratio))
        frames.append(frame(ordinal, points))
    return frames


def test_drop_recovers_damped_frequency_from_centered_quadrupole():
    omega = 2.024
    decay = 0.12
    result = evaluate_drop_frequency(drop_samples(omega, decay, offset=0.5))
    assert result["passed"]
    assert result["angular_frequency_rad_per_s"] == pytest.approx(omega, rel=0.001)
    assert result["damping_ratio"] == pytest.approx(decay / math.hypot(omega, decay), rel=0.005)
    assert result["damping_corrected_angular_frequency_rad_per_s"] == pytest.approx(math.hypot(omega, decay), rel=0.001)
    assert result["period_count_by_sign"] == {"positive": 1, "negative": 2}
    assert result["quadrupole_samples"][0]["quadrupole_m2"] == pytest.approx(0.001)
    assert result["fit_time_rms_residual_seconds"] < 0.001
    assert result["damping_log_amplitude_rms_residual"] < 0.001


def test_drop_frequency_outside_five_percent():
    result = evaluate_drop_frequency(drop_samples(2.20, 0.12))
    assert not result["passed"]
    assert result["relative_frequency_error"] > 0.05


def test_drop_rejects_excessive_damping_with_correct_frequency():
    result = evaluate_drop_frequency(drop_samples(2.024, 0.5))
    assert not result["passed"]
    assert result["frequency_passed"]
    assert result["damping_ratio"] > 0.20


def test_drop_rejects_amplitude_growth_with_correct_frequency():
    result = evaluate_drop_frequency(drop_samples(2.024, -0.8, amplitude=1e-6))
    assert result["frequency_passed"]
    assert result["log_amplitude_decrement"] < 0
    assert result["damping_ratio"] == pytest.approx(-0.8 / math.hypot(2.024, 0.8), rel=0.005)
    assert not result["damping_passed"]
    assert not result["passed"]


@pytest.mark.parametrize("values", [
    (1.50e308, 1.79e308, 1.78e308),  # The finite samples fit a vertex above the largest float.
    (-1.79e308, 1.79e308, 1.78e308),  # The local slope and curvature overflow.
])
def test_drop_rejects_overflow_from_finite_extremum_samples(values):
    samples = [
        {"time_seconds": ordinal / 24, "quadrupole_m2": value}
        for ordinal, value in enumerate(values, 24)
    ]
    with pytest.raises(ValueError, match="nonfinite interpolated drop extremum"):
        _interpolated_extrema(samples)


def test_drop_rejects_signal_without_complete_same_sign_periods():
    result = evaluate_drop_frequency(drop_samples(0.5, 0.03))
    assert not result["passed"]
    assert result["angular_frequency_rad_per_s"] is None


def test_drop_rejects_nonfinite_point():
    samples = drop_samples(2.024, 0.12)
    samples[6]["points"][0]["world_position"] = (0, math.nan, 0)
    with pytest.raises(ValueError):
        evaluate_drop_frequency(samples)


def capillary_steps(bound, *, first_dt=0.01, last_frame=24):
    steps = []
    time = 0.0
    for frame_ordinal in range(1, last_frame + 1):
        target_time = frame_ordinal / 24
        substep = 0
        while time < target_time:
            step_limit = first_dt if not steps else 0.01
            dt = min(target_time - time, step_limit)
            time = target_time if dt == target_time - time else time + dt
            substep += 1
            steps.append({
                "accepted_step": len(steps) + 1,
                "frame": frame_ordinal if time == target_time else frame_ordinal - 1,
                "substep": 0 if time == target_time else substep,
                "time_seconds": time,
                "accepted_measurement": {
                    "accepted_dt_seconds": dt,
                    "step_start_capillary_bound_seconds": bound,
                },
            })
    return steps


@pytest.mark.parametrize("case, bound", [
    ("static-sphere-coarse", 0.0758714187911166),
    ("static-sphere-fine", 0.03526184897173477),
    ("oscillating-drop-l2", 0.03526184897173477),
])
def test_capillary_limits_match_analytic_coarse_and_fine_values(case, bound):
    result = verify_capillary_steps(capillary_steps(bound, last_frame=216 if case == "oscillating-drop-l2" else 24),
                                    case=case)
    assert result["passed"]
    assert result["analytic_capillary_bound_seconds"] == pytest.approx(bound, rel=1e-14)


def test_fine_capillary_step_violation_is_detected_independently_of_receipt():
    steps = capillary_steps(0.05, first_dt=0.04)
    result = verify_capillary_steps(steps, case="static-sphere-fine")
    assert not result["passed"]
    assert result["first_capillary_violation_step"] == 1
    assert result["first_reported_bound_mismatch_step"] == 1


@pytest.mark.parametrize("bound", [None, math.nan, 0.05])
def test_capillary_report_must_be_present_finite_and_match(bound):
    result = verify_capillary_steps(capillary_steps(bound), case="static-sphere-fine")
    assert not result["passed"]
    assert result["first_capillary_violation_step"] is None
    assert result["first_reported_bound_mismatch_step"] == 1


def dam_samples(front_offsets, speeds=(0, 0, 0)):
    h = 0.0625
    positions = [
        ((x + 0.5) * h, (y + 0.5) * h, (z + 0.5) * h)
        for z in range(4)
        for y in range(8)
        for x in range(8)
    ]
    return {
        energy: [
            frame(ordinal, [
                point(particle_id, (x + (front_offset if ordinal else 0), y, z), h**3,
                      (speed if ordinal else 0, 0, 0))
                for particle_id, (x, y, z) in enumerate(positions, 1)
            ])
            for ordinal in range(49)
        ]
        for energy, front_offset, speed in zip((0, 0.6, 1), front_offsets, speeds)
    }


def test_dam_break_reports_nonmonotone_curve_and_endpoint_difference_by_stable_id():
    samples = dam_samples((0, 0.03, 0.012))
    samples[1][24]["points"].reverse()
    result = compare_dam_break_energy_blends(samples)
    assert result["passed"]
    assert result["initial_potential_energy_j"] == pytest.approx(153.28125)
    assert [sample["frame"] for sample in result["endpoint_differences"]] == [0, 12, 24, 48]
    assert result["endpoint_differences"][2]["front_difference_m"] == pytest.approx(0.012)
    assert result["endpoint_differences"][2]["maximum_position_difference_m"] == pytest.approx(0.012)
    assert result["blends"][0.6][2]["front_p95_x_m"] == pytest.approx(0.49875)
    assert result["blends"][0.6][2]["material_volume_m3"] == pytest.approx(0.0625)


def test_dam_break_accepts_kinetic_energy_difference_with_common_front():
    result = compare_dam_break_energy_blends(dam_samples((0, 0, 0), (0.1, 0.2, 0.3)))
    assert result["passed"]
    assert result["endpoint_differences"][1]["kinetic_energy_difference_j"] == pytest.approx(2.5)


def test_dam_break_does_not_qualify_negligible_endpoint_differences():
    assert not compare_dam_break_energy_blends(dam_samples((0, 0.03, 0.009)))["passed"]


def test_dam_break_requires_strict_front_excess():
    samples = dam_samples((0, 0, 0))
    for blend, front in ((0, 0), (0.6, 0.03), (1, 0.01)):
        for sample in samples[blend][1:]:
            for point_record in sample["points"]:
                _, y, z = point_record["world_position"]
                point_record["world_position"] = (front, y, z)
    result = compare_dam_break_energy_blends(samples)
    assert result["endpoint_differences"][1]["front_difference_m"] == 0.01
    assert not result["passed"]


def test_dam_break_rejects_duplicated_endpoint_data_in_another_blend():
    samples = dam_samples((0, 0.03, 0.02))
    samples[0.6] = copy.deepcopy(samples[1])
    samples[0.6][24]["points"].reverse()
    result = compare_dam_break_energy_blends(samples)
    assert not result["passed"]
    assert result["duplicated_blend_pairs"] == [(0.6, 1)]


@pytest.mark.parametrize("field, wrong_value", [
    ("world_position", (0.1, 0.1, 0.1)),
    ("radius", 0.1),
])
def test_dam_break_rejects_incompatible_initial_geometry(field, wrong_value):
    samples = dam_samples((0, 0.03, 0.02))
    samples[0.6][0]["points"][0][field] = wrong_value
    with pytest.raises(ValueError):
        compare_dam_break_energy_blends(samples)


def test_dam_break_rejects_duplicate_frame():
    samples = dam_samples((0, 0.03, 0.02))
    samples[1][12] = samples[1][11]
    with pytest.raises(ValueError):
        compare_dam_break_energy_blends(samples)
