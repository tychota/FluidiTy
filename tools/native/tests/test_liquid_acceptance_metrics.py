"""Independent controls for the real free-fall acceptance measurements."""

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquid_acceptance_metrics import evaluate_free_fall, uniform_history_velocities
from run_liquid_acceptance import (
    matches_identity, read_records, run_acceptance, verify_device, verify_job_measurement, verify_solver,
)


def frames():
    return [
        {
            "frame": ordinal,
            "time_seconds": ordinal / 24,
            "points": [{
                "id": 1,
                "world_position": [2.5, 2.5 - 0.5 * 9.81 * (ordinal / 24) ** 2, 2.5],
                "world_velocity": [0, -9.81 * (ordinal / 24), 0],
            }],
        }
        for ordinal in range(7)
    ]


def test_independent_ballistic_samples():
    result = evaluate_free_fall(frames())
    assert result["maximum_relative_acceleration_error"] < 1e-14
    assert result["maximum_transverse_speed_m_per_s"] == 0
    assert result["world_displacement_m"] == pytest.approx([0, -0.3065625, 0])
    assert len(result["intervals"]) == 6
    assert result["passed"]


@pytest.mark.parametrize("corruption", ["missing", "wrong-ordinal", "wrong-clock", "duplicate-id", "missing-id", "nan"])
def test_incomplete_or_invalid_samples(corruption):
    samples = frames()
    if corruption == "missing":
        samples.pop(3)
    elif corruption == "wrong-ordinal":
        samples[3]["frame"] = 2
    elif corruption == "wrong-clock":
        samples[3]["time_seconds"] = samples[2]["time_seconds"]
    elif corruption == "duplicate-id":
        samples[3]["points"].append(copy.deepcopy(samples[3]["points"][0]))
    elif corruption == "missing-id":
        samples[3]["points"][0]["id"] = 2
    else:
        samples[3]["points"][0]["world_velocity"][1] = float("nan")
    with pytest.raises(ValueError):
        evaluate_free_fall(samples)


def test_acceleration_error_cannot_hide_in_final_velocity():
    samples = frames()
    samples[3]["points"][0]["world_velocity"][1] += 0.001
    result = evaluate_free_fall(samples)
    assert result["maximum_relative_acceleration_error"] > 0.001
    assert not result["passed"]


def test_transverse_error_is_independent_of_gravity_axis():
    samples = frames()
    samples[3]["points"][0]["world_velocity"][0] = 1.01e-10
    result = evaluate_free_fall(samples)
    assert result["maximum_transverse_speed_m_per_s"] == 1.01e-10
    assert not result["passed"]


@pytest.mark.parametrize("energy", [0, 0.5, 0.6, 1])
def test_uniform_history_endpoints_and_interior(energy):
    assert uniform_history_velocities(energy, 3.0, 3.0, 2.0) == (2.0, 2.0)


def test_missing_uniform_history():
    with pytest.raises(ValueError, match="old history"):
        uniform_history_velocities(0.6, 3.0, None, 2.0)
    # An inconsistent old history makes the retained correction nonzero.
    assert uniform_history_velocities(1, 3.0, 2.0, 2.0) == pytest.approx((2.99, 2.99))


def test_receipt_identity_is_external_to_the_cache():
    expected = {"resolved_fingerprint": "a" * 64, "execution_id": "b" * 64, "backend": "cpu"}
    for field, wrong_value in (
        ("resolved_fingerprint", "c" * 64), ("execution_id", "d" * 64), ("backend", "cuda")
    ):
        with pytest.raises(ValueError, match=field):
            matches_identity({**expected, field: wrong_value, "event": "frame"}, expected)


@pytest.mark.parametrize("stdout", [
    '{"protocol_version":1,"event":"result"}',  # Incomplete final line.
    '{"protocol_version":1,"event":"result","frame":1,"frame":2}\n',
    '{"protocol_version":true,"event":"result"}\n',
    '{"protocol_version":1,"event":"result","time_seconds":NaN}\n',
    '{"protocol_version":1,"event":"result","time_seconds":1e999}\n',
])
def test_malformed_receipt_is_rejected(stdout):
    with pytest.raises(ValueError):
        read_records(stdout, "protocol_version")


def test_absent_cuda_cannot_be_qualified_as_a_cpu_fallback():
    with pytest.raises(ValueError, match="real CUDA device unavailable"):
        verify_device({
            "command": "device-info", "backend": "cuda", "cpu_available": True,
            "cuda_compiled": True, "cuda_available": False,
        }, "cuda")


def test_stale_output_is_rejected_before_launch(tmp_path):
    output = tmp_path / "stale"
    output.mkdir()
    stale_receipt = output / "free-fall-measurements.json"
    stale_receipt.write_text('{"passed":true}', encoding="utf-8")
    missing_program = tmp_path / "should-never-launch"
    with pytest.raises(FileExistsError):
        run_acceptance(missing_program, missing_program, "cpu", "free-fall-pic", output)
    assert stale_receipt.read_text(encoding="utf-8") == '{"passed":true}'


def solver_receipts():
    authored_scene_identity = {"resolved_fingerprint": "a" * 64, "execution_id": "b" * 64, "backend": "cpu"}
    manifest_clocks = [{"frame": 0, "substep": 0, "epoch": 0, "accepted_step": 0, "time_seconds": 0}]
    receipts = [{"event": "frame", "published": True, "frame": 0}]
    for frame_ordinal in range(1, 7):
        for accepted_in_frame in range(1, 6):
            step_ordinal = (frame_ordinal - 1) * 5 + accepted_in_frame
            completed = accepted_in_frame == 5
            trace_clock = {
                "frame": frame_ordinal if completed else frame_ordinal - 1,
                "substep": 0 if completed else accepted_in_frame,
                "epoch": step_ordinal,
                "accepted_step": step_ordinal,
                "time_seconds": step_ordinal / 120,
            }
            trace = {
                **trace_clock, "event": "trace", "published": False, "state_digest": "c" * 64,
                "particle_count": 1, "next_particle_id": 2, "block_count": 4, "cell_count": 7, "face_count": 768,
                "source_next_child_ordinals": [], "source_fractional_remainders": [],
                "attempts": accepted_in_frame, "accepted_attempts": accepted_in_frame,
                "dt_retries": 0, "capacity_replays": 0,
                "accepted_measurement": {
                    "accepted_dt_seconds": step_ordinal / 120 - (step_ordinal - 1) / 120,
                    "step_start_capillary_bound_seconds": None,
                    "pressure_iterations": 2,
                    "pressure_true_residual_l2_m3_per_s": 1e-6,
                    "pressure_rhs_l2_m3_per_s": 1,
                    "pressure_max_volume_scaled_residual_per_s": 1e-6,
                    "pressure_max_constraint_defect_per_s": 1e-6,
                    "step_start_pressure_cell_volume_m3": 4,
                    "step_start_inward_piston_flux_m3_per_s": 0,
                    "step_start_piston_low_x_m": None,
                    "step_end_piston_low_x_m": None,
                },
            }
            receipts.append(trace)
        manifest_clocks.append(trace_clock)
        receipts.append({"event": "frame", "published": True, "frame": frame_ordinal})
    receipts.append({
        **trace, **authored_scene_identity, "event": "result", "command": "run",
        "read_only": False, "frames_per_second": 24,
    })
    return receipts, manifest_clocks, authored_scene_identity


def with_attempt_ledger(receipts):
    output = []
    frame_attempts = []
    frame_traces = []
    prior_clock = {"frame": 0, "substep": 0, "epoch": 0, "accepted_step": 0, "time_seconds": 0}
    prior_retries = 0
    for receipt in receipts:
        if receipt["event"] == "frame":
            output.extend(frame_attempts)
            output.extend(frame_traces)
            frame_attempts.clear()
            frame_traces.clear()
            prior_retries = 0
        if receipt["event"] == "trace":
            target_frame = prior_clock["frame"] + 1
            attempt = {
                "event": "attempt", "target_frame": target_frame,
                "attempt_in_frame": receipt["attempts"], "capacity_pass": 0,
                "start_clock": prior_clock, "work_wall_seconds": 0.001,
                "attempted_dt_seconds": receipt["accepted_measurement"]["accepted_dt_seconds"],
                "outcome": "accepted", "failure_code": "OK", "retry_dt_seconds": None,
                "retained": True, "restarted_frame": False,
            }
            if receipt["dt_retries"] > prior_retries:
                frame_attempts.append({
                    **attempt, "attempt_in_frame": receipt["attempts"] - 1,
                    "outcome": "retry_dt", "failure_code": "ABORTED", "retained": False,
                    "attempted_dt_seconds": 2 * attempt["attempted_dt_seconds"],
                    "retry_dt_seconds": attempt["attempted_dt_seconds"],
                })
            frame_attempts.append(attempt)
            prior_retries = receipt["dt_retries"]
            prior_clock = {key: receipt[key] for key in prior_clock}
            frame_traces.append(receipt)
        else:
            output.append(receipt)
    return output


def test_valid_trace_counts_reset_at_publication_and_include_retries():
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    for receipt in receipts:
        if receipt["event"] == "trace" and 2 <= receipt["accepted_step"] <= 5:
            receipt["dt_retries"] = 1
            receipt["attempts"] += 1
    assert verify_solver(with_attempt_ledger(receipts), manifest_clocks, authored_scene_identity) == receipts[-1]


@pytest.mark.parametrize("accepted_step", [1, 5, 30])
def test_trace_counts_must_match_actual_accepted_records(accepted_step):
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    receipts = with_attempt_ledger(receipts)
    for receipt in receipts:
        if receipt.get("accepted_step") == accepted_step:
            # Corrupt the early trace, a frame-completion trace, or both copies
            # of the final summary without breaking attempts = accepted + retries.
            receipt.update(attempts=500, accepted_attempts=500)
    with pytest.raises(ValueError, match="accepted attempt count"):
        verify_solver(receipts, manifest_clocks, authored_scene_identity)


def test_retry_count_cannot_regress_before_publication():
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    receipts[1].update(attempts=2, dt_retries=1)
    with pytest.raises(ValueError, match="retry count regressed"):
        verify_solver(receipts, manifest_clocks, authored_scene_identity)


def test_final_summary_counts_must_match_the_final_trace():
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    receipts = with_attempt_ledger(receipts)
    receipts[-1].update(attempts=6, dt_retries=1)
    with pytest.raises(ValueError, match="final and trace attempts differ"):
        verify_solver(receipts, manifest_clocks, authored_scene_identity)


@pytest.mark.parametrize(
    "field, value",
    [
        ("pressure_rhs_l2_m3_per_s", None),
        ("pressure_true_residual_l2_m3_per_s", math.inf),
        ("pressure_true_residual_l2_m3_per_s", 1e-2),
        ("pressure_max_volume_scaled_residual_per_s", 1e-2),
        ("pressure_max_constraint_defect_per_s", 1e-2),
        ("step_start_capillary_bound_seconds", 0.001),
        ("accepted_dt_seconds", 0.001),
        ("step_start_pressure_cell_volume_m3", -1),
        ("step_start_inward_piston_flux_m3_per_s", -0.125),
    ],
)
def test_rejects_bad_accepted_measurements(field, value):
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    receipt = next(record for record in receipts if record["event"] == "trace")
    receipt["accepted_measurement"][field] = value
    with pytest.raises(ValueError, match=field):
        verify_solver(receipts, manifest_clocks, authored_scene_identity)


def test_rejects_missing_accepted_measurement():
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    del receipts[1]["accepted_measurement"]
    with pytest.raises(ValueError, match="accepted_measurement"):
        verify_solver(receipts, manifest_clocks, authored_scene_identity)


def test_rejects_incomplete_attempt_ledger():
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    with pytest.raises(ValueError, match="attempt ledger"):
        verify_solver(receipts, manifest_clocks, authored_scene_identity)


@pytest.mark.parametrize(
    "field, value",
    [
        ("attempt_in_frame", 2),
        ("capacity_pass", 1),
        ("retained", False),
        ("restarted_frame", True),
        ("work_wall_seconds", -1),
        ("attempted_dt_seconds", 0.001),
        ("failure_code", "ABORTED"),
    ],
)
def test_rejects_inconsistent_attempt_ledger(field, value):
    receipts, manifest_clocks, authored_scene_identity = solver_receipts()
    receipts = with_attempt_ledger(receipts)
    next(record for record in receipts if record["event"] == "attempt")[field] = value
    with pytest.raises(ValueError, match="attempt ledger"):
        verify_solver(receipts, manifest_clocks, authored_scene_identity)


@pytest.mark.parametrize(
    "compute_capability, compiled_architecture, diagnostic",
    [((11, 10), 1200, "wrong CUDA compute capability"), ((12, 0), 1210, "wrong CUDA architecture")],
)
def test_cuda_compute_capability_and_executed_architecture_are_independent(
    compute_capability, compiled_architecture, diagnostic
):
    compute_major, compute_minor = compute_capability
    probe = {
        "command": "device-info", "backend": "cuda", "cpu_available": True,
        "cuda_compiled": True, "cuda_available": True, "device_index": 0,
        "device_name": "CUDA qualification control", "compute_major": compute_major,
        "compute_minor": compute_minor, "compiled_architecture": compiled_architecture,
        "sample_count": 17, "driver_version": 13030, "runtime_version": 13030,
        "launch_result": 0, "copy_result": 0, "synchronize_result": 0,
    }
    with pytest.raises(ValueError, match=diagnostic):
        verify_device(probe, "cuda")
def job_receipt(backend):
    receipt = {
        "event": "job_measurement", "command": "run", "backend": backend, "status": "OK",
        "attempt_count": 6, "settled_attempt_count": 5,
        "host_wall": {
            "cold_setup_seconds": 0.125, "scheduling_seconds": 0.125,
            "emit_activation_field_seconds": 0.125, "deposit_normalization_seconds": 0.125,
            "forces_geometry_pressure_seconds": 0.125, "gather_contact_acceptance_seconds": 0.125,
            "snapshot_cache_seconds": 0.125, "drain_seconds": 0.125, "total_seconds": 1.0,
        },
        "cuda_events": None, "cuda_memory": None,
    }
    if backend == "cuda":
        receipt["cuda_events"] = {
            f"{phase}_seconds": 0.025 for phase in (
                "emit_activation_field", "deposit_normalization", "forces_geometry_pressure",
                "gather_contact_acceptance", "snapshot_d2h",
            )
        }
        receipt["cuda_events"].update({
            f"{phase}_interval_count": 6 for phase in (
                "emit_activation_field", "deposit_normalization", "forces_geometry_pressure",
                "gather_contact_acceptance", "snapshot_d2h",
            )
        })
        receipt["cuda_events"].update(pre_topology_field_seconds=0.01, pre_topology_field_interval_count=6)
        receipt["cuda_memory"] = {
            "total_device_bytes": 1000, "free_device_bytes_at_start": 800,
            "budget_bytes": 640, "device_byte_limit": 600, "current_device_reserved_bytes": 300,
            "peak_device_reserved_bytes": 480, "current_pinned_reserved_bytes": 190,
            "peak_pinned_reserved_bytes": 250, "peak_snapshot_pinned_reserved_bytes": 80,
            "current_snapshot_pinned_reserved_bytes": 70, "current_snapshot_output_bytes": 0,
            "current_physical_state_bytes": 70, "peak_physical_state_bytes": 100,
            "current_device_scratch_bytes": 210, "peak_device_scratch_bytes": 220,
            "peak_active_device_scratch_bytes": 110, "peak_snapshot_output_bytes": 60,
            "device_reservation_attempts": 2, "budget_refusals": 0, "quarantined_owner_count": 0,
            "remaining_device_reserved_bytes_after_drain": 0, "remaining_pinned_reserved_bytes_after_drain": 0,
            "free_device_bytes_after_drain": 800,
        }
    return receipt


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_accepts_opt_in_job_measurement(backend):
    receipts, frames, identity = solver_receipts()
    receipts = with_attempt_ledger(receipts)
    identity["backend"] = backend
    for record in receipts:
        if "backend" in record:
            record["backend"] = backend
        if backend == "cuda" and record["event"] in ("frame", "result"):
            record.update(compiled_architecture=1200, snapshot_copy_milliseconds=0.01)
    if backend == "cuda":
        receipts[-1].update(graph_replay_count=5, device_reserved_bytes=300)
    receipts.insert(-1, job_receipt(backend))
    receipts[-2].update(attempt_count=30, settled_attempt_count=30)
    if backend == "cuda":
        for field in receipts[-2]["cuda_events"]:
            if field.endswith("_interval_count"):
                receipts[-2]["cuda_events"][field] = 7 if field.startswith("snapshot_d2h") else 30
    assert verify_solver(receipts, frames, identity, require_job_measurement=True) == receipts[-1]
    with pytest.raises(ValueError, match="missing job measurement"):
        verify_solver(receipts[:-2] + receipts[-1:], frames, identity, require_job_measurement=True)
    receipts[-2]["attempt_count"] += 1
    with pytest.raises(ValueError, match="job measurement disagrees"):
        verify_solver(receipts, frames, identity, require_job_measurement=True)


@pytest.mark.parametrize("section, field, value, message", (
    ("host_wall", "drain_seconds", float("nan"), "invalid drain_seconds"),
    ("host_wall", "cold_setup_seconds", 0.25, "host wall phases"),
    ("cuda_events", "emit_activation_field_interval_count", 0, "CUDA event interval"),
    ("cuda_events", "deposit_normalization_seconds", -1.0, "invalid deposit_normalization_seconds"),
    ("cuda_memory", "budget_bytes", 641, "fixed device budget"),
    ("cuda_memory", "peak_device_reserved_bytes", 601, "aggregate device reservation"),
    ("cuda_memory", "peak_snapshot_pinned_reserved_bytes", 251, "aggregate pinned reservation"),
    ("cuda_memory", "peak_snapshot_output_bytes", 81, "snapshot output extent"),
))
def test_rejects_inconsistent_job_measurement(section, field, value, message):
    receipt = job_receipt("cuda")
    receipt[section][field] = value
    with pytest.raises(ValueError, match=message):
        verify_job_measurement(receipt, "cuda")


def test_rejects_missing_or_fabricated_cuda_job_measurement():
    with pytest.raises(ValueError, match="missing job measurement"):
        verify_job_measurement(None, "cuda")
    with pytest.raises(ValueError, match="CUDA evidence on a CPU job"):
        verify_job_measurement(job_receipt("cuda") | {"backend": "cpu"}, "cpu")
    with pytest.raises(ValueError, match="missing CUDA event"):
        verify_job_measurement(job_receipt("cpu") | {"backend": "cuda"}, "cuda")


def test_topology_and_face_events_cannot_stand_in_for_pre_topology_field_sampling():
    receipt = job_receipt("cuda")
    del receipt["cuda_events"]["pre_topology_field_interval_count"]
    with pytest.raises(ValueError, match="pre_topology_field"):
        verify_job_measurement(receipt, "cuda")

def test_hydrostatic_pressure_uses_signed_cell_center_depth():
    from liquid_acceptance_metrics import evaluate_hydrostatic_column

    samples = column_frames()
    checkpoints = column_checkpoints()
    result = evaluate_hydrostatic_column(samples, checkpoints)
    assert result["passed"]
    assert result["pressure"][0]["depth_bands"][0]["max_absolute_error_pa"] == 0
    for cell in checkpoints[0]["pressure_cells"]:
        cell["pressure_pa"] += 613.125
    wrong_depth = evaluate_hydrostatic_column(samples, checkpoints)
    assert wrong_depth["pressure"][0]["depth_bands"][0]["max_absolute_error_pa"] == pytest.approx(613.125)


def test_hydrostatic_frame_48_uses_the_actual_two_second_clock():
    from liquid_acceptance_metrics import evaluate_hydrostatic_column

    samples = column_frames()
    samples[48]["time_seconds"] = 48 / 25
    with pytest.raises(ValueError, match="time"):
        evaluate_hydrostatic_column(samples, column_checkpoints())


def test_hydrostatic_speed_equal_to_the_limit_fails():
    from liquid_acceptance_metrics import evaluate_hydrostatic_column

    samples = column_frames()
    samples[48]["points"][0]["world_velocity"] = [0.001, 0, 0]
    result = evaluate_hydrostatic_column(samples, column_checkpoints())
    assert result["frame_48_maximum_speed_m_per_s"] == 0.001
    assert not result["passed"]


def test_volume_equal_to_seed_cannot_hide_lost_id():
    from liquid_acceptance_metrics import evaluate_volume_budget

    samples = column_frames()[:2]
    missing = samples[1]["points"].pop()
    samples[1]["points"][0]["radius"] = missing["radius"] * 2 ** (1 / 3)
    with pytest.raises(ValueError, match="ID"):
        evaluate_volume_budget(samples, [0, 0])


@pytest.mark.parametrize("field", ["source_volume_m3", "compression_volume_m3"])
def test_volume_rejects_undeclared_source_or_compression(field):
    from liquid_acceptance_metrics import evaluate_volume_budget

    samples = column_frames()[:2]
    samples[1][field] = 0.001
    with pytest.raises(ValueError, match=field):
        evaluate_volume_budget(samples, [0, 0])


def test_volume_cannot_hide_a_transient_deviation_with_zero_final_drift():
    from liquid_acceptance_metrics import evaluate_volume_budget

    samples = column_frames()[:3]
    for point in samples[1]["points"]:
        point["radius"] *= 1.02 ** (1 / 3)
    result = evaluate_volume_budget(samples, [0, 0, 0])
    assert result["maximum_relative_drift"] == pytest.approx(0.02)
    assert not result["passed"]


def test_piston_planar_sweep_recovers_exact_displaced_volume_and_orientation():
    from liquid_acceptance_metrics import integrate_piston_sweep

    # A planar liquid, wet to the 0.5 m reference and with fixed 0.5 m width.
    def sections(_x):
        return (0.25, 0), (0, 0)

    inward = integrate_piston_sweep(sections, -0.5, -0.46875, [], 1e-10)
    outward = integrate_piston_sweep(sections, -0.46875, -0.5, [], 1e-10)
    assert inward["swept_below_m3"] == pytest.approx(0.0078125)
    assert outward["swept_below_m3"] == pytest.approx(-0.0078125)
    assert inward["swept_above_m3"] == 0


def test_piston_thin_rise_defeats_coincident_midpoint_grids():
    from liquid_acceptance_metrics import split_interval_lengths

    h = 0.0625
    rise = (8 / 31) * h
    samples = [h / n * sum((i + 0.5) * h / n < rise for i in range(n)) for n in (8, 16)]
    assert samples[0] == samples[1]
    assert (rise - samples[0]) / rise == pytest.approx(0.03125)
    below, above = split_interval_lengths([(-h, 0.5 + rise)], 0, 0.5)
    assert below == 0.5
    assert above == pytest.approx(rise)


def test_piston_sphere_sections_merge_overlap_and_clip_wetted_area():
    from liquid_acceptance_metrics import SphereUnionSections

    # Coincident x/z centers produce two overlapping vertical intervals.
    geometry = SphereUnionSections([(0, 0, 0), (0, 0.5, 0)], 1, (-1, 1), 0, 0.5)
    (below, above), (below_error, above_error) = geometry.section(0, 1e-8)
    # The intervals from both ends cover the full layer when their common
    # half-height is >= 0.25 m. Integrate the two circle tails beyond that.
    assert below == pytest.approx(math.sqrt(15) / 8 + 2 * math.asin(0.25))
    assert above == pytest.approx(math.pi / 2)
    assert 0 <= below_error < 1e-7 and 0 <= above_error < 1e-7


@pytest.mark.parametrize("solver_flux", [-0.0078125, 0.006])
def test_piston_signed_solver_flux_exposes_wrong_orientation_or_aperture(solver_flux):
    from liquid_acceptance_metrics import compare_piston_estimates

    result = compare_piston_estimates(0.0078125, 0.0078125, 0.0078125, solver_flux)
    assert not result["passed"]
    assert result["solver_to_wall_relative_mismatch"] > 0.01


def column_frames():
    radius = (3 * 0.125 ** 3 / (4 * math.pi)) ** (1 / 3)
    points = [
        {"id": pid, "world_position": [(x + 0.5) / 8, (y + 0.5) / 8, (z + 0.5) / 8],
         "world_velocity": [0, 0, 0], "radius": radius, "volume_ratio": 1}
        for pid, (x, y, z) in enumerate(
            ((x, y, z) for z in range(-2, 2) for y in range(8) for x in range(-2, 2)), 1)
    ]
    return [
        {"frame": ordinal, "time_seconds": ordinal / 24, "points": copy.deepcopy(points),
         "source_volume_m3": 0, "compression_volume_m3": 0}
        for ordinal in range(241)
    ]


def column_checkpoints():
    cells = [
        {"cell_index": [x, y, z], "occupancy": 1, "pressure_pa": 9810 * (1 - (y + 0.5) / 8)}
        for z in range(-2, 2) for y in range(8) for x in range(-2, 2)
    ]
    return [{"frame": ordinal, "time_seconds": ordinal / 24, "pressure_cells": copy.deepcopy(cells)}
            for ordinal in (230, 240)]


def test_numerical_j_drift_remains_measured_with_zero_physical_compression():
    from liquid_acceptance_metrics import evaluate_volume_budget

    samples = column_frames()[:2]
    for point in samples[1]["points"]:
        point["volume_ratio"] = 1.000004
        point["radius"] *= 1.000004 ** (1 / 3)
    result = evaluate_volume_budget(samples, [0, 0])
    assert result["passed"]
    assert result["maximum_relative_drift"] == pytest.approx(0.000004)
    assert result["frames"][1]["compression_volume_m3"] == 0


def test_particle_spatial_reordering_preserves_complete_id_inventory():
    from liquid_acceptance_metrics import evaluate_volume_budget

    samples = column_frames()[:2]
    samples[1]["points"].reverse()
    assert evaluate_volume_budget(samples, [0, 0])["passed"]


def test_piston_validates_same_step_pose_and_rejects_temporally_shifted_end():
    from liquid_acceptance_metrics import verify_piston_step

    previous = {"accepted_step": 0, "time_seconds": 0}
    step = {
        "accepted_step": 1,
        "accepted_dt_seconds": 0.008333333333333333,
        "time_seconds": 0.008333333333333333,
        "step_start_piston_low_x_m": -0.5,
        "step_end_piston_low_x_m": -0.4998697916666667,
        "step_start_inward_piston_flux_m3_per_s": 0.00390625,
    }
    verify_piston_step(step, previous, 0.01)
    step["step_end_piston_low_x_m"] = math.nextafter(step["step_end_piston_low_x_m"], 0)
    with pytest.raises(ValueError, match="pose"):
        verify_piston_step(step, previous, 0.01)


def test_piston_temporal_refinement_compares_receiving_and_all_flux_estimates():
    from liquid_acceptance_metrics import compare_piston_time_refinement

    coarse = {"receiving_m3": 0.008, "full_wall_flux_m3": 0.008, "full_wetted_sweep_m3": 0.008,
              "solver_inward_flux_m3": 0.008, "denominator_m3": 0.008}
    primary = {"passed": True, "maximum_dt_seconds": 0.01, "tight_spatial": coarse}
    refined = {"passed": True, "maximum_dt_seconds": 0.005, "tight_spatial": coarse.copy()}
    assert compare_piston_time_refinement(primary, refined)["passed"]
    refined["tight_spatial"]["receiving_m3"] += 0.000021
    result = compare_piston_time_refinement(primary, refined)
    assert not result["passed"]
    assert result["relative_changes"]["receiving_m3"] == pytest.approx(0.002625)


def test_transport_inventory_tracks_split_and_both_material_loss_classes():
    from liquid_acceptance_metrics import evaluate_transport_components

    def point(particle_id, x, radius):
        return {"id": particle_id, "world_position": [x, 0, 0], "radius": radius}

    frames = [
        {"frame": 0, "points": [point(2, 0, 0.1), point(1, 0.1, 0.4), point(3, 2, 0.1)]},
        {"frame": 1, "points": [point(1, 0.1, 0.4), point(2, 1, 0.1), point(3, 2, 0.1)]},
        {"frame": 2, "points": [point(2, 1, 0.1)]},
    ]
    result = evaluate_transport_components(frames, 0.2, 1)
    assert result["frames"][0]["components"][0]["particle_ids"] == [1, 2]
    assert [c["previous_component_overlap_counts"] for c in result["frames"][1]["components"]] == [
        {1: 1}, {1: 1}, {3: 1},
    ]
    assert result["frames"][2]["lost_below_threshold"][0]["particle_ids"] == [3]
    assert result["frames"][2]["lost_at_or_above_threshold"][0]["particle_ids"] == [1]
