"""Independent controls for the real free-fall acceptance measurements."""

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquid_acceptance_metrics import evaluate_free_fall, uniform_history_velocities
from run_liquid_acceptance import matches_identity, read_records, run_acceptance, verify_device, verify_solver


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
