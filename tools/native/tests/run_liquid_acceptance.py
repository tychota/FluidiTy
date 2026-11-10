"""Run and independently verify public liquid acceptance scenes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquid_acceptance_metrics import (
    FREE_FALL_CASES,
    FREE_FALL_LAST_FRAME,
    FRAMES_PER_SECOND,
    HYDROSTATIC_CASE,
    PISTON_CASE,
    PISTON_PARTICLE_RADIUS_M,
    PISTON_REFINED_CASE,
    TANK_CASES,
    check_ids,
    compare_piston_time_refinement,
    evaluate_hydrostatic_column,
    evaluate_piston_displacement,
    evaluate_transport_components,
    vector3,
    evaluate_free_fall,
    finite_number,
)

FRAME_ORDINALS = tuple(range(FREE_FALL_LAST_FRAME + 1))
TANK_LAYOUT_CASE = "tank-layout"
TANK_LAYOUT_INITIAL_POSITION = [0.625, 0.625, 0.625]
CUDA_ARCHITECTURE = 1200
CLOCK_FIELDS = ("frame", "substep", "epoch", "accepted_step", "time_seconds")
IDENTITY_FIELDS = ("resolved_fingerprint", "execution_id", "backend")
MAXIMUM_COUNTS = {"block_count": 128, "cell_count": 4096, "face_count": 24576}
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
CAPTURE_MANIFEST = "capture-manifest.json"
CAPTURE_MANIFEST_VERSION = 1
CUDA_COMPATIBILITY_FILE = ".simulate-cuda-compatibility"
CUDA_COMPATIBILITY_RECORD_SIZE = len(b"FLCUDA1\n") + 64 + 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    record = {}
    for key, value in pairs:
        require(key not in record, f"duplicate JSON key {key}")
        record[key] = value
    return record


def finite_float(number):
    value = float(number)
    require(finite_number(value), f"nonfinite JSON number {number}")
    return value


def reject_constant(number):
    raise ValueError(f"nonfinite JSON number {number}")


def read_records(stdout, version_field):
    require(stdout.endswith("\n"), "receipt lacks a complete newline-terminated record")
    records = []
    for line in stdout.splitlines():
        record = json.loads(
            line, object_pairs_hook=unique_object, parse_float=finite_float, parse_constant=reject_constant
        )
        require(isinstance(record, dict), "receipt must contain JSON objects")
        require(type(record.get(version_field)) is int and record[version_field] == 1, "wrong receipt version")
        require(isinstance(record.get("event"), str), "missing receipt event")
        records.append(record)
    return records


def run_child(output, name, executable, *arguments, fixture=False, timeout=60):
    stdout_path = output / f"{name}.stdout.jsonl"
    stderr_path = output / f"{name}.stderr.txt"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            [str(executable), *map(str, arguments)], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            timeout=timeout, check=False,
        )
    require(completed.returncode == 0, f"{name} exited {completed.returncode}; see {stderr_path}")
    require(stderr_path.stat().st_size == 0, f"{name} produced diagnostics; see {stderr_path}")
    version_field = "fixture_version" if fixture else "protocol_version"
    return read_records(stdout_path.read_text(encoding="utf-8"), version_field)


def one_record(records, event):
    require(len(records) == 1 and records[0]["event"] == event, f"expected one {event} receipt")
    return records[0]


def digest(record, field):
    value = record.get(field)
    require(isinstance(value, str) and DIGEST_PATTERN.fullmatch(value), f"invalid {field}")
    return value


def integer(record, field, *, minimum=0):
    value = record.get(field)
    require(type(value) is int and value >= minimum, f"invalid integer {field}")
    return value


def matches_identity(record, expected):
    for field in IDENTITY_FIELDS:
        require(record.get(field) == expected[field], f"wrong {field} in {record.get('event')} receipt")


def clock(record):
    ordinals = tuple(integer(record, field) for field in CLOCK_FIELDS[:-1])
    time = record.get("time_seconds")
    require(finite_number(time) and time >= 0, "invalid time_seconds")
    require(ordinals[2] == ordinals[3], "epoch differs from accepted step")
    return (*ordinals, time)


def verify_device(record, backend):
    require(record.get("command") == "device-info", "wrong device probe command")
    require(record.get("backend") == backend and record.get("cpu_available") is True, "wrong probe backend")
    require(type(record.get("cuda_compiled")) is bool, "missing CUDA build status")
    if backend == "cpu":
        require("cuda_available" in record and record["cuda_available"] is None, "unexpected CPU probe CUDA status")
        return
    require(record["cuda_compiled"] is True and record.get("cuda_available") is True, "real CUDA device unavailable")
    require(integer(record, "device_index") == 0, "wrong CUDA device index")
    require(isinstance(record.get("device_name"), str) and record["device_name"].strip(), "missing CUDA device name")
    compute_capability = (integer(record, "compute_major"), integer(record, "compute_minor"))
    require(compute_capability == (12, 0), "wrong CUDA compute capability")
    require(integer(record, "compiled_architecture") == CUDA_ARCHITECTURE, "wrong CUDA architecture")
    require(integer(record, "sample_count") == 17, "incomplete CUDA probe")
    for field in ("driver_version", "runtime_version"):
        integer(record, field, minimum=1)
    for field in ("launch_result", "copy_result", "synchronize_result"):
        require(integer(record, field) == 0, f"failed CUDA {field}")


def verify_snapshot_receipt(record, backend):
    if backend == "cuda":
        require(integer(record, "compiled_architecture") == CUDA_ARCHITECTURE, "wrong execution CUDA architecture")
        milliseconds = record.get("snapshot_copy_milliseconds")
        require(finite_number(milliseconds) and milliseconds > 0, "missing CUDA snapshot time")
    else:
        require("compiled_architecture" not in record, "unexpected CUDA execution for CPU case")


def verify_extent(record, particle_count=1):
    require(integer(record, "particle_count") == particle_count, "wrong physical particle count")
    require(integer(record, "next_particle_id") == particle_count + 1, "unexpected particle allocation")
    for field, maximum in MAXIMUM_COUNTS.items():
        require(integer(record, field) <= maximum, f"{field} exceeds the frozen capacity")
    require(record.get("source_next_child_ordinals") == [], "unexpected source ordinals")
    require(record.get("source_fractional_remainders") == [], "unexpected source remainder")


def advanced_clock(start_clock, dt, target_time):
    frame, substep, epoch, step, start_time = start_clock
    accepted_time = start_time + dt
    if dt == target_time - start_time and math.nextafter(accepted_time, target_time) == target_time:
        accepted_time = target_time
    if accepted_time == target_time:
        return frame + 1, 0, epoch + 1, step + 1, accepted_time
    return frame, substep + 1, epoch + 1, step + 1, accepted_time


def verify_accepted_measurement(record, previous_clock, target_time, *, tank):
    measurement = record.get("accepted_measurement")
    require(isinstance(measurement, dict), "missing accepted_measurement")
    dt = measurement.get("accepted_dt_seconds")
    require(finite_number(dt) and 0 < dt <= target_time - previous_clock[-1]
            and advanced_clock(previous_clock, dt, target_time) == clock(record), "invalid accepted_dt_seconds")
    capillary_bound = measurement.get("step_start_capillary_bound_seconds")
    require("step_start_capillary_bound_seconds" in measurement
            and (capillary_bound is None or finite_number(capillary_bound) and capillary_bound >= dt),
            "invalid step_start_capillary_bound_seconds")
    require(integer(measurement, "pressure_iterations") <= 256, "invalid pressure_iterations")
    fields = ("pressure_rhs_l2_m3_per_s", "pressure_true_residual_l2_m3_per_s",
              "pressure_max_volume_scaled_residual_per_s", "pressure_max_constraint_defect_per_s",
              "step_start_pressure_cell_volume_m3")
    for field in fields:
        value = measurement.get(field)
        require(finite_number(value) and value >= 0, f"invalid {field}")
    rhs, residual, scaled, defect, volume = (measurement[field] for field in fields)
    require(residual <= 2e-6 + 2e-6 * rhs, "pressure_true_residual_l2_m3_per_s exceeds solver bound")
    require(scaled <= 2e-5, "pressure_max_volume_scaled_residual_per_s exceeds solver bound")
    require(defect <= 2e-5, "pressure_max_constraint_defect_per_s exceeds solver bound")
    require((volume == 0) == (record["cell_count"] == 0), "step_start_pressure_cell_volume_m3 disagrees with rows")
    for pose_field in ("step_start_piston_low_x_m", "step_end_piston_low_x_m"):
        require(pose_field in measurement
                and (finite_number(measurement[pose_field]) if tank else measurement[pose_field] is None),
                f"invalid {pose_field}")
    piston_flux = measurement.get("step_start_inward_piston_flux_m3_per_s")
    require(finite_number(piston_flux) and (tank or piston_flux == 0),
            "invalid step_start_inward_piston_flux_m3_per_s")


def verify_attempt_ledger(attempts, traces, frame_start, target_frame):
    """Reconcile actual work and survivor receipts for one settled frame."""
    require(attempts and traces, "missing attempt ledger or accepted trace")
    target_time = target_frame / FRAMES_PER_SECOND
    settled_pass = integer(traces[-1], "capacity_replays")
    expected_start = frame_start
    active_pass = accepted_attempts = dt_retries = retries_this_step = trace_index = 0
    next_retry_dt = None
    for attempt_index, attempt in enumerate(attempts, start=1):
        require(attempt.get("event") == "attempt", "invalid attempt ledger event")
        for field, expected in (
            ("attempt_in_frame", attempt_index), ("target_frame", target_frame), ("capacity_pass", active_pass)
        ):
            require(type(attempt.get(field)) is int and attempt[field] == expected, f"wrong attempt ledger {field}")
        start = attempt.get("start_clock")
        require(isinstance(start, dict) and clock(start) == expected_start, "wrong attempt ledger start clock")
        dt = attempt.get("attempted_dt_seconds")
        work = attempt.get("work_wall_seconds")
        require(finite_number(dt) and 0 < dt <= target_time - expected_start[-1], "wrong attempt ledger duration")
        require(next_retry_dt is None or dt == next_retry_dt, "wrong attempt ledger retry duration")
        require(finite_number(work) and work >= 0, "invalid attempt ledger work time")
        outcome = attempt.get("outcome")
        retained = outcome == "accepted" and active_pass == settled_pass
        restarted = outcome == "capacity" and active_pass < settled_pass
        require(attempt.get("retained") is retained and attempt.get("restarted_frame") is restarted,
                "wrong attempt ledger retention or restart")
        require("retry_dt_seconds" in attempt, "missing attempt ledger retry duration")
        retry_dt = attempt["retry_dt_seconds"]
        next_retry_dt = None
        if outcome == "retry_dt":
            require(attempt.get("failure_code") == "ABORTED" and retries_this_step < 2
                    and finite_number(retry_dt) and retry_dt == dt * 0.5
                    and expected_start[-1] + retry_dt > expected_start[-1], "invalid attempt ledger retry")
            next_retry_dt = retry_dt
            dt_retries += 1
            retries_this_step += 1
            continue
        require(retry_dt is None, "unexpected attempt ledger retry duration")
        retries_this_step = 0
        if restarted:
            require(attempt.get("failure_code") == "RESOURCE_EXHAUSTED", "invalid attempt ledger capacity failure")
            active_pass += 1
            expected_start = frame_start
            continue
        require(outcome == "accepted" and attempt.get("failure_code") == "OK", "invalid settled attempt ledger outcome")
        accepted_attempts += 1
        expected_start = advanced_clock(expected_start, dt, target_time)
        if retained:
            require(trace_index < len(traces), "attempt ledger has an extra retained acceptance")
            trace = traces[trace_index]
            require(clock(trace) == expected_start and trace["accepted_measurement"]["accepted_dt_seconds"] == dt,
                    "attempt ledger differs from accepted clock or duration")
            require(trace["attempts"] == attempt_index and trace["accepted_attempts"] == accepted_attempts
                    and trace["dt_retries"] == dt_retries and trace["capacity_replays"] == active_pass,
                    "attempt ledger differs from accepted diagnostic counts")
            trace_index += 1
    require(trace_index == len(traces) and active_pass == settled_pass and expected_start[:2] == (target_frame, 0),
            "incomplete attempt ledger")


WORK_PHASES = (
    "emit_activation_field", "deposit_normalization", "forces_geometry_pressure", "gather_contact_acceptance",
)
JOB_WALL_PHASES = ("cold_setup", "scheduling", *WORK_PHASES, "snapshot_cache", "drain")


def nonnegative_seconds(receipt, field):
    value = receipt.get(field)
    require(finite_number(value) and value >= 0, f"invalid {field}")
    return value


def verify_job_measurement(receipt, backend):
    require(isinstance(receipt, dict) and receipt.get("event") == "job_measurement", "missing job measurement")
    require(receipt.get("backend") == backend and receipt.get("command") == "run" and receipt.get("status") == "OK",
            "job measurement does not belong to a successful run")
    attempts = integer(receipt, "attempt_count")
    settled = integer(receipt, "settled_attempt_count")
    require(settled <= attempts, "job attempt counts are inconsistent")
    wall = receipt.get("host_wall")
    require(isinstance(wall, dict), "missing job host wall measurement")
    phase_seconds = [nonnegative_seconds(wall, f"{phase}_seconds") for phase in JOB_WALL_PHASES]
    total = nonnegative_seconds(wall, "total_seconds")
    require(total > 0 and math.isclose(math.fsum(phase_seconds), total, rel_tol=1e-12, abs_tol=1e-12),
            "host wall phases do not partition total seconds")
    if settled:
        require(all(wall[f"{phase}_seconds"] > 0 for phase in WORK_PHASES),
                "missing attempted work in host wall phases")
    events = receipt.get("cuda_events")
    memory = receipt.get("cuda_memory")
    if backend == "cpu":
        require("cuda_events" in receipt and "cuda_memory" in receipt and events is None and memory is None,
                "CUDA evidence on a CPU job")
        return
    require(isinstance(events, dict), "missing CUDA event evidence")
    for phase in (*WORK_PHASES, "snapshot_d2h"):
        seconds = nonnegative_seconds(events, f"{phase}_seconds")
        intervals = integer(events, f"{phase}_interval_count")
        require(intervals > 0 or seconds == 0, f"CUDA event interval missing for {phase}")
        if phase in WORK_PHASES and settled:
            require(seconds > 0 and intervals >= settled, f"missing completed CUDA event intervals for {phase}")
    field_seconds = nonnegative_seconds(events, "pre_topology_field_seconds")
    field_intervals = integer(events, "pre_topology_field_interval_count")
    require(field_seconds <= events["emit_activation_field_seconds"]
            and settled <= field_intervals <= events["emit_activation_field_interval_count"]
            and (settled == 0 or field_seconds > 0)
            and (field_intervals > 0 or field_seconds == 0), "invalid pre_topology_field event subset")
    require(isinstance(memory, dict), "missing CUDA memory evidence")
    device_total = integer(memory, "total_device_bytes", minimum=1)
    free_start = integer(memory, "free_device_bytes_at_start", minimum=1)
    budget = integer(memory, "budget_bytes", minimum=1)
    limit = integer(memory, "device_byte_limit", minimum=1)
    require(free_start <= device_total and budget == min(7 * device_total // 10, 4 * free_start // 5)
            and limit <= budget, "invalid fixed device budget")
    for extent, description in (
        ("device_reserved", "aggregate device reservation"),
        ("pinned_reserved", "aggregate pinned reservation"),
        ("snapshot_pinned_reserved", "aggregate pinned reservation"),
        ("physical_state", "physical state extent"),
        ("device_scratch", "device scratch extent"),
        ("snapshot_output", "snapshot output extent"),
    ):
        require(integer(memory, f"current_{extent}_bytes") <= integer(memory, f"peak_{extent}_bytes"),
                f"inconsistent {description}")
    device_peak = memory["peak_device_reserved_bytes"]
    require(0 < device_peak <= limit, "aggregate device reservation exceeds the fixed ceiling")
    require(0 < memory["peak_snapshot_pinned_reserved_bytes"] <= memory["peak_pinned_reserved_bytes"]
            and memory["current_snapshot_pinned_reserved_bytes"] <= memory["current_pinned_reserved_bytes"],
            "snapshot reservation exceeds aggregate pinned reservation")
    require(0 < memory["peak_physical_state_bytes"] <= device_peak
            and memory["current_physical_state_bytes"] <= memory["current_device_reserved_bytes"],
            "physical state extent exceeds its reservation")
    require(0 < memory["peak_device_scratch_bytes"] <= device_peak
            and memory["current_physical_state_bytes"] + memory["current_device_scratch_bytes"]
            <= memory["current_device_reserved_bytes"], "device scratch extent exceeds its reservation")
    active_scratch_peak = integer(memory, "peak_active_device_scratch_bytes", minimum=int(settled > 0))
    require(active_scratch_peak <= memory["peak_device_scratch_bytes"], "active device scratch extent exceeds reservation")
    require(0 < memory["peak_snapshot_output_bytes"] <= memory["peak_snapshot_pinned_reserved_bytes"]
            and memory["current_snapshot_output_bytes"] <= memory["current_snapshot_pinned_reserved_bytes"],
            "snapshot output extent exceeds its reservation")
    integer(memory, "device_reservation_attempts", minimum=1)
    require(all(integer(memory, field) == 0 for field in (
        "budget_refusals", "quarantined_owner_count",
        "remaining_device_reserved_bytes_after_drain", "remaining_pinned_reserved_bytes_after_drain",
    )), "successful CUDA run did not release its job reservations")
    require("free_device_bytes_after_drain" in memory and (
        memory["free_device_bytes_after_drain"] is None
        or integer(memory, "free_device_bytes_after_drain") <= device_total
    ), "invalid optional free-device observation")


def verify_solver(
    records, frames, authored_scene_identity, frame_ordinals=FRAME_ORDINALS, *,
    tank_layout=False, require_job_measurement=False, particle_count=1, maximum_dt=0.01,
):
    final = records[-1]
    backend = authored_scene_identity["backend"]
    require(final.get("event") == "result" and final.get("command") == "run", "missing successful run result")
    matches_identity(final, authored_scene_identity)
    require(final.get("read_only") is False, "run did not publish")
    require(integer(final, "frames_per_second") == FRAMES_PER_SECOND, "wrong receipt frame rate")
    verify_extent(final, particle_count)
    verify_snapshot_receipt(final, backend)
    if backend == "cuda":
        integer(final, "graph_replay_count", minimum=1)
        integer(final, "device_reserved_bytes", minimum=1)

    measurement = records[-2] if len(records) > 1 and records[-2]["event"] == "job_measurement" else None
    if measurement is not None or require_job_measurement:
        verify_job_measurement(measurement, backend)
    events_end = -2 if measurement is not None else -1

    # Each accepted step emits one trace. Attempt counters are cumulative within
    # a frame and start again after publication.
    published_count = 0
    accepted_trace_count = 0
    previous_dt_retry_count = 0
    last_clock = None
    last_trace = None
    frame_start = None
    frame_attempts = []
    frame_traces = []
    job_attempt_count = 0
    job_settled_attempt_count = 0
    for record in records[:events_end]:
        event = record["event"]
        if event == "heartbeat":
            require("backend" in record and record["backend"] in (None, backend), "wrong heartbeat backend")
            integer(record, "elapsed_milliseconds")
            last_published = record["last_published_frame"]
            if last_published is not None:
                require(type(last_published) is int and 0 <= last_published < published_count,
                        "heartbeat names an unpublished frame")
            settled = record["last_settled_clock"]
            if settled is not None:
                frame = integer(settled, "frame")
                require(frame <= min(published_count, frame_ordinals[-1]), "invalid heartbeat frame")
                for field in CLOCK_FIELDS[:-1]:
                    require(integer(settled, field) == frames[frame][field], "heartbeat and manifest clocks differ")
            continue
        if event == "frame":
            require(published_count < len(frame_ordinals), "extra publication receipt")
            frame_ordinal = frame_ordinals[published_count]
            require(record.get("published") is True and integer(record, "frame") == frame_ordinal,
                    "wrong publication ordinal")
            verify_snapshot_receipt(record, backend)
            manifest_clock = clock(frames[frame_ordinal])
            require(manifest_clock == (last_clock or (0, 0, 0, 0, 0.0)), "publication and manifest clocks differ")
            if published_count:
                verify_attempt_ledger(frame_attempts, frame_traces, frame_start, frame_ordinal)
            frame_start = last_clock = manifest_clock
            frame_attempts.clear()
            frame_traces.clear()
            published_count += 1
            accepted_trace_count = 0
            previous_dt_retry_count = 0
            continue
        if event == "attempt":
            job_attempt_count += 1
            require(0 < published_count < len(frame_ordinals) and not frame_traces, "attempt ledger outside frame")
            frame_attempts.append(record)
            continue
        require(event == "trace" and record.get("published") is False, "unexpected run event")
        require(0 < published_count < len(frame_ordinals), "trace outside expected run frames")
        trace_clock = clock(record)
        frame, substep, _, accepted_step, time = trace_clock
        previous_frame, previous_substep, _, previous_step, previous_time = last_clock
        require(frame == previous_frame + (substep == 0), "wrong trace frame transition")
        require(substep == 0 or substep == previous_substep + 1, "wrong trace substep")
        require(accepted_step == previous_step + 1, "missing or duplicate accepted step")
        require(0 < time - previous_time <= maximum_dt + 1e-15, "invalid accepted duration")
        if substep == 0:
            require(frame == published_count and time == frame / FRAMES_PER_SECOND, "wrong completed trace clock")
        else:
            require(frame == published_count - 1 and time < published_count / FRAMES_PER_SECOND, "wrong interior trace clock")
        verify_extent(record, particle_count)
        verify_accepted_measurement(
            record, last_clock, published_count / FRAMES_PER_SECOND, tank=tank_layout,
        )
        require(record["accepted_measurement"]["accepted_dt_seconds"] <= maximum_dt,
                "accepted duration exceeds frozen maximum dt")
        frame_traces.append(record)
        accepted_trace_count += 1
        dt_retry_count = integer(record, "dt_retries")
        require(dt_retry_count <= 2, "pressure retry bound exceeded")
        require(dt_retry_count >= previous_dt_retry_count, "pressure retry count regressed within the frame")
        require(integer(record, "capacity_replays") == 0, "unexpected capacity replay")
        require(integer(record, "accepted_attempts") == accepted_trace_count,
                "accepted attempt count differs from the accepted trace count")
        require(integer(record, "attempts") == accepted_trace_count + dt_retry_count, "inconsistent attempt counts")
        previous_dt_retry_count = dt_retry_count
        job_settled_attempt_count += 1
        digest(record, "state_digest")
        last_clock = trace_clock
        last_trace = record

    if measurement is not None:
        require(measurement["attempt_count"] == job_attempt_count
                and measurement["settled_attempt_count"] == job_settled_attempt_count,
                "job measurement disagrees with the attempt ledger")
        if backend == "cuda":
            require(measurement["cuda_events"]["snapshot_d2h_interval_count"] == len(frame_ordinals)
                    and measurement["cuda_events"]["snapshot_d2h_seconds"] > 0,
                    "CUDA snapshot event intervals differ from completed frames")
            require(measurement["cuda_memory"]["current_device_reserved_bytes"] == final["device_reserved_bytes"],
                    "CUDA job reservation differs from the result")
    require(published_count == len(frame_ordinals), "missing publication receipt")
    require(clock(final) == last_clock == clock(frames[-1]), "final clock and manifest differ")
    require(last_trace is not None and digest(final, "state_digest") == last_trace["state_digest"], "final state differs")
    for field in (*MAXIMUM_COUNTS, "attempts", "accepted_attempts", "dt_retries", "capacity_replays"):
        require(integer(final, field) == last_trace[field], f"final and trace {field} differ")
    return final


def verify_frames(records, authored_scene_identity, frame_ordinals):
    require(len(records) == len(frame_ordinals), "missing or extra verified frame")
    for ordinal, record in zip(frame_ordinals, records):
        require(record["event"] == "frame", "unexpected frame reader event")
        matches_identity(record, authored_scene_identity)
        frame, substep, _, _, time = clock(record)
        require(frame == ordinal and substep == 0 and time == ordinal / FRAMES_PER_SECOND, "wrong manifest clock")
        digest(record, "points_digest")
        integer(record, "points_byte_length", minimum=1)
        for point in record.get("points", []):
            require(integer(point, "level") == 0 and integer(point, "tag") == 0, "wrong point level or tag")
            for field in ("radius", "volume_ratio"):
                require(finite_number(point.get(field)) and point[field] > 0, f"invalid point {field}")


def verify_checkpoint(checkpoint, frame_zero, authored_scene_identity, initial_position):
    matches_identity(checkpoint, authored_scene_identity)
    require(clock(checkpoint) == clock(frame_zero), "retained checkpoint must match frame zero")
    require(digest(checkpoint, "state_digest") == authored_scene_identity["seed_digest"], "retained state differs from seed")
    require(integer(checkpoint, "random_seed") == 0 and integer(checkpoint, "next_particle_id") == 2, "wrong seed metadata")
    require(checkpoint.get("capacities") == [1024, 128, 4096, 24576, 256], "wrong capacity reservations")
    require(checkpoint.get("pressure_cells") == [], "initial checkpoint contains pressure cells")
    expected_particle = {
        "id": 1, "tag": 0, "world_position": initial_position, "world_velocity": [0, 0, 0],
        "velocity_slopes": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "side_tag": 0,
        "mass": 1000, "reference_volume": 1, "volume_ratio": 1, "affine_enabled": True, "pic_initialized": False,
    }
    require(checkpoint.get("particles") == [expected_particle], "initial checkpoint particle differs from frozen seed")
    point = frame_zero["points"][0]
    point_volume = 4 * math.pi / 3 * point["radius"] ** 3
    require(abs(point_volume - 1) <= 1e-14 and point["volume_ratio"] == 1, "initial point volume differs from seed")


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_cuda_compatibility_record(cache):
    # Match the pinned CLI's exact byte format, including LF and lowercase hex.
    try:
        with (cache / CUDA_COMPATIBILITY_FILE).open("rb") as stream:
            record = stream.read(CUDA_COMPATIBILITY_RECORD_SIZE + 1)
    except FileNotFoundError as error:
        raise ValueError("missing CUDA compatibility record") from error
    require(re.fullmatch(rb"FLCUDA1\n[0-9a-f]{64}\n", record) is not None, "malformed CUDA compatibility record")
    return record


def snapshot(root):
    """Record content and modification time to detect reader writes."""
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*") if path.is_file()
    }


def verify_tank_layout(frames, solver_records):
    # Four signed wet pressure cells at y=0. Each 4^3 block owns 192
    # staggered slots. Four wet blocks plus three seed-support blocks are active.
    expected = {"block_count": 7, "cell_count": 4, "face_count": 1344}
    for record in solver_records:
        if record["event"] in ("trace", "result"):
            for field, count in expected.items():
                require(integer(record, field) == count, f"unexpected tank-layout {field}")
    traces = [record for record in solver_records if record["event"] == "trace"]
    require(bool(traces), "missing tank-layout accepted trace")
    first = traces[0]["accepted_measurement"]
    # Two unit-width y-z patches are wet up to 0.75 m and move at +0.125 m/s.
    # Four selected unit-base cells start with liquid height 0.75 m.
    require(first["step_start_inward_piston_flux_m3_per_s"] == 0.1875, "wrong first inward piston flux")
    require(first["step_start_pressure_cell_volume_m3"] == 3, "wrong first frozen pressure volume")
    require(first["step_start_piston_low_x_m"] == -1, "wrong first piston start pose")
    require(first["step_end_piston_low_x_m"] == -1 + 0.125 * first["accepted_dt_seconds"],
            "wrong first piston end pose")
    for trace in traces:
        particles = trace.get("geometry", {}).get("particles")
        require(isinstance(particles, list) and len(particles) == 1, "missing accepted geometry particles")
        require(integer(particles[0], "id") == 1, "wrong geometry particle ID")
        position = particles[0].get("world_position_m")
        require(isinstance(position, list) and len(position) == 3 and all(map(finite_number, position)),
                "invalid accepted world position")
    require(traces[-1]["geometry"]["particles"][0]["world_position_m"] == frames[-1]["points"][0]["world_position"],
            "settled trace geometry differs from verified points")
    for frame in frames:
        points = frame["points"]
        require(len(points) == 1 and integer(points[0], "id") == 1, "wrong tank-layout point inventory")
        for field in ("world_position", "world_velocity"):
            values = points[0][field]
            require(len(values) == 3 and all(finite_number(value) for value in values), f"invalid point {field}")
    require(frames[0]["points"][0]["world_position"] == TANK_LAYOUT_INITIAL_POSITION, "wrong tank-layout seed point")
    return {"passed": True, "observed_layout": expected, "point_ids": [1]}


def verify_tank_refusals(output, solver, backend, scene, seed):
    # Keep the plane tag and omit its optional table (Field id 3).
    malformed = bytearray(scene.read_bytes())
    root = struct.unpack_from("<I", malformed)[0]
    root_vtable = root - struct.unpack_from("<i", malformed, root)[0]
    field_slot = root + struct.unpack_from("<H", malformed, root_vtable + 4 + 2 * 11)[0]
    field = field_slot + struct.unpack_from("<I", malformed, field_slot)[0]
    field_vtable = field - struct.unpack_from("<i", malformed, field)[0]
    struct.pack_into("<H", malformed, field_vtable + 4 + 2 * 3, 0)
    malformed_scene = output / "missing-plane.flrs"
    malformed_scene.write_bytes(malformed)

    # At frame 385 the positive piston stroke has exceeded the two-meter width.
    controls = (
        ("missing-plane", malformed_scene, 1, "scene", "DATA_LOSS"),
        ("closed-tank", scene, 385, "frame", "FAILED_PRECONDITION"),
    )
    for control, scene_path, frame_end, field_name, status_name in controls:
        cache = output / f"{control}-cache"
        stdout_path = output / f"{control}.stdout.jsonl"
        stderr_path = output / f"{control}.stderr.txt"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.run(
                [str(solver), "run", "--stage", "simulate", "--scene", str(scene_path),
                 "--initial-state", str(seed), "--backend", backend, "--cache", str(cache),
                 "--frame-end", str(frame_end)],
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=60, check=False,
            )
        require(process.returncode == 1, f"{control} returned {process.returncode}")
        diagnostic = one_record(read_records(stdout_path.read_text(encoding="utf-8"), "protocol_version"), "error")
        require(read_records(stderr_path.read_text(encoding="utf-8"), "protocol_version") == [diagnostic],
                f"{control} diagnostic streams differ")
        require(diagnostic.get("field") == field_name and diagnostic.get("name") == status_name,
                f"unexpected {control} diagnostic")
        require(not cache.exists() and not cache.is_symlink(), f"{control} published a cache")


def verify_resolved_tank(resolved, authored, case, maximum_dt):
    matches_identity(resolved, authored)
    for field in ("solver_digest", "seed_digest"):
        require(digest(resolved, field) == digest(authored, field), f"resolved tank has wrong {field}")
    require(integer(resolved, "seed_byte_length", minimum=1) == integer(authored, "seed_byte_length", minimum=1),
            "resolved tank has wrong seed byte length")
    hydrostatic = case == HYDROSTATIC_CASE
    expected = {
        "world_origin_m": [0, 0, 0],
        "cell_size_m": 0.125 if hydrostatic else 0.0625,
        "random_seed": 0,
        "gravity_m_per_s2": [0, -9.81, 0],
        "density_kg_per_m3": 1000,
        "surface_tension_n_per_m": 0,
        "ambient_gauge_pa": 0,
        "target_divergence_per_s": 0,
        "source_count": 0,
        "friction_coefficient": 0,
        "penetration_tolerance_m": 0,
        "minimum_face_mass_kg": 0,
        "mass_kg_per_quantum": 2 ** -40,
        "momentum_kg_m_per_s_per_quantum": 2 ** -40,
        "pressure_iteration_limit": 256,
        "energy_blend": 0.6,
        "maximum_dt_seconds": maximum_dt,
        "tank_low_x_m": -0.25 if hydrostatic else -0.5,
        "tank_high_x_m": 0.25 if hydrostatic else 0.5,
        "tank_floor_y_m": 0,
        "tank_low_z_m": -0.25,
        "tank_high_z_m": 0.25,
        "piston_velocity_x_m_per_s": 0 if hydrostatic else 0.015625,
    }
    expected.update({"surface_height_y_m": 1} if hydrostatic else {"sphere_union_radius_m": PISTON_PARTICLE_RADIUS_M})
    for field, value in expected.items():
        require(resolved.get(field) == value, f"resolved tank has wrong {field}")
    require(resolved.get("incompressible") is True, "resolved tank declares compression")


def verify_tank_state(state, frame, particle_count, h, *, initial=False):
    require(clock(state) == clock(frame), "tank state and committed frame clocks differ")
    require(integer(state, "random_seed") == 0 and integer(state, "next_particle_id") == particle_count + 1,
            "wrong tank state seed or next ID")
    particles = state.get("particles")
    points = frame.get("points")
    check_ids(particles, particle_count, "tank state")
    check_ids(points, particle_count, "tank state point frame")
    reference_volume = h ** 3
    state_volumes = []
    for ordinal, (particle, point) in enumerate(zip(particles, points), 1):
        particle_id = particle["id"]
        require(particle_id == point["id"], "tank state and point ID orders differ")
        require(particle["reference_volume"] == reference_volume and particle["mass"] == 1000 * reference_volume,
                f"wrong reference volume or mass for ID {particle_id}")
        for field in ("tag", "world_position", "world_velocity", "volume_ratio"):
            require(particle[field] == point[field], f"tank state and points disagree on {field} for ID {particle_id}")
        ratio = particle["volume_ratio"]
        require(finite_number(ratio) and ratio > 0, f"invalid numerical volume ratio for ID {particle_id}")
        material_volume = reference_volume * ratio
        point_volume = 4 * math.pi / 3 * point["radius"] ** 3
        require(math.isclose(point_volume, material_volume, rel_tol=1e-14),
                f"point radius disagrees with V0*J for ID {particle_id}")
        state_volumes.append(material_volume)
        if initial:
            require(ratio == 1, f"wrong seed volume ratio for ID {particle_id}")
            require(particle_id == ordinal, "wrong initial lexicographic seed ID order")
            x_count = 4 if h == 0.125 else 16
            z_count = 4 if h == 0.125 else 8
            y_layer, x_offset = divmod(particle_id - 1, x_count)
            z_offset, y_index = divmod(y_layer, 8)
            expected_position = [(x_offset - x_count / 2 + 0.5) * h, (y_index + 0.5) * h,
                                 (z_offset - z_count / 2 + 0.5) * h]
            require(particle["world_position"] == expected_position, f"wrong seed position for ID {particle_id}")
            require(particle["world_velocity"] == [0, 0, 0] and particle["velocity_slopes"] == [[0, 0, 0]] * 3,
                    f"nonzero seed velocity or slopes for ID {particle_id}")
            require(particle["side_tag"] == 0 and particle["affine_enabled"] is True
                    and particle["pic_initialized"] is False, f"wrong seed transfer flags for ID {particle_id}")
    return {"frame": frame["frame"], "state_material_volume_m3": math.fsum(state_volumes),
            "mass_kg": math.fsum(particle["mass"] for particle in particles)}


def prepare_tank_steps(traces, frames, initial_state, particle_count, *, piston):
    steps = []
    start_positions = [particle["world_position"] for particle in initial_state["particles"]]
    for trace in traces:
        measurement = trace["accepted_measurement"]
        require(measurement["step_start_capillary_bound_seconds"] is None, "unexpected capillary bound")
        if not piston:
            require(measurement["step_start_inward_piston_flux_m3_per_s"] == 0
                    and measurement["step_start_piston_low_x_m"] == measurement["step_end_piston_low_x_m"] == -0.25,
                    "stationary column reported moving-wall flux or pose")
            continue
        geometry = trace.get("geometry")
        require(isinstance(geometry, dict), "missing accepted piston geometry")
        particles = geometry.get("particles")
        check_ids(particles, particle_count, f"accepted trace {trace['accepted_step']}")
        end_positions = [vector3(particle.get("world_position_m")) for particle in particles]
        if trace["substep"] == 0:
            points = frames[trace["frame"]]["points"]
            require([(particle["id"], position) for particle, position in zip(particles, end_positions)]
                    == [(point["id"], point["world_position"]) for point in points],
                    "accepted geometry and verified frame points differ in ID order or position")
        steps.append({
            "accepted_step": trace["accepted_step"],
            "time_seconds": trace["time_seconds"],
            "start_positions": start_positions,
            "accepted_dt_seconds": measurement["accepted_dt_seconds"],
            "step_start_piston_low_x_m": measurement["step_start_piston_low_x_m"],
            "step_end_piston_low_x_m": measurement["step_end_piston_low_x_m"],
            "step_start_inward_piston_flux_m3_per_s": measurement["step_start_inward_piston_flux_m3_per_s"],
        })
        start_positions = end_positions
    return steps


def verify_frozen_tank(output, fixture, backend, case, device, authored, records, *,
                       capture, resolved, verify_inputs, cache_files):
    last_frame, particle_count, maximum_dt, retained_frames = TANK_CASES[case]
    frame_ordinals = tuple(range(last_frame + 1))
    seed = capture / "scene.flrs.flst"
    cache = capture / "cache"
    fingerprint = authored["resolved_fingerprint"]
    seed_digest = authored["seed_digest"]
    require(seed.stat().st_size == integer(authored, "seed_byte_length", minimum=1), "wrong seed byte length")
    expected_manifests = {
        f"frames/{frame:020d}.{suffix}" for frame in frame_ordinals for suffix in ("flmf", "json")
    }
    # Immutable references remain for historical replay. Only the two
    # eligible checkpoint payloads are retained as .flst blobs.
    expected_checkpoints = {f"checkpoints/{frame:020d}.flcr" for frame in range(0, last_frame + 1, 10)}
    checkpoint_payloads = [name for name in cache_files if Path(name).parts[0] == "blobs" and name.endswith(".flst")]
    require(len(checkpoint_payloads) == len(retained_frames), "wrong retained checkpoint payload count")
    for directory, expected in (("frames", expected_manifests), ("checkpoints", expected_checkpoints)):
        actual = {name for name in cache_files if Path(name).parts[0] == directory}
        require(actual == expected, f"unexpected committed {directory} inventory")
    initial_state = one_record(
        run_child(output, "verified-initial-state", fixture, "read-seed", seed, seed_digest, fixture=True), "seed",
    )
    require(digest(initial_state, "state_digest") == seed_digest, "wrong initial state digest")
    frames = run_child(output, "verified-frames", fixture, "read-frames", cache, fingerprint, 0, last_frame,
                       fixture=True, timeout=180)
    verify_frames(frames, authored, frame_ordinals)
    for frame in frames:
        check_ids(frame.get("points"), particle_count, f"frame {frame['frame']}")
        # The verified scene has no source or compliance. Accepted receipts
        # check the source inventory, while numeric J evolves independently.
        frame["source_volume_m3"] = frame["compression_volume_m3"] = 0
    final = verify_solver(records, frames, authored, frame_ordinals, tank_layout=True, require_job_measurement=True,
                          particle_count=particle_count, maximum_dt=maximum_dt)
    h = resolved["cell_size_m"]
    material_checks = [verify_tank_state(initial_state, frames[0], particle_count, h, initial=True)]
    traces = [record for record in records if record["event"] == "trace"]
    settled_by_frame = {trace["frame"]: trace for trace in traces if trace["substep"] == 0}
    checkpoints = []
    for ordinal in retained_frames:
        checkpoint = one_record(
            run_child(output, f"verified-checkpoint-{ordinal}", fixture, "read-checkpoint", cache, fingerprint,
                      ordinal, fixture=True), "checkpoint",
        )
        matches_identity(checkpoint, authored)
        require(checkpoint["capacities"] == [1024, 128, 4096, 24576, 256], "wrong checkpoint capacity reservations")
        require(digest(checkpoint, "state_digest") == settled_by_frame[ordinal]["state_digest"],
                "retained checkpoint differs from the settled trace")
        material_checks.append(verify_tank_state(checkpoint, frames[ordinal], particle_count, h))
        checkpoints.append(checkpoint)
    verify_inputs()

    piston = case != HYDROSTATIC_CASE
    steps = prepare_tank_steps(traces, frames, initial_state, particle_count, piston=piston)
    if piston:
        with (output / "piston-spatial-progress.jsonl").open("x", encoding="utf-8", buffering=1) as progress:
            def record_progress(sample):
                progress.write(json.dumps(sample, allow_nan=False) + "\n")

            measurements = evaluate_piston_displacement(frames, steps, maximum_dt=maximum_dt, progress=record_progress)
        measurements["component_inventory"] = evaluate_transport_components(frames, PISTON_PARTICLE_RADIUS_M, h)
    else:
        measurements = evaluate_hydrostatic_column(frames, checkpoints)
    measurements.update({
        "case": case,
        "backend": backend,
        "device_probe": device,
        "scene_author": authored,
        "resolved_tank": resolved,
        "final_receipt": final,
        "job_measurement": records[-2],
        "material_state_checks": material_checks,
        "retained_checkpoint_frames": list(retained_frames),
        "verified_frame_count": len(frames),
        "verified_point_digests": [frame["points_digest"] for frame in frames],
        "verified_immutable_cache_file_count": len(cache_files),
        "observed_maximum_extents": {field: max(trace[field] for trace in traces) for field in MAXIMUM_COUNTS},
        "step_start_pressure_cell_volume_m3": [trace["accepted_measurement"]["step_start_pressure_cell_volume_m3"]
                                               for trace in traces],
    })
    return measurements


def write_measurements(path, measurements):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(measurements, stream, indent=2, allow_nan=False)
        stream.write("\n")


def is_link(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def regular_file(path):
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing file {path}") from error
    require(not is_link(info), f"symlink or reparse point in evidence: {path}")
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
    return path


def capture_inventory(capture):
    """Hash regular files in path order, retaining mtimes for reader checks."""
    files = {}

    def visit(directory):
        info = directory.lstat()
        require(not is_link(info), f"symlink or reparse point in capture: {directory}")
        require(stat.S_ISDIR(info.st_mode), f"not a capture directory: {directory}")
        for path in sorted(directory.iterdir()):
            if path == capture / CAPTURE_MANIFEST:
                continue
            info = path.lstat()
            require(not is_link(info), f"symlink or reparse point in capture: {path}")
            if stat.S_ISDIR(info.st_mode):
                visit(path)
            else:
                require(stat.S_ISREG(info.st_mode), f"not a regular capture file: {path}")
                files[path.relative_to(capture).as_posix()] = (info.st_mtime_ns, file_sha256(path))

    visit(capture)
    return files


def inventory_hashes(inventory):
    return [[name, inventory[name][1]] for name in sorted(inventory)]


def verify_binary_pins(solver, fixture, pins):
    require(isinstance(pins, dict), "frozen tank requires solver and fixture SHA-256 pins")
    for kind, path in (("solver", solver), ("fixture", fixture)):
        expected = digest(pins, kind)
        require(file_sha256(regular_file(path)) == expected, f"{kind} SHA-256 differs from the expected binary")


def pinned_json(path, expected_sha256, description):
    require(isinstance(expected_sha256, str) and DIGEST_PATTERN.fullmatch(expected_sha256),
            f"invalid trusted {description} SHA-256")
    raw = regular_file(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256,
            f"{description} SHA-256 differs from the trusted digest")
    record = json.loads(raw, object_pairs_hook=unique_object, parse_float=finite_float, parse_constant=reject_constant)
    require(isinstance(record, dict), f"{description} must be a JSON object")
    return record


def schema_build_digest(records):
    result = one_record(records, "result")
    require(result.get("command") == "schema" and result.get("execution_available") is True,
            "missing solver schema result")
    return digest(result, "build_digest")


def verify_authored_tank(authored, case, backend, solver_digest):
    require(authored.get("case") == case and authored.get("backend") == backend, "wrong authored tank case or backend")
    require(digest(authored, "solver_digest") == solver_digest, "authored solver_digest differs from pinned solver schema")
    for field in ("resolved_fingerprint", "execution_id", "seed_digest"):
        digest(authored, field)


def measured_execution(final, backend):
    require(final.get("event") == "result" and final.get("command") == "run" and final.get("read_only") is False,
            "missing successful writable run result")
    if backend == "cpu":
        require(not any(field.startswith("cuda_execution_device_") or field == "compiled_architecture" for field in final),
                "unexpected CUDA execution identity on CPU result")
        return {}
    verify_snapshot_receipt(final, backend)
    return {
        "cuda_execution_device_digest": digest(final, "cuda_execution_device_digest"),
        "cuda_execution_device_index": integer(final, "cuda_execution_device_index"),
        "compiled_architecture": final["compiled_architecture"],
    }


def inspect_capture(capture, inventory, pins, backend, case, solver_digest):
    receipts = []
    for name, version in (("solver-schema", "protocol_version"), ("device-probe", "protocol_version"),
                          ("scene-author", "fixture_version"), ("simulate-run", "protocol_version")):
        stdout_name, stderr_name = f"{name}.stdout.jsonl", f"{name}.stderr.txt"
        require(stdout_name in inventory and stderr_name in inventory, f"missing {name} capture receipts")
        require((capture / stderr_name).stat().st_size == 0, f"captured {name} produced diagnostics")
        receipts.append(read_records((capture / stdout_name).read_text(encoding="utf-8"), version))
    schema, device, authored, records = receipts
    require(schema_build_digest(schema) == solver_digest, "captured solver_digest differs from pinned solver schema")
    device = one_record(device, "result")
    authored = one_record(authored, "scene")
    verify_device(device, backend)
    verify_authored_tank(authored, case, backend, solver_digest)
    measurement = records[-2] if len(records) >= 2 else {}
    require(measurement.get("event") == "job_measurement" and measurement.get("backend") == backend
            and measurement.get("command") == "run" and measurement.get("status") == "OK",
            "missing captured successful job measurement")
    final = records[-1]
    matches_identity(final, authored)
    require(integer(final, "frame") == TANK_CASES[case][0], "captured run did not complete the frozen frames")
    execution = measured_execution(final, backend)
    compatibility_path = f"cache/{CUDA_COMPATIBILITY_FILE}"
    if backend == "cuda":
        require(execution["cuda_execution_device_index"] == device["device_index"], "actual CUDA device index differs from probe")
        require(compatibility_path in inventory, "missing CUDA compatibility record")
        execution["cuda_compatibility_record"] = read_cuda_compatibility_record(capture / "cache").decode("ascii")
    else:
        require(compatibility_path not in inventory, "unexpected CUDA compatibility record on CPU capture")
    for name in ("scene.flrs", "scene.flrs.flst", "cache/resolved.flrs"):
        require(name in inventory, f"missing captured {name}")
    require(inventory["scene.flrs"][1] == inventory["cache/resolved.flrs"][1],
            "cached and authored resolved tank differ")
    provenance = {
        "manifest_version": CAPTURE_MANIFEST_VERSION,
        "case": case,
        "backend": backend,
        "solver_sha256": pins["solver"],
        "fixture_sha256": pins["fixture"],
        **{field: authored[field] for field in ("solver_digest", "seed_digest", "resolved_fingerprint", "execution_id")},
        **execution,
    }
    return provenance, device, authored, records


def check_manifest_inventory(manifest, inventory):
    require(type(manifest.get("manifest_version")) is int and manifest["manifest_version"] == CAPTURE_MANIFEST_VERSION,
            "wrong capture manifest version")
    require(manifest.get("files") == inventory_hashes(inventory), "capture file inventory differs from the manifest")


def evaluate_frozen_capture(capture, output, solver, fixture, pins, backend, case, solver_digest, inventory,
                            *, manifest=None, manifest_sha256=None):
    provenance, device, authored, records = inspect_capture(capture, inventory, pins, backend, case, solver_digest)
    expected_manifest = provenance | {"files": inventory_hashes(inventory)}
    if manifest is not None:
        # Serializing distinguishes JSON booleans from integers and rejects unexpected fields.
        require(json.dumps(manifest, sort_keys=True) == json.dumps(expected_manifest, sort_keys=True),
                "capture manifest identity differs from the pinned run and scene")
    resolved = one_record(
        run_child(output, "verified-resolved-tank", fixture, "read-tank-scene", capture / "cache",
                  authored["resolved_fingerprint"], fixture=True),
        "resolved_tank",
    )
    require(digest(resolved, "solver_digest") == solver_digest, "decoded solver_digest differs from pinned solver schema")
    verify_resolved_tank(resolved, authored, case, TANK_CASES[case][2])

    def verify_inputs():
        require(capture_inventory(capture) == inventory, "verified readers changed capture files")
        verify_binary_pins(solver, fixture, pins)
        if manifest_sha256 is not None:
            require(file_sha256(regular_file(capture / CAPTURE_MANIFEST)) == manifest_sha256,
                    "verified readers changed capture manifest")

    verify_inputs()
    if manifest is None:
        manifest_path = capture / CAPTURE_MANIFEST
        require(not manifest_path.exists() and not manifest_path.is_symlink(), "capture manifest path must be fresh")
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            pending = Path(temporary) / CAPTURE_MANIFEST
            write_measurements(pending, expected_manifest)
            manifest_sha256 = file_sha256(pending)
            pending.replace(manifest_path)
        print(f"capture manifest SHA-256 {manifest_sha256}", flush=True)
    cache_files = [name.removeprefix("cache/") for name in inventory if name.startswith("cache/")]
    measurements = verify_frozen_tank(output, fixture, backend, case, device, authored, records, capture=capture,
                                      resolved=resolved, verify_inputs=verify_inputs, cache_files=cache_files)
    verify_binary_pins(solver, fixture, pins)
    measurements["capture_provenance"] = provenance | {"manifest_sha256": manifest_sha256}
    write_measurements(output / f"{case}-measurements.json", measurements)
    return measurements


def verify_captured_tank(capture, output, solver, fixture, backend, case, *, pins, manifest_sha256):
    """Verify a capture under an external digest and write receipts to a fresh directory."""
    require(not output.resolve().is_relative_to(capture.resolve()), "verification output must be outside the capture")
    require(not is_link(capture.lstat()), "symlink or reparse point in capture root")
    manifest = pinned_json(capture / CAPTURE_MANIFEST, manifest_sha256, "capture manifest")
    inventory = capture_inventory(capture)
    check_manifest_inventory(manifest, inventory)
    verify_binary_pins(solver, fixture, pins)
    output.mkdir(parents=True, exist_ok=False)
    solver_digest = schema_build_digest(run_child(output, "verified-solver-schema", solver, "schema"))
    return evaluate_frozen_capture(capture, output, solver, fixture, pins, backend, case, solver_digest, inventory,
                                   manifest=manifest, manifest_sha256=manifest_sha256)


def run_frozen_tank(solver, fixture, backend, case, output, pins):
    verify_binary_pins(solver, fixture, pins)
    last_frame, particle_count, _, _ = TANK_CASES[case]
    capture = output / "capture"
    capture.mkdir()
    scene = capture / "scene.flrs"
    seed = scene.with_suffix(".flrs.flst")
    solver_digest = schema_build_digest(run_child(capture, "solver-schema", solver, "schema"))
    device = one_record(run_child(capture, "device-probe", solver, "device-info", "--backend", backend), "result")
    verify_device(device, backend)
    authored = one_record(
        run_child(capture, "scene-author", fixture, "write-scene", case, backend, solver, scene, fixture=True), "scene",
    )
    verify_authored_tank(authored, case, backend, solver_digest)
    launch_inputs = capture_inventory(capture)
    require(scene.name in launch_inputs and seed.name in launch_inputs, "missing authored scene or seed")
    arguments = ("run", "--stage", "simulate", "--scene", scene, "--initial-state", seed, "--backend", backend,
                 "--trace" if case == HYDROSTATIC_CASE else "--trace-geometry", "--measure-job")
    preflight = run_child(output, "capacity-preflight", solver, *arguments, "--cache", output / "preflight-cache",
                          "--frame-end", 1, timeout=None)
    require(preflight and preflight[-1].get("event") == "result" and preflight[-1].get("frame") == 1,
            "missing tank preflight result")
    matches_identity(preflight[-1], authored)
    for record in preflight:
        if record["event"] in ("trace", "result"):
            verify_extent(record, particle_count)
    verify_snapshot_receipt(preflight[-1], backend)
    require(capture_inventory(capture) == launch_inputs, "preflight changed launch inputs")
    verify_binary_pins(solver, fixture, pins)
    run_child(capture, "simulate-run", solver, *arguments, "--cache", capture / "cache",
              "--frame-end", last_frame, timeout=None)
    inventory = capture_inventory(capture)
    require(all(inventory.get(name) == file for name, file in launch_inputs.items()), "run changed launch inputs")
    verify_binary_pins(solver, fixture, pins)
    return evaluate_frozen_capture(capture, output, solver, fixture, pins, backend, case, solver_digest, inventory)


def verify_temporal_reference(path, expected_sha256, refined):
    primary = pinned_json(path, expected_sha256, "temporal reference")
    backend = refined["backend"]
    provenances = []
    for measurements, case in ((primary, PISTON_CASE), (refined, PISTON_REFINED_CASE)):
        require(measurements.get("case") == case and measurements.get("backend") == backend,
                "wrong temporal reference case or backend")
        fields = ("capture_provenance", "scene_author", "resolved_tank", "final_receipt")
        for field in fields:
            require(isinstance(measurements.get(field), dict), f"invalid temporal reference {field}")
        provenance, authored, resolved, final = (measurements[field] for field in fields)
        require(type(provenance.get("manifest_version")) is int and provenance["manifest_version"] == CAPTURE_MANIFEST_VERSION
                and provenance.get("case") == case and provenance.get("backend") == backend,
                "invalid temporal reference capture provenance")
        for field in ("manifest_sha256", "solver_sha256", "fixture_sha256"):
            digest(provenance, field)
        for field in ("solver_digest", "seed_digest", "resolved_fingerprint", "execution_id"):
            require(digest(provenance, field) == digest(authored, field) == digest(resolved, field),
                    f"temporal reference capture and scene disagree on {field}")
        verify_authored_tank(authored, case, backend, provenance["solver_digest"])
        matches_identity(resolved, authored)
        matches_identity(final, authored)
        execution = measured_execution(final, backend)
        if backend == "cuda":
            for field, value in execution.items():
                require(type(provenance.get(field)) is type(value) and provenance[field] == value,
                        f"temporal reference capture and execution disagree on {field}")
            record = provenance.get("cuda_compatibility_record")
            require(isinstance(record, str) and re.fullmatch(r"FLCUDA1\n[0-9a-f]{64}\n", record),
                    "invalid temporal reference CUDA compatibility record")
        else:
            require(not any(field.startswith("cuda_") or field == "compiled_architecture" for field in provenance),
                    "unexpected CUDA identity on CPU temporal reference")
        provenances.append(provenance)
    base, fine = provenances
    same_fields = ("solver_sha256", "fixture_sha256", "solver_digest", "seed_digest")
    if backend == "cuda":
        same_fields += ("cuda_execution_device_digest",)
    for field in same_fields:
        require(base[field] == fine[field], f"temporal refinement has a different {field}")
    for field in ("manifest_sha256", "resolved_fingerprint", "execution_id"):
        require(base[field] != fine[field], f"temporal refinement reused {field}")
    try:
        return compare_piston_time_refinement(primary, refined)
    except KeyError as error:
        raise ValueError(f"missing temporal reference measurement {error}") from error



def run_acceptance(solver_executable_path, fixture, backend, case, output, *, pins=None):
    # Refuse even an empty pre-existing directory. A deterministic execution ID
    # alone cannot distinguish a previous same-artifact run from the current one.
    output.mkdir(parents=True, exist_ok=False)
    if case in TANK_CASES:
        return run_frozen_tank(solver_executable_path, fixture, backend, case, output, pins)
    tank_layout = case == TANK_LAYOUT_CASE
    frame_ordinals = (0, 1) if tank_layout else FRAME_ORDINALS
    last_frame = frame_ordinals[-1]
    scene = output / "scene.flrs"
    seed = scene.with_suffix(".flrs.flst")
    cache = output / "cache"
    device_records = run_child(output, "device-probe", solver_executable_path, "device-info", "--backend", backend)
    device = one_record(device_records, "result")
    verify_device(device, backend)
    authored_scene_records = run_child(
        output, "scene-author", fixture, "write-scene", case, backend, solver_executable_path, scene, fixture=True,
    )
    authored_scene_identity = one_record(authored_scene_records, "scene")
    for field in ("resolved_fingerprint", "execution_id", "solver_digest", "seed_digest"):
        digest(authored_scene_identity, field)
    require(
        authored_scene_identity.get("case") == case and authored_scene_identity.get("backend") == backend,
        "wrong authored case or backend",
    )
    energy_blend = authored_scene_identity.get("energy_blend")
    require(finite_number(energy_blend) and energy_blend == (0.6 if tank_layout else FREE_FALL_CASES[case]), "wrong blend")
    require(scene.is_file() and scene.stat().st_size > 0, "missing resolved scene bytes")
    require(seed.stat().st_size == integer(authored_scene_identity, "seed_byte_length", minimum=1), "wrong seed byte length")
    require(not cache.exists() and not cache.is_symlink(), "cache path must be fresh")
    records = run_child(
        output, "simulate-run", solver_executable_path, "run", "--stage", "simulate", "--scene", scene,
        "--initial-state", seed,
        "--backend", backend, "--cache", cache, "--frame-end", last_frame,
        "--trace-geometry" if tank_layout else "--trace", "--measure-job", timeout=180,
    )
    before = snapshot(cache)
    fingerprint = authored_scene_identity["resolved_fingerprint"]
    frames = run_child(
        output, "verified-frames", fixture, "read-frames", cache, fingerprint, 0, last_frame, fixture=True,
    )
    checkpoint = one_record(
        run_child(output, "verified-checkpoint", fixture, "read-checkpoint", cache, fingerprint,
                  last_frame, fixture=True), "checkpoint"
    )
    require(snapshot(cache) == before, "verified readers changed cache files")
    expected_manifests = {
        f"frames/{frame:020d}.{suffix}" for frame in frame_ordinals for suffix in ("flmf", "json")
    }
    require({Path(name).as_posix() for name in before if Path(name).parts[0] == "frames"} == expected_manifests,
            "unexpected on-disk committed-frame inventory")
    require((cache / "resolved.flrs").read_bytes() == scene.read_bytes(), "cache resolved scene differs from authored bytes")
    verify_frames(frames, authored_scene_identity, frame_ordinals)
    final = verify_solver(records, frames, authored_scene_identity, frame_ordinals, tank_layout=tank_layout,
                          require_job_measurement=True)
    if tank_layout:
        measurements = verify_tank_layout(frames, records)
        verify_tank_refusals(output, solver_executable_path, backend, scene, seed)
        initial_position = TANK_LAYOUT_INITIAL_POSITION
    else:
        measurements = evaluate_free_fall(frames)
        initial_position = [2.5, 2.5, 2.5]
    verify_checkpoint(checkpoint, frames[0], authored_scene_identity, initial_position)
    measurements.update({
        "case": case, "backend": backend, "energy_blend": energy_blend, "device_probe": device,
        "scene_author": authored_scene_identity, "final_receipt": final, "job_measurement": records[-2],
        "verified_frame_count": len(frames),
        "verified_point_digests": [frame["points_digest"] for frame in frames],
        "verified_immutable_cache_file_count": len(before),
    })
    (output / ("tank-layout-measurements.json" if tank_layout else "free-fall-measurements.json")).write_text(
        json.dumps(measurements, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--case", required=True, choices=(*FREE_FALL_CASES, TANK_LAYOUT_CASE, *TANK_CASES))
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--temporary-output-parent", type=Path)
    destination.add_argument("--verify-capture", type=Path)
    parser.add_argument("--solver-sha256", help="trusted solver binary SHA-256 for frozen cases")
    parser.add_argument("--fixture-sha256", help="trusted fixture binary SHA-256 for frozen cases")
    parser.add_argument("--capture-manifest-sha256", help="trusted launch manifest SHA-256 for offline verification")
    parser.add_argument("--verification-output", type=Path, help="fresh directory outside the capture")
    parser.add_argument("--temporal-reference", type=Path, help="qualified primary piston measurements")
    parser.add_argument("--temporal-reference-sha256", help="trusted SHA-256 of the qualified primary measurements")
    args = parser.parse_args()
    if args.executable is None:
        parser.error("--executable is required")
    frozen = args.case in TANK_CASES
    offline = args.verify_capture is not None
    if offline and not frozen:
        parser.error("--verify-capture requires a frozen tank case")
    if offline and (args.verification_output is None or args.capture_manifest_sha256 is None):
        parser.error("--verify-capture requires --verification-output and --capture-manifest-sha256")
    if not offline and (args.verification_output is not None or args.capture_manifest_sha256 is not None):
        parser.error("--verification-output and --capture-manifest-sha256 require --verify-capture")
    if frozen and (args.solver_sha256 is None or args.fixture_sha256 is None):
        parser.error("frozen tank cases require --solver-sha256 and --fixture-sha256")
    if (args.temporal_reference is None) != (args.temporal_reference_sha256 is None):
        parser.error("--temporal-reference and --temporal-reference-sha256 are required together")
    if args.temporal_reference is not None and args.case != PISTON_REFINED_CASE:
        parser.error("--temporal-reference requires the refined piston case")
    output = args.verification_output if offline else args.output
    try:
        if output is None:
            args.temporary_output_parent.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=f"ctest-{args.case}-", dir=args.temporary_output_parent)) / "evidence"
        output = output.absolute()
        # Preserve the supplied frozen paths so pin checks can reject symlinks.
        solver = args.executable.absolute() if frozen else args.executable.resolve(strict=True)
        fixture = args.fixture.absolute() if frozen else args.fixture.resolve(strict=True)
        pins = {"solver": args.solver_sha256, "fixture": args.fixture_sha256} if frozen else None
        if offline:
            measurements = verify_captured_tank(args.verify_capture.absolute(), output, solver, fixture,
                                                args.backend, args.case, pins=pins,
                                                manifest_sha256=args.capture_manifest_sha256)
        else:
            measurements = run_acceptance(solver, fixture, args.backend, args.case, output, pins=pins)
        print(f"{args.backend} {args.case}: {output}")
        if args.temporal_reference is not None:
            comparison = verify_temporal_reference(args.temporal_reference, args.temporal_reference_sha256, measurements)
            write_measurements(output / "piston-temporal-refinement.json", comparison)
            require(comparison["passed"], "piston exceeds the frozen physical or temporal tolerance")

        require(measurements["passed"], f"{args.case} exceeds the frozen physical tolerance")
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"liquid acceptance failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
