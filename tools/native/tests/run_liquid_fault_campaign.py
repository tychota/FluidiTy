"""Run the required 50-trial F9 interruption campaign on the built CLI artifacts."""

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic

from test_simulate_replay import RunningSimulation, execute, invoke, make_scene, records, snapshot


SEED_FRAME_COUNT = 2
FINAL_FRAME_ORDINAL = 10
TRIAL_COUNTS = {"cancel": 17, "kill": 17, "device_error": 16}
STATUS_AFTER_STOP = {"cancel": "cancelled", "kill": "crashed", "device_error": "failed"}


def immutable_frame_inventory(cache_directory, minimum_frame_count, maximum_frame_count):
    """Verify canonical manifest ordinals before opening every published frame."""
    manifest_paths = sorted((cache_directory / "frames").glob("*.flmf"))
    assert minimum_frame_count <= len(manifest_paths) <= maximum_frame_count, manifest_paths
    assert [path.name for path in manifest_paths] == [
        f"{frame_ordinal:020d}.flmf" for frame_ordinal in range(len(manifest_paths))
    ]
    return manifest_paths


def verify_published_frames(executable, cache, reference, job_status, minimum_frame_count=SEED_FRAME_COUNT):
    reference_count = len(list((reference / "frames").glob("*.flmf")))
    frame_receipts = []
    for frame_ordinal, manifest_path in enumerate(
        immutable_frame_inventory(cache, minimum_frame_count, reference_count)
    ):
        verified_frame = invoke(executable, "check", "--cache", cache, "--frame", frame_ordinal)
        assert verified_frame["job_status"] == job_status, verified_frame
        assert verified_frame["frame"] == frame_ordinal and verified_frame["point_count"] == 1
        expected_manifest = reference / "frames" / manifest_path.name
        assert manifest_path.read_bytes() == expected_manifest.read_bytes(), manifest_path.name
        points_name = verified_frame["points_relative_name"]
        assert (cache / points_name).read_bytes() == (reference / points_name).read_bytes(), points_name
        frame_receipts.append({
            "frame": frame_ordinal, "epoch": verified_frame["epoch"], "points": points_name,
        })
    return frame_receipts


def physical_files(cache_directory):
    # Job markers and transient crash residues have no content identity.
    return {
        name: digest for name, (_, digest) in snapshot(cache_directory).items()
        if Path(name).suffix in (".flmf", ".flpt", ".flrs", ".flst", ".flcr")
    }


def run_arguments(scene, initial_state_args, backend, cache_directory):
    return [
        "run", "--stage", "simulate", "--scene", scene, "--backend", backend,
        "--cache", cache_directory, *initial_state_args,
    ]


def announced_frame_ordinals(stdout_records):
    frame_records = [record for record in stdout_records if record["event"] == "frame"]
    assert all(record["published"] is True and type(record["frame"]) is int for record in frame_records)
    ordinals = [record["frame"] for record in frame_records]
    assert ordinals == sorted(set(ordinals))
    return ordinals


def interrupt_process(executable, arguments, fault):
    process = RunningSimulation(executable, [*arguments, "--resume", "--frame-end", 100])
    try:
        _, initial_frame = process.next_event("frame")
        assert initial_frame["frame"] == 0 and initial_frame["published"] is True
        interruption_frame = initial_frame
        while interruption_frame["frame"] < SEED_FRAME_COUNT:
            _, interruption_frame = process.next_event("frame")
        assert interruption_frame["published"] is True
        fault_sent_at = monotonic()
        if fault == "cancel":
            process.process.stdin.write("cancel\n")
            process.process.stdin.flush()
            _, error = process.next_event("error")
            assert error["name"] == "CANCELLED" and error["field"] == "stdin", error
        else:
            process.process.kill()
        return_code = process.process.wait(timeout=10)
        assert return_code != 0 and not process.process.stdin.closed
        if fault == "cancel":
            assert records(process.process.stderr.read()) == [error]
        fault_to_exit_seconds = monotonic() - fault_sent_at
        stdout, discarded_trailing_bytes = process.stdout_records_after_exit(allow_trailing_partial=fault == "kill")
        published_ordinals = announced_frame_ordinals(stdout)
        assert interruption_frame["frame"] in published_ordinals
        return {
            "exit_code": return_code, "fault_to_exit_seconds": fault_to_exit_seconds,
            "interruption_frame_ordinal": interruption_frame["frame"],
            "announced_frame_ordinals": published_ordinals,
            "discarded_trailing_stdout_bytes": discarded_trailing_bytes,
        }
    finally:
        process.close()


def interrupt_after_device_copy(fault_executable, arguments):
    fault_started_at = monotonic()
    child = subprocess.run(
        [str(fault_executable), "--fixture-device-error", *map(str, arguments), "--resume", "--frame-end", str(FINAL_FRAME_ORDINAL)],
        capture_output=True, encoding="utf-8", timeout=30,
    )
    stdout = records(child.stdout)
    stderr = records(child.stderr)
    assert child.returncode == 1 and stdout[-1]["name"] == "INTERNAL", (stdout, stderr)
    assert stdout[-1]["field"] == "frame" and stdout[-1]["event"] == "error"
    assert len(stderr) == 2 and stderr[-1] == stdout[-1], stderr
    injected = stderr[0]
    assert injected == {
        "protocol_version": 1, "event": "fixture_device_fault",
        "completed_snapshot_count": 3, "actual_cuda_event_result": 0,
        "injected_cuda_error": "cudaErrorLaunchFailure",
    }, injected
    published_frames = [entry for entry in stdout if entry["event"] == "frame"]
    published_ordinals = announced_frame_ordinals(stdout)
    assert published_ordinals == list(range(SEED_FRAME_COUNT)), published_frames
    for entry in published_frames:
        assert entry["compiled_architecture"] == 1200
        assert math.isfinite(entry["snapshot_copy_milliseconds"]) and entry["snapshot_copy_milliseconds"] > 0
    return {
        "exit_code": child.returncode, "child_elapsed_seconds": monotonic() - fault_started_at,
        "producing_frames": published_frames, "injected_fault": injected,
        "announced_frame_ordinals": published_ordinals,
    }


def verify_slow_device_copy(executable, arguments, baseline_state_digest):
    output = execute(executable, "--fixture-delay-copy", *arguments, "--frame-end", FINAL_FRAME_ORDINAL)
    initial_frame = next(record for record in output if record["event"] == "frame")
    assert initial_frame["frame"] == 0 and initial_frame["compiled_architecture"] == 1200
    assert 1100 <= initial_frame["snapshot_copy_milliseconds"] <= 10000, initial_frame
    heartbeat_times = [
        record["elapsed_milliseconds"] for record in output[:output.index(initial_frame)]
        if record["event"] == "heartbeat"
    ]
    assert len(heartbeat_times) >= 2
    assert all(0 < later - earlier <= 1000 for earlier, later in zip(heartbeat_times, heartbeat_times[1:]))
    assert output[-1]["state_digest"] == baseline_state_digest
    return {"initial_frame": initial_frame, "heartbeat_milliseconds": heartbeat_times}


def trial(executable, fixture, arguments, seed_cache, reference_cache, baseline_result, trial_cache, fault, trial_ordinal):
    shutil.copytree(seed_cache, trial_cache)
    if fault == "device_error":
        fault_receipt = interrupt_after_device_copy(executable, arguments)
    else:
        fault_receipt = interrupt_process(executable, arguments, fault)
    surviving_frames = verify_published_frames(executable, trial_cache, reference_cache, STATUS_AFTER_STOP[fault])
    surviving_ordinals = {frame["frame"] for frame in surviving_frames}
    for ordinal in fault_receipt["announced_frame_ordinals"]:
        assert ordinal in surviving_ordinals, f"announced frame {ordinal} missing before resume"
    if fault == "device_error":
        assert len(surviving_frames) == SEED_FRAME_COUNT, surviving_frames
    pre_resume_files = physical_files(trial_cache)
    resumed = execute(executable, *arguments, "--resume", "--frame-end", FINAL_FRAME_ORDINAL)[-1]
    assert resumed["state_digest"] == baseline_result["state_digest"], resumed
    recovered_frames = verify_published_frames(
        executable, trial_cache, reference_cache, "complete", FINAL_FRAME_ORDINAL + 1
    )
    checkpoint = json.loads(subprocess.check_output(
        [str(fixture), "checkpoint", str(trial_cache)], encoding="utf-8", timeout=30,
    ))
    assert checkpoint["frame"] == FINAL_FRAME_ORDINAL
    assert checkpoint["state_digest"] == resumed["state_digest"]
    post_resume_files = physical_files(trial_cache)
    assert all(post_resume_files[name] == digest for name, digest in pre_resume_files.items())
    assert post_resume_files == physical_files(reference_cache)
    return {
        "fault": fault, "trial_ordinal": trial_ordinal, "stop": fault_receipt,
        "surviving_frames": surviving_frames, "recovered_frames": recovered_frames,
        "recovered_state_digest": resumed["state_digest"], "accepted_steps": resumed["accepted_step"],
    }


def verify_inventory(receipts):
    assert Counter(receipt["fault"] for receipt in receipts) == TRIAL_COUNTS, receipts
    assert len(receipts) == 50 and {
        (receipt["fault"], receipt["trial_ordinal"]) for receipt in receipts
    } == {(fault, ordinal) for fault, count in TRIAL_COUNTS.items() for ordinal in range(1, count + 1)}
    for receipt in receipts:
        assert len(receipt["surviving_frames"]) >= SEED_FRAME_COUNT
        assert len(receipt["recovered_frames"]) == FINAL_FRAME_ORDINAL + 1
        stop = receipt["stop"]
        surviving_ordinals = {frame["frame"] for frame in receipt["surviving_frames"]}
        assert set(stop["announced_frame_ordinals"]) <= surviving_ordinals
        if receipt["fault"] != "device_error":
            assert stop["interruption_frame_ordinal"] >= SEED_FRAME_COUNT
            assert stop["interruption_frame_ordinal"] in stop["announced_frame_ordinals"]
        assert re.fullmatch(r"[0-9a-f]{64}", receipt["recovered_state_digest"])
        assert receipt["accepted_steps"] > 0
        for key in ("surviving_frames", "recovered_frames"):
            assert [frame["frame"] for frame in receipt[key]] == list(range(len(receipt[key])))


def run_campaign(executable, fault_executable, fixture, directory, device_control_only):
    started_at = monotonic()
    device = invoke(fault_executable, "device-info", "--backend", "cuda")
    assert device["device_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert (device["compute_major"], device["compute_minor"], device["compiled_architecture"]) == (12, 0, 1200)
    assert device["cuda_available"] and device["backend"] == "cuda"
    production_rejects_fault = invoke(executable, "--fixture-device-error", success=False)
    assert production_rejects_fault["code"] == "E-CLI-INVALID_ARGUMENT"
    assert production_rejects_fault["object"] == "--fixture-device-error"
    receipts = []
    configurations = [(fault_executable, "cuda", ("device_error",))]
    if not device_control_only:
        configurations.insert(0, (executable, "cpu", ("cancel", "kill")))
    for solver, backend, faults in configurations:
        backend_directory = directory / backend
        backend_directory.mkdir()
        scene, initial_state_args = make_scene(solver, fixture, backend_directory, backend)
        baseline_cache = backend_directory / "uninterrupted"
        baseline = execute(
            solver, *run_arguments(scene, initial_state_args, backend, baseline_cache), "--frame-end", FINAL_FRAME_ORDINAL
        )[-1]
        seed_cache = backend_directory / "initial-published-frames"
        execute(solver, *run_arguments(scene, initial_state_args, backend, seed_cache), "--frame-end", SEED_FRAME_COUNT - 1)
        seed_frames = verify_published_frames(solver, seed_cache, baseline_cache, "complete")
        assert len(seed_frames) == SEED_FRAME_COUNT
        for fault in faults:
            count = 1 if device_control_only else TRIAL_COUNTS[fault]
            for trial_ordinal in range(1, count + 1):
                trial_cache = backend_directory / f"{fault}-{trial_ordinal:02}"
                arguments = run_arguments(scene, initial_state_args, backend, trial_cache)
                receipts.append(trial(solver, fixture, arguments, seed_cache, baseline_cache, baseline, trial_cache,
                                      fault, trial_ordinal))
        if backend == "cuda" and not device_control_only:
            slow_cache = backend_directory / "slow-copy"
            slow_copy = verify_slow_device_copy(
                solver, run_arguments(scene, initial_state_args, backend, slow_cache), baseline["state_digest"]
            )
            verify_published_frames(solver, slow_cache, baseline_cache, "complete", FINAL_FRAME_ORDINAL + 1)
    if device_control_only:
        return receipts[0]
    verify_inventory(receipts)
    return {
        "schema_version": 1, "device": device, "trial_counts": TRIAL_COUNTS, "final_frame": FINAL_FRAME_ORDINAL,
        "trials": receipts, "slow_copy": slow_copy, "elapsed_seconds": monotonic() - started_at,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fault-executable", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--device-control-only", action="store_true")
    options = parser.parse_args()
    for executable_path in (options.executable, options.fault_executable, options.fixture):
        assert executable_path.is_file(), executable_path
    with tempfile.TemporaryDirectory(prefix="fluidity-f9-") as temporary_directory:
        receipt = run_campaign(
            options.executable.resolve(), options.fault_executable.resolve(), options.fixture.resolve(),
            Path(temporary_directory), options.device_control_only,
        )
    print("CTEST_FULL_OUTPUT")
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
