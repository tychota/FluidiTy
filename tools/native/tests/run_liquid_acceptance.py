"""Run the public Simulate CLI and qualify one independently measured free fall."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquid_acceptance_metrics import (
    FREE_FALL_CASES,
    FREE_FALL_LAST_FRAME,
    FRAMES_PER_SECOND,
    evaluate_free_fall,
    finite_number,
)

FRAME_ORDINALS = tuple(range(FREE_FALL_LAST_FRAME + 1))
CUDA_ARCHITECTURE = 1200
CLOCK_FIELDS = ("frame", "substep", "epoch", "accepted_step", "time_seconds")
IDENTITY_FIELDS = ("resolved_fingerprint", "execution_id", "backend")
MAXIMUM_COUNTS = {"block_count": 128, "cell_count": 4096, "face_count": 24576}
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def verify_extent(record):
    require(integer(record, "particle_count") == 1, "wrong physical particle count")
    require(integer(record, "next_particle_id") == 2, "unexpected particle allocation")
    for field, maximum in MAXIMUM_COUNTS.items():
        require(integer(record, field) <= maximum, f"{field} exceeds the frozen capacity")
    require(record.get("source_next_child_ordinals") == [], "unexpected source ordinals")
    require(record.get("source_fractional_remainders") == [], "unexpected source remainder")


def verify_solver(records, frames, authored_scene_identity):
    final = records[-1]
    backend = authored_scene_identity["backend"]
    require(final.get("event") == "result" and final.get("command") == "run", "missing successful run result")
    matches_identity(final, authored_scene_identity)
    require(final.get("read_only") is False, "run did not publish")
    require(integer(final, "frames_per_second") == FRAMES_PER_SECOND, "wrong receipt frame rate")
    verify_extent(final)
    verify_snapshot_receipt(final, backend)
    if backend == "cuda":
        integer(final, "graph_replay_count", minimum=1)
        integer(final, "device_reserved_bytes", minimum=1)

    # Each accepted step emits one trace. Attempt counters are cumulative within
    # a frame and start again after publication.
    published_count = 0
    accepted_trace_count = 0
    previous_dt_retry_count = 0
    last_clock = None
    last_trace = None
    for record in records[:-1]:
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
                require(frame <= min(published_count, FREE_FALL_LAST_FRAME), "invalid heartbeat frame")
                for field in CLOCK_FIELDS[:-1]:
                    require(integer(settled, field) == frames[frame][field], "heartbeat and manifest clocks differ")
            continue
        if event == "frame":
            require(published_count < len(FRAME_ORDINALS), "extra publication receipt")
            frame_ordinal = FRAME_ORDINALS[published_count]
            require(record.get("published") is True and integer(record, "frame") == frame_ordinal,
                    "wrong publication ordinal")
            verify_snapshot_receipt(record, backend)
            manifest_clock = clock(frames[frame_ordinal])
            require(manifest_clock == (last_clock or (0, 0, 0, 0, 0.0)), "publication and manifest clocks differ")
            last_clock = manifest_clock
            published_count += 1
            accepted_trace_count = 0
            previous_dt_retry_count = 0
            continue
        require(event == "trace" and record.get("published") is False, "unexpected run event")
        require(0 < published_count < len(FRAME_ORDINALS), "trace outside expected run frames")
        trace_clock = clock(record)
        frame, substep, _, accepted_step, time = trace_clock
        previous_frame, previous_substep, _, previous_step, previous_time = last_clock
        require(frame == previous_frame + (substep == 0), "wrong trace frame transition")
        require(substep == 0 or substep == previous_substep + 1, "wrong trace substep")
        require(accepted_step == previous_step + 1, "missing or duplicate accepted step")
        require(0 < time - previous_time <= 0.01 + 1e-15, "invalid accepted duration")
        if substep == 0:
            require(frame == published_count and time == frame / FRAMES_PER_SECOND, "wrong completed trace clock")
        else:
            require(frame == published_count - 1 and time < published_count / FRAMES_PER_SECOND, "wrong interior trace clock")
        verify_extent(record)
        accepted_trace_count += 1
        dt_retry_count = integer(record, "dt_retries")
        require(dt_retry_count <= 2, "pressure retry bound exceeded")
        require(dt_retry_count >= previous_dt_retry_count, "pressure retry count regressed within the frame")
        require(integer(record, "capacity_replays") == 0, "unexpected capacity replay")
        require(integer(record, "accepted_attempts") == accepted_trace_count,
                "accepted attempt count differs from the accepted trace count")
        require(integer(record, "attempts") == accepted_trace_count + dt_retry_count, "inconsistent attempt counts")
        previous_dt_retry_count = dt_retry_count
        digest(record, "state_digest")
        last_clock = trace_clock
        last_trace = record

    require(published_count == len(FRAME_ORDINALS), "missing publication receipt")
    require(clock(final) == last_clock == clock(frames[-1]), "final clock and manifest differ")
    require(last_trace is not None and digest(final, "state_digest") == last_trace["state_digest"], "final state differs")
    for field in (*MAXIMUM_COUNTS, "attempts", "accepted_attempts", "dt_retries", "capacity_replays"):
        require(integer(final, field) == last_trace[field], f"final and trace {field} differ")
    return final


def verify_frames(records, authored_scene_identity):
    require(len(records) == len(FRAME_ORDINALS), "missing or extra verified frame")
    for ordinal, record in zip(FRAME_ORDINALS, records):
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


def verify_checkpoint(checkpoint, frame_zero, authored_scene_identity):
    matches_identity(checkpoint, authored_scene_identity)
    require(clock(checkpoint) == clock(frame_zero), "retained checkpoint must match frame zero")
    require(digest(checkpoint, "state_digest") == authored_scene_identity["seed_digest"], "retained state differs from seed")
    require(integer(checkpoint, "random_seed") == 0 and integer(checkpoint, "next_particle_id") == 2, "wrong seed metadata")
    require(checkpoint.get("capacities") == [1024, 128, 4096, 24576, 256], "wrong capacity reservations")
    require(checkpoint.get("pressure_cells") == [], "initial checkpoint contains pressure cells")
    expected_particle = {
        "id": 1, "tag": 0, "world_position": [2.5, 2.5, 2.5], "world_velocity": [0, 0, 0],
        "velocity_slopes": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "side_tag": 0,
        "mass": 1000, "reference_volume": 1, "volume_ratio": 1, "affine_enabled": True, "pic_initialized": False,
    }
    require(checkpoint.get("particles") == [expected_particle], "initial checkpoint particle differs from frozen seed")
    point = frame_zero["points"][0]
    point_volume = 4 * math.pi / 3 * point["radius"] ** 3
    require(abs(point_volume - 1) <= 1e-14 and point["volume_ratio"] == 1, "initial point volume differs from seed")


def snapshot(root):
    """Record content and modification time to detect reader writes."""
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*") if path.is_file()
    }


def run_acceptance(solver_executable_path, fixture, backend, case, output):
    # Refuse even an empty pre-existing directory. A deterministic execution ID
    # alone cannot distinguish a previous same-artifact run from the current one.
    output.mkdir(parents=True, exist_ok=False)
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
    require(finite_number(energy_blend) and energy_blend == FREE_FALL_CASES[case], "wrong blend")
    require(scene.is_file() and scene.stat().st_size > 0, "missing resolved scene bytes")
    require(seed.stat().st_size == integer(authored_scene_identity, "seed_byte_length", minimum=1), "wrong seed byte length")
    require(not cache.exists() and not cache.is_symlink(), "cache path must be fresh")
    records = run_child(
        output, "simulate-run", solver_executable_path, "run", "--stage", "simulate", "--scene", scene,
        "--initial-state", seed,
        "--backend", backend, "--cache", cache, "--frame-end", FREE_FALL_LAST_FRAME, "--trace", timeout=180,
    )
    before = snapshot(cache)
    fingerprint = authored_scene_identity["resolved_fingerprint"]
    frames = run_child(
        output, "verified-frames", fixture, "read-frames", cache, fingerprint, 0, FREE_FALL_LAST_FRAME, fixture=True,
    )
    checkpoint = one_record(
        run_child(output, "verified-checkpoint", fixture, "read-checkpoint", cache, fingerprint,
                  FREE_FALL_LAST_FRAME, fixture=True), "checkpoint"
    )
    require(snapshot(cache) == before, "verified readers changed cache files")
    expected_manifests = {
        f"frames/{frame:020d}.{suffix}" for frame in FRAME_ORDINALS for suffix in ("flmf", "json")
    }
    require({Path(name).as_posix() for name in before if Path(name).parts[0] == "frames"} == expected_manifests,
            "unexpected on-disk committed-frame inventory")
    require((cache / "resolved.flrs").read_bytes() == scene.read_bytes(), "cache resolved scene differs from authored bytes")
    measurements = evaluate_free_fall(frames)
    verify_frames(frames, authored_scene_identity)
    verify_checkpoint(checkpoint, frames[0], authored_scene_identity)
    final = verify_solver(records, frames, authored_scene_identity)
    measurements.update({
        "case": case, "backend": backend, "energy_blend": energy_blend, "device_probe": device,
        "scene_author": authored_scene_identity, "final_receipt": final, "verified_frame_count": len(frames),
        "verified_point_digests": [frame["points_digest"] for frame in frames],
        "verified_immutable_cache_file_count": len(before),
    })
    (output / "free-fall-measurements.json").write_text(
        json.dumps(measurements, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--case", required=True, choices=FREE_FALL_CASES)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--temporary-output-parent", type=Path)
    args = parser.parse_args()
    output = args.output
    try:
        if output is None:
            args.temporary_output_parent.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="ctest-free-fall-", dir=args.temporary_output_parent)) / "evidence"
        output = output.absolute()
        measurements = run_acceptance(args.executable.resolve(strict=True), args.fixture.resolve(strict=True),
                                      args.backend, args.case, output)
        print(f"{args.backend} {args.case}: {output / 'free-fall-measurements.json'}")
        require(measurements["passed"], "free fall exceeds the frozen physical tolerance")
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"liquid acceptance failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
