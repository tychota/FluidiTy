"""Required real-process CLI checks. Invoked by CTest, never conditionally skipped.

This module intentionally has no pytest tests: CTest supplies the built binaries
and each named case fails immediately when either executable is absent.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import struct
import subprocess
import sys
import tempfile
from threading import Thread
from time import monotonic

# These process checks use assertions. This explicit guard survives -O/-OO and
# rejects optimization before argument parsing, executable launches, or receipts.
if not __debug__:
    raise SystemExit("Python optimization is unsupported by Simulate CLI process checks")


def records(text):
    def reject(value):
        raise AssertionError(f"non-JSON number {value}")
    result = [json.loads(line, parse_constant=reject) for line in text.splitlines()]
    assert result, "missing protocol record"
    for record in result:
        assert record["protocol_version"] == 1
        assert isinstance(record["event"], str)
    return result


def invoke(executable, *args, success=True):
    process = subprocess.run([str(executable), *map(str, args)], capture_output=True, encoding="utf-8", timeout=30)
    output = records(process.stdout)
    assert (process.returncode == 0) == success, (process.returncode, process.stdout, process.stderr)
    if success:
        assert len(output) == 1, "unexpected protocol stdout"
        assert process.stderr == "", process.stderr
        assert output[-1]["event"] == "result"
    else:
        errors = records(process.stderr)
        assert len(output) == len(errors) == 1
        assert output[0] == errors[0]
        assert errors[0]["event"] == "error"
        assert errors[0]["code"].startswith("E-")
        for key in ("domain", "name", "field", "object", "cause"):
            assert isinstance(errors[0][key], str)
    return output[-1]


def snapshot(root, live_job=False):
    # Windows denies file reads through the writer's byte-range lock. Verify
    # the status marker and immutable content while the owner holds that lock.
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file() and not (live_job and p == root / ".simulate-job.lock")}


def immutable_cache_files(files):
    return {path: receipt for path, receipt in files.items()
            if path not in (".simulate-job.lock", ".simulate-job-status")}


def case_arguments(executable, fixture, directory, cuda):
    # An optimized child would erase the mismatching build assertion and print PASS.
    # Exercise explicit flags and inherited optimization through real interpreters.
    for options, inherited_optimization in ((["-O"], None), (["-OO"], None), ([], "1"), ([], "2")):
        environment = os.environ.copy()
        environment.pop("PYTHONOPTIMIZE", None)
        if inherited_optimization is not None:
            environment["PYTHONOPTIMIZE"] = inherited_optimization
        child = subprocess.run(
            [sys.executable, *options, str(Path(__file__).resolve()), "--executable", str(executable),
             "--fixture", str(fixture), "--case", "cpu_device", *([] if cuda else ["--cuda"])],
            env=environment, capture_output=True, encoding="utf-8", timeout=30,
        )
        assert child.returncode != 0, (options, inherited_optimization, child.stdout, child.stderr)
        assert "PASS" not in child.stdout
        assert "Python optimization is unsupported" in child.stderr
    invalid = [[], ["unknown"], ["schema", "--bogus"], ["schema", "--backend", "cpu"],
               ["device-info", "--backend", "metal"], ["device-info", "--device", "-1"],
               ["device-info", "--backend", "cpu", "--backend", "cpu"],
               ["check"], ["check", "--cache", "x", "--frame", "-1"],
               ["check", "--cache", "x", "--frame", "18446744073709551616"],
               ["run", "--stage", "mesh", "--scene", "x", "--cache", "y"],
               ["run", "--stage", "simulate", "--scene", "x", "--cache", "y", "--frame-start", "2", "--frame-end", "1"],
               ["replay", "--cache", "x", "--frame", "1"],
               ["step", "--scene", "x", "--cache", "y", "--substep", "1"],
               ["schema", 'bad\n\t"\\\x01']]
    for args in invalid:
        result = invoke(executable, *args, success=False)
        assert result["code"] == "E-CLI-INVALID_ARGUMENT", result
    scene = str(directory / "input.flrs")
    cache = str(directory / "output cache")
    for args in (["run", "--stage", "simulate", "--scene", scene, "--cache", cache],
                 ["step", "--scene", scene, "--cache", cache],
                 ["replay", "--cache", cache, "--frame", "1", "--substep", "0"]):
        assert invoke(executable, *args, success=False)["code"] == "E-SIMULATE-NOT_FOUND"
    controls = 'bad\n\t"\\\x01'
    assert invoke(executable, "schema", controls, success=False)["object"] == controls
    assert invoke(executable, "run", "--stage", "simulate", "--scene", scene, "--cache", cache,
                  "--backend", "cuda", "--device", "2147483647", "--frame-start", "0", "--frame-end", "18446744073709551615",
                  "--resume", "--trace", "--preview", success=False)["code"] == "E-SIMULATE-NOT_FOUND"
    assert not list(directory.iterdir())


def case_schema(executable, fixture, directory, cuda):
    result = invoke(executable, "schema")
    assert result["command"] == "schema"
    assert result["identifiers"] == {"state": "FLST", "points": "FLPT", "manifest": "FLMF", "resolved_scene": "FLRS", "checkpoint_reference": "FLCR"}
    defaults = result["defaults"]
    assert defaults["material"]["bulk_modulus"] == "Infinity"
    assert defaults["material"]["density"] == 1000
    assert defaults["options"]["energy_blend"] == 0.6
    assert defaults["options"]["maximum_step_duration"] == 0.01
    assert defaults["options"]["mass_per_quantum"] == 2**-40
    assert defaults["options"]["pressure_iterations"] == 256
    assert defaults["acceleration"] == {"x": 0, "y": -9.81, "z": 0}
    assert defaults["sources"] == defaults["upstream"] == []
    assert "schema_version:uint = 1 (id:0)" in result["resolved_schema"]
    assert result["execution_available"] is True


def case_identity(executable, fixture, directory, cuda):
    original = invoke(executable, "schema")["build_digest"]
    expected = subprocess.check_output([str(fixture), "digest", str(executable)], text=True).strip()
    assert original == expected and len(original) == 64
    copy = directory / ("résumé 液体" + executable.suffix)
    shutil.copy2(executable, copy)
    assert invoke(copy, "schema")["build_digest"] == original
    with copy.open("ab") as stream:
        stream.write(b"fluidity identity regression")
    assert invoke(copy, "schema")["build_digest"] != original


def case_inspection(executable, fixture, directory, cuda):
    cache = directory / "résumé 液体"
    subprocess.run([str(fixture), "cache", str(cache)], check=True, capture_output=True, timeout=30)
    before = snapshot(directory)
    result = invoke(executable, "check", "--cache", cache, "--frame", "0")
    assert result["inspection_only"] and result["resume_authorized"] is False
    assert result["job_status"] == "idle"
    assert result["frame"] == 0 and result["substep"] == result["epoch"] == 0
    assert result["points_relative_name"].startswith("blobs/")
    assert not Path(result["points_relative_name"]).is_absolute()
    assert invoke(executable, "check", "--scene", cache / "resolved.flrs")["inspection_only"]
    assert invoke(executable, "check", "--cache", cache, "--scene", cache / "resolved.flrs")["inspection_only"]
    invoke(executable, "check", "--cache", directory / "missing", success=False)
    invoke(executable, "check", "--cache", cache, "--frame", "1", success=False)
    invalid = directory / "invalid.flrs"
    invalid.write_bytes(b"not a resolved scene")
    invoke(executable, "check", "--scene", invalid, success=False)
    # Matching portable content is checked independently of this executable build.
    altered = bytearray((cache / "resolved.flrs").read_bytes())
    root_offset = struct.unpack_from("<I", altered)[0]
    vtable_offset = root_offset - struct.unpack_from("<i", altered, root_offset)[0]
    seed_offset = struct.unpack_from("<H", altered, vtable_offset + 4 + 7 * 2)[0]
    assert seed_offset
    altered[root_offset + seed_offset] = 1
    invalid.write_bytes(altered)
    assert invoke(executable, "check", "--cache", cache, "--scene", invalid, success=False)["code"] == "E-CACHE-FAILED_PRECONDITION"
    # Unsupported shared-schema enum is rejected by the real checked decoder.
    altered = bytearray((cache / "resolved.flrs").read_bytes())
    root_offset = struct.unpack_from("<I", altered)[0]
    vtable_offset = root_offset - struct.unpack_from("<i", altered, root_offset)[0]
    backend_offset = struct.unpack_from("<H", altered, vtable_offset + 4 + 3 * 2)[0]
    assert backend_offset
    altered[root_offset + backend_offset] = 255
    invalid.write_bytes(altered)
    assert invoke(executable, "check", "--scene", invalid, success=False)["code"] == "E-CACHE-DATA_LOSS"
    with invalid.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    assert invoke(executable, "check", "--scene", invalid, success=False)["code"] == "E-CACHE-RESOURCE_EXHAUSTED"
    invalid.unlink()
    if os.name == "posix":
        os.mkfifo(invalid)
        invoke(executable, "check", "--scene", invalid, success=False)
        invalid.unlink()
    assert snapshot(directory) == before, "inspection changed cache bytes, mtimes, or inventory"
    (cache / ".simulate-job-status").write_bytes(b"unknown operational record")
    invalid_marker = snapshot(directory)
    unknown = invoke(executable, "check", "--cache", cache, "--frame", "0")
    assert unknown["job_status"] == "unknown" and unknown["frame"] == 0
    assert snapshot(directory) == invalid_marker

    # An orphaned legacy marker without a lease is still a read-only crash
    # observation. A re-probe must neither create a lease nor erase the marker.
    job_marker = cache / ".simulate-job-status"
    job_lease = cache / ".simulate-job.lock"
    job_marker.write_bytes(b"v1 baking\n")
    abandoned_files = snapshot(directory)
    orphan = invoke(executable, "check", "--cache", cache, "--frame", "0")
    assert orphan["job_status"] == "crashed" and orphan["frame"] == 0
    assert not job_lease.exists() and snapshot(directory) == abandoned_files

    # A malformed operational lease must not block checked physical data.
    # A directory exercises the error on both Windows and POSIX.
    job_lease.mkdir()
    malformed_lease = invoke(executable, "check", "--cache", cache, "--frame", "0")
    assert malformed_lease["job_status"] == "unknown"
    assert malformed_lease["points_relative_name"] == result["points_relative_name"]
    assert malformed_lease["job_status_diagnostic"]["name"] in {"DATA_LOSS", "UNAVAILABLE"}
    assert malformed_lease["job_status_diagnostic"]["cause"]
    assert job_lease.is_dir() and snapshot(directory) == abandoned_files

    job_lease.rmdir()
    job_marker.unlink()
    job_marker.mkdir()
    physical_files = snapshot(directory)
    malformed_marker = invoke(executable, "check", "--cache", cache, "--frame", "0")
    assert malformed_marker["job_status"] == "unknown"
    assert malformed_marker["points_relative_name"] == result["points_relative_name"]
    assert malformed_marker["job_status_diagnostic"]["name"] in {"DATA_LOSS", "UNAVAILABLE"}
    assert job_marker.is_dir() and not job_lease.exists()
    assert snapshot(directory) == physical_files

    manifest = next((cache / "frames").glob("*.flmf"))
    manifest.write_bytes(b"corrupt")
    corrupt = snapshot(directory)
    invoke(executable, "check", "--cache", cache, "--frame", "0", success=False)
    assert snapshot(directory) == corrupt


def case_cpu_device(executable, fixture, directory, cuda):
    result = invoke(executable, "device-info", "--backend", "cpu")
    assert invoke(executable, "device-info") == result
    assert result["backend"] == "cpu" and result["cpu_available"] is True
    assert result["cuda_compiled"] is cuda
    assert result["cuda_available"] is None  # Explicitly not probed for a CPU request.
    if not cuda:
        assert invoke(executable, "device-info", "--backend", "cuda", success=False)["code"] == "E-DEVICE-UNAVAILABLE"


def case_cuda_device(executable, fixture, directory, cuda):
    unavailable_index = invoke(executable, "device-info", "--backend", "cuda", "--device", "2147483647", success=False)
    assert unavailable_index["code"] == "E-DEVICE-INVALID_ARGUMENT"
    assert unavailable_index["field"] == "device"
    assert unavailable_index["object"] == "2147483647"
    result = invoke(executable, "device-info", "--backend", "cuda", "--device", "0")
    assert result["cpu_available"] and result["cuda_compiled"] and result["cuda_available"]
    assert result["device_name"] and result["compiled_architecture"] == result["compute_major"] * 100 + result["compute_minor"] * 10
    assert result["driver_version"] > 0 and result["runtime_version"] > 0
    assert result["launch_result"] == result["copy_result"] == result["synchronize_result"] == 0
    assert result["sample_count"] == 17
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


def execute(executable, *args, success=True):
    process = subprocess.run([str(executable), *map(str, args)], capture_output=True, encoding="utf-8", timeout=120)
    output = records(process.stdout)
    assert (process.returncode == 0) == success, (process.returncode, process.stdout, process.stderr)
    if success:
        assert process.stderr == "", process.stderr
        assert output[-1]["event"] == "result"
        events = {"heartbeat", "attempt", "trace", "frame", "result"}
        if "--measure-job" in args:
            events.add("job_measurement")
            assert output[-2]["event"] == "job_measurement" and output[-2]["status"] == "OK"
        assert all(r["event"] in events for r in output)
    else:
        errors = records(process.stderr)
        assert errors == [output[-1]] and errors[0]["event"] == "error"
    return output


def make_scene(executable, fixture, directory, backend, mode="initial"):
    scene = directory / (backend + "-" + mode + ".flrs")
    subprocess.run([str(fixture), "scene", str(scene), str(executable), backend, mode],
                   check=True, capture_output=True, timeout=30)
    initial = ["--initial-state", str(scene) + ".flst"] if mode in ("initial", "slow", "effects", "whitewater-effects", "whitewater-beach", "ocean", "ocean-level", "graded-tank") else []
    return scene, initial


def execution_case(executable, fixture, directory, backend):
    scene, initial = make_scene(executable, fixture, directory, backend)
    cache = directory / "cache"
    args = ["run", "--stage", "simulate", "--scene", scene, "--backend", backend, "--cache", cache, *initial]
    output = execute(executable, *args, "--frame-end", 2, "--trace")
    result = output[-1]
    assert result["frame"] == 2 and result["substep"] == 0
    assert result["accepted_step"] == result["epoch"] == 4
    assert result["time_seconds"] == 2 / 24 and result["frames_per_second"] == 24
    assert result["particle_count"] == 1 and result["next_particle_id"] == 2
    assert result["backend"] == backend
    assert "cuda_execution_device_digest" not in result and "cuda_execution_device_index" not in result
    traces = [r for r in output if r["event"] == "trace"]
    assert [(r["frame"], r["substep"]) for r in traces] == [(0, 1), (1, 0), (1, 1), (2, 0)]
    assert all("geometry" not in trace for trace in traces)
    assert all("accepted_measurement" in trace for trace in traces), "missing accepted physical measurements"
    before = snapshot(cache)
    assert (".simulate-cuda-compatibility" in before) == (backend == "cuda")
    resumed = execute(executable, *args, "--resume", "--frame-end", 2)[-1]
    assert resumed["state_digest"] == result["state_digest"]
    assert immutable_cache_files(snapshot(cache)) == immutable_cache_files(before), "resume rewrote committed content"
    before = snapshot(cache)
    preview = execute(executable, *args, "--resume", "--preview", "--frame-end", 3)[-1]
    assert preview["frame"] == 3 and preview["read_only"]
    assert snapshot(cache) == before
    step = execute(executable, "step", "--scene", scene, "--backend", backend, "--cache", cache, *initial)[-1]
    assert step["frame"] == 0 and step["substep"] == 1 and step["accepted_step"] == 1
    assert step["read_only"] and snapshot(cache) == before
    fresh = directory / "fresh"
    rerun = execute(executable, *[fresh if x == cache else x for x in args], "--frame-end", 2, "--measure-job")
    assert not any(record["event"] in ("attempt", "trace") for record in rerun)
    measured_result = rerun[-1]
    if backend == "cuda":
        digest = measured_result["cuda_execution_device_digest"]
        assert len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
        device_index = measured_result["cuda_execution_device_index"]
        assert type(device_index) is int and device_index == 0
        assert measured_result["compiled_architecture"] == result["compiled_architecture"] == 1200
    else:
        assert "cuda_execution_device_digest" not in measured_result
        assert "cuda_execution_device_index" not in measured_result
    geometry_cache = directory / "geometry"
    geometry = execute(executable, *[geometry_cache if x == cache else x for x in args],
                       "--frame-end", 2, "--trace-geometry")
    geometry_traces = [record for record in geometry if record["event"] == "trace"]
    assert len(geometry_traces) == len(traces)
    for expected, observed in zip(traces, geometry_traces):
        particles = observed["geometry"]["particles"]
        assert [particle["id"] for particle in particles] == [1]
        assert all(math.isfinite(value) for value in particles[0]["world_position_m"])
        assert {key: value for key, value in observed.items() if key != "geometry"} == expected
    physical_bytes = {p: v[1] for p, v in immutable_cache_files(before).items()}
    for output_cache, records in ((fresh, rerun), (geometry_cache, geometry)):
        assert records[-1]["state_digest"] == result["state_digest"]
        assert records[-1]["execution_id"] == result["execution_id"]
        assert {p: v[1] for p, v in immutable_cache_files(snapshot(output_cache)).items()} == physical_bytes
        replay = execute(executable, "replay", "--cache", output_cache, "--frame", 2, "--substep", 0)[-1]
        assert replay["state_digest"] == result["state_digest"] and replay["read_only"]
    empty, _ = make_scene(executable, fixture, directory, backend, "empty")
    assert execute(executable, "run", "--stage", "simulate", "--scene", empty, "--cache", directory / "empty",
                   "--backend", backend, "--frame-end", 1)[-1]["particle_count"] == 0
    source, _ = make_scene(executable, fixture, directory, backend, "source")
    emitted = execute(executable, "run", "--stage", "simulate", "--scene", source, "--cache", directory / "source",
                      "--backend", backend, "--frame-end", 2)[-1]
    assert emitted["particle_count"] == 2 and emitted["next_particle_id"] == 3
    assert emitted["source_next_child_ordinals"] == [2]
    # Independent source-count arithmetic at the four known accepted durations.
    remainder = 0.0
    count = 0
    for duration in (1 / 48, 1 / 48, 1 / 48, 2 / 24 - 1 / 16):
        available = remainder + 25 * duration
        births = math.floor(available)
        count += births
        remainder = available - births
    assert count == 2 and emitted["source_fractional_remainders"] == [remainder]
    source_cache = directory / "source"
    source_before = snapshot(source_cache)
    repeated_source = execute(executable, "run", "--stage", "simulate", "--scene", source, "--cache", source_cache,
                              "--backend", backend, "--frame-end", 2, "--resume")[-1]
    assert repeated_source["state_digest"] == emitted["state_digest"]
    assert immutable_cache_files(snapshot(source_cache)) == immutable_cache_files(source_before)
    grown_cache = directory / "grown-cache"
    subprocess.run([str(fixture), "growth-cache", str(grown_cache), str(executable), backend, "source"], check=True)
    if backend == "cuda":
        # The fixture creates capacity metadata directly. Supply the actual
        # CUDA profile from the same artifact, scene and device.
        profile = ".simulate-cuda-compatibility"
        shutil.copy2(source_cache / profile, grown_cache / profile)
    grown = execute(executable, "run", "--stage", "simulate", "--scene", source, "--cache", grown_cache,
                    "--backend", backend, "--frame-end", 2, "--resume", "--trace-geometry")
    assert grown[-1]["state_digest"] == emitted["state_digest"]
    assert grown[-1]["capacity_replays"] == 1
    grown_traces = [record for record in grown if record["event"] == "trace"]
    assert [(r["frame"], r["substep"]) for r in grown_traces] == [(0, 1), (1, 0), (1, 1), (2, 0)]
    assert [record["accepted_step"] for record in grown_traces] == [1, 2, 3, 4]
    assert [record["accepted_attempts"] for record in grown_traces] == [1, 2, 2, 3]
    assert [[p["id"] for p in record["geometry"]["particles"]] for record in grown_traces] == [[], [1], [1], [1, 2]]
    attempts = [record for record in grown if record["event"] == "attempt" and record["target_frame"] == 2]
    assert [record["attempt_in_frame"] for record in attempts] == [1, 2, 3, 4]
    assert [record["capacity_pass"] for record in attempts] == [0, 0, 1, 1]
    assert [record["outcome"] for record in attempts] == ["accepted", "capacity", "accepted", "accepted"]
    assert [record["retained"] for record in attempts] == [False, False, True, True]
    assert [record["restarted_frame"] for record in attempts] == [False, True, False, False]
    assert [record["start_clock"]["accepted_step"] for record in attempts] == [2, 3, 2, 3]
    assert all(math.isfinite(record["work_wall_seconds"]) and record["work_wall_seconds"] >= 0
               for record in attempts)
    grown_ten = execute(executable, "run", "--stage", "simulate", "--scene", source, "--cache", grown_cache,
                        "--backend", backend, "--frame-end", 10, "--resume")[-1]
    source_ten = execute(executable, "run", "--stage", "simulate", "--scene", source, "--cache", source_cache,
                         "--backend", backend, "--frame-end", 10, "--resume")[-1]
    assert grown_ten["state_digest"] == source_ten["state_digest"]
    grown_metadata = json.loads(subprocess.check_output([str(fixture), "checkpoint", str(grown_cache)], text=True))
    assert grown_metadata["state_digest"] == grown_ten["state_digest"]
    assert grown_metadata["capacities"] == [16, 128, 4096, 24576, 1]
    assert grown_metadata["capacity_replay_particles"] == 0
    assert grown_metadata["source_next_child"] == 10 and grown_metadata["next_particle_id"] == 11
    boundary, _ = make_scene(executable, fixture, directory, backend, "boundary")
    boundary_cache = directory / "boundary"
    boundary_args = ["run", "--stage", "simulate", "--scene", boundary, "--cache", boundary_cache,
                     "--backend", backend, "--frame-end", 2]
    boundary_result = execute(executable, *boundary_args)[-1]
    # Four binary accepted durations leave the second integer birth just below 2.
    assert boundary_result["source_next_child_ordinals"] == [1]
    assert boundary_result["source_fractional_remainders"] == [0.9999999999999999]
    boundary_before = snapshot(boundary_cache)
    assert execute(executable, *boundary_args, "--resume")[-1]["state_digest"] == boundary_result["state_digest"]
    assert immutable_cache_files(snapshot(boundary_cache)) == immutable_cache_files(boundary_before)
    defaults, _ = make_scene(executable, fixture, directory, backend, "defaults")
    default_result = execute(executable, "run", "--stage", "simulate", "--scene", defaults,
                             "--cache", directory / "defaults", "--backend", backend, "--frame-end", 1)[-1]
    assert default_result["epoch"] == 5 and default_result["particle_count"] == 0
    if backend == "cuda":
        assert result["compiled_architecture"] == 1200 and result["graph_replay_count"] >= 4
        assert result["snapshot_copy_milliseconds"] > 0 and result["device_reserved_bytes"] > 0
        print(json.dumps(result, sort_keys=True))


def replay_case(executable, fixture, directory, backend):
    scene, initial = make_scene(executable, fixture, directory, backend)
    cache = directory / "cache"
    run = ["run", "--stage", "simulate", "--scene", scene, "--backend", backend, "--cache", cache, *initial]
    output = execute(executable, *run, "--frame-end", 20, "--trace")
    stored = json.loads(subprocess.check_output([str(fixture), "checkpoint", str(cache)], text=True))
    assert stored["state_digest"] == output[-1]["state_digest"]
    assert stored["frame"] == 20 and stored["seed"] == 123 and stored["epoch"] == stored["accepted_step"] == 40
    assert stored["particle_count"] == 1 and stored["next_particle_id"] == 2 and stored["first_tag"] == 19
    assert stored["capacities"] == [1024, 128, 4096, 24576, 256]
    before = snapshot(cache)
    replay = ["replay", "--scene", scene, "--backend", backend, "--cache", cache]
    result = execute(executable, *replay, "--frame", 20, "--substep", 0)[-1]
    assert result["state_digest"] == output[-1]["state_digest"] and result["read_only"]
    selected = execute(executable, *replay, "--frame", 10, "--substep", 1)[-1]
    trace = next(r for r in output if r["event"] == "trace" and r["frame"] == 10 and r["substep"] == 1)
    assert selected["state_digest"] == trace["state_digest"]
    assert selected["frame"] == 10 and selected["substep"] == 1 and selected["accepted_step"] == 21
    assert execute(executable, *replay, "--frame", 1, "--substep", 0, success=False)[-1]["name"] == "NOT_FOUND"
    assert execute(executable, *replay, *initial, "--frame", 1, "--substep", 0)[-1]["frame"] == 1
    execute(executable, *replay, "--frame", 10, "--substep", 9, success=False)
    assert snapshot(cache) == before, "replay changed files or modified times"
    # Resume at ten must verify/reuse eleven through twenty byte-for-byte.
    execute(executable, *run, "--resume", "--frame-start", 10, "--frame-end", 20)
    assert immutable_cache_files(snapshot(cache)) == immutable_cache_files(before)
    if backend == "cuda":
        assert selected["compiled_architecture"] == 1200 and selected["graph_replay_count"] > 0
        verify_cuda_compatibility_refusals(executable, cache, directory)
        print(json.dumps(selected, sort_keys=True))



def verify_cuda_compatibility_refusals(executable, cache, directory):
    # A foreign but syntactically valid profile must block all numerical
    # consumers, even replay of a retained checkpoint that needs no substeps.
    for name, record, expected in (
        ("foreign", b"FLCUDA1\n" + b"0" * 64 + b"\n", "FAILED_PRECONDITION"),
        ("missing", None, "FAILED_PRECONDITION"),
        ("malformed", b"FLCUDA1\n" + b"z" * 64 + b"\n", "DATA_LOSS"),
    ):
        candidate = directory / name
        shutil.copytree(cache, candidate)
        profile = candidate / ".simulate-cuda-compatibility"
        if record is None:
            profile.unlink()
        else:
            profile.write_bytes(record)
        # A rejected writable resume must not even create its operational lease.
        (candidate / ".simulate-job.lock").unlink()
        before = snapshot(candidate)
        before_directories = {path.relative_to(candidate) for path in candidate.rglob("*") if path.is_dir()}
        scene = candidate / "resolved.flrs"
        run = ["run", "--stage", "simulate", "--scene", scene, "--cache", candidate, "--resume", "--frame-end", 21]
        for args in (
            ["replay", "--cache", candidate, "--frame", 20, "--substep", 0],
            ["step", "--scene", scene, "--cache", candidate],
            [*run, "--preview"],
            run,
        ):
            result = execute(executable, *args, success=False)[-1]
            assert result["name"] == expected and result["field"] == "cuda-compatibility", result
            assert snapshot(candidate) == before, (name, args, "rejection mutated cache")
        checked = invoke(executable, "check", "--cache", candidate, "--frame", 20)
        assert checked["inspection_only"] and not checked["resume_authorized"] and checked["frame"] == 20
        assert snapshot(candidate) == before, (name, "inspection mutated cache")
        assert {path.relative_to(candidate) for path in candidate.rglob("*") if path.is_dir()} == before_directories


def case_cpu_execution(executable, fixture, directory, cuda):
    execution_case(executable, fixture, directory, "cpu")


def graded_replay_case(executable, fixture, directory, backend):
    directory.mkdir(exist_ok=True)
    scene, initial = make_scene(executable, fixture, directory, backend, "graded-tank")
    uninterrupted_cache = directory / "graded-continuous"
    resumed_cache = directory / "graded-resumed"
    run = ["run", "--stage", "simulate", "--scene", scene, "--backend", backend, "--trace"]
    records = execute(executable, *run, "--cache", uninterrupted_cache, *initial, "--frame-end", 11)
    uninterrupted = records[-1]
    assert any("accepted_measurement" in record for record in records)
    child = RunningSimulation(executable, [*run, "--cache", resumed_cache, *initial, "--frame-end", 19])
    try:
        _, published = child.next_event("frame")
        while published["frame"] < 10:
            _, published = child.next_event("frame")
        assert published["frame"] == 10 and published["published"]
        child.process.kill()
        exit_code = child.process.wait(timeout=10)
        assert exit_code != 0
    finally:
        child.close()
    checkpoint = json.loads(subprocess.check_output([str(fixture), "checkpoint", str(resumed_cache)], text=True))
    uninterrupted_checkpoint = json.loads(
        subprocess.check_output([str(fixture), "checkpoint", str(uninterrupted_cache)], text=True)
    )
    assert checkpoint["frame"] == 10
    assert checkpoint == uninterrupted_checkpoint
    assert checkpoint["capacities"] == [5, 0, 5, uninterrupted["face_count"], 0]
    # The resumed process has no initial-state argument. The only available
    # restart state is the authenticated frame-ten checkpoint.
    resumed = execute(executable, *run, "--cache", resumed_cache, "--resume", "--frame-end", 11)[-1]
    assert resumed["frame"] == uninterrupted["frame"] == 11
    assert resumed["state_digest"] == uninterrupted["state_digest"], (resumed, uninterrupted)
    assert resumed["cell_count"] == 5 and resumed["particle_count"] == 5
    assert resumed["face_count"] == uninterrupted["face_count"]
    if backend == "cuda":
        assert resumed["compiled_architecture"] == 1200, resumed
    print(json.dumps({"adaptive_checkpoint": checkpoint["state_digest"], "backend": backend,
                      "frame": resumed["frame"], "interrupted_exit_code": exit_code,
                      "state_digest": resumed["state_digest"]}, sort_keys=True))


def case_cpu_graded_replay(executable, fixture, directory, cuda):
    graded_replay_case(executable, fixture, directory, "cpu")


def case_cuda_graded_replay(executable, fixture, directory, cuda):
    graded_replay_case(executable, fixture, directory, "cuda")

def case_cpu_replay(executable, fixture, directory, cuda):
    replay_case(executable, fixture, directory, "cpu")
    graded_replay_case(executable, fixture, directory / "graded", "cpu")


def case_cuda_execution(executable, fixture, directory, cuda):
    execution_case(executable, fixture, directory, "cuda")


def case_cuda_replay(executable, fixture, directory, cuda):
    replay_case(executable, fixture, directory, "cuda")
    graded_replay_case(executable, fixture, directory / "graded", "cuda")


def case_execution_validation(executable, fixture, directory, cuda):
    scene, initial = make_scene(executable, fixture, directory, "cpu")
    cache = directory / "cache"
    args = ["run", "--stage", "simulate", "--scene", scene, "--cache", cache]
    execute(executable, *args, success=False)  # Required exact initial upstream.
    execute(executable, *args, *initial, "--backend", "cuda", success=False)
    execute(executable, *args, *initial, "--frame-end", 18446744073709551615, success=False)
    assert not cache.exists()
    preview_cache = directory / "preview"
    execute(executable, *[preview_cache if value == cache else value for value in args], *initial, "--preview")
    assert not preview_cache.exists()
    completed = execute(executable, *args, *initial, "--frame-start", 2, "--frame-end", 2)[-1]
    assert not (cache / "frames" / "00000000000000000001.flmf").exists()
    reconstructed = execute(executable, "replay", "--scene", scene, "--cache", cache, "--frame", 2, "--substep", 0)[-1]
    assert reconstructed["state_digest"] == completed["state_digest"]
    before = snapshot(cache)
    invalid_geometry = (
        ("check", "--cache", cache, "--trace-geometry"),
        ("replay", "--cache", cache, "--frame", 2, "--substep", 0, "--trace-geometry"),
        (*args, *initial, "--resume", "--trace-geometry", "--trace-geometry"),
    )
    for invalid_args in invalid_geometry:
        result = invoke(executable, *invalid_args, success=False)
        assert result["code"] == "E-CLI-INVALID_ARGUMENT"
        if invalid_args == invalid_geometry[-1]:
            assert result["field"] == "--trace-geometry"
        else:
            assert result["object"] == "--trace-geometry"
        assert snapshot(cache) == before
    execute(executable, *args, *initial, success=False)  # Existing cache requires resume.
    moved = directory / ("copied" + executable.suffix)
    shutil.copy2(executable, moved)
    assert execute(moved, *args, *initial, "--resume", "--frame-start", 2, "--frame-end", 2)[-1]["frame"] == 2
    with moved.open("ab") as stream:
        stream.write(b"new build")
    execute(moved, *args, *initial, "--resume", success=False)
    original = scene.read_bytes()
    altered = bytearray(original)
    root = struct.unpack_from("<I", altered)[0]
    vtable = root - struct.unpack_from("<i", altered, root)[0]
    upstream_field = root + struct.unpack_from("<H", altered, vtable + 4 + 14 * 2)[0]
    upstream_vector = upstream_field + struct.unpack_from("<I", altered, upstream_field)[0]
    upstream_entry = upstream_vector + 4
    upstream_table = upstream_entry + struct.unpack_from("<I", altered, upstream_entry)[0]
    upstream_vtable = upstream_table - struct.unpack_from("<i", altered, upstream_table)[0]
    upstream_role = upstream_table + struct.unpack_from("<H", altered, upstream_vtable + 4)[0]
    for unsupported_role in (1, 2, 255):
        altered[upstream_role] = unsupported_role
        scene.write_bytes(altered)
        execute(executable, *args, "--resume", success=False)
    altered = bytearray(original)
    seed = struct.unpack_from("<H", altered, vtable + 4 + 7 * 2)[0]
    altered[root + seed] ^= 1
    scene.write_bytes(altered)
    execute(executable, *args, *initial, "--resume", success=False)
    scene.write_bytes(original)
    state = Path(initial[1]); state.write_bytes(state.read_bytes() + b"corrupt")
    execute(executable, *args, *initial, "--resume", success=False)
    assert immutable_cache_files(snapshot(cache)) == immutable_cache_files(before)
    missing_lock = directory / "missing-lock"
    shutil.copytree(cache, missing_lock)
    (missing_lock / ".writer.lock").unlink()
    missing_before = snapshot(missing_lock)
    execute(executable, "replay", "--scene", scene, "--cache", missing_lock, "--frame", 2, "--substep", 0, success=False)
    assert snapshot(missing_lock) == missing_before
    wrong_clock = directory / "wrong-clock"
    subprocess.run([str(fixture), "wrong-clock", str(wrong_clock), str(executable), "cpu", "empty"], check=True)
    wrong_before = snapshot(wrong_clock)
    execute(executable, "replay", "--cache", wrong_clock, "--frame", 10, "--substep", 0, success=False)
    assert snapshot(wrong_clock) == wrong_before
    corrupt_state = directory / "corrupt-state"
    shutil.copytree(cache, corrupt_state)
    next((corrupt_state / "blobs").glob("*.flst")).write_bytes(b"corrupt checkpoint")
    corrupt_before = snapshot(corrupt_state)
    execute(executable, "replay", "--cache", corrupt_state, "--frame", 2, "--substep", 0, success=False)
    assert snapshot(corrupt_state) == corrupt_before
    manifest = next((cache / "frames").glob("*.flmf")); manifest.write_bytes(b"corrupt")
    corrupt = snapshot(cache)
    execute(executable, "replay", "--cache", cache, "--frame", 2, "--substep", 0, success=False)
    assert snapshot(cache) == corrupt
    oversized_scene, _ = make_scene(executable, fixture, directory, "cpu", "capacity-failure")
    failed_cache = directory / "failed-cache"
    failed = execute(executable, "run", "--stage", "simulate", "--scene", oversized_scene, "--cache", failed_cache,
                     "--backend", "cpu", "--frame-end", 1, "--trace-geometry", success=False)
    assert failed[-1]["name"] == "RESOURCE_EXHAUSTED"
    assert not any(record["event"] == "trace" for record in failed)
    rejected, = [record for record in failed if record["event"] == "attempt"]
    assert rejected["outcome"] == "capacity" and rejected["failure_code"] == "RESOURCE_EXHAUSTED"
    assert rejected["attempt_in_frame"] == 1 and rejected["start_clock"]["accepted_step"] == 0
    assert not rejected["retained"] and not rejected["restarted_frame"]
    assert math.isfinite(rejected["work_wall_seconds"]) and rejected["work_wall_seconds"] >= 0
    assert invoke(executable, "check", "--cache", failed_cache, "--frame", 0)["job_status"] == "failed"
    assert not (failed_cache / "frames" / "00000000000000000001.flmf").exists()


class RunningSimulation:
    def __init__(self, executable, args):
        self.process = subprocess.Popen(
            [str(executable), *map(str, args)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, encoding="utf-8",
        )
        self.process.stdin.reconfigure(newline="")
        self.lines = Queue()
        self.stdout_lines = []
        self.reader = Thread(target=self.read_lines)
        self.reader.start()

    def read_lines(self):
        # Keep bytes until a full JSONL record is available. A hard process
        # kill can truncate even the final UTF-8 code point.
        for line in self.process.stdout.buffer:
            self.stdout_lines.append(line)
            self.lines.put((monotonic(), line))
        self.lines.put((monotonic(), b""))

    def stdout_records_after_exit(self, allow_trailing_partial=False):
        """Read every complete record, including those consumed by next_event."""
        assert self.process.poll() is not None
        self.reader.join(timeout=10)
        assert not self.reader.is_alive()
        stdout_lines = self.stdout_lines
        discarded_trailing_bytes = 0
        if stdout_lines and not stdout_lines[-1].endswith(b"\n"):
            assert allow_trailing_partial, "partial stdout record after normal exit"
            discarded_trailing_bytes = len(stdout_lines[-1])
            stdout_lines = stdout_lines[:-1]
        return records(b"".join(stdout_lines).decode("utf-8")), discarded_trailing_bytes

    def next_event(self, event):
        deadline = monotonic() + 10
        while monotonic() < deadline:
            try:
                received_at, line = self.lines.get(timeout=max(0, deadline - monotonic()))
            except Empty:
                break
            assert line, f"child exited before {event}"
            record, = records(line.decode("utf-8"))
            assert record["event"] != "error" or event == "error", record
            if record["event"] == event:
                return received_at, record
        raise AssertionError(f"missing {event} from running simulation")

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.reader.join(timeout=10)
        assert not self.reader.is_alive()
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()


def lifecycle_case(executable, fixture, directory, backend):
    # A 1 us maximum step makes the first frame long using real solver work.
    # The committed frame zero remains available for inspection during the wait.
    scene, initial = make_scene(executable, fixture, directory, backend, "slow")
    cache = directory / "lifecycle"
    run = ["run", "--stage", "simulate", "--scene", scene, "--backend", backend, "--cache", cache, *initial]
    frame_zero = ["check", "--cache", cache, "--frame", 0]
    cancellable_simulation = RunningSimulation(executable, [*run, "--frame-end", 1, "--trace", "--measure-job"])
    try:
        _, published = cancellable_simulation.next_event("frame")
        assert published["frame"] == 0
        first_heartbeat_received_at, first_heartbeat = cancellable_simulation.next_event("heartbeat")
        # The first heartbeat may precede the initial snapshot. Once frame zero
        # has appeared on stdout, every later heartbeat must know it is published.
        assert first_heartbeat["last_published_frame"] == 0 and first_heartbeat["backend"] == backend
        assert first_heartbeat["last_settled_clock"] == {"frame": 0, "substep": 0, "epoch": 0, "accepted_step": 0}
        before = snapshot(cache, live_job=True)
        active = invoke(executable, *frame_zero)
        assert active["job_status"] == "baking"
        assert active["epoch"] == 0 and active["point_count"] == 1
        assert snapshot(cache, live_job=True) == before, "live inspection mutated the cache"
        competing = execute(executable, *run, "--resume", "--frame-end", 0, success=False)[-1]
        assert competing["name"] == "ABORTED", competing
        # Discard an unrelated command and retain a command split across reads.
        cancellable_simulation.process.stdin.write("cancelled\ncan")
        cancellable_simulation.process.stdin.flush()
        second_heartbeat_received_at, second_heartbeat = cancellable_simulation.next_event("heartbeat")
        assert second_heartbeat["last_published_frame"] == 0
        assert second_heartbeat["last_settled_clock"] == first_heartbeat["last_settled_clock"]
        assert 0 < second_heartbeat["elapsed_milliseconds"] - first_heartbeat["elapsed_milliseconds"] <= 1000
        heartbeat_interval_seconds = second_heartbeat_received_at - first_heartbeat_received_at
        assert 0 < heartbeat_interval_seconds <= 1.0, heartbeat_interval_seconds
        cancel_sent = monotonic()
        cancellable_simulation.process.stdin.write("cel\r\n")
        cancellable_simulation.process.stdin.flush()
        _, cancelled = cancellable_simulation.next_event("error")
        assert cancelled["name"] == "CANCELLED", cancelled
        assert cancellable_simulation.process.wait(timeout=10) != 0
        cancel_exit_seconds = monotonic() - cancel_sent
        assert 0 < cancel_exit_seconds <= 10, cancel_exit_seconds
        assert not cancellable_simulation.process.stdin.closed, "cancellation should not require EOF"
        assert records(cancellable_simulation.process.stderr.read()) == [cancelled]
        complete_records, _ = cancellable_simulation.stdout_records_after_exit()
        assert not any(record["event"] == "trace" for record in complete_records)
        attempts = [record for record in complete_records if record["event"] == "attempt"]
        assert attempts, "cancelled frame lost actual attempt work"
        measurement = complete_records[-2]
        assert measurement["event"] == "job_measurement" and measurement["status"] == "CANCELLED"
        assert measurement["attempt_count"] == len(attempts) and measurement["settled_attempt_count"] == 0
        assert measurement["host_wall"]["emit_activation_field_seconds"] > 0
        if backend == "cuda":
            assert measurement["cuda_events"]["pre_topology_field_interval_count"] >= len(attempts)
            assert measurement["cuda_events"]["snapshot_d2h_interval_count"] == 1
            assert measurement["cuda_memory"]["remaining_device_reserved_bytes_after_drain"] == 0
            assert measurement["cuda_memory"]["remaining_pinned_reserved_bytes_after_drain"] == 0
        else:
            assert measurement["cuda_events"] is None and measurement["cuda_memory"] is None
        for index, attempt in enumerate(attempts, start=1):
            assert attempt["target_frame"] == 1 and attempt["attempt_in_frame"] == index
            assert attempt["outcome"] == "accepted" and not attempt["retained"]
            assert attempt["start_clock"]["accepted_step"] == index - 1
        _, final_line = cancellable_simulation.lines.get(timeout=10)
        assert final_line == b"", "unexpected stdout after the terminal record"
    finally:
        cancellable_simulation.close()
    assert invoke(executable, *frame_zero)["job_status"] == "cancelled"
    assert invoke(executable, *frame_zero)["points_relative_name"] == active["points_relative_name"]
    assert immutable_cache_files(snapshot(cache)) == immutable_cache_files(before)
    assert not (cache / "frames" / "00000000000000000001.flmf").exists()
    assert execute(executable, *run, "--resume", "--frame-end", 0)[-1]["frame"] == 0
    assert invoke(executable, *frame_zero)["job_status"] == "complete"

    interruptible_simulation = RunningSimulation(executable, [*run, "--resume", "--frame-end", 1])
    try:
        interruptible_simulation.next_event("frame")
        assert invoke(executable, *frame_zero)["job_status"] == "baking"
        interruptible_simulation.process.kill()
        assert interruptible_simulation.process.wait(timeout=10) != 0
    finally:
        interruptible_simulation.close()
    abandoned = snapshot(cache)
    assert invoke(executable, *frame_zero)["job_status"] == "crashed"
    assert snapshot(cache) == abandoned, "crash classification must be read-only"
    assert execute(executable, *run, "--resume", "--frame-end", 0)[-1]["frame"] == 0
    assert invoke(executable, *frame_zero)["job_status"] == "complete"
    if backend == "cuda":
        assert published["compiled_architecture"] == 1200 and published["snapshot_copy_milliseconds"] > 0
    print(json.dumps({
        "backend": backend, "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "cancel_exit_seconds": cancel_exit_seconds, "frame_zero": published,
    }, sort_keys=True))


def case_cpu_lifecycle(executable, fixture, directory, cuda):
    lifecycle_case(executable, fixture, directory, "cpu")


def effects_case(executable, fixture, directory, backend):
    scene, initial = make_scene(executable, fixture, directory, backend, "effects")
    cache = directory / "effects-cache"
    run = ["run", "--stage", "simulate", "--scene", scene, "--backend", backend, "--cache", cache, *initial]
    result = execute(executable, *run, "--frame-end", 3)[-1]
    products = cache / "effects" / "source"
    assert len(list((products / "frames").glob("*.flem"))) == 4
    before = snapshot(products)
    execute(executable, "replay", "--scene", scene, "--cache", cache, *initial, "--frame", 3, "--substep", 0)
    assert snapshot(products) == before
    execute(executable, *run, "--resume", "--frame-end", 5)
    assert len(list((products / "frames").glob("*.flem"))) == 6
    assert all(snapshot(products)[path] == value for path, value in before.items())
    payloads = {path.name: path.read_bytes() for path in (products / "blobs").glob("*.flef")}
    # Directory enumeration order is filesystem-dependent. Corrupt a payload in
    # the requested replay prefix, rather than an arbitrary later frame.
    seed_manifest = json.loads((products / "frames" / "00000000000000000000.json").read_text())
    source_blob = products / "blobs" / (seed_manifest["payload_digest"] + ".flef")
    source_blob.write_bytes(b"corrupt")
    invoke(executable, "replay", "--scene", scene, "--cache", cache, *initial, "--frame", 3, "--substep", 0,
           success=False)
    if backend == "cuda":
        assert result["compiled_architecture"] == 1200 and result["graph_replay_count"] > 0
        assert result["snapshot_copy_milliseconds"] > 0 and result["device_reserved_bytes"] > 0
        print(json.dumps(result, sort_keys=True))
    return payloads


def case_cpu_effects(executable, fixture, directory, cuda):
    effects_case(executable, fixture, directory, "cpu")
    whitewater_source_case(executable, fixture, directory, "cpu")
    whitewater_source_case(executable, fixture, directory, "cpu", "whitewater-beach")


def whitewater_source_case(executable, fixture, directory, backend, mode="whitewater-effects"):
    root = directory / mode
    root.mkdir()
    scene, initial = make_scene(executable, fixture, root, backend, mode)
    cache = root / "cache"
    args = ["run", "--stage", "simulate", "--scene", scene, "--cache", cache, "--backend", backend, *initial]
    execute(executable, *args, "--frame-end", 2)
    receipt = json.loads(subprocess.run([str(fixture), "source-births", str(cache)], check=True,
                                        capture_output=True, encoding="utf-8", timeout=30).stdout)
    assert receipt["birth_count"] > 0 and receipt["sample_count"] == 2
    assert 0 < receipt["first_birth_step"] < receipt["last_accepted_step"]
    before = snapshot(cache)
    invoke(executable, "replay", "--scene", scene, "--cache", cache, *initial, "--frame", 1, "--substep", 0)
    assert snapshot(cache) == before, "whitewater source replay must be read-only"
    points = root / "whitewater"
    command = ["run", "--stage", "whitewater", "--cache", cache, "--output-cache", points]
    result = execute(executable, *command, "--frame-end", 1)[-1]
    assert result["stage"] == "whitewater" and result["backend"] == "cpu" and result["frame"] == 1
    assert result["source_backend"] == backend
    assert result["foam_count"] > 0 and result["birth_count"] > 0
    assert result["point_count"] == result["foam_count"] + result["spray_count"] + result["bubble_count"]
    assert result["frame_wall_seconds"] > 0 and result["peak_process_resident_bytes"] > 0
    points_before = snapshot(points)
    replay = ["replay", "--stage", "whitewater", "--cache", cache, "--output-cache", points,
              "--frame", 1, "--substep", 0]
    assert invoke(executable, *replay)["payload_digest"] == result["payload_digest"]
    assert snapshot(points) == points_before, "whitewater replay must be read-only"
    resumed = execute(executable, *command, "--resume", "--frame-end", 2)[-1]
    assert resumed["frame"] == 2 and resumed["restored_prefix_frames"] == 2
    points_after = snapshot(points)
    assert all(points_after[path] == value for path, value in immutable_cache_files(points_before).items())
    assert snapshot(cache) == before, "whitewater must leave Simulate data unchanged"
    volume_settings = root / "volume.flvc"
    subprocess.run([str(fixture), "volume-settings", str(volume_settings)], check=True, timeout=30)
    volume = root / "volumes"
    volume_command = ["run", "--stage", "volumes", "--cache", points, "--output-cache", volume,
                      "--volume-settings", volume_settings]
    volume_records = execute(executable, *volume_command, "--frame-end", 1)
    empty = next(record for record in volume_records if record["event"] == "frame" and record["frame"] == 0)
    assert empty["known_voxels"] == 24 ** 3 and empty["nonzero_voxels"] == 0
    density = volume_records[-1]
    assert density["stage"] == "volumes" and density["nonzero_voxels"] > 0
    assert math.isclose(density["foam_inventory_count"], result["foam_count"], rel_tol=1e-12)
    assert math.isclose(density["bubble_inventory_count"], result["bubble_count"], rel_tol=1e-12)
    assert math.isclose(density["gas_inventory_m3"], result["bubble_count"] * 4 * math.pi / 3 * 0.05 ** 3,
                        rel_tol=1e-12)
    volume_before = snapshot(volume)
    volume_replay = invoke(executable, "replay", "--stage", "volumes", "--cache", points,
                           "--output-cache", volume, "--frame", 1, "--substep", 0)
    assert volume_replay["payload_digest"] == density["payload_digest"]
    assert snapshot(volume) == volume_before
    volume_resume = execute(executable, *volume_command, "--resume", "--frame-end", 2)[-1]
    assert volume_resume["restored_prefix_frames"] == 2 and volume_resume["frame"] == 2
    assert snapshot(points) == points_after, "volumes must leave Whitewater data unchanged"
    if mode == "whitewater-beach":
        mesh = root / "mesh"
        bake = ["run", "--stage", "mesh", "--cache", cache, "--output-cache", mesh,
                "--surface-spacing", ".125", "--surface-sigma", ".25"]
        surface = execute(executable, *bake, "--frame-end", 2)[-1]
        assert surface["retained_ids"] == surface["vertex_count"] > 0
        assert abs(surface["volume_relative_error"]) <= 0.005
        assert snapshot(cache) == before
    (points / "blobs" / (result["payload_digest"] + ".flef")).write_bytes(b"corrupt")
    assert invoke(executable, *replay, success=False)["code"] == "E-WHITEWATER-DATA_LOSS"


def case_cpu_mesh(executable, fixture, directory, cuda):
    scene, initial = make_scene(executable, fixture, directory, "cpu", "effects")
    source = directory / "simulate"
    mesh = directory / "mesh"
    execute(executable, "run", "--stage", "simulate", "--scene", scene, "--cache", source, *initial,
            "--frame-end", 2)
    source_before = snapshot(source)
    command = ["run", "--stage", "mesh", "--cache", source, "--output-cache", mesh,
               "--surface-spacing", "0.125", "--surface-sigma", "0.25"]
    result = execute(executable, *command, "--frame-end", 1)[-1]
    assert result["stage"] == "mesh" and result["backend"] == "cpu" and result["frame"] == 1
    assert result["vertex_count"] > 0 and result["triangle_count"] > 0
    assert result["retained_ids"] == result["vertex_count"]
    assert abs(result["volume_relative_error"]) <= 0.005
    assert result["frame_wall_seconds"] > 0 and result["peak_process_resident_bytes"] > 0
    before = snapshot(mesh)
    replay = invoke(executable, "replay", "--stage", "mesh", "--cache", source, "--output-cache", mesh,
                    "--frame", 1, "--substep", 0)
    assert replay["payload_digest"] == result["payload_digest"]
    assert snapshot(mesh) == before, "Mesh replay must be entirely read-only"
    resumed = execute(executable, *command, "--resume", "--frame-end", 2)[-1]
    assert resumed["frame"] == 2
    assert resumed["restored_prefix_frames"] == 2 and resumed["restore_wall_seconds"] > 0
    after = snapshot(mesh)
    assert all(after[path] == value for path, value in immutable_cache_files(before).items())
    assert snapshot(source) == source_before, "Mesh must leave source cache unchanged"
    invoke(executable, *command, "--frame-end", 1, success=False)
    invalid = invoke(executable, "run", "--stage", "mesh", "--cache", source, "--output-cache", mesh,
                     "--surface-spacing", "nan", success=False)
    assert invalid["code"] == "E-CLI-INVALID_ARGUMENT"


def case_cuda_effects(executable, fixture, directory, cuda):
    cuda_payloads = effects_case(executable, fixture, directory, "cuda")
    control = directory / "cpu-control"
    control.mkdir()
    scene, initial = make_scene(executable, fixture, control, "cpu", "effects")
    cache = control / "cache"
    execute(executable, "run", "--stage", "simulate", "--scene", scene, "--backend", "cpu", "--cache", cache,
            *initial, "--frame-end", 5)
    blobs = cache / "effects" / "source" / "blobs"
    cpu_payloads = {path.name: path.read_bytes() for path in blobs.glob("*.flef")}
    assert cpu_payloads == cuda_payloads, "manufactured static source intervals differ across CPU/CUDA"
    whitewater_source_case(executable, fixture, directory, "cuda")
    whitewater_source_case(executable, fixture, directory, "cuda", "whitewater-beach")


def ocean_case(executable, fixture, directory, backend, mode):
    directory.mkdir()
    scene, initial = make_scene(executable, fixture, directory, backend, mode)
    cache = directory / "cache"
    execute(executable, "run", "--stage", "simulate", "--scene", scene, "--backend", backend,
            "--cache", cache, *initial, "--frame-end", 10)
    checkpoint = json.loads(subprocess.run([str(fixture), "checkpoint", str(cache)],
                                          capture_output=True, check=True, timeout=30).stdout)
    expected_particles = 3 if mode == "ocean-level" else 1
    assert checkpoint["frame"] == 10 and checkpoint["particle_count"] == expected_particles, checkpoint
    assert checkpoint["next_particle_id"] == expected_particles + 1, checkpoint
    before = snapshot(cache)
    invoke(executable, "replay", "--scene", scene, "--cache", cache, *initial,
           "--frame", 9, "--substep", 0)
    assert snapshot(cache) == before
    return checkpoint


def case_cpu_ocean(executable, fixture, directory, cuda):
    active = ocean_case(executable, fixture, directory / "ocean", "cpu", "ocean")
    control = ocean_case(executable, fixture, directory / "control", "cpu", "initial")
    assert math.dist(active["first_velocity"], control["first_velocity"]) > 0.001
    assert math.dist(active["first_position"], control["first_position"]) > 0.00001
    ocean_case(executable, fixture, directory / "level", "cpu", "ocean-level")


def case_cuda_ocean(executable, fixture, directory, cuda):
    cpu = ocean_case(executable, fixture, directory / "cpu", "cpu", "ocean")
    device = ocean_case(executable, fixture, directory / "cuda", "cuda", "ocean")
    assert cpu["accepted_step"] == device["accepted_step"]
    assert math.dist(cpu["first_position"], device["first_position"]) < 1e-10
    assert math.dist(cpu["first_velocity"], device["first_velocity"]) < 1e-9
    cpu_level = ocean_case(executable, fixture, directory / "cpu-level", "cpu", "ocean-level")
    device_level = ocean_case(executable, fixture, directory / "cuda-level", "cuda", "ocean-level")
    assert cpu_level["state_digest"] == device_level["state_digest"]


def case_cuda_lifecycle(executable, fixture, directory, cuda):
    lifecycle_case(executable, fixture, directory, "cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--case", choices=["arguments", "schema", "identity", "inspection", "cpu_device", "cuda_device", "cpu_execution", "cpu_replay", "cpu_graded_replay", "cuda_graded_replay", "execution_validation", "cuda_execution", "cuda_replay", "cpu_lifecycle", "cuda_lifecycle", "cpu_effects", "cuda_effects", "cpu_mesh", "cpu_ocean", "cuda_ocean"], required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    assert args.executable.is_file() and args.fixture.is_file(), "required process executable is missing"
    with tempfile.TemporaryDirectory(prefix="fluidity-cli-") as directory:
        globals()["case_" + args.case](args.executable.resolve(), args.fixture.resolve(), Path(directory), args.cuda)
    print(f"PASS {args.case}")
