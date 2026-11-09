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
import shutil
import struct
import subprocess
import sys
import tempfile

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


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


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
        assert all(r["event"] in ("trace", "frame", "result") for r in output)
    else:
        errors = records(process.stderr)
        assert errors == [output[-1]] and errors[0]["event"] == "error"
    return output


def make_scene(executable, fixture, directory, backend, mode="initial"):
    scene = directory / (backend + "-" + mode + ".flrs")
    subprocess.run([str(fixture), "scene", str(scene), str(executable), backend, mode],
                   check=True, capture_output=True, timeout=30)
    initial = ["--initial-state", str(scene) + ".flst"] if mode == "initial" else []
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
    traces = [r for r in output if r["event"] == "trace"]
    assert [(r["frame"], r["substep"]) for r in traces] == [(0, 1), (1, 0), (1, 1), (2, 0)]
    before = snapshot(cache)
    resumed = execute(executable, *args, "--resume", "--frame-end", 2)[-1]
    assert resumed["state_digest"] == result["state_digest"]
    assert snapshot(cache) == before, "resume rewrote committed content"
    preview = execute(executable, *args, "--resume", "--preview", "--frame-end", 3)[-1]
    assert preview["frame"] == 3 and preview["read_only"]
    assert snapshot(cache) == before
    step = execute(executable, "step", "--scene", scene, "--backend", backend, "--cache", cache, *initial)[-1]
    assert step["frame"] == 0 and step["substep"] == 1 and step["accepted_step"] == 1
    assert step["read_only"] and snapshot(cache) == before
    fresh = directory / "fresh"
    rerun = execute(executable, *[fresh if x == cache else x for x in args], "--frame-end", 2)[-1]
    assert rerun["state_digest"] == result["state_digest"]
    assert {p: v[1] for p, v in snapshot(fresh).items()} == {p: v[1] for p, v in before.items()}
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
    assert repeated_source["state_digest"] == emitted["state_digest"] and snapshot(source_cache) == source_before
    grown_cache = directory / "grown-cache"
    subprocess.run([str(fixture), "growth-cache", str(grown_cache), str(executable), backend, "source"], check=True)
    grown = execute(executable, "run", "--stage", "simulate", "--scene", source, "--cache", grown_cache,
                    "--backend", backend, "--frame-end", 2, "--resume", "--trace")
    assert grown[-1]["state_digest"] == emitted["state_digest"]
    assert grown[-1]["capacity_replays"] == 1
    assert [(r["frame"], r["substep"]) for r in grown if r["event"] == "trace"] == [(0, 1), (1, 0), (1, 1), (2, 0)]
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
    assert snapshot(boundary_cache) == boundary_before
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
    assert snapshot(cache) == before
    if backend == "cuda":
        assert selected["compiled_architecture"] == 1200 and selected["graph_replay_count"] > 0
        print(json.dumps(selected, sort_keys=True))


def case_cpu_execution(executable, fixture, directory, cuda):
    execution_case(executable, fixture, directory, "cpu")


def case_cpu_replay(executable, fixture, directory, cuda):
    replay_case(executable, fixture, directory, "cpu")


def case_cuda_execution(executable, fixture, directory, cuda):
    execution_case(executable, fixture, directory, "cuda")


def case_cuda_replay(executable, fixture, directory, cuda):
    replay_case(executable, fixture, directory, "cuda")


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
    assert snapshot(cache) == before
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--case", choices=["arguments", "schema", "identity", "inspection", "cpu_device", "cuda_device", "cpu_execution", "cpu_replay", "execution_validation", "cuda_execution", "cuda_replay"], required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    assert args.executable.is_file() and args.fixture.is_file(), "required process executable is missing"
    with tempfile.TemporaryDirectory(prefix="fluidity-cli-") as directory:
        globals()["case_" + args.case](args.executable.resolve(), args.fixture.resolve(), Path(directory), args.cuda)
    print(f"PASS {args.case}")
