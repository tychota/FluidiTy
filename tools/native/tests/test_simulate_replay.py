"""Required real-process CLI checks. Invoked by CTest, never conditionally skipped.

This module intentionally has no pytest tests: CTest supplies the built binaries
and each named case fails immediately when either executable is absent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


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
        assert invoke(executable, *args, success=False)["code"] == "E-CLI-UNIMPLEMENTED"
    controls = 'bad\n\t"\\\x01'
    assert invoke(executable, "schema", controls, success=False)["object"] == controls
    assert invoke(executable, "run", "--stage", "simulate", "--scene", scene, "--cache", cache,
                  "--backend", "cuda", "--device", "2147483647", "--frame-start", "0", "--frame-end", "18446744073709551615",
                  "--resume", "--trace", "--preview", success=False)["code"] == "E-CLI-UNIMPLEMENTED"
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
    assert result["execution_available"] is False


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--case", choices=["arguments", "schema", "identity", "inspection", "cpu_device", "cuda_device"], required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    assert args.executable.is_file() and args.fixture.is_file(), "required process executable is missing"
    with tempfile.TemporaryDirectory(prefix="fluidity-cli-") as directory:
        globals()["case_" + args.case](args.executable.resolve(), args.fixture.resolve(), Path(directory), args.cuda)
    print(f"PASS {args.case}")
