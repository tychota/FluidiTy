"""Exercise the public CLI and the existing machine interface together."""

import json
import shutil
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from test_simulate_replay import execute, invoke, make_scene, snapshot


def run(executable, *arguments, cwd=None):
    return subprocess.run(
        [executable, *arguments], cwd=cwd, capture_output=True, encoding="utf-8", timeout=30, check=False
    )


def graded_effects_case(executable, fixture, root):
    scene, initial = make_scene(executable, fixture, root, "cpu", "graded-effects")
    initial_path = Path(initial[1])
    initial_digest = subprocess.check_output(
        [fixture, "digest", initial_path], text=True, timeout=30
    ).strip()
    tampered = root / "tampered.flst"
    tampered_bytes = bytearray(initial_path.read_bytes())
    tampered_bytes[-1] ^= 1
    tampered.write_bytes(tampered_bytes)
    refused_cache = root / "refused graded cache"
    refused = execute(
        executable, "simulate", scene, "--cache", refused_cache, "--initial-state", tampered,
        "--frame-end", 2, success=False,
    )[-1]
    assert refused["name"] == "DATA_LOSS" and refused["field"] == "initial-state", refused
    assert not refused_cache.exists()

    cache = root / "graded effects cache"
    output = execute(executable, "simulate", scene, "--cache", cache, *initial, "--frame-end", 2, "--trace")
    baked = output[-1]
    assert baked["frame"] == 2 and baked["particle_count"] == baked["cell_count"] == 5
    assert baked["face_count"] > 0
    assert [item["frame"] for item in output if item["event"] == "frame"] == [0, 1, 2]

    before = snapshot(cache)
    checkpoint = json.loads(subprocess.check_output([fixture, "checkpoint", cache], text=True, timeout=30))
    assert checkpoint["frame"] == 0 and checkpoint["state_digest"] == initial_digest
    assert checkpoint["capacities"] == [5, 0, 5, baked["face_count"], 0]
    traces = {
        (item["frame"], item["substep"]): item["state_digest"]
        for item in output if item["event"] == "trace"
    }
    traces[0, 0] = initial_digest

    frames = json.loads(subprocess.check_output([fixture, "graded-source", cache], text=True, timeout=30))
    assert [len(frame["samples"]) for frame in frames] == [1, 3, 3]
    for ordinal, frame in enumerate(frames):
        samples = frame["samples"]
        assert (samples[-1]["frame"], samples[-1]["substep"]) == (ordinal, 0)
        for sample in samples:
            assert sample["adaptive_cell_count"] == sample["particle_count"] == 5
            assert sample["adaptive_face_count"] == baked["face_count"]
            assert sample["state_digest"] == traces[sample["frame"], sample["substep"]]

    complete = execute(executable, "replay", cache, "--frame", 2, "--substep", 0)[-1]
    assert complete["read_only"] and complete["state_digest"] == baked["state_digest"]
    intermediate = execute(executable, "replay", cache, "--frame", 1, "--substep", 1)[-1]
    assert intermediate["read_only"] and intermediate["state_digest"] == traces[1, 1]
    assert snapshot(cache) == before, "graded replay changed parent cache or effects source files"
    print(json.dumps({
        "graded_initial_state_digest": initial_digest,
        "graded_state_digest": baked["state_digest"],
        "graded_source_payload_digests": [frame["payload_digest"] for frame in frames],
        "graded_samples": [len(frame["samples"]) for frame in frames],
    }, sort_keys=True))


def main():
    fluidity, legacy, fixture = (Path(argument).resolve() for argument in sys.argv[1:])
    help_result = run(fluidity, "--help")
    assert help_result.returncode == 0, help_result
    assert all(
        command in help_result.stdout
        for command in ("simulate", "validate", "inspect", "step", "replay", "devices", "schema")
    ), help_result
    simulate_help = run(fluidity, "simulate", "--help")
    assert simulate_help.returncode == 0, simulate_help
    assert "--frame-end" in simulate_help.stdout, simulate_help

    for arguments in (
        ("simulate",),
        ("simulate", "scene.flrs", "--cache", "--trace"),
        ("unknown",),
        ("simulate", "scene.flrs", "--cache", "bake", "--frame-end", "-1"),
        ("devices", "--backend", "cpu", "--backend", "cuda"),
        ("devices", "--device", "0"),
        ("inspect", "cache", "--trace"),
        ("simulate", "scene.flrs", "--cache", "bake", "--trace", "--trace"),
        ("simulate", "scene.flrs", "--cache", "bake", "--trace=true"),
        ("devices", "--backend=", "cpu"),
        ("schema", "--", "--help"),
    ):
        invalid = run(fluidity, *arguments)
        assert invalid.returncode == 2, invalid
        assert invalid.stdout == invalid.stderr, invalid
        error = json.loads(invalid.stderr)
        assert error["event"] == "error" and error["code"] == "E-CLI-INVALID_ARGUMENT", error

    for command, legacy_command in (("schema", "schema"), ("devices", "device-info")):
        public = run(fluidity, command)
        machine = run(legacy, legacy_command)
        assert public.returncode == machine.returncode == 0, (public, machine)
        assert json.loads(public.stdout) == json.loads(machine.stdout), (public, machine)
    with TemporaryDirectory() as directory:
        root = Path(directory) / "résumé 液体"
        root.mkdir()
        scene, initial = make_scene(legacy, fixture, root, "cpu")
        cache = root / "output cache"
        validated = invoke(legacy, "check", "--scene", scene)
        assert invoke(fluidity, "validate", scene) == validated
        shutil.copyfile(scene, root / "--scene.flrs")
        dash_scene = run(fluidity, "validate", "--", "--scene.flrs", cwd=root)
        assert dash_scene.returncode == 0, dash_scene
        assert json.loads(dash_scene.stdout) == validated
        dash_cache = run(
            fluidity, "simulate", scene, "--cache=--bake", *initial, "--frame-end=0", cwd=root
        )
        assert dash_cache.returncode == 0, dash_cache
        assert (root / "--bake").is_dir()
        single_dash_cache = run(
            fluidity, "simulate", scene, "--cache", "-h=cache", *initial, "--frame-end=0", cwd=root
        )
        assert single_dash_cache.returncode == 0, single_dash_cache
        assert (root / "-h=cache").is_dir()
        baked = execute(
            fluidity, "simulate", scene, "--cache", cache, *initial, "--frame-end=1"
        )[-1]
        assert baked["frame"] == 1 and baked["backend"] == "cpu"
        before = snapshot(cache)
        assert invoke(fluidity, "inspect", cache, "--frame", "1") == invoke(
            legacy, "check", "--cache", cache, "--frame", "1"
        )
        replay = execute(fluidity, "replay", cache, "--frame", 1, "--substep", 0)[-1]
        assert replay["state_digest"] == baked["state_digest"]
        assert snapshot(cache) == before
        graded_effects_case(fluidity, fixture, root)

    print("PASS public_interface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
