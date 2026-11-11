"""Exercise the public CLI and the existing machine interface together."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from test_simulate_replay import execute, invoke, make_scene, snapshot


def run(executable, *arguments):
    return subprocess.run(
        [executable, *arguments], capture_output=True, encoding="utf-8", timeout=30, check=False
    )


def main():
    fluidity, legacy, fixture = map(Path, sys.argv[1:])
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
        ("unknown",),
        ("simulate", "scene.flrs", "--cache", "bake", "--frame-end", "-1"),
        ("devices", "--backend", "cpu", "--backend", "cuda"),
        ("devices", "--device", "0"),
        ("inspect", "cache", "--trace"),
    ):
        invalid = run(fluidity, *arguments)
        assert invalid.returncode == 2, invalid
        assert invalid.stderr, invalid

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
        assert invoke(fluidity, "validate", scene) == invoke(legacy, "check", "--scene", scene)
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

    print("PASS public_interface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
