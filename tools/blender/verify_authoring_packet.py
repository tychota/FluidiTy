"""Validate the checked authoring bridge and original packet byte compatibility."""

import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    bridge, solver, previous_bridge, output = (Path(value).resolve() for value in sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        root = Path(temporary)

        def resolve(name, packet, tool=bridge, valid=True):
            source, scene, state = (root / f"{name}.{suffix}" for suffix in ("txt", "flrs", "flst"))
            source.write_text(packet, encoding="ascii")
            result = subprocess.run(
                [tool, "resolve", solver, "cpu", source, scene, state], capture_output=True, check=False
            )
            assert (result.returncode == 0) == valid, result.stderr
            if valid:
                check = subprocess.run([solver, "check", "--scene", scene], capture_output=True, check=False)
                assert check.returncode == 0, check.stdout
                return scene.read_bytes(), state.read_bytes()
            assert not scene.exists() and not state.exists()

        legacy = "FLUIDITY-BLENDER-1\n0 0.125\n0 -9.81 0\n-0.5 0.05 0.1\n1\n0.0625 0.25 -0.0625\n"
        previous = resolve("previous", legacy, previous_bridge)
        assert resolve("legacy", legacy) == previous
        base = "FLUIDITY-BLENDER-2\n0 0.125\n0 -9.81 0\n1000 0 0\n0.6 0.01 256\n0 0 0\n"
        assert resolve("equivalent", base + "2\n1 0 0 -0.5 0.05 0.1 0 0\n1\n0.0625 0.25 -0.0625\n") == previous
        for name, collider in (
            ("free", "0"),
            ("plane", "1\n0 1 0 0 0 0 0 0"),
            ("slab", "2\n1 0 0 -0.5 0.05 0 0 0"),
            ("tank", "3\n-1 1 0 -1 1 0"),
        ):
            resolve(name, base + collider + "\n1\n0.0625 0.25 -0.0625\n")
        free = base + "0\n1\n0.0625 0.25 -0.0625\n"
        for index, (old, new) in enumerate(
            (
                ("1000 0 0", "-1 0 0"),
                ("1000 0 0", "1000 nan 0"),
                ("1000 0 0", "1000 0 -1"),
                ("0.6 0.01 256", "1.1 0.01 256"),
                ("0.6 0.01 256", "0.6 0 256"),
                ("0.6 0.01 256", "0.6 0.01 257"),
            )
        ):
            resolve("invalid" + str(index), free.replace(old, new), valid=False)
    print("PASS: checked native authoring, all collision kinds, malformed parameters, v1 byte compatibility")


if __name__ == "__main__":
    main()
