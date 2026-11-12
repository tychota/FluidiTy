"""Process acceptance for fixed-domain gas run, resume, inspection and replay."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(executable, *args, expected=0):
    process = subprocess.run([executable, *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    if (process.returncode == 0) != (expected == 0):
        raise AssertionError((process.args, process.returncode, process.stdout, process.stderr))
    return [json.loads(line) for line in process.stdout.splitlines()]


def main():
    executable, flatc, schema, *options = sys.argv[1:]
    backend = options[0] if options else "cpu"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        identity = run(executable, "schema")[-1]
        digest = list(bytes.fromhex(identity["build_digest"]))
        zero = dict(x=0.0, y=0.0, z=0.0)
        packet = dict(
            backend=backend.capitalize(),
            build_digest=dict(bytes=digest),
            world_origin_m=zero,
            extent=dict(x=2, y=2, z=2),
            open_boundaries=3,
            initial_velocity_m_per_s=zero,
            gravity_m_per_s2=zero,
            substeps_per_frame=2,
            emitters=[dict(
                lower_m=zero,
                upper_m=dict(x=1.0, y=1.0, z=1.0),
                keys=[
                    dict(time_s=0.0, fuel_kg_per_m3_s=0.0, heat_w_per_m3=0.0),
                    dict(time_s=0.04, fuel_kg_per_m3_s=0.1, heat_w_per_m3=1500.0),
                ],
            )],
            draft=[dict(time_s=0.0, acceleration_m_per_s2=dict(x=0.1, y=0.0, z=0.0))],
            reaction={},
        )
        source = root / "gas.json"
        source.write_text(json.dumps(packet), encoding="utf-8")
        subprocess.run([flatc, "-b", "-o", str(root), schema, str(source)], check=True)
        scene = root / "gas.flgs"
        continuous = root / "continuous"
        resumed = root / "resumed"
        checked = run(executable, "validate", scene)[-1]
        assert checked["model"] == "gas"
        assert run(executable, "simulate", scene, "--cache", continuous, "--frame-end", 2)[-1]["frame"] == 2
        run(executable, "simulate", scene, "--cache", resumed, "--frame-end", 1)
        run(executable, "simulate", scene, "--cache", resumed, "--frame-end", 2, "--resume")
        expected_frames = sorted((continuous / "checkpoints").glob("*.flgc"))
        actual_frames = sorted((resumed / "checkpoints").glob("*.flgc"))
        assert len(expected_frames) == len(actual_frames) == 3
        assert [item.read_bytes() for item in actual_frames] == [item.read_bytes() for item in expected_frames]
        before = {path.name: path.stat().st_mtime_ns for path in actual_frames}
        assert run(executable, "inspect", resumed, "--frame", 2)[-1]["cells"] == 8
        assert run(executable, "replay", resumed, "--frame", 2, "--substep", 0)[-1]["frame"] == 2
        assert {path.name: path.stat().st_mtime_ns for path in actual_frames} == before
        damaged = bytearray(actual_frames[-1].read_bytes())
        damaged[-1] ^= 1
        actual_frames[-1].write_bytes(damaged)
        run(executable, "replay", resumed, "--frame", 2, "--substep", 0, expected=1)
    print(f"Gas {backend} public run, resume, inspect, and replay passed")


if __name__ == "__main__":
    main()
