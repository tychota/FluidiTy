"""Verify the native bridge used by the Blender rigid-box frame workflow."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(solver, fixture):
    with tempfile.TemporaryDirectory(prefix="fluidity-rigid-") as directory:
        root = Path(directory)
        scene = root / "tank.flrs"
        initial = root / "tank.flrs.flst"
        packet = root / "rigid.txt"
        subprocess.run([fixture, "scene", scene, solver, "cpu", "rigid-tank"], check=True)
        packet.write_text(
            "FLUIDITY-RIGID-1 4 2 1 1 1 0.125 0.125 0.125 0.5 0.6875 0.5 1 0 0 0 0 0.125 0 0 0 0.0625",
            encoding="ascii",
        )

        def frame(previous, output, check=True):
            return subprocess.run(
                [solver, "rigid-frame", scene, initial, packet, previous, output],
                check=check,
                capture_output=True,
                text=True,
            )

        first = root / "first.chk"
        second = root / "second.chk"
        replay = root / "replayed.chk"
        first_record = json.loads(frame("-", first).stdout)
        second_record = json.loads(frame(first, second).stdout)
        assert first_record["event"] == "rigid_frame" and first_record["frame"] == 1
        assert second_record["frame"] == 2 and second_record["orientation"][3] > first_record["orientation"][3]
        assert json.loads(frame(first, replay).stdout) == second_record
        assert second.read_bytes() == replay.read_bytes()
        corrupt = root / "corrupt.chk"
        corrupt.write_bytes(first.read_bytes() + b"\x00")
        refused = root / "refused.chk"
        result = frame(corrupt, refused, check=False)
        assert result.returncode != 0 and not refused.exists()
        assert "checkpoint payload digest" in result.stdout
        packet.write_text(packet.read_text(encoding="ascii") + " ", encoding="ascii")
        refused = root / "refused.chk"
        result = frame(first, refused, check=False)
        assert result.returncode != 0 and not refused.exists()
        assert json.loads(result.stdout)["name"] == "FAILED_PRECONDITION"


if __name__ == "__main__":
    run(*(Path(path).resolve() for path in sys.argv[1:]))
    print("PASS rigid_frame")
