"""Exercise launcher failures with controlled Blender process results."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "qualify_point_cache", Path(__file__).parents[1] / "blender" / "qualify_point_cache.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def qualification_output(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    archive = tmp_path / "FluidiTyPoints.abc"
    output = tmp_path / "evidence"
    blender.touch()
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(launcher, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["qualify", "--blender", str(blender), "--archive", str(archive), "--out-dir", str(output)]
    )
    return output


def playback_samples():
    return [
        {
            "frame": frame,
            "count": count,
            "points": [{"position": [frame, i, -(frame + i)], "radius": (i + 1) / 16} for i in range(count)],
            "velocity": {"matches_fixture": True},
        }
        for frame, count in [(1, 3), (2, 5), (3, 0), (4, 4), (3, 0), (1, 3)]
    ]


def simulate_process_reports(monkeypatch, playback, missing):
    def run(command, **kwargs):
        report_path = Path(command[command.index("--report") + 1])
        scene_path = report_path.with_name("FluidiTyPoints.blend")
        if "--scene" in command:
            scene_path.touch()
            report, exit_code = {"status": "passed"}, 0
        elif report_path.with_name("FluidiTyPoints.abc").is_file():
            report, exit_code = playback, 0
        else:
            report, exit_code = missing, 23
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(launcher.subprocess, "run", run)


def read_receipt(output):
    return json.loads((output / "receipt.json").read_text(encoding="utf-8"))


def test_timeout_leaves_failed_receipt_with_raw_log(qualification_output, monkeypatch):
    def timed_out(command, **kwargs):
        kwargs["stdout"].write(b"Blender stalled here\n")
        assert kwargs["timeout"] == 90
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(launcher.subprocess, "run", timed_out)
    assert launcher.main() == 1
    receipt = read_receipt(qualification_output)
    assert receipt["status"] == "failed"
    (process,) = receipt["processes"]
    assert process["timeout_seconds"] == 90
    assert process["timed_out"] is True
    assert process["seconds"] >= 0
    log = Path(process["log_path"])
    assert str(log) in receipt["error"]
    assert log.read_bytes() == b"Blender stalled here\n"


@pytest.mark.parametrize("change", ["seek_order", "velocity_only", "within_tolerance", "count", "position", "radius"])
def test_missing_control_requires_correct_seek_and_required_geometry_mismatch(
    qualification_output, monkeypatch, change
):
    playback = playback_samples()
    missing = copy.deepcopy(playback)
    if change == "seek_order":
        missing[0], missing[1] = missing[1], missing[0]
        missing[0]["points"][0]["radius"] = 1.0
    elif change == "velocity_only":
        missing[0]["velocity"]["matches_fixture"] = False
    elif change == "within_tolerance":
        missing[0]["points"][0]["radius"] += 1e-7
    elif change == "count":
        missing[0]["points"].pop()
        missing[0]["count"] -= 1
    elif change == "position":
        missing[0]["points"][0]["position"][0] = 100.0
    elif change == "radius":
        missing[0]["points"][0]["radius"] = 1.0
    simulate_process_reports(
        monkeypatch,
        {"status": "passed", "samples": playback},
        {"status": "failed", "binding": {"cache_present": False}, "samples": missing},
    )

    if change in {"count", "position", "radius"}:
        assert launcher.main() == 0
        assert read_receipt(qualification_output)["status"] == "passed"
    else:
        assert launcher.main() == 1
        assert read_receipt(qualification_output)["status"] == "failed"


@pytest.mark.parametrize("invalid_playback", [None, [], {"status": "passed"}], ids=["null", "list", "missing-samples"])
def test_invalid_playback_report_names_report_path(qualification_output, monkeypatch, invalid_playback):
    simulate_process_reports(monkeypatch, invalid_playback, {"status": "failed"})
    assert launcher.main() == 1
    receipt = read_receipt(qualification_output)
    assert receipt["status"] == "failed"
    assert receipt["processes"][1]["report_path"] in receipt["error"]
