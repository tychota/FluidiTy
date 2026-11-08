"""Exercise launcher failures with controlled Blender process results."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER_MODULE_SPEC = importlib.util.spec_from_file_location(
    "qualify_point_cache", Path(__file__).parents[1] / "blender" / "qualify_point_cache.py"
)
launcher = importlib.util.module_from_spec(LAUNCHER_MODULE_SPEC)
LAUNCHER_MODULE_SPEC.loader.exec_module(launcher)


@pytest.fixture
def qualification_evidence_directory(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    archive = tmp_path / "FluidiTyPoints.abc"
    output = tmp_path / "evidence"
    blender.touch()
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(launcher, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["qualify", "--blender", str(blender), "--archive", str(archive), "--out-dir", str(output)]
    )
    return output


def expected_timeline_samples():
    return [
        {
            "frame": frame,
            "count": count,
            "points": [
                {
                    "position": [frame, sample_point_index, -(frame + sample_point_index)],
                    "radius": (sample_point_index + 1) / 16,
                }
                for sample_point_index in range(count)
            ],
            "velocity": {"matches_fixture": True},
        }
        for frame, count in [(1, 3), (2, 5), (3, 0), (4, 4), (3, 0), (1, 3)]
    ]


def mock_blender_process_reports(monkeypatch, relocated_playback_report, missing_archive_report):
    def run(command, **kwargs):
        report_path = Path(command[command.index("--report") + 1])
        scene_path = report_path.with_name("FluidiTyPoints.blend")
        if "--scene" in command:
            scene_path.touch()
            report, exit_code = {"status": "passed"}, 0
        elif report_path.with_name("FluidiTyPoints.abc").is_file():
            report, exit_code = relocated_playback_report, 0
        else:
            report, exit_code = missing_archive_report, 23
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(launcher.subprocess, "run", run)


def read_qualification_receipt(evidence_directory):
    return json.loads((evidence_directory / "receipt.json").read_text(encoding="utf-8"))


def test_timeout_leaves_failed_receipt_with_raw_log(qualification_evidence_directory, monkeypatch):
    def timed_out(command, **kwargs):
        kwargs["stdout"].write(b"Blender stalled here\n")
        assert kwargs["timeout"] == 90
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(launcher.subprocess, "run", timed_out)
    assert launcher.main() == 1
    receipt = read_qualification_receipt(qualification_evidence_directory)
    assert receipt["status"] == "failed"
    (process,) = receipt["processes"]
    assert process["timeout_seconds"] == 90
    assert process["timed_out"] is True
    assert process["seconds"] >= 0
    log = Path(process["log_path"])
    assert str(log) in receipt["error"]
    assert log.read_bytes() == b"Blender stalled here\n"


@pytest.mark.parametrize(
    "missing_cache_case", ["seek_order", "velocity_only", "within_tolerance", "count", "position", "radius"]
)
def test_missing_control_requires_correct_seek_and_required_geometry_mismatch(
    qualification_evidence_directory, monkeypatch, missing_cache_case
):
    relocated_samples = expected_timeline_samples()
    missing_archive_samples = copy.deepcopy(relocated_samples)
    if missing_cache_case == "seek_order":
        missing_archive_samples[0], missing_archive_samples[1] = missing_archive_samples[1], missing_archive_samples[0]
        missing_archive_samples[0]["points"][0]["radius"] = 1.0
    elif missing_cache_case == "velocity_only":
        missing_archive_samples[0]["velocity"]["matches_fixture"] = False
    elif missing_cache_case == "within_tolerance":
        missing_archive_samples[0]["points"][0]["radius"] += 1e-7
    elif missing_cache_case == "count":
        missing_archive_samples[0]["points"].pop()
        missing_archive_samples[0]["count"] -= 1
    elif missing_cache_case == "position":
        missing_archive_samples[0]["points"][0]["position"][0] = 100.0
    elif missing_cache_case == "radius":
        missing_archive_samples[0]["points"][0]["radius"] = 1.0
    mock_blender_process_reports(
        monkeypatch,
        {"status": "passed", "samples": relocated_samples},
        {"status": "failed", "binding": {"cache_present": False}, "samples": missing_archive_samples},
    )

    if missing_cache_case in {"count", "position", "radius"}:
        assert launcher.main() == 0
        assert read_qualification_receipt(qualification_evidence_directory)["status"] == "passed"
    else:
        assert launcher.main() == 1
        assert read_qualification_receipt(qualification_evidence_directory)["status"] == "failed"


@pytest.mark.parametrize("invalid_playback", [None, [], {"status": "passed"}], ids=["null", "list", "missing-samples"])
def test_invalid_playback_report_names_report_path(qualification_evidence_directory, monkeypatch, invalid_playback):
    mock_blender_process_reports(monkeypatch, invalid_playback, {"status": "failed"})
    assert launcher.main() == 1
    receipt = read_qualification_receipt(qualification_evidence_directory)
    assert receipt["status"] == "failed"
    assert receipt["processes"][1]["report_path"] in receipt["error"]
