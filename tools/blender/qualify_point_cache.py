# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = []
# ///

"""Qualify the relocated native Blender point-cache reader."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

PLAYBACK_SCRIPT_PATH = Path(__file__).resolve().with_name("point_cache_playback.py")
EVIDENCE_ROOT = PLAYBACK_SCRIPT_PATH.parents[2] / "out"
ARCHIVE_NAME = "FluidiTyPoints.abc"
SCENE_NAME = "FluidiTyPoints.blend"
BLENDER_PYTHON_FAILURE_EXIT_CODE = 23
BLENDER_TIMEOUT_SECONDS = 90
TIMELINE_SEEK_ORDER = (1, 2, 3, 4, 3, 1)
GEOMETRY_ABS_TOLERANCE = 1e-6


def run_blender_playback_process(blender_executable, process_directory, mode, process_receipts):
    scene = process_directory / SCENE_NAME
    report_path = process_directory / f"{mode}.json"
    log_path = process_directory / f"{mode}.log"
    command = [
        str(blender_executable),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
    ]
    if mode == "verify":
        command.append(str(scene))
    command.extend(
        (
            "--python-exit-code",
            str(BLENDER_PYTHON_FAILURE_EXIT_CODE),
            "--python",
            str(PLAYBACK_SCRIPT_PATH),
            "--",
            mode,
            "--report",
            str(report_path),
        )
    )
    if mode == "prepare":
        command.extend(("--scene", str(scene)))

    blender_environment = {
        name: value for name, value in os.environ.items() if not name.startswith(("BLENDER_USER_", "BLENDER_SYSTEM_"))
    }
    blender_environment["BLENDER_USER_RESOURCES"] = str(process_directory / "user-resources")
    process_receipt = {
        "mode": mode,
        "command": command,
        "user_resources": blender_environment["BLENDER_USER_RESOURCES"],
        "log_path": str(log_path),
        "report_path": str(report_path),
        "timeout_seconds": BLENDER_TIMEOUT_SECONDS,
    }
    process_receipts.append(process_receipt)
    process_start_seconds = time.perf_counter()
    try:
        with log_path.open("wb") as log:
            subprocess_result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=process_directory,
                env=blender_environment,
                check=False,
                timeout=BLENDER_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as error:
        process_receipt["timed_out"] = True
        process_receipt["exit_code"] = None
        raise RuntimeError(
            f"Blender {mode} timed out after {BLENDER_TIMEOUT_SECONDS} seconds. See {log_path}"
        ) from error
    finally:
        process_receipt["seconds"] = time.perf_counter() - process_start_seconds
    process_receipt["exit_code"] = subprocess_result.returncode
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError(f"Could not read Blender report {report_path}: {error}") from error
        if not isinstance(report, dict):
            raise ValueError(f"Expected a JSON object in Blender report {report_path}")
        process_receipt["report"] = report
    return process_receipt


def require_passed_blender_report(process_receipt):
    report = process_receipt.get("report", {})
    if process_receipt["exit_code"] != 0 or report.get("status") != "passed":
        raise RuntimeError(
            f"Blender qualification step failed with exit {process_receipt['exit_code']}. See {process_receipt['log_path']}"
        )


def has_required_point_fields(point):
    if not isinstance(point, dict):
        return False
    position = point.get("position")
    return (
        isinstance(position, list)
        and len(position) == 3
        and all(type(coordinate) in (int, float) for coordinate in position)
        and type(point.get("radius")) in (int, float)
    )


def read_reported_seek_samples(process_receipt):
    samples = process_receipt["report"].get("samples")
    report_path = process_receipt["report_path"]
    if not isinstance(samples, list) or any(not isinstance(sample, dict) for sample in samples):
        raise ValueError(f"Expected playback samples in Blender report {report_path}")
    frames = tuple(sample.get("frame") for sample in samples)
    if frames != TIMELINE_SEEK_ORDER:
        raise ValueError(
            f"Expected timeline frames {TIMELINE_SEEK_ORDER}, got {frames} in Blender report {report_path}"
        )
    for sample in samples:
        points = sample.get("points")
        count = sample.get("count")
        if (
            not isinstance(points, list)
            or type(count) is not int
            or count != len(points)
            or not all(has_required_point_fields(point) for point in points)
        ):
            raise ValueError(f"Invalid required geometry at frame {sample['frame']} in Blender report {report_path}")
    return samples


def required_point_geometry_matches(sample, reference):
    if sample["count"] != reference["count"]:
        return False
    for point, reference_point in zip(sample["points"], reference["points"], strict=True):
        if not math.isclose(point["radius"], reference_point["radius"], rel_tol=0.0, abs_tol=GEOMETRY_ABS_TOLERANCE):
            return False
        for coordinate, reference_coordinate in zip(point["position"], reference_point["position"], strict=True):
            if not math.isclose(coordinate, reference_coordinate, rel_tol=0.0, abs_tol=GEOMETRY_ABS_TOLERANCE):
                return False
    return True


def qualify_relocated_point_cache(blender_executable, archive, output, receipt):
    original_scene_directory = output / "original"
    relocated_scene_directory = output / "Relocated cache – échantillons"
    missing_archive_scene_directory = output / "Missing cache – échantillons"
    for process_directory in (original_scene_directory, relocated_scene_directory, missing_archive_scene_directory):
        process_directory.mkdir()
    staged_archive = Path(shutil.copyfile(archive, original_scene_directory / ARCHIVE_NAME))
    receipt["archive"] = {
        "source_path": str(archive),
        "sha256": hashlib.sha256(staged_archive.read_bytes()).hexdigest(),
    }

    process_receipts = receipt["processes"]
    require_passed_blender_report(
        run_blender_playback_process(blender_executable, original_scene_directory, "prepare", process_receipts)
    )
    shutil.copyfile(staged_archive, relocated_scene_directory / ARCHIVE_NAME)
    shutil.copyfile(original_scene_directory / SCENE_NAME, relocated_scene_directory / SCENE_NAME)
    relocated_playback = run_blender_playback_process(
        blender_executable, relocated_scene_directory, "verify", process_receipts
    )
    require_passed_blender_report(relocated_playback)
    relocated_samples = read_reported_seek_samples(relocated_playback)

    shutil.copyfile(relocated_scene_directory / SCENE_NAME, missing_archive_scene_directory / SCENE_NAME)
    missing_archive_playback = run_blender_playback_process(
        blender_executable, missing_archive_scene_directory, "verify", process_receipts
    )
    missing_report = missing_archive_playback.get("report", {})
    binding = missing_report.get("binding")
    if (
        missing_archive_playback["exit_code"] != BLENDER_PYTHON_FAILURE_EXIT_CODE
        or missing_report.get("status") != "failed"
        or not isinstance(binding, dict)
        or binding.get("cache_present") is not False
    ):
        raise RuntimeError(
            f"Missing-archive control did not fail as expected. See {missing_archive_playback['log_path']}"
        )
    missing_archive_samples = read_reported_seek_samples(missing_archive_playback)
    if all(
        required_point_geometry_matches(sample, reference)
        for sample, reference in zip(missing_archive_samples, relocated_samples, strict=True)
    ):
        raise RuntimeError(
            f"Missing-archive control reproduced the required playback geometry. See {missing_archive_playback['log_path']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path, help="pinned Blender executable")
    parser.add_argument("--archive", required=True, type=Path, help="archive produced by fluidity_write_point_cache")
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="new evidence directory inside the repository's out/"
    )
    args = parser.parse_args()
    blender_executable = args.blender.resolve()
    archive = args.archive.resolve()
    output = args.out_dir.resolve()
    if not blender_executable.is_file() or not archive.is_file():
        parser.error("--blender and --archive must name existing files")
    if not output.is_relative_to(EVIDENCE_ROOT.resolve()) or output == EVIDENCE_ROOT.resolve():
        parser.error("--out-dir must be inside the repository's out/")
    try:
        output.mkdir(parents=True)
    except OSError as error:
        parser.error(f"Could not create a new --out-dir: {error}")

    receipt_path = output / "receipt.json"
    receipt = {"status": "failed", "processes": []}
    try:
        qualify_relocated_point_cache(blender_executable, archive, output, receipt)
        receipt["status"] = "passed"
    except (OSError, RuntimeError, ValueError) as error:
        receipt["error"] = str(error)
        print(error)
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{receipt['status'].upper()}: {receipt_path}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
