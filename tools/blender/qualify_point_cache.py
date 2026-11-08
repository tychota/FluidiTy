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

BLENDER_SCRIPT = Path(__file__).resolve().with_name("point_cache_playback.py")
OUTPUT_ROOT = BLENDER_SCRIPT.parents[2] / "out"
ARCHIVE_NAME = "FluidiTyPoints.abc"
SCENE_NAME = "FluidiTyPoints.blend"
PYTHON_FAILURE = 23
BLENDER_TIMEOUT_SECONDS = 90
FRAME_ORDER = (1, 2, 3, 4, 3, 1)
GEOMETRY_TOLERANCE = 1e-6


def run_blender(blender, directory, mode, processes):
    scene = directory / SCENE_NAME
    report_path = directory / f"{mode}.json"
    log_path = directory / f"{mode}.log"
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
    ]
    if mode == "verify":
        command.append(str(scene))
    command.extend(
        (
            "--python-exit-code",
            str(PYTHON_FAILURE),
            "--python",
            str(BLENDER_SCRIPT),
            "--",
            mode,
            "--report",
            str(report_path),
        )
    )
    if mode == "prepare":
        command.extend(("--scene", str(scene)))

    env = {
        name: value for name, value in os.environ.items() if not name.startswith(("BLENDER_USER_", "BLENDER_SYSTEM_"))
    }
    env["BLENDER_USER_RESOURCES"] = str(directory / "user-resources")
    process = {
        "mode": mode,
        "command": command,
        "user_resources": env["BLENDER_USER_RESOURCES"],
        "log_path": str(log_path),
        "report_path": str(report_path),
        "timeout_seconds": BLENDER_TIMEOUT_SECONDS,
    }
    processes.append(process)
    start = time.perf_counter()
    try:
        with log_path.open("wb") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=directory,
                env=env,
                check=False,
                timeout=BLENDER_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as error:
        process["timed_out"] = True
        process["exit_code"] = None
        raise RuntimeError(
            f"Blender {mode} timed out after {BLENDER_TIMEOUT_SECONDS} seconds. See {log_path}"
        ) from error
    finally:
        process["seconds"] = time.perf_counter() - start
    process["exit_code"] = result.returncode
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError(f"Could not read Blender report {report_path}: {error}") from error
        if not isinstance(report, dict):
            raise ValueError(f"Expected a JSON object in Blender report {report_path}")
        process["report"] = report
    return process


def require_passed(process):
    report = process.get("report", {})
    if process["exit_code"] != 0 or report.get("status") != "passed":
        raise RuntimeError(
            f"Blender qualification step failed with exit {process['exit_code']}. See {process['log_path']}"
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


def playback_samples(process):
    samples = process["report"].get("samples")
    report_path = process["report_path"]
    if not isinstance(samples, list) or any(not isinstance(sample, dict) for sample in samples):
        raise ValueError(f"Expected playback samples in Blender report {report_path}")
    frames = tuple(sample.get("frame") for sample in samples)
    if frames != FRAME_ORDER:
        raise ValueError(f"Expected timeline frames {FRAME_ORDER}, got {frames} in Blender report {report_path}")
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


def required_geometry_matches(sample, reference):
    if sample["count"] != reference["count"]:
        return False
    for point, reference_point in zip(sample["points"], reference["points"], strict=True):
        if not math.isclose(point["radius"], reference_point["radius"], rel_tol=0.0, abs_tol=GEOMETRY_TOLERANCE):
            return False
        for coordinate, reference_coordinate in zip(point["position"], reference_point["position"], strict=True):
            if not math.isclose(coordinate, reference_coordinate, rel_tol=0.0, abs_tol=GEOMETRY_TOLERANCE):
                return False
    return True


def qualify(blender, archive, output, receipt):
    prepare_dir = output / "original"
    playback_dir = output / "Relocated cache – échantillons"
    missing_dir = output / "Missing cache – échantillons"
    for directory in (prepare_dir, playback_dir, missing_dir):
        directory.mkdir()
    staged_archive = Path(shutil.copyfile(archive, prepare_dir / ARCHIVE_NAME))
    receipt["archive"] = {
        "source_path": str(archive),
        "sha256": hashlib.sha256(staged_archive.read_bytes()).hexdigest(),
    }

    processes = receipt["processes"]
    require_passed(run_blender(blender, prepare_dir, "prepare", processes))
    shutil.copyfile(staged_archive, playback_dir / ARCHIVE_NAME)
    shutil.copyfile(prepare_dir / SCENE_NAME, playback_dir / SCENE_NAME)
    playback = run_blender(blender, playback_dir, "verify", processes)
    require_passed(playback)
    reference_samples = playback_samples(playback)

    shutil.copyfile(playback_dir / SCENE_NAME, missing_dir / SCENE_NAME)
    missing = run_blender(blender, missing_dir, "verify", processes)
    missing_report = missing.get("report", {})
    binding = missing_report.get("binding")
    if (
        missing["exit_code"] != PYTHON_FAILURE
        or missing_report.get("status") != "failed"
        or not isinstance(binding, dict)
        or binding.get("cache_present") is not False
    ):
        raise RuntimeError(f"Missing-archive control did not fail as expected. See {missing['log_path']}")
    missing_samples = playback_samples(missing)
    if all(
        required_geometry_matches(sample, reference)
        for sample, reference in zip(missing_samples, reference_samples, strict=True)
    ):
        raise RuntimeError(
            f"Missing-archive control reproduced the required playback geometry. See {missing['log_path']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path, help="pinned Blender executable")
    parser.add_argument("--archive", required=True, type=Path, help="archive produced by fluidity_write_point_cache")
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="new evidence directory inside the repository's out/"
    )
    args = parser.parse_args()
    blender = args.blender.resolve()
    archive = args.archive.resolve()
    output = args.out_dir.resolve()
    if not blender.is_file() or not archive.is_file():
        parser.error("--blender and --archive must name existing files")
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        parser.error("--out-dir must be inside the repository's out/")
    try:
        output.mkdir(parents=True)
    except OSError as error:
        parser.error(f"Could not create a new --out-dir: {error}")

    receipt_path = output / "receipt.json"
    receipt = {"status": "failed", "processes": []}
    try:
        qualify(blender, archive, output, receipt)
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
