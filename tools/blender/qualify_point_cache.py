# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = []
# ///

"""Qualify the relocated native Blender point-cache reader."""

import argparse
import hashlib
import json
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
    }
    processes.append(process)
    start = time.perf_counter()
    with log_path.open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=directory, env=env, check=False)
    process["seconds"] = time.perf_counter() - start
    process["exit_code"] = result.returncode
    if report_path.is_file():
        process["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return process


def passed_report(process):
    report = process.get("report", {})
    if process["exit_code"] != 0 or report.get("status") != "passed":
        raise RuntimeError(
            f"Blender qualification step failed with exit {process['exit_code']}. See {process['log_path']}"
        )
    return report


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
    passed_report(run_blender(blender, prepare_dir, "prepare", processes))
    shutil.copyfile(staged_archive, playback_dir / ARCHIVE_NAME)
    shutil.copyfile(prepare_dir / SCENE_NAME, playback_dir / SCENE_NAME)
    playback = passed_report(run_blender(blender, playback_dir, "verify", processes))

    shutil.copyfile(playback_dir / SCENE_NAME, missing_dir / SCENE_NAME)
    missing = run_blender(blender, missing_dir, "verify", processes)
    missing_report = missing.get("report", {})
    if (
        missing["exit_code"] != PYTHON_FAILURE
        or missing_report.get("status") != "failed"
        or missing_report.get("binding", {}).get("cache_present") is not False
        or len(missing_report.get("samples", [])) != len(playback["samples"])
    ):
        raise RuntimeError(f"Missing-archive control did not fail as expected. See {missing['log_path']}")


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
