"""Cross-check native fixed-cache reconstruction against the independent Python baseline."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess

from tools.native.m2_surface_baseline import (
    DROP_AXES_M, FINGERPRINT, RUN_ARTIFACTS, SELECTED_FRAMES, SPACINGS_M,
    mesh_metrics, reconstruct, require_sha256,
)


def parse_measurement(stdout):
    prefix = "native_surface_measurement "
    records = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(records) != 1:
        raise ValueError("expected exactly one native measurement; skipped tests are not evidence")
    return json.loads(records[0])


def validate_measurement(record, frame, spacing, digest, oracle):
    if (record["frame"] != frame or record["spacing_m"] != spacing
            or record["source_points_digest"] != digest):
        raise ValueError("native measurement names different input content")
    for value in record.values():
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError("nonfinite native measurement")
    material = record["material_volume_m3"]
    if material <= 0 or not math.isclose(material, oracle["reference_volume_m3"], rel_tol=1e-12, abs_tol=0):
        raise ValueError("material volume differs from the independent oracle")
    for name, volume in (("raw", record["raw_volume_m3"]), ("corrected", record["corrected_volume_m3"])):
        if not math.isclose(record[f"{name}_volume_error"], volume / material - 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("reported volume error differs from measured geometry")
    if abs(record["corrected_volume_error"]) > 0.005:
        raise ValueError("corrected volume fails the 0.5% gate")
    if record["vertices"] != oracle["vertices"] or record["triangles"] != oracle["triangles"]:
        raise ValueError("native extraction differs from independent topology counts")
    if abs(record["raw_volume_m3"] - oracle["signed_volume_m3"]) > 1e-14:
        raise ValueError("native raw volume differs from the independent geometry")
    if record["components"] != 1 or record["euler"] != 2 or record["self_intersections"] != 0:
        raise ValueError("fixed drop fails topology/intersection controls")
    if not 0 <= record["maximum_offset_m"] <= record["offset_limit_m"] or record["offset_limit_m"] <= 0:
        raise ValueError("native correction exceeds its total displacement limit")
    if record["runtime_seconds"] <= 0 or record["peak_process_rss_bytes"] <= 0:
        raise ValueError("missing native resource measurement")
    if frame == 0:
        for statistic in ("rms", "max"):
            if abs(record[f"raw_radial_{statistic}_m"] - oracle[f"vertex_radial_error_{statistic}_m"]) > 1e-14:
                raise ValueError("native radial discrepancy differs from independent analytic control")


def benchmark(executable, run, logs):
    executable = executable.resolve(strict=True)
    executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    run = run.resolve(strict=True)
    require_sha256(run / "frames.jsonl", RUN_ARTIFACTS["frames.jsonl"])
    frames = {frame["frame"]: frame for frame in map(json.loads, (run / "frames.jsonl").read_text().splitlines())}
    logs.mkdir(parents=True, exist_ok=False)
    samples, failures = [], []
    for spacing in SPACINGS_M:
        for ordinal in SELECTED_FRAMES:
            environment = dict(os.environ, FLUIDITY_FIXED_CACHE_ROOT=str(run / "cache"),
                               FLUIDITY_FIXED_CACHE_FRAME=str(ordinal), FLUIDITY_FIXED_CACHE_SPACING=str(spacing))
            result = subprocess.run([str(executable), "--gtest_filter=SurfaceFixedCache3D.*"], env=environment,
                                    capture_output=True, timeout=180, encoding="utf-8", errors="replace", check=False)
            log = logs / f"frame-{ordinal:04d}-spacing-{spacing}.log"
            log.write_text(result.stdout + result.stderr, encoding="utf-8")
            require_sha256(executable, executable_digest)
            try:
                if result.returncode:
                    raise ValueError(f"native process failed with exit code {result.returncode}")
                record = parse_measurement(result.stdout)
                frame = frames[ordinal]
                mesh, volume, _ = reconstruct(frame["points"], spacing, "density")
                oracle = mesh_metrics(mesh, volume, DROP_AXES_M if ordinal == 0 else None)
                validate_measurement(record, ordinal, spacing, frame["points_digest"], oracle)
                record["python_raw_volume_m3"] = oracle["signed_volume_m3"]
                samples.append(record)
                print(f"frame={ordinal} spacing={spacing}: native/Python controls passed", flush=True)
            except (ValueError, KeyError, TypeError) as error:
                failures.append({"frame": ordinal, "spacing_m": spacing, "cause": str(error)})
                print(f"frame={ordinal} spacing={spacing}: {error}", flush=True)
    return {
        "executable_sha256": executable_digest,
        "source_fingerprint": FINGERPRINT,
        "source_archive_sha256": RUN_ARTIFACTS["frames.jsonl"],
        "platform": platform.platform(),
        "samples": samples,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve the output before expensive work; never replace an earlier receipt.
    with args.output.open("x", encoding="utf-8") as output:
        report = benchmark(args.executable, args.run, args.logs)
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    raise SystemExit(bool(report["failures"]))


if __name__ == "__main__":
    main()
