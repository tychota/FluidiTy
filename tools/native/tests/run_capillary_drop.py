"""Run the frozen l=2 capillary gate on a public solver and verify committed frames."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


LAST_FRAME = 216
FRAME_RATE = 24
RADIUS_M = 0.125
DENSITY = 1000
TENSION = 1
POINT_COUNT = 552


def execute(output, name, command):
    with (output / f"{name}.jsonl").open("wb") as stdout, (output / f"{name}.stderr").open("wb") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)


def quadrupole(points):
    # The frozen seed uses identical V0 for every particle, so J gives the
    # relative current-volume weight. Center each frame before measuring shape.
    weights = [point["volume_ratio"] for point in points]
    positions = [point["world_position"] for point in points]
    total = math.fsum(weights)
    center = [
        math.fsum(weight * position[axis] for position, weight in zip(positions, weights, strict=True)) / total
        for axis in range(3)
    ]
    moments = [
        math.fsum(weight * (position[axis] - center[axis]) ** 2
                  for position, weight in zip(positions, weights, strict=True)) / total
        for axis in range(3)
    ]
    return moments[1] - 0.5 * (moments[0] + moments[2]), total


def measure(samples):
    extrema = {1: [], -1: []}
    for index in range(1, len(samples) - 1):
        left, middle, right = samples[index - 1:index + 2]
        sign = 1 if middle > 0 else -1
        if sign * (middle - left) <= 0 or sign * (middle - right) <= 0:
            continue
        offset = 0.5 * (left - right) / (left - 2 * middle + right)
        time = (index + offset) / FRAME_RATE
        amplitude = sign * (middle - 0.25 * (left - right) * offset)
        if 0.25 <= time <= 9 and math.isfinite(amplitude) and amplitude > 0:
            extrema[sign].append((time, amplitude))

    periods = []
    decrements = []
    for peaks in extrema.values():
        for (first_time, first_amplitude), (last_time, last_amplitude) in zip(peaks, peaks[1:]):
            periods.append(last_time - first_time)
            decrements.append(math.log(first_amplitude / last_amplitude))
    if len(periods) < 2:
        raise ValueError("drop has fewer than two complete same-sign periods")

    period = statistics.median(periods)
    decrement = statistics.median(decrements)
    damping = decrement / math.hypot(2 * math.pi, decrement)
    omega = 2 * math.pi / period
    reference = math.sqrt(8 * TENSION / (DENSITY * RADIUS_M**3))
    return {
        "omega_rad_per_s": omega,
        "reference_rad_per_s": reference,
        "relative_error": abs(omega / reference - 1),
        "damping_ratio": damping,
        "damping_corrected_rad_per_s": omega / math.sqrt(1 - damping**2),
        "period_residual_rms_seconds": math.sqrt(statistics.fmean((value - period) ** 2 for value in periods)),
        "same_sign_periods": periods,
        "extrema": extrema,
    }


def verify(output):
    scene = json.loads((output / "scene.jsonl").read_text())
    samples, volumes = [], []
    ids = set(range(1, POINT_COUNT + 1))
    with (output / "frames.jsonl").open() as source:
        for frame, line in enumerate(source):
            record = json.loads(line)
            if (record["event"], record["frame"], record["backend"], record["execution_id"]) != (
                "frame", frame, scene["backend"], scene["execution_id"]
            ):
                raise ValueError(f"unexpected frame identity at ordinal {frame}")
            if record["time_seconds"] != frame / FRAME_RATE:
                raise ValueError(f"unexpected time at frame {frame}")
            points = record["points"]
            if len(points) != POINT_COUNT or {point["id"] for point in points} != ids:
                raise ValueError(f"drop changed particle ownership at frame {frame}")
            if any(not math.isfinite(point["volume_ratio"]) or point["volume_ratio"] <= 0
                   or not all(map(math.isfinite, point["world_position"])) for point in points):
                raise ValueError(f"invalid drop inventory at frame {frame}")
            shape, volume = quadrupole(points)
            samples.append(shape)
            volumes.append(volume)
    if len(samples) != LAST_FRAME + 1:
        raise ValueError("drop capture does not cover frames 0 through 216")
    steps = 0
    bounded_steps = 0
    max_capillary_ratio = 0.0
    with (output / "solver.jsonl").open() as source:
        for line in source:
            record = json.loads(line)
            if record["event"] != "trace":
                continue
            accepted = record["accepted_measurement"]
            dt = accepted["accepted_dt_seconds"]
            bound = accepted["step_start_capillary_bound_seconds"]
            if not math.isfinite(dt) or not 0 < dt <= 0.01:
                raise ValueError("drop accepted an invalid timestep")
            steps += 1
            if bound is not None:
                if not math.isfinite(bound) or dt > bound:
                    raise ValueError("drop exceeded its accepted capillary bound")
                max_capillary_ratio = max(max_capillary_ratio, dt / bound)
                bounded_steps += 1
    if steps < LAST_FRAME or bounded_steps == 0:
        raise ValueError("drop trace does not certify accepted capillary steps")
    result = measure(samples)
    result["accepted_steps"] = steps
    result["capillary_bounded_steps"] = bounded_steps
    result["maximum_capillary_timestep_ratio"] = max_capillary_ratio
    result["peak_relative_volume_drift"] = max(abs(volume / volumes[0] - 1) for volume in volumes)
    result["backend"] = scene["backend"]
    result["execution_id"] = scene["execution_id"]
    result["passed"] = (
        result["relative_error"] <= 0.05 and 0 <= result["damping_ratio"] <= 0.20
        and result["peak_relative_volume_drift"] <= 0.01
    )
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not result["passed"]:
        raise ValueError("drop exceeded the frozen frequency, damping, or volume gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    solver, fixture = args.solver.resolve(), args.fixture.resolve()
    provenance = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                  for key, path in (("solver", solver), ("fixture", fixture))}
    (output / "binaries.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    scene, cache = output / "scene.flrs", output / "cache"
    execute(output, "scene", [fixture, "write-scene", "oscillating-drop-l2", args.backend, solver, scene])
    identity = json.loads((output / "scene.jsonl").read_text())
    execute(output, "solver", [
        solver, "run", "--stage", "simulate", "--scene", scene, "--initial-state", str(scene) + ".flst",
        "--cache", cache, "--backend", args.backend, "--frame-end", str(LAST_FRAME), "--trace",
    ])
    execute(output, "frames", [
        fixture, "read-frames", cache, identity["resolved_fingerprint"], "0", str(LAST_FRAME),
    ])
    verify(output)


if __name__ == "__main__":
    main()
