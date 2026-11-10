"""Manufactured MotionInterval sampling study, independent of the M1 particle cache."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


FRAME_SECONDS = 1 / 24
CELL_WIDTH_M = 0.025
MAX_ERROR_CELLS = 0.1
RK2_STEPS = 256
SAMPLING_INTERVALS = (1, 2, 4, 8, 16)


def time_factor(fraction):
    return 1 + 0.8 * np.sin(math.pi * fraction) + 0.2 * fraction


def rotation_rate(radius):
    return 45 * (0.5 + radius / 0.22)


def endpoint(initial, fixture, integrated_seconds):
    if fixture != "folding_sheet":
        return initial + np.array([4 * integrated_seconds, 0, 0])
    result = initial.copy()
    # The seeded sheet starts on the positive x axis, and radius remains constant.
    angle = rotation_rate(initial[:, 0]) * integrated_seconds
    result[:, 0] *= np.cos(angle)
    result[:, 1] = initial[:, 0] * np.sin(angle)
    return result


def base_velocity(position, fixture):
    if fixture != "folding_sheet":
        return np.broadcast_to([4, 0, 0], position.shape)
    rate = rotation_rate(np.linalg.norm(position[:, :2], axis=1))
    return np.column_stack((-position[:, 1] * rate, position[:, 0] * rate, np.zeros(len(position))))


def error_metrics(candidate, reference, prefix):
    error = np.linalg.norm(candidate - reference, axis=1)
    maximum = float(np.max(error))
    return {
        f"{prefix}_rms_m": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_max_m": maximum,
        f"{prefix}_max_cells": maximum / CELL_WIDTH_M,
    }


def study() -> dict:
    radius, height = np.meshgrid(np.linspace(0.025, 0.22, 41), [-0.01, 0.01], indexing="ij")
    initial = np.column_stack((radius.ravel(), np.zeros(radius.size), height.ravel()))
    exact_integral = 1.1 + 1.6 / math.pi
    samples = []
    for fixture in ("constant_translation", "translation", "folding_sheet"):
        exact = endpoint(initial, fixture, FRAME_SECONDS * (1 if fixture == "constant_translation" else exact_integral))
        for intervals in SAMPLING_INTERVALS:
            timestamps = np.linspace(0, 1, intervals + 1)
            factors = np.ones_like(timestamps) if fixture == "constant_translation" else time_factor(timestamps)
            for method in ("endpoint_hold", "linear"):
                if method == "endpoint_hold":
                    represented_integral = np.mean(factors[1:])
                else:
                    represented_integral = np.trapezoid(factors, timestamps)
                represented = endpoint(initial, fixture, FRAME_SECONDS * represented_integral)

                def sample(fraction):
                    if method == "endpoint_hold":
                        return factors[min(int(fraction * intervals) + 1, intervals)]
                    return np.interp(fraction, timestamps, factors)

                position = initial.copy()
                dt = FRAME_SECONDS / RK2_STEPS
                max_spatial_step = 0.0
                for step in range(RK2_STEPS):
                    velocity = base_velocity(position, fixture) * sample(step / RK2_STEPS)
                    midpoint = position + 0.5 * dt * velocity
                    displacement = dt * base_velocity(midpoint, fixture) * sample((step + 0.5) / RK2_STEPS)
                    max_spatial_step = max(max_spatial_step, float(np.linalg.norm(displacement, axis=1).max()))
                    position += displacement
                samples.append({
                    "fixture": fixture,
                    "method": method,
                    "intervals": intervals,
                    "bytes_per_velocity_node": 12 * len(timestamps),
                    "shared_timestamp_bytes": 8 * len(timestamps),
                    "max_rk2_displacement_cells": max_spatial_step / CELL_WIDTH_M,
                    **error_metrics(represented, exact, "temporal"),
                    **error_metrics(position, represented, "integration"),
                    **error_metrics(position, exact, "total"),
                })
    return {
        "frame_seconds": FRAME_SECONDS,
        "cell_width_m": CELL_WIDTH_M,
        "rk2_steps": RK2_STEPS,
        "budget_m": MAX_ERROR_CELLS * CELL_WIDTH_M,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(study(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report)


if __name__ == "__main__":
    main()
