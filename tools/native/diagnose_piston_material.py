"""Compare the piston material mesh with the frozen fixed-radius measurements."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from piston_material_mesh import BELOW_AREA, CELLS, H, HIGH_X, HIGH_Z, LOW, measure, snapshot, sweep

PISTON_SPEED = 0.015625
CLOCK = ("frame", "substep", "epoch", "accepted_step", "time_seconds")
RUN_ID = ("backend", "resolved_fingerprint", "execution_id")
WALL_STEP = ("accepted_dt_seconds", "step_start_piston_low_x_m", "step_end_piston_low_x_m")
OFFSETS_M = (0.000125, 0.00025, 0.0005, 0.001, 0.002)
CASES = {"open-tank-piston": 5, "open-tank-piston-refined": 9}
SATURATION_LIMIT = 0.0025
SEED_CENTERS = {index: tuple(low + (i + 0.5) * H for low, i in zip(LOW, index)) for index in CELLS}


def _require(condition, description):
    if not condition:
        raise ValueError(description)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _records(path):
    with path.open(encoding="utf-8-sig") as source:
        yield from map(json.loads, source)


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _fields(record, keys):
    return tuple(record[key] for key in keys)


def _capture_reader(capture, pin, official):
    manifest_path = capture / "capture-manifest.json"
    actual_pin = _sha256(manifest_path)
    _require(actual_pin == pin, "capture manifest differs from the trusted SHA-256")
    manifest = _json(manifest_path)
    inventory = dict(manifest.pop("files"))
    _require(official["capture_provenance"] == manifest | {"manifest_sha256": actual_pin},
             "capture identity differs from the frozen measurements")

    def read(name):
        path = capture / name
        _require(_sha256(path) == inventory.get(name), f"capture hash differs for {name}")
        return path

    return manifest, read


def _published_frames(files, official, tank, read_capture, steps_per_frame):
    frames = list(_records(files["frames"]))
    digests = official["verified_point_digests"]
    _require(len(frames) == len(digests) == official["verified_frame_count"] == 49,
             "expected 49 verified piston frames")
    for frame, (row, expected_digest) in enumerate(zip(frames, digests, strict=True)):
        expected_clock = (frame, 0, frame * steps_per_frame, frame * steps_per_frame, frame / 24)
        _require(row["event"] == "frame" and _fields(row, RUN_ID) == _fields(tank, RUN_ID)
                 and _fields(row, CLOCK) == expected_clock, f"verified frame {frame} identity differs")
        metadata = _json(read_capture(f"cache/frames/{frame:020}.json"))
        points = metadata["points"]
        _require(metadata["status"] == "complete" and _fields(metadata["clock"], CLOCK) == expected_clock
                 and _fields(metadata, RUN_ID[1:]) == _fields(tank, RUN_ID[1:])
                 and points["digest"] == row["points_digest"] == expected_digest
                 and points["relative_name"] == f"blobs/{expected_digest}.flpt"
                 and points["byte_length"] == row["points_byte_length"],
                 f"captured frame {frame} identity differs")
        blob = read_capture(f'cache/{points["relative_name"]}')
        _require(blob.stat().st_size == points["byte_length"], f"captured frame {frame} length differs")
        ratios = [point["volume_ratio"] for point in row["points"]]
        _require(all(math.isfinite(ratio) and ratio > 0 for ratio in ratios),
                 f"invalid material volume ratio at frame {frame}")
        point_material_volume_m3 = math.fsum(ratio * H**3 for ratio in ratios)
        recorded = official["volume_budget"]["frames"][frame]["point_material_volume_m3"]
        _require(abs(point_material_volume_m3 - recorded) <= 5e-16, f"frame {frame} particle volume differs")
        yield {
            "positions": snapshot(row["points"], "world_position"),
            "points_digest": expected_digest,
            "j_weighted_m3": point_material_volume_m3,
        }


def _accepted_steps(records, official_steps, frame_positions, steps_per_frame):
    traces = [row for row in records if row["event"] == "trace"]
    _require(len(traces) == len(official_steps) == 48 * steps_per_frame, "incomplete accepted piston steps")
    last_x = LOW[0]
    last_time = 0.0
    for step, (trace, official) in enumerate(zip(traces, official_steps, strict=True), 1):
        frame, substep = divmod(step, steps_per_frame)
        time = trace["time_seconds"]
        _require(_fields(trace, CLOCK) == (frame, substep, step, step, official["time_seconds"])
                 and official["accepted_step"] == step, f"accepted clock differs at step {step}")
        native = trace["accepted_measurement"]
        dt, x0, x1 = _fields(native, WALL_STEP)
        _require((dt, x0, x1) == _fields(official, WALL_STEP) and abs(x0 - last_x) <= 2e-15
                 and abs(x1 - LOW[0] - PISTON_SPEED * time) <= 2e-15
                 and abs(x1 - x0 - PISTON_SPEED * dt) <= 2e-15
                 and abs(time - last_time - dt) <= 2e-15, f"wall motion differs at step {step}")
        solver = native["step_start_inward_piston_flux_m3_per_s"] * dt
        _require(abs(solver - official["solver_inward_flux_m3"]) <= 2e-17,
                 f"signed solver flux differs at step {step}")
        positions = snapshot(trace["geometry"]["particles"], "world_position_m")
        if substep == 0:
            _require(positions == frame_positions[frame], f"trace and published positions differ at frame {frame}")
        yield x0, x1, positions, solver
        last_x, last_time = x1, time
    _require(last_time == 2, "accepted steps did not reach the two-second endpoint")


def _mesh_budget(frames, steps, fixed):
    x_final = steps[-1][1]
    sampled = []
    for frame, row in enumerate(frames):
        x = LOW[0] + PISTON_SPEED * frame / 24
        mesh = measure(row["positions"], x)
        _require(mesh["injectivity_certified"], f"uncertified published material mesh at frame {frame}")
        sampled.append({"frame": frame, "piston_x_m": x, "mesh": mesh,
                        **{key: value for key, value in row.items() if key != "positions"}})

    previous = frames[0]["positions"]
    sections = []
    for x0, x1, current, _ in steps:
        left = sweep(previous, x0, x1, offsets=OFFSETS_M)
        middle_positions = {index: tuple((a + b) / 2 for a, b in zip(previous[index], current[index]))
                            for index in CELLS}
        middle = sweep(middle_positions, (x0 + x1) / 2)
        sections.append((left, middle))
        previous = current
    last_section = sweep(previous, x_final)

    left_steps = [left for left, _ in sections]
    strip = {side: math.fsum(left[f"strip_{side}_m3"] for left in left_steps) for side in ("below", "above")}
    volumes = {method: {} for method in ("step_start", "midpoint", "trapezoid", "simpson")}
    for side in ("below", "above"):
        area_key = f"area_{side}_m2"
        parts = {method: [] for method in volumes}
        for (x0, x1, _, _), (left, middle), right in zip(
            steps, sections, left_steps[1:] + [last_section], strict=True
        ):
            dx = x1 - x0
            a, b, c = left[area_key], middle[area_key], right[area_key]
            parts["step_start"].append(dx * a)
            parts["midpoint"].append(dx * b)
            parts["trapezoid"].append(dx * (a + c) / 2)
            parts["simpson"].append(dx * (a + 4 * b + c) / 6)
        for method in volumes:
            volumes[method][side] = math.fsum(parts[method])
    for integral in (strip, *volumes.values()):
        integral["full"] = math.fsum(integral.values())

    offsets = {}
    for i, offset in enumerate(OFFSETS_M):
        offsets[offset] = {
            side: math.fsum((x1 - x0) * section["offset_area_below_above_m2"][i][j]
                            for (x0, x1, _, _), section in zip(steps, left_steps, strict=True))
            for j, side in enumerate(("below", "above"))
        }
    initial, final = sampled[0]["mesh"], sampled[-1]["mesh"]
    gain = final["tank_clipped_above_m3"] - initial["tank_clipped_above_m3"]
    total_change = math.fsum(final[f"tank_clipped_{side}_m3"] - initial[f"tank_clipped_{side}_m3"]
                             for side in ("below", "above"))
    initial_void = initial["tank_clipped_below_plane_void_m3"]
    final_void = final["tank_clipped_below_plane_void_m3"]
    nominal_below = (x_final - LOW[0]) * BELOW_AREA
    dry_below = nominal_below - strip["below"]
    saturation = math.fsum(map(abs, (initial_void, final_void, dry_below)))
    receiving = gain + strip["above"]
    solver = math.fsum(value for *_, value in steps)
    _require(abs(solver - fixed["solver_inward_flux_m3"]) <= 2e-15, "integrated solver flux differs")
    wall = fixed["full_wall_flux_m3"]
    return sampled, {
        "nominal_below_sweep_m3": nominal_below,
        "tank_clipped_above_gain_m3": gain,
        "tank_clipped_total_change_m3": total_change,
        "unclipped_total_change_m3": final["signed_total_m3"] - initial["signed_total_m3"],
        "initial_void_below_m3": initial_void,
        "final_void_below_m3": final_void,
        "frozen_strip_dry_below_m3": dry_below,
        "frozen_wetted_strip_below_above_full_m3": strip,
        "wall_integrals_below_above_full_m3": volumes,
        "step_start_offset_integrals_below_above_m3": offsets,
        "maximum_accepted_displacement_gradient_frobenius": max(
            section["max_displacement_gradient_frobenius"] for section in (*left_steps, last_section)
        ),
        "above_gain_plus_above_strip_m3": receiving,
        "above_gain_plus_strip_minus_wetted_strip_m3": receiving - strip["full"],
        "receiving_budget_identity_residual_m3": math.fsum(
            (receiving, -strip["full"], -total_change, -final_void, initial_void, -dry_below)
        ),
        "saturation_contribution_m3": saturation,
        "saturation_contribution_over_frozen_wall": saturation / wall,
        "saturation_contribution_over_mesh_step_start_wall": saturation / max(volumes["step_start"]["full"], 1e-15),
        "frozen_saturation_limit": SATURATION_LIMIT,
        "frozen_wall_relative_mismatch_of_above_gain_plus_strip": abs(receiving - wall) / wall,
        "frozen_receiving_difference_of_above_gain_plus_strip_over_frozen_wall": abs(receiving - fixed["receiving_m3"]) / wall,
        "native_signed_solver_integral_m3": solver,
    }


def _fixed_radius_receiving_budget(fixed):
    """Split the signed receiving discrepancy of a verified fixed-radius result."""
    initial = fixed["initial_reconstruction"]
    final = fixed["final_reconstruction"]
    total_volume_change = math.fsum(
        final[f"swept_{side}_m3"] - initial[f"swept_{side}_m3"] for side in ("below", "above")
    )
    void_change = fixed["final_void_below_reference_m3"] - fixed["initial_void_below_reference_m3"]
    dry_sweep = fixed["swept_dry_below_reference_m3"]
    receiving_difference = fixed["receiving_m3"] - fixed["full_wetted_sweep_m3"]
    # Below-plane capacity shrinks by the nominal sweep. Subtracting the dry
    # sweep gives the wet part. The remaining terms are inventory changes.
    return {
        "receiving_minus_wetted_sweep_m3": receiving_difference,
        "reconstructed_total_volume_change_m3": total_volume_change,
        "below_plane_void_change_m3": void_change,
        "swept_dry_below_plane_m3": dry_sweep,
        "identity_residual_m3": math.fsum((receiving_difference, -total_volume_change, -void_change, -dry_sweep)),
    }


def diagnose(run, manifest_sha256, measurements_sha256):
    """Compare a pinned capture and pinned frozen measurements without changing their gate."""
    case = run.name
    _require(case in CASES, "expected a primary or refined piston evaluation directory")
    steps_per_frame = CASES[case]
    files = {
        "seed": run / "verified-initial-state.stdout.jsonl",
        "tank": run / "verified-resolved-tank.stdout.jsonl",
        "frames": run / "verified-frames.stdout.jsonl",
        "measurements": run / f"{case}-measurements.json",
    }
    measurement_bytes = files["measurements"].read_bytes()
    measurement_digest = hashlib.sha256(measurement_bytes).hexdigest()
    _require(measurement_digest == measurements_sha256, "frozen measurements differ from the trusted SHA-256")
    official = json.loads(measurement_bytes)
    manifest, read_capture = _capture_reader(run / "capture", manifest_sha256, official)
    seed, = _records(files["seed"])
    tank, = _records(files["tank"])
    _require(manifest["case"] == official["case"] == case and tank == official["resolved_tank"]
             and _fields(tank, (*RUN_ID, "solver_digest", "seed_digest"))
             == _fields(manifest, (*RUN_ID, "solver_digest", "seed_digest")), "resolved tank identity differs")
    geometry = ("cell_size_m", "tank_low_x_m", "tank_high_x_m", "tank_floor_y_m", "tank_low_z_m",
                "tank_high_z_m", "piston_velocity_x_m_per_s", "source_count")
    _require(_fields(tank, geometry) == (H, LOW[0], HIGH_X, LOW[1], LOW[2], HIGH_Z, PISTON_SPEED, 0),
             "unexpected piston tank geometry")
    _require(seed["event"] == "seed" and seed["state_digest"] == tank["seed_digest"]
             and _fields(seed, CLOCK) == (0, 0, 0, 0, 0)
             and all(row["reference_volume"] == H**3 and row["volume_ratio"] == 1 for row in seed["particles"])
             and snapshot(seed["particles"], "world_position") == SEED_CENTERS, "seed lattice or identity differs")
    frames = list(_published_frames(files, official, tank, read_capture, steps_per_frame))
    _require(frames[0]["positions"] == SEED_CENTERS, "published frame zero differs from seed")
    trace_path = read_capture("simulate-run.stdout.jsonl")
    records = list(_records(trace_path))
    final = records[-1]
    _require(final == official["final_receipt"] and _fields(final, RUN_ID) == _fields(tank, RUN_ID)
             and _fields(final, CLOCK) == (48, 0, 48 * steps_per_frame, 48 * steps_per_frame, 2)
             and records[-2] == official["job_measurement"], "native final run or job receipt differs")
    fixed = official["tight_spatial"]
    steps = list(_accepted_steps(records, fixed["steps"], [row["positions"] for row in frames], steps_per_frame))
    sampled, budget = _mesh_budget(frames, steps, fixed)
    return {
        "case": case,
        "capture_identity": official["capture_provenance"],
        "source_sha256": {
            label: measurement_digest if label == "measurements" else _sha256(path)
            for label, path in (files | {"trace": trace_path}).items()
        },
        "accepted_step_count": len(steps),
        "official_gate_passed": official["passed"],
        "official_saturation_passed": official["saturation_passed"],
        "fixed_radius_above_endpoint_gain_m3": (fixed["final_reconstruction"]["swept_above_m3"]
                                                - fixed["initial_reconstruction"]["swept_above_m3"]),
        "fixed_radius": {key: value for key, value in fixed.items() if key != "steps"},
        "fixed_radius_receiving_budget": _fixed_radius_receiving_budget(fixed),
        "mesh_budget": budget,
        "frames": sampled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="verified primary or refined piston evaluation directory")
    parser.add_argument("--manifest-sha256", required=True, help="trusted SHA-256 of the capture manifest")
    parser.add_argument("--measurements-sha256", required=True, help="trusted SHA-256 of the frozen measurements")
    args = parser.parse_args()
    try:
        result = diagnose(args.run, args.manifest_sha256, args.measurements_sha256)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"piston material diagnostic: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
