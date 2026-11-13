"""Record reference-wave admission and run the widest authored M2 control."""

import argparse
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time

from qualify_matched_runtime import cache_contents, digest, json_lines, run


REFERENCE = {
    "domain_width_length_height_m": [8, 20, 8],
    "inlet_depth_m": 3,
    "wave_height_m": 1,
    "wave_period_s": 4,
    "near_camera_cell_width_m": 0.035,
    "frames": 240,
    "fps": 24,
    "camera_focal_length_mm": 35,
    "camera_output_width_px": 1920,
    "approximate_camera_distance_from_breaking_m": 2,
    "required_output": ["simulate", "surface", "whitewater", "volumes", "authored_multires"],
    "maximum_seconds": 10800,
    "uniform_fallback_maximum_seconds": 21600,
}
CAPACITIES = {
    "public_uniform": {"particles": 1024, "blocks": 128, "cells": 4096, "faces": 24576},
    "graded_topology": {"leaves": 4096, "faces": 24576},
}
CONTROL_FRAME_END = 48


def geometry_admission():
    width, length, height = REFERENCE["domain_width_length_height_m"]
    inlet_depth = REFERENCE["inlet_depth_m"]
    h = REFERENCE["near_camera_cell_width_m"]
    length_columns = math.ceil(length / h)
    width_columns = math.ceil(width / h)
    # Count a uniformly fine diagnostic on the idealized linear taper. No
    # near-camera refinement footprint was authored, so this is a uniform
    # requirement only and cannot certify or disqualify a particular grading.
    wet_cells = 0
    for column in range(length_columns):
        x = (column + 0.5) * h
        depth = inlet_depth * max(0.0, 1 - x / length)
        wet_cells += width_columns * math.ceil(depth / h)
    fine_leaf_cap = CAPACITIES["graded_topology"]["leaves"]
    # The containing box of any 4,096 h-cubes could be much larger. These
    # bounds apply to their occupied volume and a single complete layer.
    return {
        "maximum_all_finest_graded_volume_m3": fine_leaf_cap * h**3,
        "maximum_all_finest_graded_one_layer_area_m2": fine_leaf_cap * h**2,
        "graded_coarser_neighbors_also_consume_leaf_capacity": True,
        "uniform_lattice_dimensions": [width_columns, length_columns, math.ceil(height / h)],
        "uniform_initial_wet_cells_at_column_centers": wet_cells,
        "idealized_initial_wet_volume_m3": width * length * inlet_depth / 2,
        "uniform_fine_cells_admitted": wet_cells <= CAPACITIES["public_uniform"]["cells"],
        "graded_camera_footprint_authored": False,
        "reference_scene_and_seed_authored": False,
        "reference_pipeline_authored": False,
        "reference_gate_admitted": False,
    }


def invoke(root, name, argv, commands):
    try:
        return json_lines(run(root, name, argv, commands))
    except RuntimeError:
        if not commands or commands[-1]["exit"] == 0:
            raise
        return json_lines((root / f"{name}.stdout").read_text(encoding="utf-8"))


def stage_summary(records, command, end_frame):
    frames = [record for record in records if record.get("event") == "frame" and record.get("published")]
    publication = [record["frame"] for record in frames]
    result = next((record for record in reversed(records) if record.get("event") == "result"), None)
    if not publication and command["exit"] == 0:
        raise RuntimeError("successful native stage reported no published frames")
    measured = frames + ([result] if result else [])
    return {
        "exit": command["exit"],
        "seconds": command["seconds"],
        "published_frames": publication,
        "complete": command["exit"] == 0 and publication == list(range(end_frame + 1)),
        "terminal_result": result,
        "maximum_reported_counts": {
            field: max(record[field] for record in measured if field in record)
            for field in ("particle_count", "block_count", "cell_count", "face_count")
            if any(field in record for record in measured)
        },
        "first_frame": frames[0] if frames else None,
        "last_frame": frames[-1] if frames else None,
    }


def qualify(root, solver, fixture, receipt):
    commands = receipt["commands"]
    receipt["device"] = json_lines(run(root, "device", [solver, "device-info", "--backend", "cuda"], commands))[-1]
    # Allocator telemetry records a separate, exact Simulate start budget.
    gpu_rows = run(root, "gpu-preflight", [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ], commands).strip().splitlines()
    if len(gpu_rows) != 1:
        raise RuntimeError("bounded control requires one unambiguous measured CUDA device")
    name, total_mib, free_mib, driver = (field.strip() for field in gpu_rows[0].split(","))
    total_bytes = int(total_mib) * 2**20
    free_bytes = int(free_mib) * 2**20
    receipt["gpu_preflight"] = {
        "device": name,
        "driver": driver,
        "device_bytes": total_bytes,
        "free_bytes_before_control": free_bytes,
        "preflight_cap_bytes": min(7 * total_bytes // 10, 4 * free_bytes // 5),
    }

    scene = root / "wave.flrs"
    seed = root / "wave.flrs.flst"
    volume_settings = root / "wave.flvc"
    run(root, "author", [fixture, "scene", scene, solver, "cuda", "bounded-wave-column-seed"], commands)
    run(root, "volume-settings", [fixture, "wave-volume-settings", volume_settings], commands)
    receipt["control_inputs_sha256"] = {path.name: digest(path) for path in (scene, seed, volume_settings)}
    stages = receipt["stages"] = {}
    receipt["simultaneous_reference_gpu_peak_measured"] = False
    simulate = root / "simulate"
    mesh = root / "mesh"
    whitewater = root / "whitewater"
    volumes = root / "volumes"
    started = time.perf_counter()
    records = invoke(root, "simulate", [
        solver, "run", "--stage", "simulate", "--scene", scene, "--initial-state", seed,
        "--cache", simulate, "--backend", "cuda", "--frame-end", CONTROL_FRAME_END, "--measure-job",
    ], commands)
    stages["simulate"] = stage_summary(records, commands[-1], CONTROL_FRAME_END)
    receipt["simulate_job_measurement"] = next(
        (record for record in records if record.get("event") == "job_measurement"), None
    )
    job = receipt["simulate_job_measurement"]
    if stages["simulate"]["complete"]:
        if job is None or stages["simulate"]["last_frame"].get("compiled_architecture") != 1200:
            raise RuntimeError("bounded control requires measured SM120 device execution")
        memory = job["cuda_memory"]
        cap = min(7 * memory["total_device_bytes"] // 10, 4 * memory["free_device_bytes_at_start"] // 5)
        receipt["simulate_arena_within_measured_budget"] = (
            0 < memory["peak_device_reserved_bytes"] <= memory["budget_bytes"] <= cap
            and memory["remaining_device_reserved_bytes_after_drain"] == 0
        )
        if not receipt["simulate_arena_within_measured_budget"]:
            raise RuntimeError("bounded control violated Simulate allocator budget or drain")
        for stage, upstream, output, options in [
            ("mesh", simulate, mesh, ["--surface-spacing", 0.125, "--surface-sigma", 0.16]),
            ("whitewater", simulate, whitewater, []),
            ("volumes", whitewater, volumes, ["--volume-settings", volume_settings]),
        ]:
            if stage == "volumes" and not stages["whitewater"]["complete"]:
                break
            rows = invoke(root, stage, [
                solver, "run", "--stage", stage, "--cache", upstream, "--output-cache", output,
                "--frame-end", CONTROL_FRAME_END, *options,
            ], commands)
            stages[stage] = stage_summary(rows, commands[-1], CONTROL_FRAME_END)
    receipt["control_launch_to_attempted_stage_drain_seconds"] = time.perf_counter() - started
    receipt["control_native_stages_complete"] = len(stages) == 4 and all(stage["complete"] for stage in stages.values())

    caches = {name: path for name, path in (
        ("simulate", simulate), ("mesh", mesh), ("whitewater", whitewater), ("volumes", volumes)
    ) if path.exists()}

    def snapshot():
        return {
            name: (cache_contents(path), {
                str(file.relative_to(path)): file.stat().st_mtime_ns
                for file in path.rglob("*") if file.is_file()
            })
            for name, path in caches.items()
        }

    before = snapshot()
    for stage, upstream in (("simulate", simulate), ("mesh", simulate), ("whitewater", simulate), ("volumes", whitewater)):
        final = stages.get(stage, {}).get("last_frame")
        if final is None:
            continue
        replay_args = [
            solver, "replay", "--stage", stage, "--cache", upstream, "--frame", final["frame"], "--substep", 0,
        ]
        replay_args += ["--backend", "cuda"] if stage == "simulate" else ["--output-cache", caches[stage]]
        replay = invoke(root, f"{stage}-replay", replay_args, commands)
        stages[stage]["replay"] = replay[-1] if replay else None
        if stage == "simulate":
            final = stages[stage]["terminal_result"] or final
        key = "state_digest" if stage == "simulate" else "payload_digest"
        stages[stage]["replay_matches"] = bool(
            replay and commands[-1]["exit"] == 0 and replay[-1].get("read_only")
            and final.get(key) and replay[-1].get(key) == final[key]
        )
    receipt["replay_preserves_cache_bytes_and_timestamps"] = before == snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.solver) != bool(args.fixture):
        parser.error("--solver and --fixture must be supplied together")
    root = args.output.resolve()
    root.mkdir(parents=True)
    receipt = {
        "host": {
            "platform": platform.platform(),
            "logical_processors": os.cpu_count(),
        },
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "preflight_sha256": digest(Path(__file__)),
        "reference": REFERENCE,
        "capacities": CAPACITIES,
        "admission": geometry_admission(),
        "commands": [],
        "control": {
            "scene_fixture": "bounded-wave-column-seed",
            "width_m": 1,
            "length_m": 5,
            "inlet_depth_m": 0.75,
            "cell_width_m": 0.25,
            "wave_height_m": 0.25,
            "wave_period_s": 2,
            "frame_end": CONTROL_FRAME_END,
            "fps": 24,
            "pipeline": "sequential native Simulate, Mesh, Whitewater, Volumes",
            "reference_gate_credit": False,
        },
    }
    try:
        if args.solver:
            solver = args.solver.resolve()
            fixture = args.fixture.resolve()
            receipt["solver_sha256"] = digest(solver)
            receipt["fixture_sha256"] = digest(fixture)
            qualify(root, solver, fixture, receipt)
    finally:
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
