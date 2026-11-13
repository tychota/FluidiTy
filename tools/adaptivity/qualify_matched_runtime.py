"""Compare public uniform and graded Simulate and Mesh on a matched rest tank."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


FRAME_END = 2
MOTION_TOLERANCE_M = 1e-9
SPEED_TOLERANCE_M_PER_S = 1e-9
VOLUME_RELATIVE_TOLERANCE = 0.005


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_contents(root):
    return {
        str(path.relative_to(root)): digest(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def run(out, name, arguments, commands):
    command = [str(argument) for argument in arguments]
    stdout = out / f"{name}.stdout"
    stderr = out / f"{name}.stderr"
    started = time.perf_counter()
    with stdout.open("wb") as output, stderr.open("wb") as errors:
        result = subprocess.run(command, stdout=output, stderr=errors, timeout=600, check=False)
    elapsed = time.perf_counter() - started
    commands.append({"argv": command, "exit": result.returncode, "seconds": elapsed})
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}. See {stdout} and {stderr}")
    return stdout.read_text(encoding="utf-8")


def json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def particle_map(frame):
    return {particle["id"]: particle for particle in frame["particles"]}


def verify_motion(frames):
    assert [frame["frame"] for frame in frames] == [0, 1, 2], frames
    initial = particle_map(frames[0])
    assert len(initial) == 12, initial
    maximum_displacement = 0
    maximum_speed = 0
    for frame in frames:
        accepted = particle_map(frame)
        assert accepted.keys() == initial.keys(), frame
        for particle_id, particle in accepted.items():
            seed = initial[particle_id]
            for field in ("mass", "reference_volume", "current_volume"):
                assert particle[field] == seed[field], (field, particle, seed)
            maximum_displacement = max(
                maximum_displacement, math.dist(particle["position"], seed["position"])
            )
            maximum_speed = max(maximum_speed, math.hypot(*particle["velocity"]))
    assert maximum_displacement <= MOTION_TOLERANCE_M, maximum_displacement
    assert maximum_speed <= SPEED_TOLERANCE_M_PER_S, maximum_speed
    return maximum_displacement, maximum_speed


def qualify_variant(root, solver, fixture, backend, variant, commands):
    out = root / variant
    out.mkdir()
    scene = out / "tank.flrs"
    seed = out / "tank.flrs.flst"
    cache = out / "simulate"
    mesh_cache = out / "mesh"
    run(out, "author", [fixture, "scene", scene, solver, backend, f"matched-{variant}-tank"], commands)
    started = time.perf_counter()
    simulate = json_lines(run(out, "simulate", [
        solver, "run", "--stage", "simulate", "--scene", scene, "--initial-state", seed,
        "--cache", cache, "--backend", backend, "--frame-end", FRAME_END, "--measure-job",
    ], commands))
    mesh = json_lines(run(out, "mesh", [
        solver, "run", "--stage", "mesh", "--cache", cache, "--output-cache", mesh_cache,
        "--surface-spacing", 0.125, "--surface-sigma", 0.25, "--frame-end", FRAME_END,
    ], commands))
    elapsed = time.perf_counter() - started

    assert simulate[-1]["frame"] == FRAME_END and simulate[-1]["particle_count"] == 12, simulate[-1]
    assert [record["frame"] for record in simulate if record["event"] == "frame"] == [0, 1, 2]
    assert mesh[-1]["frame"] == FRAME_END, mesh[-1]
    assert abs(mesh[-1]["volume_relative_error"]) <= VOLUME_RELATIVE_TOLERANCE, mesh[-1]
    assert mesh[-1]["retained_ids"] == mesh[-1]["vertex_count"] > 0, mesh[-1]
    job = next(record for record in simulate if record["event"] == "job_measurement")
    if backend == "cuda":
        memory = job["cuda_memory"]
        budget = min(7 * memory["total_device_bytes"] // 10, 4 * memory["free_device_bytes_at_start"] // 5)
        assert 0 < memory["peak_device_reserved_bytes"] <= memory["budget_bytes"] <= budget, memory
        assert memory["remaining_device_reserved_bytes_after_drain"] == 0, memory
        assert simulate[-1]["compiled_architecture"] == 1200, simulate[-1]

    before = cache_contents(out)
    replay_simulate = json_lines(run(root, f"{variant}-simulate-replay", [
        solver, "replay", "--stage", "simulate", "--cache", cache, "--backend", backend,
        "--frame", FRAME_END, "--substep", 0,
    ], commands))[-1]
    replay_mesh = json_lines(run(root, f"{variant}-mesh-replay", [
        solver, "replay", "--stage", "mesh", "--cache", cache, "--output-cache", mesh_cache,
        "--frame", FRAME_END, "--substep", 0,
    ], commands))[-1]
    assert replay_simulate["read_only"] and replay_simulate["state_digest"] == simulate[-1]["state_digest"]
    assert replay_mesh["read_only"] and replay_mesh["payload_digest"] == mesh[-1]["payload_digest"]
    assert before == cache_contents(out), f"{variant} replay changed a published file"

    frames = json.loads(run(root, f"{variant}-motion", [fixture, "matched-motion", cache], commands))
    displacement, speed = verify_motion(frames)
    return frames, {
        "scene_sha256": digest(scene),
        "seed_sha256": digest(seed),
        "simulate_and_mesh_seconds": elapsed,
        "maximum_particle_displacement_m": displacement,
        "maximum_particle_speed_m_per_s": speed,
        "simulate": simulate[-1],
        "mesh": mesh[-1],
        "job_measurement": job,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True)
    solver = args.solver.resolve()
    fixture = args.fixture.resolve()
    commands = []
    receipt = {
        "backend": args.backend,
        "solver_sha256": digest(solver),
        "fixture_sha256": digest(fixture),
        "commands": commands,
        "limits": {
            "particle_displacement_m": MOTION_TOLERANCE_M,
            "particle_speed_m_per_s": SPEED_TOLERANCE_M_PER_S,
            "mesh_volume_relative": VOLUME_RELATIVE_TOLERANCE,
        },
    }
    try:
        receipt["device"] = json_lines(run(root, "device", [solver, "device-info", "--backend", args.backend], commands))[-1]
        uniform, receipt["uniform"] = qualify_variant(root, solver, fixture, args.backend, "uniform", commands)
        graded, receipt["graded"] = qualify_variant(root, solver, fixture, args.backend, "graded", commands)
        assert particle_map(uniform[0]) == particle_map(graded[0]), "the physical seeds differ"
        for uniform_frame, graded_frame in zip(uniform, graded, strict=True):
            uniform_particles = particle_map(uniform_frame)
            graded_particles = particle_map(graded_frame)
            for particle_id, particle in uniform_particles.items():
                other = graded_particles[particle_id]
                assert math.dist(particle["position"], other["position"]) <= MOTION_TOLERANCE_M
                assert math.dist(particle["velocity"], other["velocity"]) <= SPEED_TOLERANCE_M_PER_S
        receipt["passed"] = True
    finally:
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {root / 'receipt.json'}")


if __name__ == "__main__":
    main()
