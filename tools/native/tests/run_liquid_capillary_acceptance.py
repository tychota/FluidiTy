"""Run frozen M1 capillary and dam scenes and measure verified native results."""

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import math
from pathlib import Path
import re
import subprocess
import sys

from run_liquid_acceptance import (
    MAXIMUM_COUNTS,
    clock,
    digest,
    integer,
    matches_identity,
    one_record,
    read_records,
    require,
    snapshot,
    verify_device,
    verify_frames,
    verify_solver,
    write_measurements,
)
from liquid_acceptance_metrics import (
    check_ids,
    evaluate_transport_components,
    evaluate_volume_budget,
    finite_number,
    vector3,
)
from liquid_capillary_acceptance_metrics import (
    CAPILLARY_CASES,
    DAM_CELL_WIDTH_M,
    DAM_LAST_FRAME,
    DAM_PARTICLE_REFERENCE_VOLUME_M3,
    DAM_REPORT_FRAMES,
    DROP_CASE,
    LIQUID_DENSITY_KG_PER_M3,
    MAXIMUM_DT_SECONDS,
    SPHERE_RADIUS_M,
    SPHERE_VOLUME_M3,
    STATIC_COARSE_CASE,
    STATIC_FINE_CASE,
    SURFACE_TENSION_N_PER_M,
    compare_dam_break_energy_blends,
    evaluate_drop_frequency,
    evaluate_static_sphere,
    verify_capillary_steps,
)
from liquid_drop_seed_calibration import authenticated_drop_positions, calibrate_drop_seed


DAM_BLEND_BY_CASE = {"dam-break-pic": 0, "dam-break-default": 0.6, "dam-break-retained": 1}
GROUPS = {"static": (STATIC_COARSE_CASE, STATIC_FINE_CASE), "drop": (DROP_CASE,), "dam": tuple(DAM_BLEND_BY_CASE)}
PARTICLE_COUNTS = {STATIC_COARSE_CASE: 136, STATIC_FINE_CASE: 552, DROP_CASE: 552}
CAPACITIES = [1024, 128, 4096, 24576, 256]
DAM_SUPPORT_RADIUS_M = 0.9 * DAM_CELL_WIDTH_M
ZERO_CLOCK = (0, 0, 0, 0, 0)
CUDA_COMPATIBILITY_FILE = ".simulate-cuda-compatibility"
CUDA_COMPATIBILITY_RECORD_SIZE = len(b"FLCUDA1\n") + 64 + 1
CUDA_PROBE_IDENTITY_FIELDS = (
    "device_index", "device_name", "compute_major", "compute_minor", "compiled_architecture",
    "driver_version", "runtime_version",
)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_child(output, name, executable, *arguments, fixture=False, timeout=180):
    """Retain the raw streams and process status before trusting a receipt."""
    command = [str(executable), *map(str, arguments)]
    stdout_path = output / f"{name}.stdout.jsonl"
    stderr_path = output / f"{name}.stderr.txt"
    status_path = output / f"{name}.exit.json"
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=timeout, check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        write_measurements(status_path, {"command": command, "returncode": None, "error": str(error)})
        raise
    write_measurements(status_path, {"command": command, "returncode": result.returncode})
    require(result.returncode == 0, f"{name} exited {result.returncode}; see {stderr_path}")
    require(stderr_path.stat().st_size == 0, f"{name} produced diagnostics; see {stderr_path}")
    return read_records(stdout_path.read_text(encoding="utf-8"), "fixture_version" if fixture else "protocol_version")


def frozen_scene(case):
    if case in DAM_BLEND_BY_CASE:
        return DAM_CELL_WIDTH_M, 256, DAM_PARTICLE_REFERENCE_VOLUME_M3, DAM_LAST_FRAME
    cells_per_radius, last_frame = CAPILLARY_CASES[case]
    particle_count = PARTICLE_COUNTS[case]
    return SPHERE_RADIUS_M / cells_per_radius, particle_count, SPHERE_VOLUME_M3 / particle_count, last_frame


def verify_resolved_scene(resolved, authored, case, support_radius_m):
    matches_identity(resolved, authored)
    for field in ("seed_digest", "solver_digest"):
        require(digest(resolved, field) == digest(authored, field), f"wrong resolved {field}")
    require(integer(resolved, "seed_byte_length", minimum=1) == authored["seed_byte_length"],
            "wrong resolved seed length")
    h, _, _, _ = frozen_scene(case)
    dam = case in DAM_BLEND_BY_CASE
    blend = DAM_BLEND_BY_CASE[case] if dam else 0.6
    expected = {
        "world_origin_m": [0, 0, 0],
        "cell_size_m": h,
        "random_seed": 0,
        "gravity_m_per_s2": [0, -9.81, 0] if dam else [0, 0, 0],
        "density_kg_per_m3": LIQUID_DENSITY_KG_PER_M3,
        "surface_tension_n_per_m": 0 if dam else SURFACE_TENSION_N_PER_M,
        "ambient_gauge_pa": 0,
        "target_divergence_per_s": 0,
        "source_count": 0,
        "friction_coefficient": 0,
        "penetration_tolerance_m": 0,
        "minimum_face_mass_kg": 0,
        "mass_kg_per_quantum": 2 ** -40,
        "momentum_kg_m_per_s_per_quantum": 2 ** -40,
        "pressure_iteration_limit": 256,
        "energy_blend": blend,
        "maximum_dt_seconds": MAXIMUM_DT_SECONDS,
    }
    if dam:
        expected.update({
            "tank_low_x_m": 0, "tank_high_x_m": 2, "tank_floor_y_m": 0,
            "tank_low_z_m": 0, "tank_high_z_m": 0.25, "piston_velocity_x_m_per_s": 0,
            "sphere_union_radius_m": DAM_SUPPORT_RADIUS_M,
        })
    elif case == DROP_CASE:
        # repr round-trips the calibrated binary64 value through native authoring.
        expected["sphere_union_radius_m"] = support_radius_m
    else:
        expected.update({"sphere_center_m": [0, 0, 0], "sphere_radius_m": SPHERE_RADIUS_M})
    for field, value in expected.items():
        require(resolved.get(field) == value, f"wrong resolved {field}")
    require(resolved.get("incompressible") is True, "resolved scene declares compression")
    require(authored.get("energy_blend") == blend, "wrong authored energy blend")


def verify_point_material(frame, particle_count, reference_volume_m3):
    points = frame.get("points")
    check_ids(points, particle_count, f"point frame {frame['frame']}")
    for point in points:
        vector3(point.get("world_position"))
        vector3(point.get("world_velocity"))
        require(finite_number(point.get("volume_ratio")) and point["volume_ratio"] > 0,
                f"invalid J at frame {frame['frame']}, ID {point['id']}")
        point_volume_m3 = 4 * math.pi / 3 * point["radius"] ** 3
        require(point_volume_m3 > 0
                and math.isclose(point_volume_m3, reference_volume_m3 * point["volume_ratio"], rel_tol=1e-14),
                f"point radius disagrees with V0*J at frame {frame['frame']}, ID {point['id']}")


def verify_material_state(state, frame, particle_count, reference_volume_m3):
    require(clock(state) == clock(frame), "state and point clocks differ")
    require(integer(state, "random_seed") == 0 and integer(state, "next_particle_id") == particle_count + 1,
            "wrong state random seed or next ID")
    particles = state.get("particles")
    points = frame["points"]
    check_ids(particles, particle_count, "state")
    require([p["id"] for p in particles] == [p["id"] for p in points], "state and point ID orders differ")
    for particle, point in zip(particles, points):
        require(particle.get("reference_volume") == reference_volume_m3, f"wrong V0 for ID {point['id']}")
        require(particle.get("mass") == LIQUID_DENSITY_KG_PER_M3 * reference_volume_m3,
                f"wrong mass for ID {point['id']}")
        for field in ("tag", "world_position", "world_velocity", "volume_ratio"):
            require(particle.get(field) == point[field], f"state differs from point {field} for ID {point['id']}")
        slopes = particle.get("velocity_slopes")
        require(isinstance(slopes, list) and len(slopes) == 3, f"invalid velocity slopes for ID {point['id']}")
        for row in slopes:
            vector3(row)
    return {
        "frame": frame["frame"],
        "state_material_volume_m3": math.fsum(p["reference_volume"] * p["volume_ratio"] for p in particles),
        "mass_kg": math.fsum(p["mass"] for p in particles),
    }


def verify_frozen_seed(seed, expected_digest, case):
    require(digest(seed, "state_digest") == expected_digest, "seed digest differs from authored state")
    require(clock(seed) == ZERO_CLOCK and integer(seed, "random_seed") == 0, "wrong seed clock or random seed")
    h, particle_count, reference_volume_m3, _ = frozen_scene(case)
    require(integer(seed, "next_particle_id") == particle_count + 1, "wrong seed next ID")
    check_ids(seed.get("particles"), particle_count, "authored seed")
    if case == DROP_CASE:
        authenticated_drop_positions(seed, expected_digest)
        return
    if case in DAM_BLEND_BY_CASE:
        lattice = product(range(4), range(8), range(8))
    else:
        n, _ = CAPILLARY_CASES[case]
        lattice = (
            (z, y, x) for z, y, x in product(range(-n, n), repeat=3)
            if (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2 < n**2
        )
    for particle_id, (particle, (z, y, x)) in enumerate(zip(seed["particles"], lattice, strict=True), 1):
        expected = {
            "id": particle_id,
            "tag": 0,
            "side_tag": 0,
            "world_position": [(coordinate + 0.5) * h for coordinate in (x, y, z)],
            "world_velocity": [0, 0, 0],
            "velocity_slopes": [[0, 0, 0]] * 3,
            "reference_volume": reference_volume_m3,
            "mass": LIQUID_DENSITY_KG_PER_M3 * reference_volume_m3,
            "volume_ratio": 1,
            "affine_enabled": True,
            "pic_initialized": False,
        }
        require(particle == expected, f"seed ID {particle_id} differs from the frozen lattice")


def check_preflight_pressure(case, traces):
    require(traces, "missing preflight pressure trace")
    h, _, _, _ = frozen_scene(case)
    dam = case in DAM_BLEND_BY_CASE
    bound = None if dam else math.sqrt(LIQUID_DENSITY_KG_PER_M3 * h**3 / (4 * math.pi * SURFACE_TENSION_N_PER_M))
    first_empty_pressure_step = first_capillary_bound_mismatch_step = None
    pressure_volumes_m3 = []
    for trace in traces:
        step = trace["accepted_step"]
        measurement = trace["accepted_measurement"]
        pressure_volume_m3 = measurement["step_start_pressure_cell_volume_m3"]
        pressure_volumes_m3.append(pressure_volume_m3)
        if first_empty_pressure_step is None and (trace["cell_count"] == 0 or pressure_volume_m3 <= 0):
            first_empty_pressure_step = step
        reported_bound = measurement["step_start_capillary_bound_seconds"]
        if dam:
            require(reported_bound is None, "zero-tension dam reports a capillary bound")
            require(measurement["step_start_piston_low_x_m"] == measurement["step_end_piston_low_x_m"] == 0,
                    "dam wall moved")
            require(measurement["step_start_inward_piston_flux_m3_per_s"] == 0, "nonzero dam wall flux")
        elif first_capillary_bound_mismatch_step is None and (
            not finite_number(reported_bound) or not math.isclose(reported_bound, bound, rel_tol=1e-14)
            or measurement["accepted_dt_seconds"] > min(reported_bound, bound)
        ):
            first_capillary_bound_mismatch_step = step
    return {
        "passed": first_empty_pressure_step is None and first_capillary_bound_mismatch_step is None,
        "first_empty_pressure_step": first_empty_pressure_step,
        "first_capillary_bound_mismatch_step": first_capillary_bound_mismatch_step,
        "minimum_pressure_cell_count": min(trace["cell_count"] for trace in traces),
        "minimum_pressure_cell_volume_m3": min(pressure_volumes_m3),
        "analytic_capillary_bound_seconds": bound,
    }


def cuda_probe_identity(device):
    verify_device(device, "cuda")
    require("RTX" in device["device_name"], "expected actual RTX device")
    return {field: device[field] for field in CUDA_PROBE_IDENTITY_FIELDS}


def read_cuda_compatibility_record(cache):
    # Match the pinned CLI's exact byte format, including LF and lowercase hex.
    try:
        with (cache / CUDA_COMPATIBILITY_FILE).open("rb") as stream:
            record = stream.read(CUDA_COMPATIBILITY_RECORD_SIZE + 1)
    except FileNotFoundError as error:
        raise ValueError("missing CUDA compatibility record") from error
    require(re.fullmatch(rb"FLCUDA1\n[0-9a-f]{64}\n", record) is not None, "malformed CUDA compatibility record")
    return record


def verify_group_identity(group, backend, members):
    require(tuple(members) == GROUPS[group], "incomplete or unexpected scene group")
    identities = []
    for case, member in members.items():
        authored = member["authored"]
        require(authored.get("backend") == backend and authored.get("case") == case, "wrong group backend or case")
        identities.append(authored)
    require(len({digest(record, "solver_digest") for record in identities}) == 1, "group solver digests differ")
    for field in ("resolved_fingerprint", "execution_id"):
        require(len({digest(record, field) for record in identities}) == len(identities), f"group reused {field}")
    seed_count = len({digest(record, "seed_digest") for record in identities})
    require(seed_count == (1 if group == "dam" else len(identities)), "wrong group seed identities")
    if backend == "cuda":
        # The CLI's compatibility digest includes the scene fingerprint. Across
        # scenes the public probe fields can be compared, while the UUID is hidden.
        devices = [cuda_probe_identity(member.get("device", {})) for member in members.values()]
        for field in CUDA_PROBE_IDENTITY_FIELDS:
            require(len({device[field] for device in devices}) == 1, f"group CUDA probe differs on {field}")


def verify_cache(output, label, cache, fixture, scene, seed, authored, case, records, last_frame, support_radius_m, *,
                 cuda_compatibility_record=None):
    h, particle_count, reference_volume_m3, _ = frozen_scene(case)
    ordinals = tuple(range(last_frame + 1))
    fingerprint = authored["resolved_fingerprint"]
    before = snapshot(cache)
    if authored["backend"] == "cuda":
        require(cuda_compatibility_record is not None, "missing CUDA compatibility evidence")
        record_sha256 = hashlib.sha256(cuda_compatibility_record).hexdigest()
        require(CUDA_COMPATIBILITY_FILE in before and before[CUDA_COMPATIBILITY_FILE][1] == record_sha256,
                "CUDA compatibility record changed before cache verification")
    names = {Path(name).as_posix() for name in before}
    expected_manifests = {f"frames/{frame:020d}.{suffix}" for frame in ordinals for suffix in ("flmf", "json")}
    checkpoint_refs = tuple(range(0, last_frame + 1, 10))
    expected_refs = {f"checkpoints/{frame:020d}.flcr" for frame in checkpoint_refs}
    require({name for name in names if name.startswith("frames/")} == expected_manifests,
            "wrong committed manifest inventory")
    require({name for name in names if name.startswith("checkpoints/")} == expected_refs,
            "wrong checkpoint reference inventory")
    require((cache / "resolved.flrs").read_bytes() == scene.read_bytes(), "cache FLRS differs from authored scene")
    if case in DAM_BLEND_BY_CASE:
        reader, scene_input, event = "read-tank-scene", cache, "resolved_tank"
    elif case == DROP_CASE:
        reader, scene_input, event = "read-drop-scene", cache / "resolved.flrs", "resolved_drop"
    else:
        reader, scene_input, event = "read-static-sphere-scene", cache / "resolved.flrs", "resolved_static_sphere"
    resolved = one_record(
        record_child(output, f"{label}-resolved", fixture, reader, scene_input, fingerprint, fixture=True), event,
    )
    verify_resolved_scene(resolved, authored, case, support_radius_m)
    seed_digest = authored["seed_digest"]
    initial = one_record(
        record_child(output, f"{label}-seed", fixture, "read-seed", seed, seed_digest, fixture=True), "seed",
    )
    verify_frozen_seed(initial, seed_digest, case)

    frames = record_child(output, f"{label}-frames", fixture, "read-frames", cache, fingerprint, 0, last_frame,
                          fixture=True)
    verify_frames(frames, authored, ordinals)
    point_blobs = set()
    for frame in frames:
        point_blob = f"blobs/{frame['points_digest']}.flpt"
        point_blobs.add(point_blob)
        require((cache / point_blob).stat().st_size == frame["points_byte_length"], "wrong point blob size")
        verify_point_material(frame, particle_count, reference_volume_m3)
        frame.update(source_volume_m3=0, compression_volume_m3=0)
    require({name for name in names if name.startswith("blobs/") and name.endswith(".flpt")} == point_blobs,
            "wrong point blob inventory")
    verify_attempt_durations(records)
    final = verify_solver(records, frames, authored, ordinals, tank_layout=case in DAM_BLEND_BY_CASE,
                          require_job_measurement=True, particle_count=particle_count, maximum_dt=MAXIMUM_DT_SECONDS)
    traces = [record for record in records if record["event"] == "trace"]
    settled_by_frame = {trace["frame"]: trace for trace in traces if trace["substep"] == 0}
    material = [verify_material_state(initial, frames[0], particle_count, reference_volume_m3)]
    checkpoints = []
    for ordinal in checkpoint_refs[-2:]:
        checkpoint = one_record(
            record_child(output, f"{label}-checkpoint-{ordinal}", fixture, "read-checkpoint", cache, fingerprint,
                         ordinal, fixture=True), "checkpoint",
        )
        matches_identity(checkpoint, authored)
        require(checkpoint.get("capacities") == CAPACITIES, "wrong checkpoint capacity reservations")
        state_digest = digest(checkpoint, "state_digest")
        expected_digest = seed_digest if ordinal == 0 else settled_by_frame[ordinal]["state_digest"]
        require(state_digest == expected_digest, "retained checkpoint differs from its seed or settled trace")
        material.append(verify_material_state(checkpoint, frames[ordinal], particle_count, reference_volume_m3))
        checkpoints.append(checkpoint)
    # The native reader authenticates each resume envelope through its FLCR.
    # The reported state digest hashes the decoded state without that metadata.
    checkpoint_blobs = {name for name in names if name.startswith("blobs/") and name.endswith(".flst")}
    require(len(checkpoint_blobs) == len(checkpoints), "wrong retained checkpoint payload inventory")
    require(snapshot(cache) == before, "verified readers changed cache files")

    return {
        "authored": authored,
        "resolved": resolved,
        "frames": frames,
        "checkpoints": checkpoints,
        "traces": traces,
        "evidence": {
            "verified_frame_count": len(frames),
            "verified_point_digests": [frame["points_digest"] for frame in frames],
            "verified_immutable_cache_file_count": len(before),
            "retained_checkpoint_frames": [checkpoint["frame"] for checkpoint in checkpoints],
            "observed_maximum_extents": {field: max(trace[field] for trace in traces) for field in MAXIMUM_COUNTS},
            "material_state_checks": material,
            "pressure_check": check_preflight_pressure(case, traces),
            "final_receipt": final,
            "job_measurement": records[-2],
        },
    }


def frozen_drop_support_radius(calibrated_radius_m):
    # Independently reproduced by the reviewed calibration on the frozen seed.
    # Use the same binary64 radius in both backend scenes.
    expected_radius_m = float.fromhex("0x1.0d70a3d70a3d8p-6")
    require(calibrated_radius_m == expected_radius_m, "drop calibration differs from the frozen binary64 radius")
    return expected_radius_m


def verify_attempt_durations(records):
    for record in records:
        if record["event"] == "attempt":
            dt = record.get("attempted_dt_seconds")
            require(finite_number(dt) and 0 < dt <= MAXIMUM_DT_SECONDS, "attempted duration exceeds frozen maximum dt")


def verify_pins(solver, fixture, pins):
    for kind, executable in (("solver", solver), ("fixture", fixture)):
        require(file_sha256(executable) == pins[kind], f"{kind} SHA-256 differs from the expected binary")


def run_scene(output, solver, fixture, backend, case, pins):
    verify_pins(solver, fixture, pins)
    device = one_record(record_child(output, "device", solver, "device-info", "--backend", backend), "result")
    probe_identity = None
    device_arguments = ()
    if backend == "cuda":
        probe_identity = cuda_probe_identity(device)
        device_arguments = ("--device", probe_identity["device_index"])
    else:
        verify_device(device, backend)
    write_measurements(output / "binary-identity.json", {"sha256": pins, "device": device})
    scene = output / "scene.flrs"
    seed = scene.with_suffix(".flrs.flst")
    support_radius_m = DAM_SUPPORT_RADIUS_M if case in DAM_BLEND_BY_CASE else None
    radius_argument = ()
    if case == DROP_CASE:
        preview = output / "drop-preview.flst"
        written = one_record(record_child(output, "calibration-author", fixture, "write-drop-seed", preview,
                                          fixture=True), "drop_seed")
        expected_digest = digest(written, "seed_digest")
        require(preview.stat().st_size == integer(written, "seed_byte_length", minimum=1), "wrong calibration seed size")
        calibration_seed = one_record(
            record_child(output, "calibration-seed", fixture, "read-seed", preview, expected_digest, fixture=True),
            "seed",
        )
        calibration = calibrate_drop_seed(calibration_seed, expected_digest)
        support_radius_m = frozen_drop_support_radius(calibration.support_radius_m)
        write_measurements(output / "drop-calibration.json",
                           {**asdict(calibration), "support_radius_binary64": support_radius_m.hex()})
        radius_argument = (repr(support_radius_m),)
    authored = one_record(
        record_child(output, "scene-author", fixture, "write-scene", case, backend, solver, scene,
                     *radius_argument, fixture=True), "scene",
    )
    require(authored.get("backend") == backend and authored.get("case") == case, "wrong authored case or backend")
    for field in ("resolved_fingerprint", "execution_id", "solver_digest", "seed_digest"):
        digest(authored, field)
    require(seed.stat().st_size == integer(authored, "seed_byte_length", minimum=1), "wrong authored seed size")
    if case == DROP_CASE:
        require(authored["seed_digest"] == expected_digest and seed.read_bytes() == preview.read_bytes(),
                "calibrated and authored drop seeds differ")
    input_hashes = {path.name: file_sha256(path) for path in (scene, seed)}
    write_measurements(output / "authored-inputs.json", {"sha256": input_hashes, "authored": authored})

    arguments = ("run", "--stage", "simulate", "--scene", scene, "--initial-state", seed, "--backend", backend,
                 *device_arguments, "--trace", "--measure-job")
    preflight_compatibility = None
    for label, last_frame in (("preflight", 1), ("full", frozen_scene(case)[-1])):
        cache = output / ("pre" if label == "preflight" else "cache")
        verify_pins(solver, fixture, pins)
        records = record_child(output, label, solver, *arguments, "--cache", cache, "--frame-end", last_frame,
                               timeout=None)
        compatibility = None
        if probe_identity is not None:
            compatibility = read_cuda_compatibility_record(cache)
            if label == "preflight":
                preflight_compatibility = compatibility
            else:
                require(compatibility == preflight_compatibility,
                        "CUDA compatibility record differs between preflight and full run")
        verified = verify_cache(output, label, cache, fixture, scene, seed, authored, case, records, last_frame,
                                support_radius_m, cuda_compatibility_record=compatibility)
        verify_pins(solver, fixture, pins)
        require({path.name: file_sha256(path) for path in (scene, seed)} == input_hashes,
                "native execution or verification changed authored inputs")
        evidence = verified["evidence"]
        if compatibility is not None:
            verified["device"] = device
            evidence.update(cuda_probe_identity=probe_identity, cuda_compatibility_record=compatibility.decode("ascii"))
        write_measurements(output / f"{label}-evidence.json", evidence)
        if label == "preflight" and not evidence["pressure_check"]["passed"]:
            material_budget_passed = None
            if support_radius_m is not None:
                frames = verified["frames"]
                material_budget = evaluate_volume_budget(frames, [0] * len(frames))
                material_budget_passed = material_budget["passed"]
                write_measurements(output / "preflight-material.json", {
                    "material_budget": material_budget,
                    "component_inventory": evaluate_transport_components(frames, support_radius_m, frozen_scene(case)[0]),
                })
            return {
                **verified, "status": "failed_preflight", "passed": False, "full_run_launched": False,
                "material_budget_passed": material_budget_passed,
            }

    measurements = {"pressure_inventory": evidence["pressure_check"]}
    if case in CAPILLARY_CASES:
        measurements["capillary_steps"] = verify_capillary_steps(verified["traces"], case=case)
    if support_radius_m is not None:
        frames = verified["frames"]
        measurements["material_budget"] = evaluate_volume_budget(frames, [0] * len(frames))
        measurements["component_inventory"] = evaluate_transport_components(frames, support_radius_m, frozen_scene(case)[0])
    if case == DROP_CASE:
        measurements["frequency"] = evaluate_drop_frequency(verified["frames"])
    # The dam endpoint and the drop frequency have separate physical gates.
    # Keep the material-budget verdict visible for comparison with its own limit.
    material_budget_passed = measurements.get("material_budget", {}).get("passed")
    passed = all(report["passed"] for name, report in measurements.items()
                 if name in ("pressure_inventory", "capillary_steps", "frequency"))
    write_measurements(output / "scene-measurements.json", measurements)
    return {
        **verified, "status": "verified", "passed": passed, "full_run_launched": True,
        "material_budget_passed": material_budget_passed,
    }


def verify_dam_comparison_clocks(frames_by_blend):
    comparison_clocks = {}
    for ordinal in DAM_REPORT_FRAMES:
        clocks = {clock(frames[ordinal]) for frames in frames_by_blend.values()}
        require(len(clocks) == 1, f"dam comparison clock differs at frame {ordinal}")
        comparison_clocks[ordinal] = clocks.pop()
    return comparison_clocks


def measure_group(group, backend, members):
    verify_group_identity(group, backend, members)
    if group == "static":
        coarse, fine = (members[case] for case in GROUPS[group])
        return evaluate_static_sphere(coarse["frames"][1:], coarse["checkpoints"], fine["frames"][1:], fine["checkpoints"])
    if group == "dam":
        frames_by_blend = {DAM_BLEND_BY_CASE[case]: member["frames"] for case, member in members.items()}
        comparison_clocks = verify_dam_comparison_clocks(frames_by_blend)
        comparison = compare_dam_break_energy_blends(frames_by_blend)
        comparison["accepted_comparison_clocks"] = comparison_clocks
        return comparison
    return None


def run_group(group, backend, solver, fixture, pins, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = {}
    members = {}
    for case in GROUPS[group]:
        directory = output / case
        directory.mkdir()
        print(f"{backend} {case}: verifying preflight and accepted frames", flush=True)
        try:
            member = run_scene(directory, solver, fixture, backend, case, pins)
            members[case] = member
            cases[case] = {key: member[key] for key in ("status", "passed", "full_run_launched", "material_budget_passed", "evidence")}
        except (ValueError, OSError, KeyError, TypeError, ArithmeticError, subprocess.TimeoutExpired) as error:
            cases[case] = {"status": "verification_error", "passed": False, "error": str(error)}
        write_measurements(directory / "status.json", cases[case])
        print(f"{backend} {case}: {cases[case]['status']}", flush=True)

    result = {"group": group, "backend": backend, "cases": cases, "passed": all(case["passed"] for case in cases.values())}
    if all(case["status"] == "verified" for case in cases.values()):
        if group == "dam":
            result["material_budget_passed"] = all(case["material_budget_passed"] for case in cases.values())
        try:
            comparison = measure_group(group, backend, members)
            if comparison is not None:
                result["comparison"] = comparison
                result["passed"] &= comparison["passed"]
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            result.update(passed=False, comparison_error=str(error))
    write_measurements(output / "group-measurements.json", result)
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--solver-sha256", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh short path for logs and Windows native cache")
    args = parser.parse_args()
    pins = {"solver": digest(vars(args), "solver_sha256"), "fixture": digest(vars(args), "fixture_sha256")}
    try:
        passed = run_group(args.group, args.backend, args.solver.resolve(strict=True), args.fixture.resolve(strict=True),
                           pins, args.output.resolve())
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
