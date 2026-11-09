"""Behavioral checks for the standalone capillary acceptance runner."""

import json
import math
import sys

import pytest


import run_liquid_capillary_acceptance as runner


def test_drop_preflight_rejects_empty_pressure_without_qualifying_dynamics():
    traces = [{
        "accepted_step": 1, "cell_count": 0,
        "accepted_measurement": {
            "accepted_dt_seconds": 0.008333333333333333,
            "step_start_pressure_cell_volume_m3": 0,
            "step_start_capillary_bound_seconds": None,
        },
    }]
    result = runner.check_preflight_pressure("oscillating-drop-l2", traces)
    assert result["passed"] is False
    assert result["first_empty_pressure_step"] == 1
    assert result["first_capillary_bound_mismatch_step"] == 1
    assert result["minimum_pressure_cell_count"] == 0


def test_dam_rejects_nonzero_wall_motion_even_with_same_tank_pose():
    traces = [{
        "accepted_step": 1, "cell_count": 256,
        "accepted_measurement": {
            "step_start_pressure_cell_volume_m3": 0.0625,
            "step_start_capillary_bound_seconds": None,
            "step_start_piston_low_x_m": 0,
            "step_end_piston_low_x_m": 0,
            "step_start_inward_piston_flux_m3_per_s": 0.01,
        },
    }]
    with pytest.raises(ValueError, match="wall flux"):
        runner.check_preflight_pressure("dam-break-pic", traces)


def test_point_inventory_rejects_swapped_checkpoint_order():
    volume = 0.0625**3
    radius = (3 * volume / (4 * math.pi)) ** (1 / 3)
    points = [
        {"id": i, "tag": 0, "world_position": [i, 0, 0], "world_velocity": [0, 0, 0],
         "volume_ratio": 1, "radius": radius}
        for i in (1, 2)
    ]
    clock = {"frame": 10, "time_seconds": 10 / 24, "substep": 0, "accepted_step": 50, "epoch": 50}
    particles = [
        {**point, "mass": 1000 * volume, "reference_volume": volume, "velocity_slopes": [[0, 0, 0]] * 3}
        for point in reversed(points)
    ]
    state = {**clock, "particles": particles, "next_particle_id": 3, "random_seed": 0}
    with pytest.raises(ValueError, match="order"):
        runner.verify_material_state(state, {**clock, "points": points}, 2, volume)


def test_dam_group_rejects_reused_execution_identity():
    members = {
        name: {"authored": {
            "case": name, "backend": "cpu", "solver_digest": "a" * 64, "seed_digest": "b" * 64,
            "resolved_fingerprint": fingerprint * 64, "execution_id": "f" * 64,
        }}
        for name, fingerprint in zip(("dam-break-pic", "dam-break-default", "dam-break-retained"), "012")
    }
    with pytest.raises(ValueError, match="execution_id"):
        runner.verify_group_identity("dam", "cpu", members)


def test_nonzero_native_exit_preserves_raw_output_and_status(tmp_path):
    with pytest.raises(ValueError, match="exited 17"):
        runner.record_child(
            tmp_path, "native", sys.executable, "-c",
            "import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(17)",
        )
    assert (tmp_path / "native.stdout.jsonl").read_text().strip() == "stdout"
    assert (tmp_path / "native.stderr.txt").read_text().strip() == "stderr"
    assert json.loads((tmp_path / "native.exit.json").read_text())["returncode"] == 17


def test_drop_support_radius_rejects_a_one_ulp_backend_difference():
    reviewed = float.fromhex("0x1.0d70a3d70a3d8p-6")
    assert runner.frozen_drop_support_radius(reviewed) == reviewed
    with pytest.raises(ValueError, match="binary64"):
        runner.frozen_drop_support_radius(math.nextafter(reviewed, math.inf))


def test_rejected_pressure_attempt_must_obey_frozen_maximum_dt():
    attempts = [
        {"event": "attempt", "attempted_dt_seconds": 0.02},
        {"event": "attempt", "attempted_dt_seconds": 0.01},
    ]
    with pytest.raises(ValueError, match="attempted duration"):
        runner.verify_attempt_durations(attempts)


def test_dam_comparison_rejects_mismatched_accepted_clocks_at_shared_time():
    frames = [
        {"frame": frame, "substep": 0, "epoch": frame * 5, "accepted_step": frame * 5, "time_seconds": frame / 24}
        for frame in range(49)
    ]
    changed = frames.copy()
    changed[24] = {**frames[24], "epoch": 121, "accepted_step": 121}
    with pytest.raises(ValueError, match="comparison clock.*24"):
        runner.verify_dam_comparison_clocks({0: frames, 0.6: frames, 1: changed})


def cuda_probe():
    return {
        "event": "result", "command": "device-info", "backend": "cuda", "cpu_available": True,
        "cuda_compiled": True, "cuda_available": True, "device_index": 0, "device_name": "NVIDIA GeForce RTX 5060 Ti",
        "compute_major": 12, "compute_minor": 0, "compiled_architecture": 1200, "sample_count": 17,
        "driver_version": 13020, "runtime_version": 13010, "launch_result": 0, "copy_result": 0,
        "synchronize_result": 0,
    }


@pytest.mark.parametrize(("field", "value"), [
    ("device_name", "NVIDIA GeForce RTX 5080"),
    ("driver_version", 13010),
    ("runtime_version", 13020),
])
def test_paired_cuda_cases_reject_changed_probe_identity(field, value):
    cases = ("static-sphere-coarse", "static-sphere-fine")
    members = {
        case: {
            "authored": {
                "case": case, "backend": "cuda", "solver_digest": "a" * 64, "seed_digest": ordinal * 64,
                "resolved_fingerprint": ordinal * 64, "execution_id": execution * 64,
            },
            "device": cuda_probe(),
        }
        for case, ordinal, execution in zip(cases, "01", "23")
    }
    runner.verify_group_identity("static", "cuda", members)
    members[cases[1]]["device"][field] = value
    with pytest.raises(ValueError, match=field):
        runner.verify_group_identity("static", "cuda", members)


CUDA_COMPATIBILITY_BYTES = b"FLCUDA1\n" + b"a" * 64 + b"\n"


@pytest.fixture
def launch_static_cuda_case(tmp_path, monkeypatch):
    """Exercise run_scene at the process boundary without a CUDA device."""
    seed_bytes = b"seed"
    case = "static-sphere-coarse"
    authored = {
        "event": "scene", "case": case, "backend": "cuda", "energy_blend": 0.6, "solver_digest": "a" * 64,
        "seed_digest": "b" * 64, "resolved_fingerprint": "c" * 64, "execution_id": "d" * 64,
        "seed_byte_length": len(seed_bytes),
    }
    fixture = tmp_path / "fixture.exe"
    solver = tmp_path / "solver.exe"
    fixture.write_bytes(b"fixture")
    solver.write_bytes(b"solver")
    pins = {"fixture": runner.file_sha256(fixture), "solver": runner.file_sha256(solver)}

    def run(full_record):
        def child(_output, name, _executable, *arguments, **_options):
            if name == "device":
                return [cuda_probe()]
            if name == "scene-author":
                scene = arguments[4]
                scene.write_bytes(b"scene")
                scene.with_suffix(".flrs.flst").write_bytes(seed_bytes)
                return [authored]
            cache = arguments[arguments.index("--cache") + 1]
            cache.mkdir()
            record = CUDA_COMPATIBILITY_BYTES if name == "preflight" else full_record
            if record is not None:
                (cache / ".simulate-cuda-compatibility").write_bytes(record)
            return []

        monkeypatch.setattr(runner, "record_child", child)
        # The verified FLPT and solver receipts are a separate native boundary.
        monkeypatch.setattr(runner, "verify_cache", lambda *args, **kwargs: {
            "evidence": {"pressure_check": {"passed": True}}, "traces": [],
        })
        monkeypatch.setattr(runner, "verify_capillary_steps", lambda *args, **kwargs: {"passed": True})
        return runner.run_scene(tmp_path, solver, fixture, "cuda", case, pins)

    return run


@pytest.mark.parametrize(("full_record", "diagnostic"), [
    (b"FLCUDA1\n" + b"b" * 64 + b"\n", "CUDA compatibility record differs"),
    (None, "missing CUDA compatibility record"),
    (CUDA_COMPATIBILITY_BYTES.upper(), "malformed CUDA compatibility record"),
    (CUDA_COMPATIBILITY_BYTES.replace(b"\n", b"\r\n"), "malformed CUDA compatibility record"),
    (CUDA_COMPATIBILITY_BYTES[:-1], "malformed CUDA compatibility record"),
    (CUDA_COMPATIBILITY_BYTES + b"extra", "malformed CUDA compatibility record"),
], ids=("changed", "missing", "uppercase", "crlf", "missing-final-lf", "trailing-bytes"))
def test_cuda_scene_rejects_changed_or_invalid_compatibility_record(launch_static_cuda_case, full_record, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        launch_static_cuda_case(full_record)


def test_cuda_scene_retains_authenticated_probe_and_compatibility_record(launch_static_cuda_case):
    result = launch_static_cuda_case(CUDA_COMPATIBILITY_BYTES)
    assert result["passed"]
    assert result["device"] == cuda_probe()
    assert result["evidence"]["cuda_compatibility_record"] == CUDA_COMPATIBILITY_BYTES.decode("ascii")
