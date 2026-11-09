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
