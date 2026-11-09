"""Independent controls for the real free-fall acceptance measurements."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquid_acceptance_metrics import evaluate_free_fall, uniform_history_velocities
from run_liquid_acceptance import matches_identity, read_records, run_acceptance, verify_device


def frames():
    return [
        {
            "frame": ordinal,
            "time_seconds": ordinal / 24,
            "points": [{
                "id": 1,
                "world_position": [2.5, 2.5 - 0.5 * 9.81 * (ordinal / 24) ** 2, 2.5],
                "world_velocity": [0, -9.81 * (ordinal / 24), 0],
            }],
        }
        for ordinal in range(7)
    ]


def test_independent_ballistic_samples():
    result = evaluate_free_fall(frames())
    assert result["maximum_relative_acceleration_error"] < 1e-14
    assert result["maximum_transverse_speed_m_per_s"] == 0
    assert result["world_displacement_m"] == pytest.approx([0, -0.3065625, 0])
    assert len(result["intervals"]) == 6
    assert result["passed"]


@pytest.mark.parametrize("corruption", ["missing", "wrong-ordinal", "wrong-clock", "duplicate-id", "missing-id", "nan"])
def test_incomplete_or_invalid_samples(corruption):
    samples = frames()
    if corruption == "missing":
        samples.pop(3)
    elif corruption == "wrong-ordinal":
        samples[3]["frame"] = 2
    elif corruption == "wrong-clock":
        samples[3]["time_seconds"] = samples[2]["time_seconds"]
    elif corruption == "duplicate-id":
        samples[3]["points"].append(copy.deepcopy(samples[3]["points"][0]))
    elif corruption == "missing-id":
        samples[3]["points"][0]["id"] = 2
    else:
        samples[3]["points"][0]["world_velocity"][1] = float("nan")
    with pytest.raises(ValueError):
        evaluate_free_fall(samples)


def test_acceleration_error_cannot_hide_in_final_velocity():
    samples = frames()
    samples[3]["points"][0]["world_velocity"][1] += 0.001
    result = evaluate_free_fall(samples)
    assert result["maximum_relative_acceleration_error"] > 0.001
    assert not result["passed"]


def test_transverse_error_is_independent_of_gravity_axis():
    samples = frames()
    samples[3]["points"][0]["world_velocity"][0] = 1.01e-10
    result = evaluate_free_fall(samples)
    assert result["maximum_transverse_speed_m_per_s"] == 1.01e-10
    assert not result["passed"]


@pytest.mark.parametrize("energy", [0, 0.5, 0.6, 1])
def test_uniform_history_endpoints_and_interior(energy):
    assert uniform_history_velocities(energy, 3.0, 3.0, 2.0) == (2.0, 2.0)


def test_missing_uniform_history():
    with pytest.raises(ValueError, match="old history"):
        uniform_history_velocities(0.6, 3.0, None, 2.0)
    # An inconsistent old history makes the retained correction nonzero.
    assert uniform_history_velocities(1, 3.0, 2.0, 2.0) == pytest.approx((2.99, 2.99))



def test_receipt_identity_is_external_to_the_cache():
    expected = {"resolved_fingerprint": "a" * 64, "execution_id": "b" * 64, "backend": "cpu"}
    for field, wrong_value in (
        ("resolved_fingerprint", "c" * 64), ("execution_id", "d" * 64), ("backend", "cuda")
    ):
        with pytest.raises(ValueError, match=field):
            matches_identity({**expected, field: wrong_value, "event": "frame"}, expected)


@pytest.mark.parametrize("stdout", [
    '{"protocol_version":1,"event":"result"}',  # Incomplete final line.
    '{"protocol_version":1,"event":"result","frame":1,"frame":2}\n',
    '{"protocol_version":true,"event":"result"}\n',
    '{"protocol_version":1,"event":"result","time_seconds":NaN}\n',
    '{"protocol_version":1,"event":"result","time_seconds":1e999}\n',
])
def test_malformed_receipt_is_rejected(stdout):
    with pytest.raises(ValueError):
        read_records(stdout, "protocol_version")


def test_absent_cuda_cannot_be_qualified_as_a_cpu_fallback():
    with pytest.raises(ValueError, match="real CUDA device unavailable"):
        verify_device({
            "command": "device-info", "backend": "cuda", "cpu_available": True,
            "cuda_compiled": True, "cuda_available": False,
        }, "cuda")


def test_stale_output_is_rejected_before_launch(tmp_path):
    output = tmp_path / "stale"
    output.mkdir()
    stale_receipt = output / "free-fall-measurements.json"
    stale_receipt.write_text('{"passed":true}', encoding="utf-8")
    missing_program = tmp_path / "should-never-launch"
    with pytest.raises(FileExistsError):
        run_acceptance(missing_program, missing_program, "cpu", "free-fall-pic", output)
    assert stale_receipt.read_text(encoding="utf-8") == '{"passed":true}'
