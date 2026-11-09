"""Negative controls for the exact F9 fault receipt inventory."""

import json
from pathlib import Path
import shutil
import sys

import pytest

import run_liquid_fault_campaign as campaign

from run_liquid_fault_campaign import verify_inventory


def complete_receipts():
    return [
        {
            "fault": fault, "trial_ordinal": ordinal,
            "surviving_frames": [{"frame": frame} for frame in range(2 if fault == "device_error" else 3)],
            "stop": {
                "announced_frame_ordinals": [0, 1] if fault == "device_error" else [0, 1, 2],
                "interruption_frame_ordinal": 2,
            },
            "recovered_frames": [{"frame": frame} for frame in range(11)],
            "recovered_state_digest": "a" * 64, "accepted_steps": 8,
        }
        for fault, count in (("cancel", 17), ("kill", 17), ("device_error", 16))
        for ordinal in range(1, count + 1)
    ]


def test_requires_each_fault_and_trial_ordinal_exactly_once():
    receipts = complete_receipts()
    verify_inventory(receipts)
    with pytest.raises(AssertionError):
        verify_inventory(receipts[:-1])
    with pytest.raises(AssertionError):
        verify_inventory([*receipts[:-1], receipts[-2]])
    extra = {**receipts[-1], "trial_ordinal": 17}
    with pytest.raises(AssertionError):
        verify_inventory([*receipts[:-1], extra])


@pytest.mark.parametrize(
    ("field", "incomplete_receipt_value"),
    (
        ("surviving_frames", []),
        ("surviving_frames", [{"frame": 1}, {"frame": 2}]),
        ("recovered_frames", []),
        ("accepted_steps", 0),
        ("recovered_state_digest", ""),
        ("recovered_state_digest", "z" * 64),
    ),
)
def test_rejects_trials_without_preservation_and_recovery_evidence(field, incomplete_receipt_value):
    receipts = complete_receipts()
    receipts[-1][field] = incomplete_receipt_value
    with pytest.raises(AssertionError):
        verify_inventory(receipts)


def test_trial_rejects_announced_frame_missing_before_resume(tmp_path, monkeypatch):
    seed_cache = tmp_path / "seed"
    reference_cache = tmp_path / "reference"
    trial_cache = tmp_path / "trial"
    for cache, frame_count in ((seed_cache, 2), (reference_cache, 11)):
        (cache / "frames").mkdir(parents=True)
        for frame in range(frame_count):
            (cache / "frames" / f"{frame:020d}.flmf").write_bytes(f"manifest {frame}".encode())
            (cache / f"frame-{frame}.flpt").write_bytes(f"points {frame}".encode())

    resumed = False
    final_state_digest = "a" * 64

    def stop_after_announcement(executable, arguments, fault):
        # Frame two was flushed to stdout, but its manifest was lost on stop.
        return {"exit_code": 1, "interruption_frame_ordinal": 2, "announced_frame_ordinals": [0, 1, 2]}

    def read_verified_frame(executable, command, cache_flag, cache, frame_flag, frame):
        assert (cache / "frames" / f"{frame:020d}.flmf").is_file()
        return {
            "job_status": "complete" if resumed else "crashed",
            "frame": frame, "epoch": frame * 2, "point_count": 1,
            "points_relative_name": f"frame-{frame}.flpt",
        }

    def resume_and_recreate_missing_frame(*args):
        nonlocal resumed
        resumed = True
        shutil.copytree(reference_cache, trial_cache, dirs_exist_ok=True)
        return [{"state_digest": final_state_digest, "accepted_step": 20}]

    monkeypatch.setattr(campaign, "interrupt_process", stop_after_announcement)
    monkeypatch.setattr(campaign, "invoke", read_verified_frame)
    monkeypatch.setattr(campaign, "execute", resume_and_recreate_missing_frame)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *args, **kwargs: json.dumps({
        "frame": 10, "state_digest": final_state_digest,
    }))
    with pytest.raises(AssertionError, match="announced frame 2 missing before resume"):
        campaign.trial(
            Path("simulator"), Path("fixture"), [], seed_cache, reference_cache,
            {"state_digest": final_state_digest}, trial_cache, "kill", 1,
        )
    assert not resumed


def test_stdout_history_keeps_consumed_and_unread_records_before_a_partial_tail():
    child_program = r"""
import sys
sys.stdout.buffer.write(
    b'{"protocol_version":1,"event":"frame","frame":2,"published":true}\n'
    b'{"protocol_version":1,"event":"frame","frame":3,"published":true}\n'
    b'{"event":"\xf0'
)
sys.stdout.buffer.flush()
"""
    child = campaign.RunningSimulation(sys.executable, ["-c", child_program])
    try:
        _, consumed_frame = child.next_event("frame")
        assert consumed_frame["frame"] == 2
        assert child.process.wait(timeout=10) == 0
        with pytest.raises(AssertionError, match="partial stdout record after normal exit"):
            child.stdout_records_after_exit()
        stdout_records, discarded_trailing_bytes = child.stdout_records_after_exit(allow_trailing_partial=True)
        assert campaign.announced_frame_ordinals(stdout_records) == [2, 3]
        assert discarded_trailing_bytes == len(b'{"event":"\xf0')
    finally:
        child.close()
