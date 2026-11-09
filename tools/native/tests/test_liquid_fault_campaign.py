"""Negative controls for the exact F9 fault receipt inventory."""

import pytest

from run_liquid_fault_campaign import verify_inventory


def complete_receipts():
    return [
        {
            "fault": fault, "trial_ordinal": ordinal,
            "surviving_frames": [{"frame": 0}, {"frame": 1}],
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
