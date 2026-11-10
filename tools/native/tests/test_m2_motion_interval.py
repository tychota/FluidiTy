import pytest

from tools.native.m2_motion_interval import study


@pytest.mark.parametrize("fixture", ["translation", "folding_sheet"])
def test_linear_samples_converge_to_the_analytic_endpoint(fixture):
    report = study()
    rows = [row for row in report["samples"] if row["fixture"] == fixture and row["method"] == "linear"]
    assert [row["intervals"] for row in rows] == [1, 2, 4, 8, 16]
    error = [row["temporal_max_m"] for row in rows]
    assert all(coarse > fine for coarse, fine in zip(error, error[1:]))
    assert rows[-1]["total_max_m"] < report["budget_m"]
    assert all(row["integration_max_m"] < row["temporal_max_m"] for row in rows)
    assert all(row["max_rk2_displacement_cells"] <= 0.5 for row in rows)
    if fixture == "folding_sheet":
        assert rows[-2]["total_max_m"] > report["budget_m"]


def test_constant_translation_and_sample_cost():
    report = study()
    rows = [row for row in report["samples"] if row["fixture"] == "constant_translation"]
    assert len(rows) == 10
    assert max(row["total_max_m"] for row in rows) < 1e-12
    for row in rows:
        assert row["bytes_per_velocity_node"] == 12 * (row["intervals"] + 1)
        assert row["shared_timestamp_bytes"] == 8 * (row["intervals"] + 1)
