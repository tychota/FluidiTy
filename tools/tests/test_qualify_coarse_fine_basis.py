"""Verify the reported quantities against independently specified fixture results."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fluidity_reference.experiments.coarse_fine_basis import BasisStencil

SCRIPT = Path(__file__).parents[1] / "adaptivity" / "qualify_coarse_fine_basis.py"
SPEC = importlib.util.spec_from_file_location("qualify_coarse_fine_basis", SCRIPT)
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_moment_defects_use_world_offsets_and_physical_gradients(dtype):
    stencil = BasisStencil(
        anchor_world_positions=np.array([[4, -3]], dtype=dtype),
        weights=np.array([0.75], dtype=dtype),
        physical_gradients=np.array([[0.25, -0.5]], dtype=dtype),
        levels=np.array([0], dtype=np.int8),
    )
    defects = qualification.measure_moment_defects(stencil, np.array([3, -2], dtype=dtype), dtype(2))

    assert defects == {"partition": 0.25, "first_moment": 0.375, "gradient_sum": 1, "affine_gradient": 0.75}
    assert all(value.dtype == dtype for value in defects.values())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("interface_axis", "expected_normal"), [(0, [1, 0]), (1, [0, 1])])
def test_pressure_fixture_uses_two_owned_subfaces(dtype, interface_axis, expected_normal):
    result = qualification.measure_pressure_fixture(dtype, interface_axis)

    assert result["status"] == "pass"
    assert result["dtype"] == np.dtype(dtype).name
    assert result["normal"] == expected_normal
    assert result["cell_areas_m2"] == [1, 0.25, 0.25]
    assert result["subface_lengths_m"] == [0.5, 0.5]
    assert result["normal_center_separations_m"] == [0.75, 0.75]
    assert result["incidence"] == [[1, 1], [-1, 0], [0, -1]]
    assert result["predicted_subface_fluxes_m2_per_s"] == [0.375, -0.125]
    assert result["predicted_cell_fluxes_m2_per_s"] == [0.25, -0.375, 0.125]
    assert result["predicted_divergence_per_s"] == [0.25, -1.5, 0.5]
    assert result["coarse_aggregate_flux_m2_per_s"] == 0.25
    np.testing.assert_allclose(
        result["integrated_pressure_matrix"],
        np.array([[4, -2, -2], [-2, 2, 0], [-2, 0, 2]]) / 3,
        rtol=1e-7,
        atol=0,
    )
    np.testing.assert_allclose(result["gauge_fixed_pressure"], [0, 0.5625, -0.1875], rtol=1e-7, atol=0)
    assert result["constant_pressure_relative_residual_floor"] == 1e-12
    assert result["zero_pressure_residual_denominator"] == float(dtype(1e-12))
    assert result["constant_pressure_relative_residual"] <= 1e-6
    assert result["max_corrected_divergence_per_s"] <= 1e-6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mask_fixtures_measure_raw_defect_and_classify_affine_support(dtype):
    masked, deficient = qualification.measure_masked_support_fixtures(dtype)

    assert masked["status"] == deficient["status"] == "pass"
    assert masked["dtype"] == deficient["dtype"] == np.dtype(dtype).name
    assert masked["original_level_counts"] == deficient["original_level_counts"] == [9, 3]
    assert masked["retained_count"] == 11
    assert masked["raw_partition_defect"] > 0.2
    assert masked["raw_partition_defect"] == pytest.approx(masked["removed_weight"], abs=1e-6)
    assert masked["affine_rank"] == 2
    assert masked["disposition"] == "pic_required_incomplete_support"
    assert deficient["retained_count"] == 3
    assert deficient["raw_partition_defect"] > 0.2
    assert deficient["affine_rank"] == 1
    assert deficient["disposition"] == "pic_required_rank_deficient_support"
    assert masked["affine_claim"] is deficient["affine_claim"] is False


@pytest.mark.parametrize("face_normal_axis", [0, 1])
def test_quadratic_nesting_probe_detects_coarse_curvature_jump(face_normal_axis):
    result = qualification.measure_quadratic_nesting(face_normal_axis)

    assert result["status"] == "pass"
    assert result["dtype"] == "float64"
    assert result["coarse_knot_over_width"] == 0.5
    assert result["adjacent_fine_knots_over_width"] == [0.25, 0.75]
    assert result["coarse_one_sided_curvatures_times_width_squared"] == [-2, 1]
    assert result["fine_one_sided_curvatures_times_width_squared"] == [-8, -8]
    assert result["nested"] is False


def test_cli_reports_the_same_timed_queries_for_both_evaluators(tmp_path):
    report_path = tmp_path / "qualification.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(report_path), "--repeats", "1"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert {fixture["dtype"] for fixture in report["uniform_cubic_normal"]} == {"float32", "float64"}
    assert {fixture["dtype"] for fixture in report["coarse_fine"]} == {"float32", "float64"}
    assert [fixture["dtype"] for fixture in report["uniform_quadratic"]] == ["float64"]
    assert len(report["pressure"]) == 4
    assert len(report["masked_support"]) == 4
    assert len(report["quadratic_nesting"]) == 2
    for uniform, transition, timing in zip(
        report["uniform_cubic_normal"], report["coarse_fine"], report["timing"]["measurements"], strict=True
    ):
        assert uniform["status"] == transition["status"] == "pass"
        assert uniform["dtype"] == transition["dtype"] == timing["dtype"]
        assert len(uniform["samples"]) == len(transition["samples"]) == timing["queries_per_repeat"] == 48
        assert timing["repeats"] == 1
        assert timing["uniform_cubic_normal_median_seconds"] > 0
        assert timing["coarse_fine_median_seconds"] > 0
        assert timing["coarse_fine_over_uniform_ratio"] == pytest.approx(
            timing["coarse_fine_median_seconds"] / timing["uniform_cubic_normal_median_seconds"]
        )
        for uniform_sample, transition_sample in zip(uniform["samples"], transition["samples"], strict=True):
            assert uniform_sample["particle_world_position"] == transition_sample["particle_world_position"]
            assert transition_sample["status"] == uniform_sample["status"] == "pass"


@pytest.mark.parametrize(
    ("nonfinite_weight", "collinear_metric"),
    [
        pytest.param(np.nan, "removed_weight", id="nan-removed"),
        pytest.param(np.inf, "removed_weight", id="inf-removed"),
        pytest.param(np.inf, "raw_partition_defect", id="inf-retained"),
    ],
)
def test_cli_replaces_passing_receipt_when_stencil_weight_is_nonfinite(
    tmp_path, monkeypatch, nonfinite_weight, collinear_metric
):
    report_path = tmp_path / "qualification.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(report_path), "--repeats", "1"])
    assert qualification.main() == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "pass"

    evaluate = qualification.evaluate_coarse_fine_basis

    def evaluate_with_nonfinite_weight(particle_world_position, **kwargs):
        stencil = evaluate(particle_world_position, **kwargs)
        if (
            kwargs["dtype"] is np.float32
            and kwargs["face_normal_axis"] == kwargs["interface_axis"] == 0
            and np.array_equal(particle_world_position, [-3.0625, -7.34375])
        ):
            weights = stencil.weights.copy()
            weight_index = 0
            if collinear_metric == "raw_partition_defect":
                interface_x = kwargs["interface_origin"][0]
                weight_index = np.flatnonzero(stencil.anchor_world_positions[:, 0] == interface_x)[0]
            weights[weight_index] = nonfinite_weight
            return replace(stencil, weights=weights)
        return stencil

    monkeypatch.setattr(qualification, "evaluate_coarse_fine_basis", evaluate_with_nonfinite_weight)
    # The injected infinity can encounter zero in the moment reduction.
    with np.errstate(invalid="ignore"):
        assert qualification.main() == 1

    def reject_nonfinite_json(token):
        pytest.fail(f"Invalid JSON number {token}")

    report = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite_json)
    assert report["status"] == "fail"
    assert report["uniform_cubic_normal"][0]["status"] == "pass"
    transition = report["coarse_fine"][0]
    assert transition["status"] == "fail"
    assert transition["max_defects"]["partition"] is None
    failed_samples = [
        (index, sample) for index, sample in enumerate(transition["samples"]) if sample["status"] == "fail"
    ]
    assert len(failed_samples) == 1
    sample_index, sample = failed_samples[0]
    assert sample["sample_name"] == "mixed_support"
    assert sample["defects"]["partition"] is None
    assert sample["defects"]["first_moment"] is None
    assert sample["defects"]["affine_gradient"] is not None
    assert f"coarse_fine[0].samples[{sample_index}].defects.partition" in report["nonfinite_metrics"]
    assert "coarse_fine[0].max_defects.partition" in report["nonfinite_metrics"]
    for fixture_index, metric in enumerate(("removed_weight", collinear_metric)):
        fixture = report["masked_support"][fixture_index]
        assert fixture["status"] == "fail"
        assert fixture[metric] is None
        assert f"masked_support[{fixture_index}].{metric}" in report["nonfinite_metrics"]
