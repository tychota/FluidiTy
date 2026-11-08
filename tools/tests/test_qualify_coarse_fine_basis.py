"""Verify the reported quantities against independently specified fixture results."""

import importlib.util
import json
import subprocess
import sys
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
