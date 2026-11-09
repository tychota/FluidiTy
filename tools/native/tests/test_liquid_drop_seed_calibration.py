"""Controls for the calibrated oscillating-drop seed and native authoring seam."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_positive_octant_volume_and_moments_match_a_sphere():
    from liquid_drop_seed_calibration import measure_symmetric_union

    radius = 0.2
    volume = 4 * math.pi * radius**3 / 3
    measured = measure_symmetric_union([(0, 0, 0)], radius, radius, volume * 0.0003)
    assert measured.volume_m3 == pytest.approx(volume, rel=0.0003)
    assert measured.second_moment_m5 == pytest.approx(volume * radius**2 / 5, rel=0.0003)


def test_large_moment_scale_keeps_a_representable_physical_moment():
    from liquid_drop_seed_calibration import measure_symmetric_union

    radius = 0.2
    volume = 4 * math.pi * radius**3 / 3
    measured = measure_symmetric_union([(0, 0, 0)], radius, 1e154, 1e-5)
    assert measured.second_moment_m5 == pytest.approx(volume * radius**2 / 5, rel=0.0003)
    assert math.isfinite(measured.estimated_second_moment_error_m5)


def test_measure_rejects_an_unrepresentable_physical_moment():
    from liquid_drop_seed_calibration import measure_symmetric_union

    with pytest.raises(ValueError, match="binary64 range"):
        measure_symmetric_union([(0, 0, 0)], 1e62, 1e62, 1e183)


def test_transposed_measure_recovers_displaced_sphere_moment():
    from liquid_drop_seed_calibration import measure_symmetric_union

    radius, offset = 0.1, 0.2
    volume = 8 * math.pi * radius**3 / 3
    measure_x = measure_symmetric_union([(0, -offset, 0), (0, offset, 0)], radius, offset, volume * 0.0003)
    measure_y = measure_symmetric_union(
        [(0, -offset, 0), (0, offset, 0)], radius, offset, volume * 0.0003, axis=1,
    )
    assert measure_x.second_moment_m5 == pytest.approx(volume * radius**2 / 5, rel=0.0003)
    assert measure_y.second_moment_m5 == pytest.approx(volume * (radius**2 / 5 + offset**2), rel=0.0003)


@pytest.mark.parametrize(
    "centers,radius,tolerance,reason",
    [
        ([(0, 0, 0), (0, 0, 0)], 0.1, 1e-6, "unique"),
        ([(0, -0.2, 0)], 0.1, 1e-6, "reflections"),
        ([(0.2, 0, 0), (-0.2, 0, 0)], 0.1, 1e-6, "x/z exchange"),
        ([(0, 0, math.nan)], 0.1, 1e-6, "finite"),
        ([(0, 0, 0)], math.inf, 1e-6, "radius"),
        ([(0, 0, 0)], 0.1, 1e-100, "quadrature"),
    ],
)
def test_octant_measure_rejects_invalid_or_unresolved_inputs(centers, radius, tolerance, reason):
    from liquid_drop_seed_calibration import measure_symmetric_union

    with pytest.raises(ValueError, match=reason):
        measure_symmetric_union(centers, radius, 0.1, tolerance)


def test_bisection_rejects_unbracketed_reference_volume():
    from liquid_drop_seed_calibration import _bisect_support

    with pytest.raises(ValueError, match="bracket"):
        _bisect_support([(0, 0, 0)], 1.0, 1.0, 4 * math.pi / 3, max_iterations=14)


def test_bisection_rejects_iteration_exhaustion():
    from liquid_drop_seed_calibration import _bisect_support

    with pytest.raises(ValueError, match="converge"):
        _bisect_support([(0, 0, 0)], 1.0, 0.6, 4 * math.pi * 0.6**3 / 3, max_iterations=1)


@pytest.fixture
def native_tools():
    fixture = os.environ.get("FLUIDITY_DROP_FIXTURE")
    solver = os.environ.get("FLUIDITY_DROP_SOLVER")
    if not fixture or not solver:
        pytest.skip("set FLUIDITY_DROP_FIXTURE and FLUIDITY_DROP_SOLVER to run native authoring controls")
    return fixture, solver


def native_command(fixture, *arguments, success=True):
    result = subprocess.run([fixture, *map(str, arguments)], capture_output=True, text=True, timeout=30)
    if success:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0
    assert result.stdout == ""
    return result.stderr


def test_native_drop_seed_roundtrips_calibrated_binary64_radius(native_tools, tmp_path):
    fixture, solver = native_tools
    preview = tmp_path / "seed.flst"
    preview_identity = native_command(fixture, "write-drop-seed", preview)
    digest = preview_identity["seed_digest"]
    seed = native_command(fixture, "read-seed", preview, digest)
    assert [particle["id"] for particle in seed["particles"]] == list(range(1, 553))

    from liquid_drop_seed_calibration import calibrate_drop_seed

    calibration = calibrate_drop_seed(seed, digest)
    assert calibration.support_radius_m.hex() == "0x1.0d70a3d70a3d8p-6"
    assert abs(calibration.refined_quadrupole_m2 / 0.000281331917475728 - 1) <= 0.10

    scene = tmp_path / "drop.flrs"
    authored = native_command(
        fixture, "write-scene", "oscillating-drop-l2", "cpu", solver, scene, repr(calibration.support_radius_m),
    )
    resolved = native_command(fixture, "read-drop-scene", scene, authored["resolved_fingerprint"])
    for key in ("resolved_fingerprint", "execution_id", "backend", "solver_digest", "seed_digest", "seed_byte_length"):
        assert resolved[key] == authored[key]
    assert resolved["backend"] == "cpu"
    assert resolved["sphere_union_radius_m"].hex() == calibration.support_radius_m.hex()
    assert resolved["cell_size_m"] == 0.025
    assert resolved["world_origin_m"] == [0, 0, 0]
    assert resolved["gravity_m_per_s2"] == [0, 0, 0]
    assert resolved["surface_tension_n_per_m"] == 1
    assert resolved["density_kg_per_m3"] == 1000
    assert resolved["source_count"] == 0
    assert resolved["maximum_dt_seconds"] == 0.01
    assert authored["seed_digest"] == digest
    assert scene.with_suffix(".flrs.flst").read_bytes() == preview.read_bytes()

    exponent_scene = tmp_path / "drop-exponent.flrs"
    exponent_authored = native_command(
        fixture, "write-scene", "oscillating-drop-l2", "cpu", solver, exponent_scene,
        format(calibration.support_radius_m, ".16e"),
    )
    assert exponent_authored["resolved_fingerprint"] == authored["resolved_fingerprint"]

    assert "fingerprint" in native_command(fixture, "read-drop-scene", scene, "0" * 64, success=False)


@pytest.mark.parametrize(
    "radius",
    [(), ("nan",), ("0.024",), ("0.016junk",), ("+0.016",), (" 0.016",), ("0.016 ",), ("0x1p-6",)],
)
def test_native_drop_rejects_missing_or_invalid_support_radius(native_tools, tmp_path, radius):
    fixture, solver = native_tools
    scene = tmp_path / "drop.flrs"
    assert "support radius" in native_command(
        fixture, "write-scene", "oscillating-drop-l2", "cpu", solver, scene, *radius, success=False,
    )
    assert not list(tmp_path.iterdir())


def test_native_drop_receipt_rejects_non_drop_geometry(native_tools, tmp_path):
    fixture, solver = native_tools
    scene = tmp_path / "free-fall.flrs"
    authored = native_command(fixture, "write-scene", "free-fall-default", "cpu", solver, scene)
    assert "sphere union" in native_command(
        fixture, "read-drop-scene", scene, authored["resolved_fingerprint"], success=False,
    )
