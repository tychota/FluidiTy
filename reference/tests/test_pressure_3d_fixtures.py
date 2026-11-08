"""Check the exported 3D oracle against hand-calculated pressure systems."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from fluidity_reference.cli import export
from fluidity_reference.pressure_3d_fixtures import (
    _PressureCell,
    _PressureFace,
    _assemble_system,
    _evaluate_correction,
    _make_case,
)


@pytest.fixture(scope="module")
def exported_fixtures(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pressure-fixtures")
    export(directory)
    return directory


def _read_pressure_fixture(directory):
    filename = "uniform_mac_pressure_3d.json"
    manifest = json.loads((directory / "manifest.json").read_bytes())
    assert filename in manifest["fixtures"]
    contents = (directory / filename).read_bytes()
    assert manifest["fixtures"][filename] == hashlib.sha256(contents).hexdigest()
    return json.loads(contents)


def _case_arrays(fixture, name):
    case_number = fixture["metadata"]["case_names"].index(name)
    arrays = fixture["arrays"]
    return {
        group: slice(*arrays[f"case_{group}_offsets"]["data"][case_number:case_number + 2])
        for group in ("cell", "face", "gauge")
    }


def _check_solution(fixture, name, cells, pressures, rhs, gauges, moving_faces):
    ranges = _case_arrays(fixture, name)
    arrays = {name: field["data"] for name, field in fixture["arrays"].items()}
    assert arrays["cell_indices"][ranges["cell"]] == [list(index) for index in cells]
    np.testing.assert_allclose(arrays["expected_cell_pressures"][ranges["cell"]], pressures, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(arrays["expected_integrated_rhs"][ranges["cell"]], rhs, rtol=0.0, atol=1e-14)
    assert arrays["expected_gauge_cell_ordinals"][ranges["gauge"]] == gauges
    faces = [tuple(index) for index in arrays["face_indices"][ranges["face"]]]
    assert set(moving_faces) <= set(faces)
    expected_velocities = [moving_faces.get(face, 0.0) for face in faces]
    np.testing.assert_allclose(arrays["expected_corrected_face_velocities"][ranges["face"]],
                               expected_velocities, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(arrays["expected_true_residuals"][ranges["cell"]], 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(arrays["expected_cell_constraint_defects"][ranges["cell"]], 0.0, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(("axis", "tail"), [
    ("x", (-1, 0, 0)), ("y", (0, -1, 0)), ("z", (0, 0, -1)),
])
def test_signed_block_seam_jump_sets_relative_pressure(exported_fixtures, axis, tail):
    pressure_fixture = _read_pressure_fixture(exported_fixtures)
    _check_solution(pressure_fixture, f"jump_across_{axis}_block", [tail, (0, 0, 0)],
                    [0, 1], [-1, 1], [0], {})
    indices = _case_arrays(pressure_fixture, f"jump_across_{axis}_block")
    arrays = pressure_fixture["arrays"]
    assert arrays["face_link_kinds"]["data"][indices["face"]].count(1) == 1
    axis_ordinal = "xyz".index(axis)
    faces = arrays["face_indices"]["data"][indices["face"]]
    interface = faces.index([axis_ordinal, 0, 0, 0])
    assert arrays["face_pressure_lifts"]["data"][indices["face"]][interface] == 1


@pytest.mark.parametrize(("side", "face", "pressure"), [
    ("low", (0, 0, 0, 0), 4.5), ("high", (0, 1, 0, 0), 1.5),
])
def test_partial_dirichlet_includes_signed_moving_wall_flux(exported_fixtures, side, face, pressure):
    pressure_fixture = _read_pressure_fixture(exported_fixtures)
    _check_solution(pressure_fixture, f"{side}_dirichlet", [(0, 0, 0)], [pressure], [pressure], [], {face: -2})


def test_zero_open_area_preserves_velocity_and_gauges_each_cell(exported_fixtures):
    pressure_fixture = _read_pressure_fixture(exported_fixtures)
    _check_solution(pressure_fixture, "disconnected_neumann", [(-1, 0, 0), (0, 0, 0)],
                    [0, 0], [0, 0], [0, 1], {(0, 0, 0, 0): 0.75})
    arrays = pressure_fixture["arrays"]
    face_range = _case_arrays(pressure_fixture, "disconnected_neumann")["face"]
    interface = arrays["face_indices"]["data"][face_range].index([0, 0, 0, 0])
    assert arrays["face_open_areas"]["data"][face_range][interface] == 0
    assert arrays["wall_face_velocities"]["data"][face_range][interface] == 0.25


def test_isolated_compliance_keeps_physical_mean(exported_fixtures):
    pressure_fixture = _read_pressure_fixture(exported_fixtures)
    _check_solution(pressure_fixture, "isolated_compliance", [(0, 0, 0)], [5], [2.5], [], {})
    arrays = pressure_fixture["arrays"]
    assert 0 in arrays["cell_bulk_moduli"]["data"]
    cell_range = _case_arrays(pressure_fixture, "isolated_compliance")["cell"]
    assert arrays["cell_bulk_moduli"]["data"][cell_range] == [2]


def test_mixed_components_preserve_local_gauges_and_anchors(exported_fixtures):
    pressure_fixture = _read_pressure_fixture(exported_fixtures)
    cells = [
        (8, 0, -1), (4, -1, 0), (-1, 0, 0), (0, 0, 0), (4, 0, 0), (8, 0, 0),
        (12, 0, 0), (16, 0, 0), (19, 0, 0), (20, 0, 0), (24, 0, 0),
    ]
    _check_solution(
        pressure_fixture, "mixed", cells,
        [0, 0, 0, 1, 1, 1, 4.5, 1.5, 0, 0, 5], [-1, -1, -1, 1, 1, 1, 4.5, 1.5, 0, 0, 2.5],
        [0, 1, 2, 8, 9], {(0, 12, 0, 0): -2, (0, 17, 0, 0): -2, (0, 20, 0, 0): 0.75},
    )


def test_export_shapes_units_and_signed_owned_face_order(exported_fixtures):
    pressure_fixture = _read_pressure_fixture(exported_fixtures)
    assert pressure_fixture["schema_version"] == 1
    metadata = pressure_fixture["metadata"]
    assert metadata["unit_system"] == "SI"
    assert len(metadata["case_names"]) == len(set(metadata["case_names"])) == 8
    arrays = pressure_fixture["arrays"]
    groups = {
        "cell": {
            "cell_indices": ("1", [3]), "cell_fluid_volumes": ("m^3", []),
            "target_cell_divergences": ("s^-1", []), "cell_bulk_moduli": ("Pa", []),
            "previous_cell_pressures": ("Pa", []), "expected_cell_pressures": ("Pa", []),
            "expected_integrated_rhs": ("m^3/s", []), "expected_true_residuals": ("m^3/s", []),
            "expected_cell_constraint_defects": ("s^-1", []),
        },
        "face": {
            "face_indices": ("1", [4]), "face_link_kinds": ("1", []), "face_open_areas": ("m^2", []),
            "face_inverse_densities": ("m^3/kg", []), "face_pressure_gradient_distances": ("m", []),
            "predicted_face_velocities": ("m/s", []), "wall_face_velocities": ("m/s", []),
            "face_pressure_lifts": ("Pa", []), "expected_corrected_face_velocities": ("m/s", []),
        },
        "gauge": {"expected_gauge_cell_ordinals": ("1", [])},
    }
    integer_arrays = {"cell_indices", "face_indices", "face_link_kinds", "expected_gauge_cell_ordinals"}
    for group, fields in groups.items():
        offsets = arrays[f"case_{group}_offsets"]
        assert offsets["shape"] == [9]
        assert offsets["units"] == "1"
        assert offsets["dtype"] == "int64"
        assert offsets["data"][0] == 0
        assert offsets["data"] == sorted(offsets["data"])
        for field, (units, trailing_shape) in fields.items():
            assert arrays[field]["units"] == units
            assert arrays[field]["shape"] == [offsets["data"][-1], *trailing_shape]
            assert arrays[field]["dtype"] == ("int64" if field in integer_arrays else "float64")
    assert arrays["step_durations"] == {"data": [1.0] * 8, "dtype": "float64", "shape": [8], "units": "s"}
    for name in metadata["case_names"]:
        ranges = _case_arrays(pressure_fixture, name)
        cells = arrays["cell_indices"]["data"][ranges["cell"]]
        faces = arrays["face_indices"]["data"][ranges["face"]]
        assert cells == sorted(cells, key=lambda index: index[::-1])
        expected_faces = set()
        for cell in cells:
            for normal in range(3):
                expected_faces.add((normal, *cell))
                high = cell.copy()
                high[normal] += 1
                expected_faces.add((normal, *high))
        assert set(map(tuple, faces)) == expected_faces
        assert len(faces) == len(expected_faces)
        slot_order = [
            (z // 4, y // 4, x // 4, normal, (z % 4) * 16 + (y % 4) * 4 + x % 4)
            for normal, x, y, z in faces
        ]
        assert slot_order == sorted(slot_order)


def test_incompatible_components_cannot_cancel_across_closed_face():
    closed_face = _PressureFace((0, 0, 0, 0), link_kind=1, wall_velocity=0.25)
    case = _make_case([_PressureCell((-1, 0, 0)), _PressureCell((0, 0, 0))], [closed_face])
    with pytest.raises(ValueError, match="incompatible"):
        _assemble_system(case)


def test_mismatched_correction_scale_breaks_the_signed_f6_identity():
    interface = _PressureFace(
        (0, 0, 0, 0),
        link_kind=1,
        open_area=1.0,
        inverse_density=1.0,
        pressure_gradient_distance=1.0,
        pressure_lift=1.0,
    )
    cells = [_PressureCell((-1, 0, 0)), _PressureCell((0, 0, 0))]
    original = _make_case(cells, [interface])
    matrix, rhs, gauges = _assemble_system(original)
    assert gauges == [0]
    candidate = np.array([0.0, 0.75])
    residuals = rhs - matrix @ candidate
    np.testing.assert_allclose(residuals, [-0.25, 0.25], rtol=0.0, atol=1e-14)

    velocities, defects = _evaluate_correction(original, candidate)
    interface_ordinal = [face.index for face in original.faces].index(interface.index)
    assert velocities[interface_ordinal] == pytest.approx(0.25)
    np.testing.assert_allclose(defects, [0.25, -0.25], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(defects + residuals, 0.0, rtol=0.0, atol=1e-14)

    mismatched = _make_case(cells, [replace(interface, inverse_density=2.0)])
    velocities, defects = _evaluate_correction(mismatched, candidate)
    assert velocities[interface_ordinal] == pytest.approx(0.5)
    np.testing.assert_allclose(defects, [0.5, -0.5], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(defects + residuals, [0.25, -0.25], rtol=0.0, atol=1e-14)


def test_tiny_cell_fails_local_gate_despite_passing_global_floor():
    case = _make_case([_PressureCell((0, 0, 0), fluid_volume=1e-6)], [
        _PressureFace((0, 1, 0, 0), link_kind=2, open_area=1.0, inverse_density=1.0, pressure_gradient_distance=1.0),
    ])
    matrix, rhs, gauges = _assemble_system(case)
    assert gauges == []
    candidate = np.array([1e-7])
    residuals = rhs - matrix @ candidate
    _, defects = _evaluate_correction(case, candidate)
    np.testing.assert_allclose(residuals, [-1e-7], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(defects, [0.1], rtol=0.0, atol=1e-14)
    assert np.linalg.norm(residuals) < 2e-6 + 2e-6 * np.linalg.norm(rhs)
    assert abs(residuals[0]) / 1e-6 > 2e-5
    assert abs(defects[0]) > 2e-5
