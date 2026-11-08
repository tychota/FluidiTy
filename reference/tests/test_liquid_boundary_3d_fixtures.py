"""Check the exported boundary oracle against analytic geometry and contact cases."""

import hashlib
import json

import pytest

from fluidity_reference.cli import export


@pytest.fixture(scope="module")
def exported_fixtures(tmp_path_factory):
    directory = tmp_path_factory.mktemp("boundary-fixtures")
    export(directory)
    return directory


def _read_boundary_fixture(exported_fixtures):
    path = exported_fixtures / "uniform_mac_liquid_boundary_3d.json"
    assert path.is_file(), f"Missing canonical fixture {path.name}"
    return json.loads(path.read_bytes())


def test_export_has_hashed_si_boundary_fixture(exported_fixtures):
    filename = "uniform_mac_liquid_boundary_3d.json"
    manifest = json.loads((exported_fixtures / "manifest.json").read_bytes())
    assert filename in manifest["fixtures"]
    contents = (exported_fixtures / filename).read_bytes()
    assert manifest["fixtures"][filename] == hashlib.sha256(contents).hexdigest()

    fixture = json.loads(contents)
    assert fixture["schema_version"] == 1
    assert fixture["name"] == "uniform_mac_liquid_boundary_3d"
    assert fixture["metadata"]["unit_system"] == "SI"
    fields = {
        "unit_cell_plane_offsets": ([3], "m"),
        "expected_open_face_areas": ([3], "m^2"),
        "expected_open_cell_volumes": ([3], "m^3"),
        "slab_half_widths": ([1], "m"),
        "expected_slab_blocked_face_areas": ([1], "m^2"),
        "free_surface_center_distances": ([2, 2], "m"),
        "expected_regularized_theta": ([2], "1"),
        "sphere_radii": ([1], "m"),
        "surface_tensions": ([1], "N/m"),
        "expected_sphere_curvatures": ([1], "m^-1"),
        "expected_sphere_pressure_lifts": ([1], "Pa"),
        "sweep_start_coordinates": ([3], "m"),
        "sweep_end_coordinates": ([3], "m"),
        "sweep_wall_start_centers": ([3], "m"),
        "sweep_wall_end_centers": ([3], "m"),
        "sweep_half_widths": ([3], "m"),
        "expected_first_contact_fractions": ([3], "1"),
    }
    assert fixture["arrays"].keys() == fields.keys()
    for name, (shape, units) in fields.items():
        field = fixture["arrays"][name]
        assert field["shape"] == shape
        assert field["units"] == units
        assert field["dtype"] == "float64"


def test_halfspace_preserves_tiny_open_area_and_volume(exported_fixtures):
    boundary_fixture = _read_boundary_fixture(exported_fixtures)
    arrays = boundary_fixture["arrays"]
    assert arrays["unit_cell_plane_offsets"]["data"] == [0.0, 0.25, 1.0 - 2**-20]
    assert arrays["expected_open_face_areas"]["data"] == [1.0, 0.75, 2**-20]
    assert arrays["expected_open_cell_volumes"]["data"] == [1.0, 0.75, 2**-20]


def test_centered_slab_blocks_face_with_no_corner_inside(exported_fixtures):
    boundary_fixture = _read_boundary_fixture(exported_fixtures)
    arrays = boundary_fixture["arrays"]
    assert arrays["slab_half_widths"]["data"] == [1 / 32]
    assert arrays["expected_slab_blocked_face_areas"]["data"] == [1 / 16]


def test_free_surface_regularization_preserves_larger_fractions(exported_fixtures):
    boundary_fixture = _read_boundary_fixture(exported_fixtures)
    arrays = boundary_fixture["arrays"]
    assert arrays["free_surface_center_distances"]["data"] == [[-0.01, 0.99], [-0.25, 0.75]]
    assert arrays["expected_regularized_theta"]["data"] == [0.05, 0.25]


def test_sphere_capillary_pressure_lift_is_positive(exported_fixtures):
    boundary_fixture = _read_boundary_fixture(exported_fixtures)
    arrays = boundary_fixture["arrays"]
    assert arrays["sphere_radii"]["data"] == [0.5]
    assert arrays["surface_tensions"]["data"] == [0.072]
    assert arrays["expected_sphere_curvatures"]["data"] == [4.0]
    assert arrays["expected_sphere_pressure_lifts"]["data"] == pytest.approx([0.288])


@pytest.mark.parametrize(("row", "particle_path", "wall_path", "incident_side"), [
    (0, (-1.0, 1.0), (0.0, 0.0), -1),
    (1, (0.0, 0.0), (-1.0, 1.0), 1),
])
def test_thin_slab_first_hit_uses_relative_motion(exported_fixtures, row, particle_path, wall_path, incident_side):
    boundary_fixture = _read_boundary_fixture(exported_fixtures)
    arrays = boundary_fixture["arrays"]
    assert arrays["sweep_start_coordinates"]["data"][row] == particle_path[0]
    assert arrays["sweep_end_coordinates"]["data"][row] == particle_path[1]
    assert arrays["sweep_wall_start_centers"]["data"][row] == wall_path[0]
    assert arrays["sweep_wall_end_centers"]["data"][row] == wall_path[1]
    assert arrays["sweep_half_widths"]["data"][row] == 1 / 32
    assert arrays["expected_first_contact_fractions"]["data"][row] == 31 / 64
    assert boundary_fixture["metadata"]["sweep_incident_sides"][row] == incident_side


def test_co_moving_particle_stays_outside_slab(exported_fixtures):
    boundary_fixture = _read_boundary_fixture(exported_fixtures)
    arrays = boundary_fixture["arrays"]
    row = 2
    assert arrays["sweep_start_coordinates"]["data"][row] == 0.0
    assert arrays["sweep_end_coordinates"]["data"][row] == 1.0
    assert arrays["sweep_wall_start_centers"]["data"][row] == -1.0
    assert arrays["sweep_wall_end_centers"]["data"][row] == 0.0
    assert arrays["sweep_half_widths"]["data"][row] == 1 / 32
    assert arrays["expected_first_contact_fractions"]["data"][row] == -1.0
    assert boundary_fixture["metadata"]["sweep_incident_sides"][row] == 0
