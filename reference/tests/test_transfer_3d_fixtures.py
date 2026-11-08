"""Independent checks on the published staggered transfer oracle."""

import hashlib
import json
from fractions import Fraction

from fluidity_reference.cli import export


def test_export_includes_hashed_si_transfer_fixture(tmp_path):
    export(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert set(manifest["fixtures"]) == {
        "affine_apic.json", "compliant_pressure.json", "stiff_viscosity.json",
        "two_cell_pressure.json", "uniform_mac_transfer_3d.json",
    }
    filename = "uniform_mac_transfer_3d.json"
    contents = (tmp_path / filename).read_bytes()
    assert manifest["fixtures"][filename] == hashlib.sha256(contents).hexdigest()
    fixture = json.loads(contents)
    assert fixture["name"] == "uniform_mac_transfer_3d"
    assert fixture["metadata"]["unit_system"] == "SI"
    assert fixture["metadata"]["provenance"]
    assert fixture["metadata"]["tolerances"] == {"atol": 2e-12, "rtol": 2e-12}
    arrays = fixture["arrays"]
    assert arrays["face_indices"]["shape"] == [81, 4]
    assert arrays["face_indices"]["dtype"] == "int64"
    assert arrays["face_displacements"]["units"] == "m"
    assert arrays["weight_gradients"]["units"] == "m^-1"
    assert arrays["expected_mass"]["units"] == "kg"
    assert arrays["expected_pure_momentum"]["units"] == "kg*m/s"


def test_exported_staggered_rows_have_exact_mass_momentum_and_spin(tmp_path):
    export(tmp_path)
    fixture_path = tmp_path / "uniform_mac_transfer_3d.json"
    assert fixture_path.is_file()
    arrays = json.loads(fixture_path.read_bytes())["arrays"]
    position = tuple(map(Fraction, arrays["position"]["data"]))
    assert position == (Fraction(15, 4), Fraction(-1, 4), Fraction(15, 4))
    rows = arrays["affine_velocity_fit"]["data"]
    velocities = arrays["velocity"]["data"]
    assert rows == [[2, 4, -1], [-5, 1, 5], [7, -2, 3]]
    assert velocities == [1, -2, 4]

    face_indices = arrays["face_indices"]["data"]
    assert len({tuple(face) for face in face_indices}) == 81
    assert any(lattice_y < 0 for _, _, lattice_y, _ in face_indices)
    mass_sums = [Fraction(0)] * 3
    momentum_sums = [Fraction(0)] * 3
    row_moments = [[Fraction(0)] * 3 for _ in range(3)]
    wrong_row_moments = [[Fraction(0)] * 3 for _ in range(3)]
    gradient_sums = [[Fraction(0)] * 3 for _ in range(3)]

    for face, offsets, weight, gradient, mass, momentum in zip(
        face_indices, arrays["face_displacements"]["data"], arrays["weights"]["data"],
        arrays["weight_gradients"]["data"], arrays["expected_mass"]["data"],
        arrays["expected_pure_momentum"]["data"], strict=True,
    ):
        axis, *lattice = face
        displacement = [
            Fraction(node) + (0 if coordinate == axis else Fraction(1, 2)) - position[coordinate]
            for coordinate, node in enumerate(lattice)
        ]
        assert displacement == list(map(Fraction, offsets))
        mass = Fraction(mass)
        momentum = Fraction(momentum)
        assert mass == 2 * Fraction(weight)
        mass_sums[axis] += mass
        momentum_sums[axis] += momentum

        wrong_row = rows[(axis + 1) % 3]
        wrong_momentum = mass * (
            velocities[axis] + sum(slope * delta for slope, delta in zip(wrong_row, displacement))
        )
        for coordinate, delta in enumerate(displacement):
            row_moments[axis][coordinate] += momentum * delta
            wrong_row_moments[axis][coordinate] += wrong_momentum * delta
            gradient_sums[axis][coordinate] += Fraction(gradient[coordinate])

    assert mass_sums == [2, 2, 2]
    assert momentum_sums == [2, -4, 8]
    assert gradient_sums == [[0, 0, 0]] * 3
    assert row_moments == [[Fraction(slope, 2) for slope in row] for row in ([2, 4, -1], [-5, 1, 5], [7, -2, 3])]
    spin = (
        row_moments[2][1] - row_moments[1][2],
        row_moments[0][2] - row_moments[2][0],
        row_moments[1][0] - row_moments[0][1],
    )
    assert spin == (Fraction(-7, 2), Fraction(-4), Fraction(-9, 2))
    assert wrong_row_moments[0] == [Fraction(-5, 2), Fraction(1, 2), Fraction(5, 2)]
    assert wrong_row_moments[0] != row_moments[0]


def test_quadratic_exact_basis_hypotheses_at_signed_boundaries():
    """Removing a tail, flipping derivative sign or shifting anchors breaks F1."""
    from fluidity_reference.transfer_3d_fixtures import _centered_quadratic

    for position in map(Fraction, ('-4.25', '-4', '-0.5', '0', '0.5', '3.75', '4', '5.5')):
        for shift in (Fraction(0), Fraction(1, 2)):
            weights = []
            offsets = []
            derivatives = []
            for node in range(-8, 10):
                displacement = Fraction(node) + shift - position
                weight, derivative = _centered_quadratic(-displacement)
                weights.append(weight)
                offsets.append(displacement)
                derivatives.append(derivative)
            assert sum(weights) == 1
            assert sum(w * r for w, r in zip(weights, offsets)) == 0
            assert sum(w * r**2 for w, r in zip(weights, offsets)) == Fraction(1, 4)
            assert sum(derivatives) == 0
            assert sum(r * g for r, g in zip(offsets, derivatives)) == 1
            # Exact central differences independently estimate each analytic derivative.
            step = Fraction(1, 2**20)
            for r, derivative in zip(offsets, derivatives):
                forward, _ = _centered_quadratic(-r + step)
                backward, _ = _centered_quadratic(-r - step)
                # At the inner knots the curvature jump is 3: central error is 3*step/4.
                assert abs((forward - backward) / (2 * step) - derivative) <= 3 * step / 4
