import pytest

from fluidity_reference.grid import Grid2D
from fluidity_reference.multiphase import InterfacePair, mix_constituent_volumes
from fluidity_reference.pressure import PressureInputs, solve_pressure


def test_two_aqueous_constituents_mix_by_volume_and_oil_stays_separate():
    aqueous = mix_constituent_volumes({14: 3.0, 8: 1.0}, {8: 1000.0, 14: 1020.0})
    oil = mix_constituent_volumes({25: 2.0}, {25: 820.0})
    assert list(aqueous.volume_fractions) == [8, 14]
    assert aqueous.volume_fractions == {8: 0.25, 14: 0.75}
    assert aqueous.volume == 4.0
    assert aqueous.mass == 4060.0
    assert aqueous.density == 1015.0
    assert oil.volume_fractions == {25: 1.0}


def test_mixture_rejects_negative_volume_and_missing_density():
    with pytest.raises(ValueError, match='volume'):
        mix_constituent_volumes({8: -1.0, 14: 3.0}, {8: 1000.0, 14: 1020.0})
    with pytest.raises(ValueError, match='density'):
        mix_constituent_volumes({8: 1.0, 14: 3.0}, {8: 1000.0})


@pytest.mark.parametrize('reverse', [False, True])
def test_shared_pair_pressure_jump_is_applied_once_with_oriented_sign(reverse):
    pair = InterfacePair(phase_a=3, phase_b=9, surface_tension=0.072)
    densities = {3: 1015.0, 9: 820.0}
    tail_phase, tail_fraction = (9, 0.75) if reverse else (3, 0.25)
    link = pair.pressure_link(tail_phase, tail_fraction, densities, curvature_a_to_b=10.0)
    assert link.mobility == pytest.approx(1.0 / (0.25 * 1015.0 + 0.75 * 820.0))
    expected_jump = 0.72 if reverse else -0.72
    assert link.jump == pytest.approx(expected_jump)

    grid = Grid2D(2, 1, 1.0)
    result = solve_pressure(PressureInputs(
        grid=grid,
        dt=0.02,
        velocity=[0.0] * grid.face_count,
        aperture=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mobility=[1.0, link.mobility, 1.0, 1.0, 1.0, 1.0, 1.0],
        jump=[0.0, link.jump, 0.0, 0.0, 0.0, 0.0, 0.0],
    ))
    assert result.pressure[1] - result.pressure[0] == pytest.approx(expected_jump)
    assert result.velocity[1] == pytest.approx(0.0, abs=1e-14)
    assert max(abs(result.constraint_defect)) < 1e-14


def test_normal_viscous_traction_jump_uses_same_pair_orientation():
    pair = InterfacePair(phase_a=3, phase_b=9, surface_tension=0.0)
    densities = {3: 1000.0, 9: 820.0}
    forward = pair.pressure_link(3, 0.5, densities, 0.0, normal_viscous_jump_a_to_b=4.0)
    reverse = pair.pressure_link(9, 0.5, densities, 0.0, normal_viscous_jump_a_to_b=4.0)
    assert forward.jump == 4.0
    assert reverse.jump == -4.0
