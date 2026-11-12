import math

import pytest

from fluidity_reference.eos import LiquidEOS


@pytest.mark.parametrize('law', ['hencky', 'tait'])
@pytest.mark.parametrize('volume_ratio', [0.8, 1.0, 1.05])
def test_eos_round_trip_and_acoustic_coefficient(law, volume_ratio):
    eos = LiquidEOS(1000.0, 50.0, law)
    pressure = eos.pressure(volume_ratio)
    state = eos.at_pressure(pressure)
    assert state.volume_ratio == pytest.approx(volume_ratio, rel=2e-15)
    assert state.density == pytest.approx(1000.0 / volume_ratio)
    assert state.density * state.sound_speed**2 == pytest.approx(state.bulk_modulus)
    expected_bulk_modulus = 2_500_000.0 + (7.0 * pressure if law == 'tait' else 0.0)
    assert state.bulk_modulus == pytest.approx(expected_bulk_modulus)


def test_tait_uses_positive_absolute_pressure_domain():
    eos = LiquidEOS(1.0, 7.0, 'tait')
    with pytest.raises(ValueError, match='Tait pressure'):
        eos.at_pressure(-7.0)
    state = eos.at_pressure(math.nextafter(-7.0, math.inf))
    assert state.volume_ratio > 1.0
    assert state.bulk_modulus > 0.0


@pytest.mark.parametrize('law', ['hencky', 'tait'])
def test_infinite_sound_speed_gives_exact_unit_volume_ratio(law):
    eos = LiquidEOS(1000.0, math.inf, law)
    state = eos.at_pressure(-2.0e6)
    assert state.volume_ratio == 1.0
    assert state.log_volume_ratio == 0.0
    assert state.density == 1000.0
    assert state.bulk_modulus == math.inf
    assert state.sound_speed == math.inf
    with pytest.raises(ValueError, match='incompressible'):
        eos.pressure(1.0)


def test_tait_preserves_small_predictor_log_volume_ratio():
    eos = LiquidEOS(7.0, 1.0, 'tait')
    state = eos.at_pressure(1e-20)
    assert state.volume_ratio == 1.0
    assert state.log_volume_ratio == pytest.approx(-1e-20 / 7.0, rel=1e-15, abs=0.0)



@pytest.mark.parametrize('law', ['hencky', 'tait'])
def test_closed_compliant_piston_converges_to_eos_compression(law):
    from fluidity_reference.grid import Grid2D
    from fluidity_reference.pressure import PressureInputs, solve_pressure

    eos = LiquidEOS(1000.0, 50.0, law)
    grid = Grid2D(1, 1, 1.0)
    piston_speed = 0.02
    exact_pressure = eos.pressure(1.0 - piston_speed)
    errors = []

    for steps in (16, 32, 64):
        dt = 1.0 / steps
        pressure = 0.0
        for step in range(steps):
            volume = 1.0 - step * dt * piston_speed
            state = eos.at_pressure(pressure)
            result = solve_pressure(PressureInputs(
                grid=grid,
                dt=dt,
                velocity=[0.0] * 4,
                aperture=[0.0] * 4,
                wall_velocity=[piston_speed, 0.0, 0.0, 0.0],
                fluid_volume=[volume],
                bulk_modulus=state.bulk_modulus,
                previous_pressure=[pressure],
            ))
            assert not result.gauges
            assert result.constraint_defect[0] == pytest.approx(0.0, abs=1e-14)
            pressure = result.pressure[0]
        errors.append(abs(pressure - exact_pressure) / exact_pressure)

    assert errors[1] < 0.55 * errors[0]
    assert errors[2] < 0.55 * errors[1]
    assert errors[2] < 0.002
