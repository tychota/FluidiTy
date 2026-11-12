"""Fixed-domain M7 heating and conservative transport acceptance checks."""

import numpy as np
import pytest

from fluidity_reference.gas import project_heating, transport_densities
from fluidity_reference.grid import Grid2D


def test_open_heating_expels_analytic_volume():
    grid = Grid2D(4, 3, 0.25)
    exterior = np.abs(np.asarray(grid.incidence.sum(axis=0)).ravel()) == 1
    heat = np.zeros(grid.cell_count)
    heat[5] = 20000
    dt = 0.02
    ambient = 101325.0
    result = project_heating(grid, dt, heat, np.full(grid.cell_count, ambient),
                             np.ones_like(exterior), exterior, ambient)
    expansion = dt * np.sum(grid.incidence @ (grid.face_areas * result.velocity))
    analytic = 0.4 * dt * (grid.cell_volumes @ heat) / (1.4 * ambient)
    assert expansion == pytest.approx(analytic, rel=0.25)
    np.testing.assert_allclose(result.p0, ambient)


def test_sealed_components_heat_independently():
    grid = Grid2D(4, 1, 0.5)
    permeable = np.zeros(grid.face_count, dtype=bool)
    permeable[[1, 3]] = True
    open_faces = np.zeros_like(permeable)
    heat = np.array([20000.0, 0, 0, 0])
    ambient = 101325.0
    dt = 0.1
    result = project_heating(grid, dt, heat, np.full(4, ambient), permeable, open_faces, ambient)
    np.testing.assert_allclose(result.p0[:2] - ambient, 0.4 * dt * 10000, rtol=0.01)
    np.testing.assert_array_equal(result.p0[2:], ambient)
    assert result.velocity[2] == 0
    np.testing.assert_allclose(grid.incidence @ (grid.face_areas * result.velocity),
                               grid.cell_volumes * result.divergence, atol=1e-12)


def test_scalar_budgets_include_sources_and_signed_boundary_flux():
    grid = Grid2D(2, 1, 0.5)
    densities = np.array([[2.0, 1, 10], [3.0, 2, 12]])
    exterior = np.array([1.0, 0.5, 20])
    rates = np.array([[0.25, 0.5, 3], [0, 0, 0]])
    face_velocity = np.zeros(grid.face_count)
    face_velocity[:3] = 0.4
    dt = 0.1
    result = transport_densities(grid, dt, densities, rates, face_velocity, exterior)
    expected_flux = 0.4 * grid.h * (densities[-1] - exterior)
    np.testing.assert_allclose(result.outward_rate, expected_flux)
    np.testing.assert_allclose(grid.cell_volumes @ (result.densities - densities),
                               dt * (grid.cell_volumes @ rates - expected_flux))
    with pytest.raises(ValueError, match="CFL"):
        transport_densities(grid, 2, densities, rates, face_velocity, exterior)


def test_burn_conserves_fuel_and_adds_soot_and_enthalpy():
    from fluidity_reference.gas import react_densities

    densities = np.array([[0.1, 0.01, 3e5], [0.0, 0.02, 4e5]])
    result = react_densities(densities, 0.25, 4.0, 2e6, 0.2)
    burned = -0.1 * np.expm1(-1.0)
    np.testing.assert_allclose(result.densities[0], [0.1 - burned, 0.01 + 0.2 * burned, 3e5 + 2e6 * burned])
    np.testing.assert_array_equal(result.densities[1], densities[1])
    np.testing.assert_allclose(result.heat_rate, [2e6 * burned / 0.25, 0])
    np.testing.assert_array_equal(densities, [[0.1, 0.01, 3e5], [0, 0.02, 4e5]])


def test_ideal_gas_density_and_residual_buoyancy():
    from fluidity_reference.gas import residual_gas_state

    temperature = np.array([300.0, 600, 300])
    p0 = np.array([101325.0, 101325, 202650])
    soot = np.array([0.0, 0, 0.1])
    result = residual_gas_state(p0, temperature, soot, 300.0, [0, -9.81, 0])
    np.testing.assert_allclose(result.density, p0 / (287.05 * temperature))
    np.testing.assert_allclose(result.acceleration[:, 1], [0, 9.81, -9.81 * 0.1 / result.density[2]])
