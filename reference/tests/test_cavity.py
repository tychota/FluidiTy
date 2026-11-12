import math

import pytest

from fluidity_reference.cavity import CavityReference


def test_submerged_sphere_initialization_and_projection_pressure():
    radius = 0.2
    volume = 4.0 * math.pi * radius**3 / 3.0
    ambient_pressure = 101325.0
    hydrostatic_pressure = 9810.0
    surface_tension = 0.072
    curvature = 2.0 / radius
    cavity = CavityReference.submerged_sphere(
        volume, ambient_pressure, hydrostatic_pressure, surface_tension)
    assert cavity.initial_pressure == pytest.approx(
        ambient_pressure + hydrostatic_pressure + surface_tension * curvature)
    state = cavity.at_volume(volume, ambient_pressure, 1000.0, surface_tension, -curvature)
    assert state.absolute_pressure == cavity.initial_pressure
    assert state.boundary_pressure == pytest.approx(hydrostatic_pressure)
    expected_omega = math.sqrt(3.0 * 1.4 * state.absolute_pressure / 1000.0) / radius
    assert state.step_limit == pytest.approx(0.5 / expected_omega)


def test_compression_uses_fixed_polytropic_inventory():
    cavity = CavityReference(0.01, 1e5)
    compressed = cavity.at_volume(0.005, 1e5, 1000.0, 0.0, 0.0)
    expanded = cavity.at_volume(0.02, 1e5, 1000.0, 0.0, 0.0)
    assert compressed.absolute_pressure == pytest.approx(1e5 * 2**1.4)
    assert expanded.absolute_pressure == pytest.approx(1e5 * 0.5**1.4)
    assert compressed.step_limit < expanded.step_limit
    assert compressed.is_resolved(compressed.equivalent_radius / 2.01)
    assert not compressed.is_resolved(compressed.equivalent_radius / 1.99)


@pytest.mark.parametrize('volume', [0.0, -1.0, math.nan, math.inf])
def test_cavity_requires_positive_representable_volume(volume):
    with pytest.raises(ValueError, match='volume'):
        CavityReference(volume, 1e5)
    cavity = CavityReference(0.01, 1e5)
    with pytest.raises(ValueError, match='volume'):
        cavity.at_volume(volume, 1e5, 1000.0, 0.0, 0.0)
