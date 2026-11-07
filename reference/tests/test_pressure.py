import numpy as np
import pytest

from fluidity_reference.grid import Grid2D
from fluidity_reference.pressure import PressureInputs, solve_pressure, bounded_band_source


def test_two_cells_jump_and_gauge():
    g = Grid2D(2, 1, 1.)
    u = np.zeros(g.face_count); u[1] = 2
    jump = np.zeros(g.face_count); jump[1] = 3
    r = solve_pressure(PressureInputs(g, 1., u, jump=jump))
    np.testing.assert_allclose(r.pressure, [0, 5])
    np.testing.assert_allclose(r.velocity, 0, atol=1e-14)
    assert r.gauges == (0,)
    np.testing.assert_allclose(r.matrix.toarray(), [[1, -1], [-1, 1]])


@pytest.mark.parametrize('face', [0, 1])
def test_dirichlet_boundary_orientation(face):
    g = Grid2D(1, 1, 1.)
    mask = np.zeros(g.face_count, bool); mask[face] = True
    p = np.zeros(g.face_count); p[face] = 7
    r = solve_pressure(PressureInputs(g, 1., np.zeros(g.face_count), dirichlet_mask=mask, dirichlet_pressure=p))
    np.testing.assert_allclose(r.pressure, [7])
    np.testing.assert_allclose(r.velocity, 0)
    assert not r.gauges


def test_disconnected_incompatible_components_rejected():
    g = Grid2D(2, 1, 1.)
    a = np.ones(g.face_count); a[1] = 0
    with pytest.raises(ValueError, match='compatib'):
        solve_pressure(PressureInputs(g, 1., np.zeros(g.face_count), aperture=a, source=[1, -1]))


def test_compliance_and_blocked_wall_residual_at_tiny_volume():
    g = Grid2D(1, 1, 1.)
    a = np.ones(g.face_count); a[0] = 0
    wall = np.zeros(g.face_count); wall[0] = 1
    r = solve_pressure(PressureInputs(g, 1., np.zeros(g.face_count), aperture=a, wall_velocity=wall,
                                      fluid_volume=[1e-12], bulk_modulus=[2.], source=[3.]))
    assert not r.gauges
    np.testing.assert_allclose(r.constraint_defect, -r.residual / 1e-12, atol=1e-3)
    np.testing.assert_allclose(r.pressure, [2e12 + 6], rtol=1e-14)


def test_invalid_boundary_and_coefficients():
    g = Grid2D(2, 1, 1.); u = np.zeros(g.face_count)
    mask = np.zeros(g.face_count, bool); mask[1] = True
    with pytest.raises(ValueError): solve_pressure(PressureInputs(g, 1., u, dirichlet_mask=mask))
    mask[1] = False; mask[0] = True
    j = np.zeros(g.face_count); j[0] = 1
    with pytest.raises(ValueError): solve_pressure(PressureInputs(g, 1., u, dirichlet_mask=mask, jump=j))
    for kw in ({'aperture': -np.ones(g.face_count)}, {'fluid_volume': [0, 1]}, {'velocity': [0]}, {'mobility': np.full(g.face_count, np.nan)}):
        args = dict(grid=g, dt=1., velocity=u); args.update(kw)
        with pytest.raises(ValueError): solve_pressure(PressureInputs(**args))


def test_bounded_band_source():
    v = np.array([1., 2., 3., 4.]); band = np.array([0, 1, 1, 1], bool)
    x = bounded_band_source([100, -10, 2, 10], v, band, 1.)
    assert x[0] == 0 and np.max(np.abs(x)) <= 1
    assert abs(v @ x) < 1e-12
