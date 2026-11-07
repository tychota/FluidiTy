import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from fluidity_reference.grid import Grid2D
from fluidity_reference.viscosity import mac_strain, viscous_step


def test_mac_strain_rigid_modes_and_couette():
    g = Grid2D(3, 2, .5)
    E, W = mac_strain(g, 2.)
    n = g.ny * (g.nx + 1)
    x, y = g.face_centers.T
    translation = np.r_[np.full(n, 3.), np.full(g.face_count-n, -2.)]
    rotation = np.r_[-y[:n], x[n:]]
    couette = np.r_[2*y[:n], np.zeros(g.face_count-n)]
    np.testing.assert_allclose(E @ translation, 0, atol=1e-14)
    np.testing.assert_allclose(E @ rotation, 0, atol=1e-14)
    np.testing.assert_allclose(E @ couette, np.r_[np.zeros(2*g.cell_count), np.ones((g.nx-1)*(g.ny-1))])
    np.testing.assert_allclose(W, np.r_[np.full(2*g.cell_count, 2*2*g.h**2), np.full((g.nx-1)*(g.ny-1), 4*2*g.h**2)])
    K = (E.T @ diags(W) @ E).toarray()
    np.testing.assert_allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-12


def test_mass_scaled_bound_and_explicit_stability():
    E = csr_matrix([[1., -1.]])
    args = ([1., 0.], [.01, 1.], E, [1.], [0.])
    r = viscous_step(*args, .1)
    assert r.method == 'implicit'
    assert r.spectral_bound == pytest.approx(110.)
    with pytest.raises(ValueError): viscous_step(*args, .1, mode='explicit')
    r = viscous_step(*args, .99*2/110, mode='explicit')
    assert r.energy_after <= r.energy_before


def test_implicit_wall_work_identity_and_no_mutation():
    u = np.array([0.]); mass = np.array([2.]); E = csr_matrix([[1.]])
    weights = np.array([3.]); wall = np.array([4.])
    r = viscous_step(u, mass, E, weights, wall, .5, mode='implicit')
    assert r.velocity[0] == pytest.approx(12/7)
    assert r.energy_after-r.energy_before + r.numerical_dissipation + .5*3*r.velocity[0]**2 == pytest.approx(r.wall_work)
    np.testing.assert_array_equal(u, [0.]); np.testing.assert_array_equal(mass, [2.])
    np.testing.assert_array_equal(weights, [3.]); np.testing.assert_array_equal(wall, [4.])
    np.testing.assert_array_equal(E.toarray(), [[1.]])


def test_zero_viscosity_identity():
    g = Grid2D(1, 1, 1.)
    E, W = mac_strain(g, 0.)
    u = np.arange(g.face_count, dtype=float)
    r = viscous_step(u, np.ones(g.face_count), E, W, np.zeros(E.shape[0]), 1.)
    np.testing.assert_array_equal(r.velocity, u)
    assert r.spectral_bound == 0


@pytest.mark.parametrize('change', [
    {'mass': [0.]}, {'mass': [np.nan]}, {'W': [-1.]}, {'W': [np.inf]},
    {'velocity': [np.nan]}, {'velocity': [1., 2.]}, {'boundary_strain': [np.inf]},
    {'E': csr_matrix([[np.nan]])}, {'E': csr_matrix([[1., 2.]])},
    {'dt': -1.}, {'dt': np.nan}, {'safety': 1.}, {'safety': -1.}, {'mode': 'bad'},
])
def test_invalid_inputs_rejected(change):
    args = dict(velocity=[1.], mass=[1.], E=csr_matrix([[1.]]), W=[1.], boundary_strain=[0.], dt=.1)
    args.update(change)
    with pytest.raises(ValueError): viscous_step(**args)


def test_invalid_viscosity_rejected():
    for mu in (-1., np.nan, np.inf):
        with pytest.raises(ValueError): mac_strain(Grid2D(1, 1, 1.), mu)
