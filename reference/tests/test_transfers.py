import numpy as np
import pytest

from fluidity_reference.grid import Grid2D
from fluidity_reference.transfers import (local_apic_fit, local_velocity_gradient,
                                           p2g, g2p, energy_velocity)


def test_local_fit_and_gradient_use_distinct_moments():
    x = np.array([0., 0.])
    samples = np.array([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
    w = np.full(4, .25)
    values = 3 + samples @ np.array([2., -4.])
    value, row = local_apic_fit(x, samples, w, values)
    np.testing.assert_allclose(value, 3)
    np.testing.assert_allclose(row, [2, -4])
    gradients = np.array([[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]])
    np.testing.assert_allclose(local_velocity_gradient(values, gradients), [2, -4])
    nonlinear = np.array([0., 0., 0., 4.])
    _, fit = local_apic_fit(x, samples, w, nonlinear)
    np.testing.assert_allclose(fit, [1, 1])
    np.testing.assert_allclose(local_velocity_gradient(nonlinear, 2 * gradients), [2, 2])


def test_p2g_mass_linear_and_angular_moments():
    grid = Grid2D(8, 8, 1.)
    positions = np.array([[3.2, 3.4], [4.3, 4.1]])
    masses = np.array([2., 3.])
    velocities = np.array([[1., 2.], [-2., 1.]])
    affine = np.array([[[0., -2.], [2., 0.]], [[0., 1.], [-1., 0.]]])
    fm, fq = p2g(grid, positions, masses, velocities, affine)
    split = grid.ny * (grid.nx + 1)
    for sl in (slice(None, split), slice(split, None)):
        np.testing.assert_allclose(fm[sl].sum(), masses.sum())
    np.testing.assert_allclose([fq[:split].sum(), fq[split:].sum()], (masses[:, None] * velocities).sum(axis=0), atol=1e-14)
    centers = grid.face_centers
    angular = (centers[split:, 0] * fq[split:]).sum() - (centers[:split, 1] * fq[:split]).sum()
    expected = np.sum(masses * (positions[:, 0]*velocities[:, 1] - positions[:, 1]*velocities[:, 0] + .25*(affine[:, 1, 0] - affine[:, 0, 1])))
    np.testing.assert_allclose(angular, expected, atol=1e-13)


def test_g2p_affine_and_independent_old_new():
    grid = Grid2D(8, 8, .7, (1., -2.))
    p = np.array([[3.2, .2], [3.5, .8]])
    centers = grid.face_centers
    split = grid.ny * (grid.nx + 1)
    A = np.array([[2., -3.], [.5, 4.]])
    b = np.array([1., -2.])
    faces = np.r_[centers[:split] @ A[0] + b[0], centers[split:] @ A[1] + b[1]]
    pic, inc, C, L = g2p(grid, p, faces, faces)
    np.testing.assert_allclose(pic, p @ A.T + b, atol=1e-14)
    np.testing.assert_allclose(inc, 0, atol=1e-14)
    np.testing.assert_allclose(C, np.broadcast_to(A, C.shape), atol=1e-14)
    np.testing.assert_allclose(L, np.broadcast_to(A, L.shape), atol=1e-14)
    particle, advect = energy_velocity(pic, np.ones_like(pic), .5, 'liquid')
    np.testing.assert_allclose(particle, pic + .495)
    np.testing.assert_allclose(advect, pic + .2475)


def test_nonlinear_field_has_distinct_fit_and_gradient():
    grid = Grid2D(8, 8, 1.)
    p = np.array([[3.2, 3.3]])
    centers = grid.face_centers
    split = grid.ny * (grid.nx + 1)
    f = np.r_[centers[:split, 0]**3, centers[split:, 1]**3]
    _, _, C, L = g2p(grid, p, f, f)
    assert not np.allclose(C, L)


@pytest.mark.parametrize('phase,alpha,beta', [('liquid', .792, .4), ('gas', 0, 0), ('elastic', .792, 0), ('snow', .792, .4), ('granular', .792, .4)])
def test_energy_policy(phase, alpha, beta):
    p, a = energy_velocity(np.array([1., 2.]), np.array([2., -1.]), .8, phase, .5)
    np.testing.assert_allclose(p, np.array([1., 2.]) + alpha * np.array([2., -1.]))
    np.testing.assert_allclose(a, np.array([1., 2.]) + beta * alpha * np.array([2., -1.]))


def test_invalid_inputs_and_incomplete_support_do_not_mutate():
    grid = Grid2D(6, 6, 1.)
    positions = np.array([[.1, .1]])
    masses = np.array([1.])
    velocities = np.ones((1, 2))
    affine = np.zeros((1, 2, 2))
    snapshots = [x.copy() for x in (positions, masses, velocities, affine)]
    with pytest.raises(ValueError):
        p2g(grid, positions, masses, velocities, affine)
    for x, original in zip((positions, masses, velocities, affine), snapshots):
        np.testing.assert_array_equal(x, original)
    with pytest.raises(ValueError):
        g2p(grid, positions, np.zeros(grid.face_count), np.zeros(grid.face_count))
    with pytest.raises(ValueError):
        energy_velocity([1., np.nan], [0., 0.], .5, 'liquid')
    for energy, phase, sep in [(1.1, 'liquid', 1), (.5, 'unknown', 1), (.5, 'liquid', -1)]:
        with pytest.raises(ValueError):
            energy_velocity([1., 2.], [0., 0.], energy, phase, sep)
    with pytest.raises(ValueError):
        p2g(grid, [[3., 3.]], [-1.], velocities, affine)
    with pytest.raises(ValueError):
        g2p(grid, [[3., 3.]], np.zeros(2), np.zeros(grid.face_count))


def test_g2p_nonzero_increment_and_success_preserves_inputs():
    grid = Grid2D(6, 6, 1.)
    positions = np.array([[3., 3.]])
    old = np.arange(grid.face_count, dtype=float)
    new = old + 1
    originals = [x.copy() for x in (positions, old, new)]
    _, increment, _, _ = g2p(grid, positions, old, new)
    np.testing.assert_allclose(increment, [[1., 1.]], atol=1e-14)
    for x, original in zip((positions, old, new), originals):
        np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize('api', ['fit_position', 'fit_samples', 'fit_weights', 'fit_values', 'gradient_values', 'gradient_gradients', 'p2g_positions', 'p2g_mass', 'p2g_velocity', 'p2g_affine', 'g2p_positions', 'g2p_old', 'g2p_new', 'energy_pic', 'energy_residual', 'energy_scalar', 'energy_separation'])
def test_transfer_apis_reject_nonfinite_inputs(api):
    grid = Grid2D(6, 6, 1.)
    bad = np.nan
    calls = {
        'fit_position': lambda: local_apic_fit([bad, 0], [[0, 0], [1, 0], [0, 1]], [1/3]*3, [1, 2, 3]),
        'fit_samples': lambda: local_apic_fit([0, 0], [[bad, 0], [1, 0], [0, 1]], [1/3]*3, [1, 2, 3]),
        'fit_weights': lambda: local_apic_fit([0, 0], [[0, 0], [1, 0], [0, 1]], [bad, 0, 0], [1, 2, 3]),
        'fit_values': lambda: local_apic_fit([0, 0], [[0, 0], [1, 0], [0, 1]], [1/3]*3, [bad, 2, 3]),
        'gradient_values': lambda: local_velocity_gradient([bad], [[1, 1]]),
        'gradient_gradients': lambda: local_velocity_gradient([1], [[bad, 1]]),
        'p2g_positions': lambda: p2g(grid, [[bad, 3]], [1], [[0, 0]], np.zeros((1, 2, 2))),
        'p2g_mass': lambda: p2g(grid, [[3, 3]], [bad], [[0, 0]], np.zeros((1, 2, 2))),
        'p2g_velocity': lambda: p2g(grid, [[3, 3]], [1], [[bad, 0]], np.zeros((1, 2, 2))),
        'p2g_affine': lambda: p2g(grid, [[3, 3]], [1], [[0, 0]], np.full((1, 2, 2), bad)),
        'g2p_positions': lambda: g2p(grid, [[bad, 3]], np.zeros(grid.face_count), np.zeros(grid.face_count)),
        'g2p_old': lambda: g2p(grid, [[3, 3]], np.full(grid.face_count, bad), np.zeros(grid.face_count)),
        'g2p_new': lambda: g2p(grid, [[3, 3]], np.zeros(grid.face_count), np.full(grid.face_count, bad)),
        'energy_pic': lambda: energy_velocity([bad, 0], [0, 0], .5, 'liquid'),
        'energy_residual': lambda: energy_velocity([0, 0], [bad, 0], .5, 'liquid'),
        'energy_scalar': lambda: energy_velocity([0, 0], [0, 0], bad, 'liquid'),
        'energy_separation': lambda: energy_velocity([0, 0], [0, 0], .5, 'liquid', bad),
    }
    with pytest.raises(ValueError):
        calls[api]()


def test_p2g_success_preserves_inputs():
    grid = Grid2D(6, 6, 1.)
    positions = np.array([[3., 3.]])
    mass = np.array([2.]); velocity = np.array([[1., -1.]])
    affine = np.array([[[0., 1.], [-1., 0.]]])
    inputs = (positions, mass, velocity, affine)
    snapshots = [x.copy() for x in inputs]
    face_mass, momentum = p2g(grid, *inputs)
    assert face_mass.sum() == pytest.approx(4.)
    assert np.all(np.isfinite(momentum))
    for x, before in zip(inputs, snapshots):
        np.testing.assert_array_equal(x, before)
