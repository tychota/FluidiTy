import itertools
import numpy as np
import pytest

from fluidity_reference.resampling import ParticleBatch, split_particle, merge_particles, reseed_cell, tait_J


def batch():
    return ParticleBatch([[1., 2., 3.]], [8.], [4.], [2.], [[2., -1., 3.]],
                         [np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])], [[.2, .8]], [7])


def test_split_cubic_moments_and_energy():
    p = batch(); h = 2.
    offsets = np.array(list(itertools.product(*[[-np.sqrt(3)*h/4, np.sqrt(3)*h/4]]*3)))
    c = split_particle(p, 0, offsets, np.arange(8)+10)
    assert np.allclose(c.mass.sum(), p.mass[0])
    assert np.allclose(c.mass @ c.positions / c.mass.sum(), p.positions[0])
    assert np.allclose(c.mass @ c.velocity, p.mass[0]*p.velocity[0])
    before_rows = []
    after_rows = []
    for a in range(3):
        e = np.eye(3)[a]; parent = h*h*(np.eye(3)/4+np.outer(e,e)/12); child = parent/4
        before = p.mass[0]*p.affine[0,a] @ parent
        after = sum(c.mass[i]*((c.velocity[i,a]-p.velocity[0,a])*offsets[i]+c.affine[i,a] @ child) for i in range(8))
        before_rows.append(before)
        after_rows.append(after)
        assert np.allclose(before, after + p.mass[0]*p.affine[0,a] @ (h*h*np.outer(e,e)/16))
    before_rows, after_rows = np.array(before_rows), np.array(after_rows)
    assert np.allclose(before_rows - before_rows.T, after_rows - after_rows.T)
    energy = lambda b, ds: sum(b.mass[i]/2*(b.velocity[i]@b.velocity[i]+sum(b.affine[i,a]@ds[a]@b.affine[i,a] for a in range(3))) for i in range(len(b.mass)))
    ds = np.array([h*h*(np.eye(3)/4+np.outer(e,e)/12) for e in np.eye(3)])
    assert np.isclose(energy(p,ds)-energy(c,ds/4), p.mass[0]*h*h*np.sum(np.diag(p.affine[0])**2)/32)



@pytest.mark.parametrize(
    "affine, expected_loss",
    [
        (np.diag([1.0, -2.0, 3.0]), 14.0),
        (np.array([[0.0, 2.0, 3.0], [4.0, 0.0, 6.0], [7.0, 8.0, 0.0]]), 0.0),
    ],
)
def test_split_isolates_normal_energy_deficit(affine, expected_loss):
    width = 2.0
    mass = 8.0
    displacement = np.sqrt(3.0) * width / 4.0
    offsets = np.array(list(itertools.product([-displacement, displacement], repeat=3)))
    parent = ParticleBatch([[0.0, 0.0, 0.0]], [mass], [4.0], [2.0], [[0.0, 0.0, 0.0]],
                           [affine], [[1.0]], [1])
    children = split_particle(parent, 0, offsets, np.arange(2, 10))
    np.testing.assert_allclose(offsets.T @ offsets / len(offsets), 3 * width**2 * np.eye(3) / 16,
                               atol=1e-14)
    assert children.reference_volume.sum() == parent.reference_volume[0]
    assert children.current_volume.sum() == parent.current_volume[0]

    quadratic = np.tile(width**2 / 4 * np.eye(3), (3, 1, 1))
    normal = width**2 / 12 * np.einsum("ai,aj->aij", np.eye(3), np.eye(3))
    for parent_moment, loss in [(quadratic, 0.0), (quadratic + normal, expected_loss)]:
        child_moment = parent_moment / 4
        parent_energy = mass / 2 * np.einsum("ai,aij,aj->", affine, parent_moment, affine)
        child_energy = np.sum(children.mass * np.sum(children.velocity**2, axis=1)) / 2
        child_energy += mass / 2 * np.einsum("ai,aij,aj->", affine, child_moment, affine)
        assert parent_energy - child_energy == pytest.approx(loss, abs=1e-12)

        parent_spatial = mass * np.einsum("ai,aij->aj", affine, parent_moment)
        child_spatial = np.einsum("n,na,nj->aj", children.mass, children.velocity, offsets)
        child_spatial += mass * np.einsum("ai,aij->aj", affine, child_moment)
        np.testing.assert_allclose(child_spatial - child_spatial.T, parent_spatial - parent_spatial.T,
                                   atol=1e-12)


def test_merge_preserves_first_moments_and_volume_weighted_composition():
    p = ParticleBatch([[0.,0.],[1.,0.]], [1.,3.], [1.,3.], [2.,2.], [[0.,0.],[2.,0.]], np.zeros((2,2,2)), [[1.,0.],[0.,1.]], [1,2])
    ds = np.tile(np.eye(2), (2,2,1,1)); target = np.tile(2*np.eye(2), (2,1,1))
    q = merge_particles(p, ds, target, 9)
    assert np.allclose(q.positions[0], [.75,0])
    assert np.allclose(q.velocity[0], [1.5,0])
    assert np.allclose(q.composition[0], [.25,.75])
    assert np.allclose(q.affine[0,0] @ (4*target[0]), [1.5,0])
    assert q.reference_volume[0] == 4 and q.current_volume[0] == 4
    with pytest.raises(ValueError): merge_particles(p, ds, np.zeros_like(target), 9)
    with pytest.raises(ValueError): merge_particles(p, ds, np.tile(np.eye(2), (2,1,1)), 9)


def test_eos_and_reseed():
    assert np.isclose(tait_J(0, 10, 2), 1)
    assert np.isclose(tait_J(5, 10, 2), 1/np.sqrt(2))
    with pytest.raises(ValueError): tait_J(-5,10,2)
    r = reseed_cell(4, 2, 2, 4, [.25,.75])
    assert np.allclose(r['mass'], 2) and np.allclose(r['reference_volume'], .5)
    with pytest.raises(ValueError): reseed_cell(4,2,2,0,[1.])


def test_invalid_data_and_split_do_not_mutate():
    p = batch(); original = p.positions.copy()
    with pytest.raises(ValueError): split_particle(p,0,[[1,0,0],[-1,0,0]],[1,1])
    with pytest.raises(ValueError): split_particle(p,0,[[1,0,0],[0,0,0]],[1,2])
    assert np.array_equal(p.positions,original)
    with pytest.raises(ValueError): ParticleBatch([[0,0]], [1], [1], [1], [[0,0]], np.zeros((1,2,2)), [[.2,.2]], [1])
    with pytest.raises(ValueError): ParticleBatch([[0,0]], [np.nan], [1], [1], [[0,0]], np.zeros((1,2,2)), [[1]], [1])


def test_merge_rejects_small_infeasible_moments_and_preserves_feasible_affine():
    p = ParticleBatch([[0., 0.]], [1.], [1.], [1.], [[0., 0.]], [np.eye(2)], [[1.]], [1])
    child = np.tile(1e-13*np.eye(2), (1,2,1,1))
    with pytest.raises(ValueError):
        merge_particles(p, child, np.tile(1e-20*np.eye(2), (2,1,1)), 2)
    q = merge_particles(p, child, child[0], 2)
    assert np.allclose(q.affine[0], np.eye(2))


def test_reseed_avoids_overflow_and_rejects_unrepresentable_results():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(ValueError):
            reseed_cell(1e308, 2, 1, 1, [1])
        r = reseed_cell(1, 1, 1e308, 2, [1])
        assert np.all(r['reference_volume'] == 5e-309)


def test_reseed_preserves_mass_with_subnormal_volume():
    volume = 3 * np.nextafter(0., 1.)
    result = reseed_cell(volume, 1e308, 1, 2, [1])
    from fractions import Fraction
    expected = float(Fraction.from_float(float(volume)) * Fraction.from_float(1e308) / 2)
    assert np.all(result['mass'] == expected)


@pytest.mark.parametrize('which', ['child', 'target'])
def test_merge_rejects_material_asymmetry_at_small_scale(which):
    p = ParticleBatch([[0., 0.]], [1.], [1.], [1.], [[0., 0.]], [np.ones((2, 2))], [[1.]], [1])
    child = np.tile(1e-14*np.eye(2), (1, 2, 1, 1))
    target = np.tile(1e-13*np.eye(2), (2, 1, 1))
    bad = np.array([[1e-13, 9e-13], [0., 1e-13]])
    if which == 'child':
        child[0, 0] = bad
        target[:] = 1e-11*np.eye(2)
    else:
        target[0] = bad
    with pytest.raises(ValueError, match='symmetric'):
        merge_particles(p, child, target, 2)


def test_merge_symmetrizes_rounding_level_asymmetry_consistently():
    p = ParticleBatch([[0., 0.]], [1.], [1.], [1.], [[0., 0.]], [np.ones((2, 2))], [[1.]], [1])
    child = np.tile(1e-14*np.eye(2), (1, 2, 1, 1))
    target = np.tile(1e-13*np.eye(2), (2, 1, 1))
    child[0, :, 0, 1] = 5e-27
    target[:, 0, 1] = 5e-26
    originals = child.copy(), target.copy()
    q = merge_particles(p, child, target, 2)
    for a in range(2):
        symmetric_child = (child[0, a] + child[0, a].T) / 2
        symmetric_target = (target[a] + target[a].T) / 2
        np.testing.assert_allclose(q.affine[0, a] @ symmetric_target, np.ones(2) @ symmetric_child, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(child, originals[0])
    np.testing.assert_array_equal(target, originals[1])
