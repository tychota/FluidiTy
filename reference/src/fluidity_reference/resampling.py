"""Uniform-support algebra; callers must separately certify geometry and EOS."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np


def _array(value, shape=None):
    a = np.array(value, dtype=float, copy=True)
    if (shape is not None and a.shape != shape) or not np.all(np.isfinite(a)):
        raise ValueError("invalid array shape or nonfinite value")
    return a


def _fractions(a):
    if np.any(a < 0) or not np.allclose(a.sum(axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("invalid composition fractions")


def _ids(value, n):
    a = np.array(value, copy=True)
    if a.shape != (n,) or a.dtype.kind not in 'iu' or len(np.unique(a)) != n:
        raise ValueError("IDs must be distinct integers")
    return a


@dataclass(frozen=True)
class ParticleBatch:
    positions: np.ndarray
    mass: np.ndarray
    current_volume: np.ndarray
    reference_volume: np.ndarray
    velocity: np.ndarray
    affine: np.ndarray
    composition: np.ndarray
    ids: np.ndarray

    def __post_init__(self):
        p = _array(self.positions)
        if p.ndim != 2 or not all(p.shape):
            raise ValueError("positions must be a nonempty matrix")
        n, d = p.shape
        values = {'positions': p, 'mass': _array(self.mass, (n,)),
                  'current_volume': _array(self.current_volume, (n,)),
                  'reference_volume': _array(self.reference_volume, (n,)),
                  'velocity': _array(self.velocity, (n,d)),
                  'affine': _array(self.affine, (n,d,d)),
                  'composition': _array(self.composition), 'ids': _ids(self.ids,n)}
        f = values['composition']
        if f.ndim != 2 or f.shape[0] != n or f.shape[1] == 0:
            raise ValueError("invalid composition shape")
        _fractions(f)
        if any(np.any(values[k] <= 0) for k in ('mass','current_volume','reference_volume')):
            raise ValueError("mass and volumes must be positive")
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def split_particle(batch, index, offsets, child_ids):
    n, d = batch.positions.shape
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < n:
        raise ValueError("invalid particle index")
    offsets = _array(offsets)
    if offsets.ndim != 2 or offsets.shape[1] != d or offsets.shape[0] == 0 or not np.allclose(offsets.mean(axis=0), 0, rtol=0, atol=1e-12):
        raise ValueError("offsets must have zero mean")
    count = len(offsets); ids = _ids(child_ids, count)
    if np.intersect1d(ids, np.delete(batch.ids, index)).size:
        raise ValueError("child IDs collide with retained particles")
    # Return children only; other particles remain the caller's responsibility.
    return ParticleBatch(batch.positions[index] + offsets, np.full(count, batch.mass[index]/count),
        np.full(count, batch.current_volume[index]/count), np.full(count, batch.reference_volume[index]/count),
        batch.velocity[index] + offsets @ batch.affine[index].T,
        np.repeat(batch.affine[index,None],count,axis=0),
        np.repeat(batch.composition[index,None],count,axis=0), ids)


def _positive_definite(m):
    if not np.allclose(m, m.T, rtol=1e-12, atol=1e-12) or np.min(np.linalg.eigvalsh(m)) <= 0:
        raise ValueError("support moments must be symmetric positive definite")


def merge_particles(batch, child_moments, target_moments, merged_id):
    n, d = batch.positions.shape
    if n > 8:
        raise ValueError("at most eight particles may merge")
    ids = _ids([merged_id], 1)
    child = _array(child_moments, (n,d,d,d)); target = _array(target_moments, (d,d,d))
    mass = batch.mass.sum()
    x = batch.mass @ batch.positions / mass; v = batch.mass @ batch.velocity / mass
    r = batch.positions - x
    affine = np.empty((d,d))
    for a in range(d):
        _positive_definite(target[a])
        for moment in child[:,a]: _positive_definite(moment)
        k = np.einsum('n,ni,nj->ij', batch.mass,r,r) + np.einsum('n,nij->ij',batch.mass,child[:,a])
        gap = mass*target[a]-k
        if np.min(np.linalg.eigvalsh(gap)) < -1e-12 * max(np.linalg.norm(mass*target[a], ord=2), np.linalg.norm(k, ord=2)):
            raise ValueError("target moments fail dissipative gate")
        h = np.einsum('n,n,ni->i',batch.mass,batch.velocity[:,a]-v[a],r) + np.einsum('n,ni,nij->j',batch.mass,batch.affine[:,a,:],child[:,a])
        affine[a] = np.linalg.solve(target[a],h)/mass
    volume = batch.current_volume.sum()
    return ParticleBatch([x], [mass], [volume], [batch.reference_volume.sum()], [v], [affine],
        [batch.current_volume @ batch.composition / volume], ids)


def tait_J(pressure, bulk_modulus, exponent):
    p, k, gamma = _array([pressure,bulk_modulus,exponent], (3,))
    if k <= 0 or gamma <= 0 or p <= -k/gamma:
        raise ValueError("invalid Tait EOS state")
    result = (1 + gamma*p/k)**(-1/gamma)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("unrepresentable Tait EOS state")
    return float(result)


def reseed_cell(volume, density, J, count, composition):
    volume, density, J = _array([volume,density,J], (3,))
    if min(volume,density,J) <= 0 or isinstance(count, (bool,np.bool_)) or not isinstance(count,(int,np.integer)) or count <= 0:
        raise ValueError("invalid reseeding state")
    f = _array(composition)
    if f.ndim != 1 or not len(f): raise ValueError("invalid composition")
    _fractions(f)
    # Round only after each complete expression, including on binary64 platforms.
    v, rho, j = (Fraction.from_float(float(x)) for x in (volume, density, J))
    try:
        values = np.array([float(v * rho / count), float(v / count),
                           float(v / (count * j))])
    except OverflowError as exc:
        raise ValueError("unrepresentable reseeding state") from exc
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("unrepresentable reseeding state")
    return {'mass': np.full(count,values[0]), 'current_volume': np.full(count,values[1]),
            'reference_volume': np.full(count,values[2]), 'composition': np.tile(f,(count,1))}
