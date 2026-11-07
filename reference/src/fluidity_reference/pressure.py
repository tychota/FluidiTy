"""Small, directly assembled finite-volume pressure reference."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from .grid import Grid2D


@dataclass
class PressureInputs:
    grid: Grid2D
    dt: float
    velocity: object
    aperture: object = None
    mobility: object = None
    distance: object = None
    wall_velocity: object = None
    dirichlet_mask: object = None
    dirichlet_pressure: object = None
    jump: object = None
    source: object = None
    fluid_volume: object = None
    bulk_modulus: object = None
    previous_pressure: object = None


@dataclass
class PressureResult:
    pressure: np.ndarray
    velocity: np.ndarray
    matrix: object
    rhs: np.ndarray
    residual: np.ndarray
    constraint_defect: np.ndarray
    gauges: tuple


def _array(value, default, size, name, *, allow_inf=False):
    try:
        a = np.asarray(default if value is None else value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if a.shape != (size,) or np.any(np.isnan(a)) or (not allow_inf and not np.all(np.isfinite(a))):
        raise ValueError(f"invalid {name} shape or values")
    return a.copy()


def solve_pressure(inputs):
    g = inputs.grid
    if not isinstance(g, Grid2D) or not np.isscalar(inputs.dt) or not np.isfinite(inputs.dt) or inputs.dt <= 0:
        raise ValueError("invalid grid or dt")
    n, f, dt = g.cell_count, g.face_count, inputs.dt
    arr = lambda v, d, size, name: _array(v, d, size, name)
    u = arr(inputs.velocity, None, f, 'velocity')
    a_fraction = arr(inputs.aperture, np.ones(f), f, 'aperture')
    beta = arr(inputs.mobility, np.ones(f), f, 'mobility')
    dist = arr(inputs.distance, g.face_distances, f, 'distance')
    wall = arr(inputs.wall_velocity, np.zeros(f), f, 'wall_velocity')
    dp = arr(inputs.dirichlet_pressure, np.zeros(f), f, 'dirichlet_pressure')
    jump = arr(inputs.jump, np.zeros(f), f, 'jump')
    source = arr(inputs.source, np.zeros(n), n, 'source')
    vol = arr(inputs.fluid_volume, g.cell_volumes, n, 'fluid_volume')
    bulk_value = inputs.bulk_modulus
    if bulk_value is not None and np.ndim(bulk_value) == 0:
        bulk_value = np.full(n, bulk_value)
    bulk = _array(bulk_value, np.full(n, np.inf), n, 'bulk_modulus', allow_inf=True)
    old = arr(inputs.previous_pressure, np.zeros(n), n, 'previous_pressure')
    raw_mask = np.zeros(f, dtype=bool) if inputs.dirichlet_mask is None else np.asarray(inputs.dirichlet_mask)
    if raw_mask.shape != (f,) or raw_mask.dtype.kind != 'b':
        raise ValueError('invalid dirichlet_mask')
    mask = raw_mask.copy()
    if np.any((a_fraction < 0) | (a_fraction > 1)) or np.any(beta <= 0) or np.any(dist <= 0) or np.any(vol <= 0) or np.any(bulk <= 0) or np.any(np.isneginf(bulk)):
        raise ValueError('invalid pressure coefficients')
    R = g.incidence
    counts = np.asarray(abs(R).sum(axis=0)).ravel()
    internal = counts == 2
    if np.any(mask & internal) or np.any(mask & (jump != 0)):
        raise ValueError('invalid Dirichlet face or jump')
    area = g.face_areas
    a = area * a_fraction
    active = (internal | mask) & (a > 0)
    k = np.where(active, dt * a * beta / dist, 0.)
    C = vol / (bulk * dt)
    # The external value is signed by the cell's outward incidence.
    external = np.asarray(R.sum(axis=0)).ravel() * dp * mask
    matrix = (R @ diags(k) @ R.T + diags(C)).tocsr()
    rhs = vol * source - R @ (a * u + (area - a) * wall) - R @ (k * jump) + R @ (k * external) + C * old
    graph = (abs(R[:, internal]) @ diags((k[internal] > 0).astype(float)) @ abs(R[:, internal]).T).tocsr()
    count, labels = connected_components(graph, directed=False)
    gauges = []
    for component in range(count):
        cells = np.flatnonzero(labels == component)
        anchored = np.any(C[cells] > 0) or np.any(np.asarray(abs(R[cells, :]) @ (k * mask)).ravel() > 0)
        if not anchored:
            total = np.sum(rhs[cells])
            scale = np.sum(np.abs(rhs[cells]))
            if abs(total) > 1e-12 * max(scale, np.finfo(float).tiny):
                raise ValueError('incompatible Neumann component')
            gauges.append(int(cells[0]))
    system = matrix.tolil(copy=True)
    solve_rhs = rhs.copy()
    for cell in gauges:
        system[cell, :] = 0
        system[:, cell] = 0
        system[cell, cell] = 1
        solve_rhs[cell] = 0
    pressure = np.asarray(spsolve(system.tocsr(), solve_rhs))
    velocity = u.copy()
    velocity[active] += (dt * beta / dist * (R.T @ pressure + jump - external))[active]
    residual = rhs - matrix @ pressure
    defect = (R @ (a * velocity + (area - a) * wall)) / vol - source + (pressure - old) / (bulk * dt)
    return PressureResult(pressure, velocity, matrix, rhs, residual, defect, tuple(gauges))


def bounded_band_source(error, volumes, band, maximum):
    error = np.asarray(error, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    band = np.asarray(band)
    if error.ndim != 1 or volumes.shape != error.shape or band.shape != error.shape or band.dtype.kind != 'b' or not np.all(np.isfinite(error)) or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0) or not np.isscalar(maximum) or not np.isfinite(maximum) or maximum < 0:
        raise ValueError('invalid band source inputs')
    result = np.zeros_like(error)
    if not np.any(band) or maximum == 0:
        return result
    # Decimal preserves differences smaller than an ulp of a large error.  The
    # finite binary64 input range requires fewer than 2200 decimal digits even
    # for products of subnormal weights and extreme differences.
    with localcontext() as ctx:
        ctx.prec = 2200
        e = [Decimal.from_float(float(x)) for x in error[band]]
        v = [Decimal.from_float(float(x)) for x in volumes[band]]
        bound = Decimal.from_float(float(maximum))
        points = sorted({x - bound for x in e} | {x + bound for x in e})

        def balance(offset):
            return sum((weight * max(-bound, min(bound, x - offset))
                        for x, weight in zip(e, v)), Decimal(0))

        lo, hi = 0, len(points) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if balance(points[mid]) > 0:
                lo = mid
            else:
                hi = mid
        left, right = points[lo], points[hi]
        left_balance, right_balance = balance(left), balance(right)
        if left_balance == 0:
            offset = left
        elif right_balance == 0:
            offset = right
        else:
            offset = left + (right - left) * left_balance / (left_balance - right_balance)
        result[band] = [float(max(-bound, min(bound, x - offset))) for x in e]

    # Check in normalized units to avoid overflowing finite input products.
    weights = volumes[band] / np.max(volumes[band])
    scaled = result[band] / maximum
    terms = weights * scaled
    imbalance = abs(math.fsum(terms))
    tolerance = 8 * np.finfo(float).eps * max(math.fsum(abs(x) for x in terms), 1.)
    if imbalance > tolerance:
        raise ArithmeticError('band source weighted balance lost to rounding')
    return result
