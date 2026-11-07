"""Uniform MAC strain and small mass-metric diffusion steps.

Boundary strain is a prescribed lifting in strain space, so the load is
E.T @ (W * boundary_strain). This is an algebraic reference, not a boundary
traction discretization. The row-sum bound is computed in float64 and is not
a production upper-rounding certificate.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import spsolve

from .grid import Grid2D


@dataclass
class ViscosityResult:
    velocity: np.ndarray
    method: str
    spectral_bound: float
    energy_before: float
    energy_after: float
    wall_work: float
    numerical_dissipation: float


def _scalar(value, name, minimum, *, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError(f'invalid {name}')
    return float(value)


def mac_strain(grid, viscosity: float):
    if not isinstance(grid, Grid2D):
        raise ValueError('invalid grid')
    mu = _scalar(viscosity, 'viscosity', 0.)
    nx, ny, h = grid.nx, grid.ny, grid.h
    cells = grid.cell_count
    vertices = (nx - 1) * (ny - 1)
    y_start = ny * (nx + 1)
    rows, cols, values = [], [], []

    def add(row, entries):
        for col, value in entries:
            rows.append(row); cols.append(col); values.append(value)

    for j in range(ny):
        for i in range(nx):
            cell = j * nx + i
            add(cell, [(j*(nx+1)+i, -1/h), (j*(nx+1)+i+1, 1/h)])
            add(cells+cell, [(y_start+j*nx+i, -1/h), (y_start+(j+1)*nx+i, 1/h)])
    for j in range(1, ny):
        for i in range(1, nx):
            row = 2*cells + (j-1)*(nx-1) + i-1
            add(row, [((j-1)*(nx+1)+i, -.5/h), (j*(nx+1)+i, .5/h),
                      (y_start+j*nx+i-1, -.5/h), (y_start+j*nx+i, .5/h)])
    E = csr_matrix((values, (rows, cols)), shape=(2*cells+vertices, grid.face_count))
    W = np.r_[np.full(2*cells, 2*mu*h*h), np.full(vertices, 4*mu*h*h)]
    if not np.all(np.isfinite(E.data)) or not np.all(np.isfinite(W)):
        raise ValueError('nonfinite strain coefficients')
    return E, W


def _vector(value, size, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid {name}') from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'invalid {name}')
    return result.copy()


def viscous_step(velocity, mass, E, W, boundary_strain, dt, mode='auto', safety=.01):
    dt = _scalar(dt, 'dt', 0.)
    safety = _scalar(safety, 'safety', 0.)
    if safety >= 1 or mode not in ('auto', 'explicit', 'implicit'):
        raise ValueError('invalid safety or mode')
    if not issparse(E) or len(E.shape) != 2 or E.shape[1] == 0:
        raise ValueError('E must be a sparse matrix with velocity columns')
    try:
        E = csr_matrix(E, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid E') from exc
    if not np.all(np.isfinite(E.data)):
        raise ValueError('invalid E')
    nrows, n = E.shape
    u = _vector(velocity, n, 'velocity')
    m = _vector(mass, n, 'mass')
    w = _vector(W, nrows, 'W')
    wall = _vector(boundary_strain, nrows, 'boundary_strain')
    if np.any(m <= 0) or np.any(w < 0):
        raise ValueError('invalid masses or weights')
    K = (E.T @ diags(w) @ E).tocsr()
    scale = diags(1 / np.sqrt(m))
    S = scale @ K @ scale
    bound = float(np.max(np.asarray(abs(S).sum(axis=1))))
    load = np.asarray(E.T @ (w * wall)).ravel()
    if not np.isfinite(bound) or not np.all(np.isfinite(load)):
        raise ValueError('nonfinite assembled coefficients')
    safe = dt * bound <= 2 * (1 - safety)
    if mode == 'explicit' and not safe:
        raise ValueError('explicit step exceeds stability bound')
    method = 'explicit' if mode == 'explicit' or (mode == 'auto' and safe) else 'implicit'
    if method == 'explicit':
        new = u + dt * (load - K @ u) / m
        wall_work = dt * float(u @ load)
    else:
        new = np.asarray(spsolve(diags(m) + dt*K, m*u + dt*load))
        wall_work = dt * float(new @ load)
    delta = new - u
    return ViscosityResult(new, method, bound, float(.5*np.dot(m*u, u)),
                           float(.5*np.dot(m*new, new)), wall_work,
                           float(.5*np.dot(m*delta, delta)))
