"""Manufactured planar extension at a grid-aligned viscosity discontinuity."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve

from .grid import Grid2D
from .pressure import PressureInputs, solve_pressure
from .viscosity import mac_strain, viscous_step


@dataclass(frozen=True)
class TractionError:
    normal_rms: float
    normal_max: float
    tangential_rms: float
    tangential_max: float


def _coupled_step(velocity, strain, weights, wall_strain, divergence, wall_flux, mass, dt):
    stiffness = strain.T @ diags(weights) @ strain
    wall_load = strain.T @ (weights * wall_strain)
    momentum = mass * eye(strain.shape[1]) + dt * stiffness
    # Eliminate the pressure at the first cell. Its divergence row is redundant.
    divergence = divergence[1:]
    system = bmat([[momentum, -dt * divergence.T], [divergence, None]], format='csc')
    rhs = np.r_[mass * velocity + dt * wall_load, -wall_flux[1:]]
    result = spsolve(system, rhs)
    return result[:velocity.size], np.r_[0.0, result[velocity.size:]]


def measure_extension(resolution, dt, viscosity_ratio, *, coupled):
    """Measure one step on the predeclared unit square extension fixture.

    Return stress errors normalized by 2 μ_a U/L. RMS is area-weighted
    on each side and the larger side error is reported.
    """
    if resolution not in (16, 32) or dt not in (0.005, 0.0025) or viscosity_ratio not in (1, 10, 1000):
        raise ValueError('case is outside the predeclared traction campaign')
    grid = Grid2D(resolution, resolution, 1.0 / resolution)
    strain, weights = mac_strain(grid, 1.0)
    midpoint = resolution // 2
    viscosity = np.ones((resolution, resolution))
    viscosity[:, midpoint:] = viscosity_ratio
    # Normal strain uses each cell's viscosity. Interior edge shear uses
    # the harmonic mean of the four contributing cells.
    inverse_viscosity = 1.0 / viscosity
    edge_viscosity = 4.0 / (inverse_viscosity[:-1, :-1] + inverse_viscosity[1:, :-1]
                            + inverse_viscosity[:-1, 1:] + inverse_viscosity[1:, 1:])
    weights *= np.r_[viscosity.ravel(), viscosity.ravel(), edge_viscosity.ravel()]

    face_x_count = resolution * (resolution + 1)
    world_x, world_y = grid.face_centers.T
    initial_velocity = np.r_[world_x[:face_x_count] - 0.5, 0.5 - world_y[face_x_count:]]
    incidence = grid.incidence
    unknown = np.asarray(abs(incidence).sum(axis=0)).ravel() == 2
    boundary = ~unknown
    active_strain = strain[:, unknown]
    wall_strain = -(strain[:, boundary] @ initial_velocity[boundary])
    active_velocity = initial_velocity[unknown]
    face_mass = grid.h**2

    if coupled:
        divergence = grid.h * incidence
        velocity, pressure = _coupled_step(
            active_velocity, active_strain, weights, wall_strain, divergence[:, unknown],
            divergence[:, boundary] @ initial_velocity[boundary], face_mass, dt)
        final_velocity = initial_velocity.copy()
        final_velocity[unknown] = velocity
    else:
        diffusion = viscous_step(active_velocity, np.full(active_velocity.size, face_mass),
                                 active_strain, weights, wall_strain, dt)
        # The manufactured field already has zero divergence with its wall
        # motion. Project the increment to avoid cancellation at near-zero RHS.
        increment = np.zeros_like(initial_velocity)
        increment[unknown] = diffusion.velocity - active_velocity
        projection = solve_pressure(PressureInputs(
            grid=grid, dt=dt, velocity=increment, aperture=unknown))
        final_velocity = initial_velocity + projection.velocity
        pressure = projection.pressure

    stress_scale = 2.0
    # The pressure pin is in phase a. Match its analytic pressure before
    # evaluating absolute normal traction on the two sides.
    pressure = pressure.reshape(viscosity.shape) + 2.0 + stress_scale
    final_strain = strain @ final_velocity
    normal_strain = final_strain[:grid.cell_count].reshape(viscosity.shape)
    normal_traction = -pressure + 2.0 * viscosity * normal_strain
    normal_error = (normal_traction[:, [midpoint - 1, midpoint]] + stress_scale) / stress_scale
    shear_strain = final_strain[2 * grid.cell_count:].reshape(resolution - 1, resolution - 1)
    # Interpolate the nearest same-phase edge shears along the interface.
    side_shear = shear_strain[:, [midpoint - 2, midpoint]]
    side_shear = np.pad(side_shear, ((1, 1), (0, 0)), mode='edge')
    tangential_error = (side_shear[:-1] + side_shear[1:]) * np.array([1.0, viscosity_ratio]) / stress_scale
    return TractionError(
        float(np.max(np.sqrt(np.mean(normal_error**2, axis=0)))),
        float(np.max(np.abs(normal_error))),
        float(np.max(np.sqrt(np.mean(tangential_error**2, axis=0)))),
        float(np.max(np.abs(tangential_error))),
    )


def measure_couette(resolution, dt, viscosity_ratio):
    """Measure the tangential block of the periodic-x Couette fixture."""
    if resolution not in (16, 32) or dt not in (0.005, 0.0025) or viscosity_ratio not in (1, 10, 1000):
        raise ValueError('case is outside the predeclared traction campaign')
    h = 1.0 / resolution
    midpoint = resolution // 2
    viscosity = np.ones(resolution)
    viscosity[midpoint:] = viscosity_ratio
    exact_traction = 1.0 / (0.5 + 0.5 / viscosity_ratio)
    y = h * (np.arange(resolution) + 0.5)
    initial_velocity = exact_traction * (np.minimum(y, 0.5) + np.maximum(y - 0.5, 0.0) / viscosity_ratio)

    # The uniform-in-x field has only shear strain. The two Dirichlet
    # wall rows are half-length intervals and the mixed row uses series resistance.
    face_indices = np.arange(resolution)
    rows = np.r_[face_indices, face_indices + 1]
    columns = np.r_[face_indices, face_indices]
    coefficients = np.r_[np.full(resolution, 1.0 / h), np.full(resolution, -1.0 / h)]
    coefficients[[0, -1]] *= 2.0
    strain = csr_matrix((coefficients, (rows, columns)), shape=(resolution + 1, resolution))
    edge_viscosity = np.r_[viscosity[0], 2.0 / (1.0 / viscosity[:-1] + 1.0 / viscosity[1:]), viscosity[-1]]
    weights = h**2 * edge_viscosity
    weights[[0, -1]] *= 0.5
    wall_strain = np.zeros(resolution + 1)
    wall_strain[-1] = -2.0 / h
    result = viscous_step(initial_velocity, np.full(resolution, h**2), strain, weights, wall_strain, dt)
    traction = edge_viscosity * (strain @ result.velocity - wall_strain)
    # Each phase's trace is constant along its side of the interface.
    tangential_error = float(np.max(np.abs(traction[midpoint - 1:midpoint + 2] - exact_traction)) / 2.0)
    return TractionError(0.0, 0.0, tangential_error, tangential_error)


if __name__ == '__main__':
    from dataclasses import asdict
    import json

    for resolution in (16, 32):
        for dt in (0.005, 0.0025):
            for ratio in (1, 10, 1000):
                cases = (
                    ('extension_fractional', measure_extension(resolution, dt, ratio, coupled=False)),
                    ('extension_coupled', measure_extension(resolution, dt, ratio, coupled=True)),
                    ('couette', measure_couette(resolution, dt, ratio)),
                )
                for scene, error in cases:
                    print(json.dumps(dict(scene=scene, resolution=resolution, dt=dt,
                                          viscosity_ratio=ratio, **asdict(error))))
