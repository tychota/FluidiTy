"""Float64 APIC transfers on complete uniform 2D MAC supports."""

import numpy as np

from .basis import quadratic_stencil
from .grid import Grid2D


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"invalid {name}")
    return result


def local_apic_fit(position, sample_positions, weights, values):
    """Gather a scalar and fit its affine row using the centered full moment."""
    position = np.asarray(position, dtype=np.float64)
    if position.ndim != 1 or not position.size or not np.all(np.isfinite(position)):
        raise ValueError("invalid position")
    samples = np.asarray(sample_positions, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != position.size or not samples.shape[0] or not np.all(np.isfinite(samples)):
        raise ValueError("invalid sample positions")
    weights = _array(weights, (len(samples),), "weights")
    values = _array(values, (len(samples),), "values")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=1e-12, atol=1e-12):
        raise ValueError("weights must form a partition")
    offsets = samples - position
    centered = offsets - weights @ offsets
    moment = (centered * weights[:, None]).T @ centered
    rhs = centered.T @ (weights * values)
    if np.linalg.matrix_rank(moment) != position.size:
        raise ValueError("singular moment")
    return float(weights @ values), np.linalg.solve(moment, rhs)


def local_velocity_gradient(values, gradients):
    """Evaluate the true interpolation gradient, independently of the affine fit."""
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or not gradients.shape[0] or not gradients.shape[1] or not np.all(np.isfinite(gradients)):
        raise ValueError("invalid gradients")
    values = _array(values, (len(gradients),), "values")
    return values @ gradients


def _positions(grid, positions):
    if not isinstance(grid, Grid2D):
        raise ValueError("invalid grid")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.all(np.isfinite(positions)):
        raise ValueError("invalid positions")
    return positions


def _stencil(grid, position, component):
    offset = (0., .5) if component == 0 else (.5, 0.)
    nodes, weights, gradients = quadratic_stencil(position - grid.origin, grid.h, offset)
    width, height = ((grid.nx + 1, grid.ny) if component == 0 else (grid.nx, grid.ny + 1))
    if np.any(nodes < 0) or np.any(nodes[:, 0] >= width) or np.any(nodes[:, 1] >= height):
        raise ValueError("incomplete support")
    indices = nodes[:, 1] * width + nodes[:, 0]
    if component:
        indices += grid.ny * (grid.nx + 1)
    samples = grid.origin + grid.h * (nodes + offset)
    return indices, samples, weights, gradients


def p2g(grid, positions, masses, velocities, affine):
    positions = _positions(grid, positions)
    n = len(positions)
    masses = _array(masses, (n,), "masses")
    velocities = _array(velocities, (n, 2), "velocities")
    affine = _array(affine, (n, 2, 2), "affine")
    if np.any(masses < 0):
        raise ValueError("masses must be nonnegative")
    face_mass = np.zeros(grid.face_count)
    momentum = np.zeros(grid.face_count)
    for p, position in enumerate(positions):
        for a in range(2):
            indices, samples, weights, _ = _stencil(grid, position, a)
            offsets = samples - position
            centered = offsets - weights @ offsets
            contributions = weights * masses[p]
            np.add.at(face_mass, indices, contributions)
            np.add.at(momentum, indices, contributions * (velocities[p, a] + centered @ affine[p, a]))
    return face_mass, momentum


def g2p(grid, positions, old_faces, new_faces):
    positions = _positions(grid, positions)
    old_faces = _array(old_faces, (grid.face_count,), "old faces")
    new_faces = _array(new_faces, (grid.face_count,), "new faces")
    pic = np.zeros((len(positions), 2))
    increments = np.zeros_like(pic)
    affine = np.zeros((len(positions), 2, 2))
    gradient = np.zeros_like(affine)
    for p, position in enumerate(positions):
        for a in range(2):
            indices, samples, weights, gradients = _stencil(grid, position, a)
            pic[p, a], affine[p, a] = local_apic_fit(position, samples, weights, new_faces[indices])
            increments[p, a] = pic[p, a] - weights @ old_faces[indices]
            gradient[p, a] = local_velocity_gradient(new_faces[indices], gradients)
    return pic, increments, affine, gradient


def energy_velocity(pic, residual, energy, phase, separation=1.):
    """Apply ASFLIP history residual to stored and advecting velocities."""
    pic = np.asarray(pic, dtype=np.float64)
    if pic.ndim == 0 or pic.shape[-1] != 2 or not np.all(np.isfinite(pic)):
        raise ValueError("invalid PIC velocity")
    residual = _array(residual, pic.shape, "residual")
    try:
        energy = float(energy)
        separation = float(separation)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid energy or separation") from exc
    if not np.isfinite(energy) or not 0 <= energy <= 1 or not np.isfinite(separation) or not 0 <= separation <= 1:
        raise ValueError("energy and separation must be in [0, 1]")
    if phase not in ('liquid', 'gas', 'elastic', 'granular', 'snow'):
        raise ValueError("invalid phase")
    alpha = 0. if phase == 'gas' else .99 * energy
    beta = 0. if phase in ('gas', 'elastic') else energy * separation
    return pic + alpha * residual, pic + beta * alpha * residual
