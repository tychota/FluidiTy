"""Complete uniform quadratic B-spline stencils."""

from itertools import product
from numbers import Real

import numpy as np


def quadratic_stencil(position, h, offset):
    """Return nodes, weights and physical-position gradients on a shifted lattice."""
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, Real) or not np.isfinite(h) or h <= 0:
        raise ValueError("spacing must be finite and positive")
    position = np.asarray(position, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if position.ndim != 1 or position.size == 0 or offset.shape != position.shape or not np.all(np.isfinite(position)) or not np.all(np.isfinite(offset)):
        raise ValueError("position and offset must be finite vectors of equal shape")
    coordinate = position / h - offset
    base = np.floor(coordinate - .5).astype(np.int64)
    indices = np.array(list(product(range(3), repeat=position.size)), dtype=np.int64)
    nodes = base + indices
    delta = coordinate - nodes
    absolute = np.abs(delta)
    axis_weights = np.where(absolute < .5, .75 - delta**2, .5 * (1.5 - absolute)**2)
    axis_gradients = np.where(absolute < .5, -2 * delta, -(1.5 - absolute) * np.sign(delta)) / h
    weights = np.prod(axis_weights, axis=1)
    gradients = np.empty_like(axis_weights)
    for axis in range(position.size):
        gradients[:, axis] = axis_gradients[:, axis] * np.prod(np.delete(axis_weights, axis, axis=1), axis=1)
    return nodes, weights, gradients
