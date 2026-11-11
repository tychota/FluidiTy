"""Conservative inventory transfer for one aligned 2:1 Cartesian remap."""

from itertools import product
from math import fsum

import numpy as np

# Inventory order is mass, reference volume, current volume, xyz momentum,
# enthalpy and one or more constituent volumes. Momentum and enthalpy are signed.
_CURRENT_VOLUME = 2
_CONSTITUENTS = 7
_CHILD_OFFSETS = np.array(list(product((-0.25, 0.25), repeat=3)))
_RELATIVE_ROUNDOFF = 64 * np.finfo(float).eps


def _real_array(values):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError("remap values must be real")
    with np.errstate(over="ignore"):
        array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("remap values must be finite")
    return array


def _finite_sum(values):
    try:
        total = fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError("remap sum is unrepresentable") from exc
    if not np.isfinite(total):
        raise ValueError("remap sum is unrepresentable")
    return total


def _validate_inventories(rows):
    if rows.shape[1] <= _CONSTITUENTS:
        raise ValueError("remap requires constituent volume channels")
    if np.any(rows[:, :3] < 0) or np.any(rows[:, _CONSTITUENTS:] < 0):
        raise ValueError("mass, volume and constituent inventories must be nonnegative")
    for row in rows:
        total = _finite_sum(row[_CONSTITUENTS:])
        current_volume = row[_CURRENT_VOLUME]
        if abs(total - current_volume) > _RELATIVE_ROUNDOFF * max(total, current_volume):
            raise ValueError("constituent volumes do not sum to current volume")


def _check_conservation(children, parent):
    for child_values, parent_value in zip(children.T, parent, strict=True):
        child_sum = _finite_sum(child_values)
        scale = max(abs(parent_value), np.max(np.abs(child_values)))
        if abs(child_sum - parent_value) > _RELATIVE_ROUNDOFF * scale:
            raise ValueError("prolongation cannot conserve an inventory at this precision")


def restrict_cell_inventories(children):
    """Sum eight child rows in lexicographic xyz order into a new parent row."""
    children = _real_array(children)
    if children.ndim != 2 or children.shape[0] != 8:
        raise ValueError("restriction requires eight child inventory rows")
    _validate_inventories(children)
    parent = np.array([_finite_sum(values) for values in children.T])
    _validate_inventories(parent[None, :])
    return parent


def _minmod(left, right):
    return np.where(
        np.sign(left) == np.sign(right), np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0,
    )


def prolong_cell_inventory(parent, neighbors):
    """Return eight extensive rows from a parent and its three same-size neighbor pairs.

    Neighbor axes are x, y, z and each pair is low, high. Child order is
    lexicographic xyz. A shared limiter preserves the relation between total
    current volume and its constituent volumes.
    """
    parent = _real_array(parent)
    neighbors = _real_array(neighbors)
    if parent.ndim != 1 or neighbors.shape != (3, 2, parent.size):
        raise ValueError("prolongation requires a parent and three low/high neighbor pairs")
    stencil = np.vstack((parent, neighbors.reshape(6, parent.size)))
    _validate_inventories(stencil)

    try:
        with np.errstate(over="raise", invalid="raise"):
            slopes = _minmod(parent - neighbors[:, 0], neighbors[:, 1] - parent)
            for axis in range(3):
                slopes[axis, _CURRENT_VOLUME] = _finite_sum(slopes[axis, _CONSTITUENTS:])
            offsets = _CHILD_OFFSETS @ slopes
            maxima = np.max(offsets, axis=0)
            minima = np.min(offsets, axis=0)
            upper = stencil.max(axis=0) - parent
            lower = stencil.min(axis=0) - parent
            limiter = 1.0
            for room, delta in zip(upper, maxima, strict=True):
                if delta > room:
                    limiter = min(limiter, room / delta)
            for room, delta in zip(lower, minima, strict=True):
                if delta < room:
                    limiter = min(limiter, room / delta)
            children = parent / 8 + (limiter / 8) * offsets
    except FloatingPointError as exc:
        raise ValueError("prolongation arithmetic is unrepresentable") from exc
    _validate_inventories(children)
    _check_conservation(children, parent)
    return children


def _orientation_signs(signs):
    signs = _real_array(signs)
    if signs.shape != (4,) or not np.all(np.abs(signs) == 1):
        raise ValueError("four subface orientations must be +1 or -1")
    return signs


def restrict_face_flux(owner_ids, fine_fluxes, normal_signs):
    """Sum four integrated subface fluxes in coarse-normal orientation."""
    owner_ids = np.asarray(owner_ids)
    if owner_ids.shape != (4,) or owner_ids.dtype.kind not in "iu" or np.unique(owner_ids).size != 4:
        raise ValueError("restriction requires four unique subface owners")
    fine_fluxes = _real_array(fine_fluxes)
    normal_signs = _orientation_signs(normal_signs)
    if fine_fluxes.shape != (4,):
        raise ValueError("restriction requires four integrated subface fluxes")
    owner_order = np.argsort(owner_ids)
    return _finite_sum((normal_signs * fine_fluxes)[owner_order])


def prolong_face_flux(coarse_flux, normal_signs):
    """Distribute integrated flux over four equal-area subfaces."""
    coarse_flux = _real_array(coarse_flux)
    if coarse_flux.ndim != 0:
        raise ValueError("prolongation requires one integrated coarse flux")
    normal_signs = _orientation_signs(normal_signs)
    share = float(coarse_flux) / 4
    if share * 4 != coarse_flux:
        raise ValueError("face prolongation cannot conserve flux at this precision")
    return normal_signs * share
