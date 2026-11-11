from itertools import product

import numpy as np
import pytest

from fluidity_reference.adaptive_remap import (
    prolong_cell_inventory,
    prolong_face_flux,
    restrict_cell_inventories,
    restrict_face_flux,
)


def test_affine_inventory_round_trip_conserves_every_channel():
    parent = np.array([8.0, 6.0, 4.0, -2.0, 0.0, 3.0, -5.0, 1.0, 3.0])
    slopes = np.array([
        [1.0, 0.0, 0.5, -0.25, 0.5, 0.0, 1.0, 0.25, 0.25],
        [0.5, 0.5, -0.5, 0.5, -0.25, 0.5, 0.0, -0.25, -0.25],
        [0.0, 0.0, 0.25, 0.25, 0.5, -0.5, 0.5, 0.0, 0.25],
    ])
    neighbors = np.stack((parent - slopes, parent + slopes), axis=1)
    children = prolong_cell_inventory(parent, neighbors)
    offsets = np.array(list(product((-0.25, 0.25), repeat=3)))
    expected = (parent + offsets @ slopes) / 8
    np.testing.assert_array_equal(children, expected)
    np.testing.assert_array_equal(restrict_cell_inventories(children), parent)


def test_shared_limiter_keeps_children_inside_stencil_range():
    parent = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75])
    neighbors = np.tile(parent, (3, 2, 1))
    for axis in range(3):
        neighbors[axis, 0, [2, 7, 8]] = [1.0, 0.0, 1.0]
        neighbors[axis, 1, [2, 7, 8]] = [1.0, 0.5, 0.5]
    children = prolong_cell_inventory(parent, neighbors)
    volumes = 8 * children[:, 7:]
    assert np.min(volumes[:, 0]) == 0.0625
    assert np.max(volumes[:, 0]) == 0.4375
    np.testing.assert_array_equal(children[:, 7:].sum(axis=1), children[:, 2])
    np.testing.assert_array_equal(restrict_cell_inventories(children), parent)


def test_signed_flux_counts_each_owner_once():
    owner_ids = [9, 2, 7, 4]
    orientation = [1, -1, 1, -1]
    flux = [0.375, 0.125, 0.25, 0.25]
    assert restrict_face_flux(owner_ids, flux, orientation) == 0.25
    fine_flux = prolong_face_flux(-0.25, orientation)
    np.testing.assert_array_equal(fine_flux, [-0.0625, 0.0625, -0.0625, 0.0625])
    assert restrict_face_flux(owner_ids, fine_flux, orientation) == -0.25
    with pytest.raises(ValueError, match="unique"):
        restrict_face_flux([9, 2, 7, 9], flux, orientation)


def test_rejected_prolongation_keeps_inputs_and_refuses_lost_inventory():
    parent = np.array([np.nextafter(0.0, 1.0), 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75])
    neighbors = np.tile(parent, (3, 2, 1))
    original = parent.copy()
    with pytest.raises(ValueError, match="conserve"):
        prolong_cell_inventory(parent, neighbors)
    np.testing.assert_array_equal(parent, original)
    np.testing.assert_array_equal(neighbors, np.tile(original, (3, 2, 1)))
    parent[7] = 0.5
    with pytest.raises(ValueError, match="constituent"):
        prolong_cell_inventory(parent, neighbors)
    with pytest.raises(ValueError, match="finite"):
        restrict_face_flux([1, 2, 3, 4], [0.0, np.inf, 0.0, 0.0], [1, 1, 1, 1])





def test_total_volume_limits_all_constituent_slopes_together():
    parent = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5])
    neighbors = np.tile(parent, (3, 2, 1))
    neighbors[0, 0, 7:] -= [0.0625, 0.125, -0.1875]
    neighbors[0, 1, 7:] += [0.125, -0.03125, -0.09375]
    children = prolong_cell_inventory(parent, neighbors)
    np.testing.assert_array_equal(children, np.tile(parent / 8, (8, 1)))


def test_fractional_limiter_preserves_constituents_and_conservation():
    parent = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5])
    neighbors = np.tile(parent, (3, 2, 1))
    neighbors[0, 0, 7:] -= [0.0625, 0.125 - 1 / 256, -0.1875]
    neighbors[0, 1, 7:] += [0.125, -0.03125 - 1 / 256, -0.09375]
    neighbors[0, :, 2] = [1 + 1 / 256, 1 - 1 / 256]
    children = prolong_cell_inventory(parent, neighbors)
    assert np.max(8 * children[:, 2]) == 1 + 1 / 256
    assert np.max(8 * children[:, 7]) == 0.25 + 1 / 128
    np.testing.assert_array_equal(children[:, 7:].sum(axis=1), children[:, 2])
    np.testing.assert_array_equal(restrict_cell_inventories(children), parent)
