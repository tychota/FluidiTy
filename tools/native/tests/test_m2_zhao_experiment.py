import itertools

import numpy as np
import pytest

from tools.native.m2_zhao_experiment import feature_patches, field_values, pruning_classes


def _scalar_patch(points):
    def kernel(delta):
        return np.prod(np.maximum(0, 1 - (delta / 5) ** 2) ** 2, axis=-1)

    density = []
    for point in points:
        cell = np.floor(point)
        fraction = point - cell
        total = 0
        for bits in itertools.product((0, 1), repeat=3):
            corner = np.array(bits)
            weight = np.prod(np.where(corner, fraction, 1 - fraction))
            total += weight * np.sum(kernel(cell + corner - points))
        density.append(total)
    values = [np.sum(kernel(np.array(node) - points) / density)
              for node in itertools.product(range(-7, 8, 2), repeat=3)]
    return np.array(values, dtype=np.float32).reshape(8, 8, 8)


def test_patch_matches_independent_density_and_feature_oracle():
    points = np.array([[0.1, 0.2, 0.3], [-3.5, 1.1, 4.2], [4.75, -2.2, 0.1]])
    query = np.array([[0, 0, 0]])
    np.testing.assert_array_equal(feature_patches(points, query)[0], _scalar_patch(points))
    shift = np.array([-29, 3, 10])
    np.testing.assert_array_equal(feature_patches(points + shift, query + shift),
                                  feature_patches(points, query))


def test_missing_density_support_outside_direct_patch_changes_features():
    points = np.array([[11.8, 0, 0], [16.2, 0, 0]])
    query = np.array([[0, 0, 0]])
    full = feature_patches(points, query)
    missing = feature_patches(points[:1], query)
    assert full.max() > 0
    assert missing.max() > full.max()
    distant = np.append(points, [[18.01, 23, 0]], axis=0)
    np.testing.assert_array_equal(full, feature_patches(distant, query))


def test_pruning_respects_occupied_cells_and_half_open_windows():
    query = np.array([[0, 0, 0]])
    assert pruning_classes(np.array([[-4, 0, 0]]), query)[0] == 1
    assert pruning_classes(np.array([[4, 0, 0], [4.1, 0, 0]]), query)[0] == 0
    points = 2 * np.array(list(itertools.product(range(-3, 5), repeat=3)))
    assert pruning_classes(points[:411], query)[0] == -1
    assert pruning_classes(points[:410], query)[0] != -1
    assert pruning_classes(np.empty((0, 3)), query)[0] == 1


def test_nonfinite_input_fails_before_splat():
    with pytest.raises(ValueError, match="finite"):
        feature_patches(np.array([[0, np.nan, 0]]), np.array([[0, 0, 0]]))


def test_tiled_field_preserves_query_order_and_fixed_batches():
    points = np.array([[0.1, 0, 0], [12.2, 0, 0], [17.2, 0, 0], [20.5, 0, 0]])
    queries = np.array([[18, 0, 0], [0, 0, 0], [4, 0, 0], [12, 0, 0], [17, 0, 0]])
    observed = []

    def predict(batch):
        assert batch.dtype == np.float32
        assert batch.shape == (1024, 1, 8, 8, 8)
        observed.append(batch.copy())
        return batch.reshape(1024, -1).sum(axis=1, keepdims=True)

    full = field_values(points, queries, predict, 0.0125)
    full_batches = observed.copy()
    observed.clear()
    tiled = field_values(points, queries, predict, 0.0125, tile_width_cells=4)
    np.testing.assert_array_equal(full["values"], tiled["values"])
    np.testing.assert_array_equal(full["classes"], tiled["classes"])
    np.testing.assert_array_equal(full_batches, observed)
    assert full["batches"] == tiled["batches"] == 2
