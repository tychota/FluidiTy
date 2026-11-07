from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr

from fluidity_reference.grid import Grid2D
from fluidity_reference.basis import quadratic_stencil


def test_mac_geometry_and_incidence():
    grid = Grid2D(2, 1, 1.0)
    assert grid.cell_count == 2
    assert grid.face_count == 7
    np.testing.assert_allclose(grid.cell_centers, [[.5, .5], [1.5, .5]])
    np.testing.assert_allclose(grid.face_centers, [[0,.5], [1,.5], [2,.5], [.5,0], [1.5,0], [.5,1], [1.5,1]])
    assert isspmatrix_csr(grid.incidence)
    np.testing.assert_array_equal(grid.incidence.toarray(), [[-1,1,0,-1,0,1,0], [0,-1,1,0,-1,0,1]])
    np.testing.assert_allclose(grid.face_areas, np.ones(7))
    np.testing.assert_allclose(grid.face_distances, [.5,1,.5,.5,.5,.5,.5])
    np.testing.assert_allclose(grid.cell_volumes, [1,1])
    with pytest.raises(FrozenInstanceError):
        grid.h = 2


def test_geometry_respects_scale_origin_and_row_major_order():
    grid = Grid2D(1, 2, 2., (3., -1.))
    np.testing.assert_allclose(grid.cell_centers, [[4,0], [4,2]])
    np.testing.assert_allclose(grid.face_centers, [[3,0],[5,0],[3,2],[5,2],[4,-1],[4,1],[4,3]])
    np.testing.assert_allclose(grid.face_distances, [1,1,1,1,1,2,1])
    np.testing.assert_allclose(grid.face_areas, 2 * np.ones(7))
    np.testing.assert_allclose(grid.cell_volumes, [4,4])
    np.testing.assert_array_equal(grid.incidence.toarray(), [[-1,1,0,0,-1,1,0],[0,0,-1,1,0,-1,1]])


@pytest.mark.parametrize('nx,ny,h,origin', [(0,1,1.,(0,0)), (1,0,1.,(0,0)), (True,1,1.,(0,0)), (1,False,1.,(0,0)), (1.5,1,1.,(0,0)), (1,1,0.,(0,0)), (1,1,-1.,(0,0)), (1,1,float('inf'),(0,0)), (1,1,float('nan'),(0,0)), (1,1,1.,(0,)), (1,1,1.,(0,float('nan')))])
def test_invalid_geometry_is_rejected(nx, ny, h, origin):
    with pytest.raises((ValueError, TypeError)):
        Grid2D(nx, ny, h, origin)


@pytest.mark.parametrize('dimension', [2, 3])
def test_quadratic_complete_stencil_moments_and_affine_reproduction(dimension):
    rng = np.random.default_rng(2026 + dimension)
    h = 1.7
    offset = np.arange(dimension) / 2
    coefficients = np.arange(1, dimension + 1)
    for _ in range(50):
        position = rng.uniform(-5, 5, dimension)
        nodes, weights, gradients = quadratic_stencil(position, h, offset)
        assert nodes.shape == (3**dimension, dimension)
        assert np.issubdtype(nodes.dtype, np.integer)
        assert gradients.shape == nodes.shape
        samples = h * (nodes + offset)
        delta = samples - position
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-14)
        np.testing.assert_allclose(weights @ delta, np.zeros(dimension), atol=1e-14)
        np.testing.assert_allclose(gradients.sum(axis=0), np.zeros(dimension), atol=1e-14)
        np.testing.assert_allclose(weights @ (samples @ coefficients + 3), position @ coefficients + 3, atol=1e-13)
        np.testing.assert_allclose((samples @ coefficients + 3) @ gradients, coefficients, atol=1e-13)
        np.testing.assert_allclose((delta.T * weights) @ delta, h*h/4 * np.eye(dimension), atol=1e-14)


@pytest.mark.parametrize('position,h,offset', [([0,0],0,[0,0]), ([0,0],float('nan'),[0,0]), ([0,float('inf')],1,[0,0]), ([0,0],1,[0]), ([0,0],1,[0,float('nan')])])
def test_invalid_stencil_inputs_are_rejected(position, h, offset):
    with pytest.raises((ValueError, TypeError)):
        quadratic_stencil(np.array(position), h, np.array(offset))
