from itertools import product

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from fluidity_reference.experiments import isolated_drop_geometry


H = 0.025
RADIUS = 0.125
VOLUME = 4 * np.pi * RADIUS**3 / 3
SEED_AXES = RADIUS * np.array([1 / np.sqrt(1.03), 1.03, 1 / np.sqrt(1.03)])


def _affine_sphere(axes=SEED_AXES, rotation=np.eye(3), center=np.zeros(3)):
    lattice = H * (np.array(list(product(range(-5, 5), repeat=3))) + 0.5)
    sphere = lattice[np.sum(lattice**2, axis=1) < RADIUS**2]
    return (sphere * (axes / RADIUS)) @ rotation.T + center


def _fit(points, volume=VOLUME):
    volumes = np.full(len(points), volume / len(points))
    return isolated_drop_geometry.fit_isolated_drops(points, volumes, H)


def test_affine_sphere_recovers_rotated_ellipsoid_and_axial_curvature():
    rotation = Rotation.from_euler("xyz", [19, 31, 47], degrees=True).as_matrix()
    center = H * np.array([0.21, -0.37, 0.43])
    points = _affine_sphere(rotation=rotation, center=center)
    drop, = _fit(points)

    np.testing.assert_allclose(drop.center, center, atol=1e-15)
    np.testing.assert_allclose(drop.semiaxes, sorted(SEED_AXES), rtol=1e-13)
    np.testing.assert_allclose(drop.level_set(center + (rotation * SEED_AXES).T), 0, atol=1e-14)

    # At the end of the y axis, the principal curvatures are b/a² and b/c².
    wet = center + 0.12 * rotation[:, 1]
    air = center + 0.14 * rotation[:, 1]
    assert drop.air_link_curvature(wet, air) == pytest.approx(16.9744, rel=1e-13)


def test_sphere_curvature_and_signed_distance():
    points = _affine_sphere(axes=np.full(3, RADIUS))
    drop, = _fit(points)

    assert drop.air_link_curvature([0.12, 0, 0], [0.145, 0, 0]) == pytest.approx(2 / RADIUS)
    np.testing.assert_allclose(
        drop.signed_distance([[0, 0, 0], [0.11, 0, 0], [0.145, 0, 0]]),
        [-0.125, -0.015, 0.02],
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="bracket"):
        drop.air_link_curvature([0.145, 0, 0], [0.155, 0, 0])


def test_ellipsoid_signed_distance_resolves_medial_axis_and_beta_band():
    # Affine sampling keeps the graph connected with these axis lengths.
    axes = np.array([0.115, 0.122, 0.129])
    points = _affine_sphere(axes=axes)
    drop, = _fit(points, 4 * np.pi * np.prod(axes) / 3)

    # For this interior major-axis point, the nearest surface point is off-axis.
    z = 0.01
    nearest_z = axes[2] ** 2 * z / (axes[2] ** 2 - axes[0] ** 2)
    nearest_x = axes[0] * np.sqrt(1 - (nearest_z / axes[2]) ** 2)
    distance = drop.signed_distance([[0, 0, z], [0, 0, 0.103], [0, 0, 0.149]])

    np.testing.assert_allclose(distance, [-np.hypot(nearest_x, nearest_z - z), -0.026, 0.02], atol=1e-14)
    assert distance[1] < -H
    assert -H <= drop.level_set([[0, 0, 0.103]])[0] <= 0


@pytest.mark.parametrize(
    ("phase", "angles", "wet_centers", "wet_corners"),
    [
        ([0, 0, 0], [0, 0, 0], 528, 511),
        ([0.21, -0.37, 0.43], [19, 31, 47], 527, 526),
    ],
)
def test_grid_visibility_at_seed_and_rotated_translated_phase(phase, angles, wet_centers, wet_corners):
    rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    points = _affine_sphere(rotation=rotation, center=H * np.asarray(phase))
    grid = H * np.array(list(product(range(-7, 8), repeat=3)))
    drop, = _fit(points)

    assert np.count_nonzero(drop.level_set(grid + 0.5 * H) < 0) == wet_centers
    assert np.count_nonzero(drop.level_set(grid) < 0) == wet_corners
    if not any(phase):
        assert len(points) == 552
        fixed_support = cKDTree(points)
        assert np.count_nonzero(fixed_support.query(grid + 0.5 * H)[0] < 0.6578 * H) == 552
        assert np.count_nonzero(fixed_support.query(grid)[0] < 0.6578 * H) == 0


@pytest.mark.parametrize("gap_in_cells", [1, 2])
def test_separated_drop_fits_leave_the_gap_dry(gap_in_cells):
    points = _affine_sphere()
    separation = 2 * SEED_AXES[0] + gap_in_cells * H
    left, right = _fit(np.concatenate((points, points + [separation, 0, 0])), 2 * VOLUME)
    gap_midpoint = [separation / 2, 0, 0]

    assert left.signed_distance(gap_midpoint) == pytest.approx(gap_in_cells * H / 2)
    assert right.signed_distance(gap_midpoint) == pytest.approx(gap_in_cells * H / 2)


def test_disconnected_clouds_with_overlapping_fits_are_rejected():
    points = _affine_sphere()
    separation = 2 * SEED_AXES[0] + H
    with pytest.raises(ValueError, match="separation"):
        _fit(np.concatenate((points, points + [separation, 0, 0])), 4 * VOLUME)


@pytest.mark.parametrize(
    ("points", "volumes"),
    [
        (np.zeros((4, 3)), np.ones(4)),
        (np.array([[0, 0, 0], [H, 0, 0], [0, H, 0], [H, H, 0]]), np.ones(4)),
        (np.array([[np.nan, 0, 0]]), np.ones(1)),
        (np.array([[0, 0, 0]]), np.zeros(1)),
    ],
)
def test_invalid_clouds_are_rejected(points, volumes):
    with pytest.raises(ValueError):
        isolated_drop_geometry.fit_isolated_drops(points, volumes, H)


def test_covariance_fit_uses_particle_material_volume():
    points = H * np.array(list(product([-0.5, 0.5], repeat=3)))
    volumes = np.where(points[:, 0] > 0, 3.0, 1.0) * VOLUME / 16
    drop, = isolated_drop_geometry.fit_isolated_drops(points, volumes, H)

    np.testing.assert_allclose(drop.center, [H / 4, 0, 0], atol=1e-16)
    assert np.prod(drop.semiaxes) == pytest.approx(RADIUS**3)
    assert drop.semiaxes[0] / drop.semiaxes[1] == pytest.approx(np.sqrt(3) / 2)
