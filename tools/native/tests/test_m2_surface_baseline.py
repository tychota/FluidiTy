import math

import numpy as np
import pytest

from tools.native import m2_surface_baseline
from tools.native.m2_surface_baseline import (
    Mesh,
    extract_surface,
    mesh_metrics,
    reconstruct,
    require_sha256,
)


def test_rejects_changed_input(tmp_path):
    artifact = tmp_path / "frames.jsonl"
    artifact.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    require_sha256(artifact, digest)
    artifact.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="SHA-256"):
        require_sha256(artifact, digest)


def test_closed_sphere_is_deterministic_and_converges():
    spacing = 0.1
    axis = np.arange(-12, 13) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    field = np.sqrt(x * x + y * y + z * z) - 1
    mesh = extract_surface(field, [axis[0]] * 3, spacing)
    repeat = extract_surface(field, [axis[0]] * 3, spacing)
    np.testing.assert_array_equal(mesh.vertices, repeat.vertices)
    np.testing.assert_array_equal(mesh.triangles, repeat.triangles)
    metrics = mesh_metrics(mesh, 4 * math.pi / 3, [1, 1, 1])
    assert metrics["components"] == 1
    assert metrics["euler_characteristic"] == 2
    assert metrics["boundary_edges"] == 0
    assert metrics["nonmanifold_edges"] == 0
    assert metrics["orientation_defects"] == 0
    assert metrics["degenerate_triangles"] == 0
    assert abs(metrics["relative_volume_error"]) < 0.02
    assert metrics["vertex_radial_error_max_m"] < 0.01
    inverted = mesh_metrics(Mesh(mesh.vertices, mesh.triangles[:, ::-1]), 4 * math.pi / 3)
    assert inverted["signed_volume_m3"] < 0


def test_disconnected_pair_and_empty_field():
    axis = np.arange(-12, 13) * 0.1
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    left = np.sqrt((x + 0.6) ** 2 + y * y + z * z)
    right = np.sqrt((x - 0.6) ** 2 + y * y + z * z)
    field = np.minimum(left, right) - 0.3
    mesh = extract_surface(field, [axis[0]] * 3, 0.1)
    metrics = mesh_metrics(mesh, 2 * 4 * math.pi * 0.3**3 / 3)
    assert metrics["components"] == 2
    assert metrics["euler_characteristic"] == 4
    assert metrics["boundary_edges"] == 0
    assert metrics["nonmanifold_edges"] == 0
    empty = extract_surface(np.ones((2, 2, 2)), [0, 0, 0], 1)
    assert mesh_metrics(empty, 1)["components"] == 0


def test_rejects_truncated_surface_and_nonfinite_samples():
    field = np.ones((3, 3, 3))
    field[0, 1, 1] = -1
    with pytest.raises(ValueError, match="lattice boundary"):
        extract_surface(field, [0, 0, 0], 1)
    field[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_surface(field, [0, 0, 0], 1)


def test_reconstruction_accepts_publication_order_and_rejects_duplicate_ids():
    points = [
        {"id": ordinal + 1, "world_position": (x / 100, y / 100, z / 100), "radius": 0.015}
        for ordinal, (x, y, z) in enumerate(np.ndindex(8, 23, 3))
    ]
    first, _, _ = reconstruct(points, 0.025, "isotropic")
    second, _, _ = reconstruct(points[::-1], 0.025, "isotropic")
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.triangles, second.triangles)
    with pytest.raises(ValueError, match="invalid isolated drop particles"):
        reconstruct(points[:-1] + [points[0]], 0.025, "isotropic")


def test_density_reference_uses_full_material_support():
    radius = m2_surface_baseline.DROP_RADIUS_M
    axis = (np.arange(-5, 5) + 0.5) * (radius / 5)
    sites = [position for position in np.ndindex(10, 10, 10)
             if sum(axis[index] ** 2 for index in position) < radius**2]
    assert len(sites) == 552
    material_radius = radius / len(sites) ** (1 / 3)
    points = [
        {"id": ordinal + 1, "world_position": [axis[index] for index in site], "radius": material_radius}
        for ordinal, site in enumerate(sites)
    ]
    mesh, volume, _ = reconstruct(points, 0.0125, "density")
    result = mesh_metrics(mesh, volume)
    assert result["components"] == 1
    assert result["euler_characteristic"] == 2
    assert result["boundary_edges"] == 0
    assert result["nonmanifold_edges"] == 0
    assert result["orientation_defects"] == 0
    assert result["signed_volume_m3"] > 0
