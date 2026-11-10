"""Analytic controls for the piston material mesh diagnostic."""

import itertools
import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piston_material_mesh as mesh
from piston_material_mesh import clipped_signed_volumes, measure, reconstruct, signed_volume

H = 0.0625
LOW = (-0.5, 0.0, -0.25)
SHAPE = (16, 8, 8)


def lattice(shift, extent_offset):
    return {
        index: tuple(lo + (i + shift) * H for lo, i in zip(LOW, index))
        for index in itertools.product(*(range(n + extent_offset) for n in SHAPE))
    }


def mapped(positions, transform):
    return {index: transform(*point) for index, point in positions.items()}


class MaterialMeshControls(unittest.TestCase):
    def test_identity(self):
        centers = lattice(0.5, 0)
        self.assertEqual(reconstruct(centers), lattice(0.0, 1))
        result = measure(centers, LOW[0])
        self.assertEqual(result["tetra_count"], 6144)
        self.assertEqual(result["signed_total_m3"], 0.25)
        self.assertEqual(result["signed_below_m3"], 0.25)
        self.assertEqual(result["signed_above_m3"], 0.0)
        self.assertEqual(result["inversion_count"], 0)
        self.assertEqual(result["near_degenerate_count"], 0)
        self.assertEqual(result["max_displacement_gradient_frobenius"], 0.0)
        self.assertTrue(result["injectivity_certified"])

    def test_determinant_one_piston(self):
        piston_x = -0.46875
        scale_x = 0.96875
        affine = lambda x, y, z: (0.5 + (x - 0.5) * scale_x, y / scale_x, z)
        result = measure(mapped(lattice(0.5, 0), affine), piston_x)
        self.assertAlmostEqual(result["signed_total_m3"], 0.25, delta=2e-14)
        self.assertAlmostEqual(result["signed_below_m3"], 0.2421875, delta=2e-14)
        self.assertAlmostEqual(result["signed_above_m3"], 0.0078125, delta=2e-14)
        self.assertAlmostEqual(result["wall_fit"]["piston"]["max_absolute_offset_m"], 0.0, delta=2e-14)
        self.assertTrue(result["injectivity_certified"])

    def test_tank_intersection_crops_translated_box_at_multiple_walls(self):
        piston_x = -0.46875
        for shift, overlap in (
            ((0.03125, 0.0625, 0.03125), (0.96875, 0.4375, 0.0625, 0.46875)),
            ((-0.03125, -0.0625, -0.03125), (0.9375, 0.4375, 0.0, 0.46875)),
        ):
            dx, dy, dz = shift
            centers = mapped(lattice(0.5, 0), lambda x, y, z: (x + dx, y + dy, z + dz))
            result = measure(centers, piston_x)
            width, below_height, above_height, depth = overlap
            self.assertAlmostEqual(result["tank_clipped_below_m3"], width * below_height * depth, delta=2e-14)
            self.assertAlmostEqual(result["tank_clipped_above_m3"], width * above_height * depth, delta=2e-14)
            self.assertAlmostEqual(result["signed_total_m3"], 0.25, delta=2e-14)

    def test_tank_clipped_identity_and_affine_fill_to_reference(self):
        initial = measure(lattice(0.5, 0), LOW[0])
        self.assertEqual(initial["tank_clipped_below_m3"], 0.25)
        self.assertEqual(initial["tank_clipped_above_m3"], 0.0)
        affine = lambda x, y, z: (0.5 + (x - 0.5) * 0.96875, y / 0.96875, z)
        final = measure(mapped(lattice(0.5, 0), affine), -0.46875)
        self.assertAlmostEqual(final["tank_clipped_below_m3"], 0.2421875, delta=2e-14)
        self.assertAlmostEqual(final["tank_clipped_above_m3"], 0.0078125, delta=2e-14)

    def test_tank_clipped_slab_and_tangent(self):
        initial = measure(lattice(0.5, 0), -0.4693)
        self.assertAlmostEqual(initial["tank_clipped_below_m3"], 0.9693 * 0.25, delta=2e-14)
        self.assertEqual(initial["tank_clipped_above_m3"], 0.0)
        tangent_centers = mapped(lattice(0.5, 0), lambda x, y, z: (x - 0.96875, y, z))
        tangent = measure(tangent_centers, -0.46875)
        self.assertEqual(tangent["tank_clipped_below_m3"], 0.0)
        self.assertEqual(tangent["tank_clipped_above_m3"], 0.0)

    def test_general_affine_shear_and_translation(self):
        affine = lambda x, y, z: (x + 0.2 * y - 0.3 * z + 0.1, y + 0.4 * z - 0.1, z + 0.025)
        centers = mapped(lattice(0.5, 0), affine)
        expected = mapped(lattice(0.0, 1), affine)
        actual = reconstruct(centers)
        self.assertLess(max(math.dist(actual[key], point) for key, point in expected.items()), 2e-14)
        result = measure(centers, LOW[0])
        self.assertAlmostEqual(result["signed_total_m3"], 0.25, delta=2e-14)
        self.assertAlmostEqual(result["max_displacement_gradient_frobenius"], math.sqrt(0.29), delta=2e-14)
        self.assertTrue(result["injectivity_certified"])

    def test_reflection_retains_inverted_signed_volumes(self):
        result = measure(mapped(lattice(0.5, 0), lambda x, y, z: (-x, y + 0.25, z)), LOW[0])
        self.assertEqual(result["signed_total_m3"], -0.25)
        self.assertEqual(result["signed_above_m3"], -0.125)
        self.assertEqual(result["signed_below_m3"], -0.125)
        self.assertEqual(result["unsigned_total_m3"], 0.25)
        self.assertEqual(result["inversion_count"], 6144)
        self.assertFalse(result["injectivity_certified"])

    def test_fold_has_both_orientation_signs_and_overlap(self):
        result = measure(mapped(lattice(0.5, 0), lambda x, y, z: (abs(x), y, z)), LOW[0])
        self.assertEqual(result["signed_total_m3"], 0.0)
        self.assertGreater(result["unsigned_total_m3"], 0.0)
        self.assertGreater(result["inversion_count"], 0)
        self.assertFalse(result["injectivity_certified"])

    def test_collapse_reports_degeneracy(self):
        result = measure(mapped(lattice(0.5, 0), lambda x, y, z: (x, y, 0.0)), LOW[0])
        self.assertEqual(result["signed_total_m3"], 0.0)
        self.assertEqual(result["inversion_count"], 0)
        self.assertEqual(result["near_degenerate_count"], 6144)
        self.assertFalse(result["injectivity_certified"])

    def test_invalid_centers_rejected(self):
        centers = lattice(0.5, 0)
        centers[(0, 0, 0)] = (float("nan"), 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            reconstruct(centers)
        del centers[(0, 0, 0)]
        with self.assertRaisesRegex(ValueError, "lattice"):
            reconstruct(centers)

    def test_stable_ids_reconstruct_shuffled_centers(self):
        expected = lattice(0.5, 0)
        particles = [
            {"id": 1 + ix + 16 * iy + 128 * iz, "world_position": point}
            for (ix, iy, iz), point in reversed(tuple(expected.items()))
        ]
        self.assertEqual(mesh.snapshot(particles, "world_position"), expected)
        particles[0]["id"] = particles[1]["id"]
        with self.assertRaisesRegex(ValueError, "material IDs"):
            mesh.snapshot(particles, "world_position")

    def test_wall_and_frozen_strip_for_identity(self):
        section = mesh.sweep(lattice(0.5, 0), -0.5, -0.46875)
        self.assertEqual(section["area_below_m2"], 0.25)
        self.assertEqual(section["area_above_m2"], 0.0)
        self.assertAlmostEqual(section["strip_below_m3"], 0.0078125, delta=2e-14)
        self.assertEqual(section["strip_above_m3"], 0.0)

    def test_affine_wall_and_frozen_strip_include_raised_surface(self):
        scale = 0.96875
        centers = mapped(lattice(0.5, 0), lambda x, y, z: (0.5 + (x - 0.5) * scale, y / scale, z))
        section = mesh.sweep(centers, -0.46875, -0.4375)
        self.assertAlmostEqual(section["area_below_m2"], 0.25, delta=2e-14)
        self.assertAlmostEqual(section["area_above_m2"], 1 / 124, delta=2e-14)
        self.assertAlmostEqual(section["strip_below_m3"], 1 / 128, delta=2e-14)
        self.assertAlmostEqual(section["strip_above_m3"], 1 / 3968, delta=2e-14)

    def test_detached_material_has_dry_wall_and_partly_wet_strip(self):
        centers = mapped(lattice(0.5, 0), lambda x, y, z: (x + 0.015625, y, z))
        section = mesh.sweep(centers, -0.5, -0.46875, offsets=(0.015625, 0.03125))
        self.assertEqual(section["area_below_m2"], 0.0)
        self.assertEqual(section["area_above_m2"], 0.0)
        self.assertAlmostEqual(section["strip_below_m3"], 0.00390625, delta=2e-14)
        self.assertEqual(section["strip_above_m3"], 0.0)
        self.assertEqual(section["offset_area_below_above_m2"], ((0.25, 0.0), (0.25, 0.0)))

    def test_uncertified_fold_refuses_wall_occupancy(self):
        centers = mapped(lattice(0.5, 0), lambda x, y, z: (abs(x), y, z))
        with self.assertRaisesRegex(ValueError, "injectivity"):
            mesh.sweep(centers, -0.5, -0.46875)

    def test_halfspace_simplex_analytic_fractions(self):
        # A uniform simplex point has uniform barycentric coordinates. The sum
        # on k upper vertices has Beta(k, 4-k) CDF when vertex heights are binary.
        above_at_quarter = (0.0, 27 / 64, 27 / 32, 63 / 64, 1.0)
        for upper_count, expected in enumerate(above_at_quarter):
            heights = [0.0] * (4 - upper_count) + [1.0] * upper_count
            for order in set(itertools.permutations(heights)):
                tetra = ((0, order[0], 0), (1, order[1], 0), (0, order[2], 1), (0, order[3], 0))
                volume = signed_volume(tetra)
                if volume == 0.0:
                    continue
                below, above = clipped_signed_volumes(tetra, 0.25)
                self.assertAlmostEqual(above / volume, expected, delta=2e-14)
                self.assertAlmostEqual(below / volume, 1.0 - expected, delta=2e-14)

    def test_skew_halfspace_distinct_and_plane_heights(self):
        # For distinct vertex heights, the simplex slice tail is the truncated
        # cubic divided difference. This evaluates a separate formula.
        for heights in itertools.permutations((-2.0, -0.75, 1.5, 3.0)):
            a, b, c, d = heights
            tetra = ((0.0, a, 0.0), (2.0, b, 0.3), (1.0, c, -1.5), (-0.4, d, 0.6))
            volume = signed_volume(tetra)
            self.assertNotEqual(volume, 0.0)
            for level in (-3.0, -1.5, -0.75, 0.25, 1.5, 2.0, 3.0, 4.0):
                above_fraction = math.fsum(
                    max(y - level, 0.0) ** 3 / math.prod(y - other for other in heights if other != y)
                    for y in heights
                )
                below, above = clipped_signed_volumes(tetra, level)
                self.assertAlmostEqual(above / volume, above_fraction, delta=2e-14)
                self.assertAlmostEqual(below / volume, 1.0 - above_fraction, delta=2e-14)


if __name__ == "__main__":
    unittest.main()
