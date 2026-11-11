"""Bounded 3D 2:1 MAC qualification on manufactured aligned patches."""

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import platform

import numpy as np

from fluidity_reference.experiments.coarse_fine_basis import (
    _active_fine_axis,
    _centered_cardinal,
    _support_radius,
    _truncated_coarse_axis,
    _uniform_axis,
)


def tensor(axes):
    """Yield anchor, weight, and gradient for each Cartesian product row."""
    for (x, wx, gx), (y, wy, gy), (z, wz, gz) in product(
        *(zip(*axis, strict=True) for axis in axes)
    ):
        yield (x, y, z), wx * wy * wz, (gx * wy * wz, wx * gy * wz, wx * wy * gz)


def contained_children(coordinate, normal, anchors):
    """Sum fine child contributions with lower support endpoint at or above zero."""
    dtype = coordinate.dtype
    if normal:
        offsets = np.array([-2, -1, 0, 1, 2], dtype=dtype)
        mask = np.array([1, 4, 6, 4, 1], dtype=dtype) / 8
    else:
        offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=dtype)
        mask = np.array([1, 3, 3, 1], dtype=dtype) / 4
    children = 2 * anchors[:, None] + offsets
    eligible = children >= _support_radius(normal)
    weights, gradients = _centered_cardinal(2 * coordinate - children, normal)
    return anchors, (weights * eligible) @ mask, 2 * ((gradients * eligible) @ mask)


def stencil(position, origin, width, signs, normal, boundary_axes):
    """Return local Greville rows as world anchors and dimensionless gradients."""
    dtype = position.dtype
    local = signs * (position - origin) / width
    uniform = [_uniform_axis(x, axis == normal) for axis, x in enumerate(local)]
    if not boundary_axes:
        rows = [(anchor, weight, gradient, 0) for anchor, weight, gradient in tensor(uniform) if weight > 0]
    else:
        coarse_rows = {}
        # Partition retained children by their first failed containment test.
        # Later axes sum to the unrefined spline by the dyadic identity.
        axes = uniform.copy()
        for axis in boundary_axes:
            axes[axis] = _truncated_coarse_axis(local[axis], axis == normal)
            for anchor, weight, gradient in tensor(axes):
                if weight <= 0:
                    continue
                old_weight, old_gradient = coarse_rows.get(anchor, (dtype.type(0), np.zeros(3, dtype=dtype)))
                coarse_rows[anchor] = old_weight + weight, old_gradient + gradient
            axes[axis] = contained_children(local[axis], axis == normal, uniform[axis][0])
        rows = [(anchor, weight, gradient, 0) for anchor, (weight, gradient) in sorted(coarse_rows.items())]
        fine = [
            (_active_fine_axis if axis in boundary_axes else _uniform_axis)(2 * x, axis == normal)
            for axis, x in enumerate(local)
        ]
        rows.extend(
            (tuple(component / 2 for component in anchor), weight, tuple(2 * g for g in gradient), 1)
            for anchor, weight, gradient in tensor(fine)
            if weight > 0
        )
    anchors, weights, gradients, levels = zip(*rows, strict=True)
    world_anchors = origin + width * signs * np.array(anchors, dtype=dtype)
    return world_anchors, np.array(weights, dtype=dtype), signs * np.array(gradients, dtype=dtype), np.array(levels)


def defects(stencil_rows, position, width):
    anchors, weights, gradients, _ = stencil_rows
    offsets = (anchors - position) / width
    slope = np.array([0.375, -0.25, 0.5], dtype=position.dtype)
    affine_samples = 0.25 + offsets @ slope
    return {
        "partition": float(abs(weights.sum() - 1)),
        "first_moment": float(np.max(np.abs(weights @ offsets))),
        "gradient_sum": float(np.max(np.abs(gradients.sum(axis=0)))),
        "affine_gradient_moment": float(np.max(np.abs(offsets.T @ gradients - np.eye(3, dtype=position.dtype)))),
        "affine_value": float(abs(weights @ affine_samples - 0.25)),
        "affine_gradient": float(np.max(np.abs(affine_samples @ gradients - slope))),
    }


def transfer_fixture(dtype, codimension):
    origin = np.array([-4.0, 2.0, -3.0], dtype=dtype)
    positions = np.array([
        [-2.25, -2.25, -2.25], [0, 0, 0], [-0.125, 0.1875, -0.375], [0.375, -0.125, 0.625],
        [0.625, 0.375, -0.125], [0.875, 0.75, 0.625], [1, 1, 1], [1.25, 0.875, 1.125],
        [1.375, 1.625, 1.125], [2.25, 2.25, 2.25],
    ], dtype=dtype)
    peaks = {"adaptive": {}, "uniform": {}}
    worst_queries = {"adaptive": {}, "uniform": {}}
    counts = {"coarse_only": 0, "mixed": 0, "fine_only": 0}
    for rotation, normal, width, local in product(range(2), range(3), (0.5, 2), positions):
        # The second transform cyclically rotates the junction and flips two axes.
        signs = np.array([1, 1, 1] if rotation == 0 else [-1, 1, -1], dtype=dtype)
        boundary_axes = tuple((axis + rotation) % 3 for axis in range(codimension))
        position = origin + width * signs * np.roll(local, rotation)
        for label, axes in (("adaptive", boundary_axes), ("uniform", ())):
            rows = stencil(position, origin, width, signs, normal, axes)
            for name, value in defects(rows, position, width).items():
                previous = peaks[label].get(name, -np.inf)
                if previous is None:
                    continue
                if not np.isfinite(value) or value > previous:
                    peaks[label][name] = value if np.isfinite(value) else None
                    worst_queries[label][name] = {
                        "position": position.tolist(), "origin": origin.tolist(), "width": width,
                        "face_normal_axis": normal, "boundary_axes": list(axes), "axis_signs": signs.tolist(),
                    }
            if axes:
                levels = rows[3]
                support = "mixed" if levels.min() != levels.max() else ("fine_only" if levels.min() else "coarse_only")
                counts[support] += 1
    return peaks, worst_queries, counts


def pressure_fixture(dtype, boundary_axes, orientation):
    """Assemble one flux per face of a sealed coarse-neighbor/fine-block graph."""
    coarse_count = len(boundary_axes)
    fine_cells = list(product(range(2), repeat=3))
    interface_faces = [
        (coarse_index, coarse_count + fine_index)
        for coarse_index, axis in enumerate(boundary_axes)
        for fine_index, cell in enumerate(fine_cells)
        if cell[axis] == (0 if orientation == 1 else 1)
    ]
    interior_faces = [
        (coarse_count + left, coarse_count + right)
        for left, right in combinations(range(len(fine_cells)), 2)
        if sum(abs(a - b) for a, b in zip(fine_cells[left], fine_cells[right], strict=True)) == 1
    ]
    faces = interface_faces + interior_faces
    incidence = np.zeros((coarse_count + len(fine_cells), len(faces)), dtype=dtype)
    conductance = np.full(len(faces), 0.25 / 0.5, dtype=dtype)
    flux = np.zeros(len(faces), dtype=dtype)
    volumes = np.array([1] * coarse_count + [0.125] * len(fine_cells), dtype=dtype)
    for face_index, (left, right) in enumerate(faces):
        incidence[left, face_index] = orientation
        incidence[right, face_index] = -orientation
    for face_index in range(len(interface_faces)):
        conductance[face_index] = 0.25 / 0.75
        flux[face_index] = orientation * (0.375, -0.125, 0.25, -0.25)[face_index % 4]

    # Sum fine-cell fluxes independently in canonical subface order.
    coarse_fluxes = []
    for coarse_index in range(coarse_count):
        face_indices = list(range(4 * coarse_index, 4 * coarse_index + 4))
        fine_outward_flux = sum(
            incidence[faces[index][1], index] * flux[index] for index in face_indices
        )
        coarse_outward_flux = incidence[coarse_index] @ flux
        coarse_fluxes.append((float(coarse_outward_flux), float(fine_outward_flux)))
    corrupt = incidence.copy()
    corrupt[faces[0][1], 0] *= -1
    matrix = (incidence * conductance) @ incidence.T
    constant_pressure = np.full(len(volumes), 1.3, dtype=dtype)
    floor = 1e-12
    denominator = max(np.linalg.norm(matrix, np.inf) * np.linalg.norm(constant_pressure, np.inf), floor)
    constant_residual = np.linalg.norm(matrix @ constant_pressure, np.inf) / denominator
    pressure = np.zeros(len(volumes), dtype=dtype)
    pressure[1:] = np.linalg.solve(matrix[1:, 1:], -(incidence @ flux)[1:])
    corrected = flux + conductance * (incidence.T @ pressure)
    return {
        "interface_faces": len(interface_faces),
        "coarse_outward_flux": [coarse for coarse, _ in coarse_fluxes],
        "fine_outward_flux": [fine for _, fine in coarse_fluxes],
        "signed_balance_defect": max(abs(coarse + fine) for coarse, fine in coarse_fluxes),
        "corrupt_signed_balance_defect": float(np.max(np.abs(corrupt.sum(axis=0) * flux))),
        "constant_pressure_relative_residual": float(constant_residual),
        "corrected_divergence": float(np.max(np.abs((incidence @ corrected) / volumes))),
    }


def masked_fixture(dtype, codimension):
    position = np.full(3, 0.875, dtype=dtype)
    origin = np.zeros(3, dtype=dtype)
    signs = np.ones(3, dtype=dtype)
    anchors, weights, gradients, levels = stencil(position, origin, 1, signs, 0, tuple(range(codimension)))
    largest = np.argmax(weights)
    masks = {
        "incomplete": np.arange(len(weights)) != largest,
        "coplanar": anchors[:, 0] == anchors[largest, 0],
    }
    result = {}
    for name, mask in masks.items():
        retained = anchors[mask]
        rank = int(np.linalg.matrix_rank(retained - retained[0]))
        metrics = defects((retained, weights[mask], gradients[mask], levels[mask]), position, 1)
        result[name] = {
            "rank": rank, "retained_rows": len(retained), "raw_partition_defect": metrics["partition"],
            "raw_first_moment_defect": metrics["first_moment"],
            "disposition": "pic_required_rank_deficient" if rank < 3 else "pic_required_incomplete",
        }
    return result


def qualify():
    rows = []
    for dtype, codimension in product((np.float32, np.float64), (1, 2, 3)):
        value_limit = 1e-6 if dtype == np.float32 else 1e-12
        gradient_limit = 2e-6 if dtype == np.float32 else 1e-12
        limits = dict.fromkeys(("partition", "first_moment", "affine_value"), value_limit)
        limits.update(dict.fromkeys(("gradient_sum", "affine_gradient_moment", "affine_gradient"), gradient_limit))
        peaks, worst_queries, counts = transfer_fixture(dtype, codimension)
        pressure = pressure_fixture(dtype, tuple(range(codimension)), 1)
        rotated_pressure = pressure_fixture(dtype, tuple((axis + 1) % 3 for axis in range(codimension)), -1)
        mask = masked_fixture(dtype, codimension)
        passed = (
            all(value is not None and value <= limits[name] for group in peaks.values() for name, value in group.items())
            and all(count > 0 for count in counts.values())
            and mask["incomplete"]["rank"] == 3 and mask["coplanar"]["rank"] == 2
            and all(item["raw_partition_defect"] > 0.01 for item in mask.values())
            and all(
                case["signed_balance_defect"] == 0
                and case["corrupt_signed_balance_defect"] == 0.75
                and case["coarse_outward_flux"] == [0.25] * codimension
                and case["constant_pressure_relative_residual"] <= 1e-6
                and case["corrected_divergence"] <= 1e-6
                for case in (pressure, rotated_pressure)
            )
        )
        rows.append({
            "dtype": np.dtype(dtype).name, "codimension": codimension, "queries": sum(counts.values()),
            "mixed_queries": counts["mixed"], "support_counts": counts, "limits": limits, "peak_defects": peaks,
            "worst_queries": worst_queries, "pressure": pressure, "rotated_pressure": rotated_pressure,
            "mask": mask, "passed": passed,
        })
    return {
        "python": platform.python_version(), "numpy": np.__version__, "pressure_relative_floor_m3_s": 1e-12,
        "pressure_relative_limit": 1e-6, "divergence_limit_s_inv": 1e-6,
        "passed": all(row["passed"] for row in rows), "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
