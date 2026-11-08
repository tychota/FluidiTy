"""Measure the 2D coarse–fine transfer basis and independent local fixtures."""

import argparse
import json
import platform
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import NamedTuple

import numpy as np
from fluidity_reference.basis import quadratic_stencil
from fluidity_reference.experiments.coarse_fine_basis import (
    BasisStencil,
    evaluate_coarse_fine_basis,
    evaluate_uniform_cubic_normal_basis,
)

RESIDUAL_FLOOR = 1e-12
PRESSURE_TOLERANCE = 1e-6
DTYPES = (np.float32, np.float64)
ROTATION_90 = ((0, -1), (1, 0))

LOCAL_SAMPLES = (
    ("coarse_interior", (-4, -0.375), (0,)),
    ("coarse_near_interface", (-0.4375, -0.3125), (0,)),
    ("interface_knot", (0, 0.5), (0,)),
    ("mixed_support", (0.375, 0.3125), (0, 1)),
    ("fine_interior", (4, 0.25), (1,)),
    ("off_knot", (0.1328125, -0.2734375), (0, 1)),
)


class TransferBasisQuery(NamedTuple):
    sample_name: str
    particle_world_position: np.ndarray
    origin: np.ndarray
    coarse_cell_width: np.floating
    face_normal_axis: int
    interface_axis: int
    expected_levels: tuple[int, ...]


def fixture_queries(dtype):
    """Use the same dyadic world geometry for both evaluators and precisions."""
    queries = []
    for interface_axis, face_normal_axis, width, sample_spec in product((0, 1), (0, 1), (0.5, 2), LOCAL_SAMPLES):
        sample_name, local_point, expected_levels = sample_spec
        local_position = np.array(local_point, dtype=dtype)
        origin = np.array([-3.25, -7.5], dtype=dtype)
        if interface_axis == 1:
            local_position = np.array(ROTATION_90, dtype=dtype) @ local_position
            origin = np.array([-8.5, -4], dtype=dtype)
        coarse_cell_width = dtype(width)
        queries.append(
            TransferBasisQuery(
                sample_name,
                origin + coarse_cell_width * local_position,
                origin,
                coarse_cell_width,
                face_normal_axis,
                interface_axis,
                expected_levels,
            )
        )
    return queries


def measure_moment_defects(stencil, particle_world_position, coarse_cell_width):
    """Return the largest absolute component defect per invariant in stencil precision."""
    dtype = stencil.weights.dtype
    offsets_over_width = (stencil.anchor_world_positions - particle_world_position) / coarse_cell_width
    dimensionless_gradients = coarse_cell_width * stencil.physical_gradients
    return {
        "partition": np.abs(stencil.weights.sum() - dtype.type(1)),
        "first_moment": np.max(np.abs(stencil.weights @ offsets_over_width)),
        "gradient_sum": np.max(np.abs(dimensionless_gradients.sum(axis=0))),
        "affine_gradient": np.max(np.abs(offsets_over_width.T @ dimensionless_gradients - np.eye(2, dtype=dtype))),
    }


def _transition(query, dtype):
    return evaluate_coarse_fine_basis(
        query.particle_world_position,
        interface_origin=query.origin,
        coarse_cell_width=query.coarse_cell_width,
        face_normal_axis=query.face_normal_axis,
        interface_axis=query.interface_axis,
        dtype=dtype,
    )


def _uniform_cubic(query, dtype):
    return evaluate_uniform_cubic_normal_basis(
        query.particle_world_position,
        lattice_origin=query.origin,
        cell_width=query.coarse_cell_width,
        face_normal_axis=query.face_normal_axis,
        dtype=dtype,
    )


def _mac_quadratic_offset(face_normal_axis):
    offset = np.full(2, 0.5)
    offset[face_normal_axis] = 0
    return offset


def _uniform_quadratic(query, _dtype):
    offset = _mac_quadratic_offset(query.face_normal_axis)
    local_position = query.particle_world_position - query.origin
    indices, weights, gradients = quadratic_stencil(local_position, query.coarse_cell_width, offset)
    active = weights > 0
    anchor_world_positions = query.origin + query.coarse_cell_width * (indices[active] + offset)
    return BasisStencil(
        anchor_world_positions, weights[active], gradients[active], np.zeros(np.count_nonzero(active), np.int8)
    )


def _status(passed):
    return "pass" if passed else "fail"


def measure_basis(evaluate, queries, dtype, *, transition=False):
    moment_limit, derivative_limit = (1e-6, 2e-6) if dtype is np.float32 else (1e-12, 1e-12)
    limits = {
        "partition": moment_limit,
        "first_moment": moment_limit,
        "gradient_sum": derivative_limit,
        "affine_gradient": derivative_limit,
    }
    samples = []
    for query in queries:
        stencil = evaluate(query, dtype)
        defects = measure_moment_defects(stencil, query.particle_world_position, query.coarse_cell_width)
        dtype_matches = all(
            values.dtype == dtype
            for values in (
                stencil.weights,
                stencil.anchor_world_positions,
                stencil.physical_gradients,
                *defects.values(),
            )
        )
        support_levels = tuple(np.unique(stencil.levels))
        expected_levels = query.expected_levels if transition else (0,)
        passed = (
            dtype_matches
            and support_levels == expected_levels
            and np.all(stencil.weights > 0)
            and all(defects[name] <= limit for name, limit in limits.items())
        )
        samples.append(
            {
                "status": _status(passed),
                "sample_name": query.sample_name,
                "face_normal_axis": query.face_normal_axis,
                "interface_axis": query.interface_axis,
                "particle_world_position": query.particle_world_position.tolist(),
                "origin": query.origin.tolist(),
                "coarse_cell_width": float(query.coarse_cell_width),
                "dtype": stencil.weights.dtype.name,
                "expected_levels": list(expected_levels),
                "level_counts": [int(np.count_nonzero(stencil.levels == level)) for level in (0, 1)],
                "defects": {name: float(value) for name, value in defects.items()},
            }
        )
    return {
        "status": _status(all(sample["status"] == "pass" for sample in samples)),
        "dtype": np.dtype(dtype).name,
        "defect_limits": limits,
        "max_defects": {
            name: float(np.array([sample["defects"][name] for sample in samples], dtype=dtype).max()) for name in limits
        },
        "samples": samples,
    }


def measure_quadratic_nesting(face_normal_axis):
    """Probe the normal curvature of one fixed coarse and one fixed fine basis function."""
    offset = _mac_quadratic_offset(face_normal_axis)
    curvatures = []
    for width, anchor_index in ((1, 0), (0.5, 1)):
        normal_derivatives = []
        for normal_coordinate in (0.375, 0.5, 0.625):
            local_position = np.full(2, -0.3125)
            local_position[face_normal_axis] = normal_coordinate
            indices, _, gradients = quadratic_stencil(local_position, width, offset)
            anchor_rows = indices[:, face_normal_axis] == anchor_index
            normal_derivatives.append(gradients[anchor_rows, face_normal_axis].sum())
        curvatures.append(np.diff(normal_derivatives) / 0.125)
    coarse_curvatures, fine_curvatures = curvatures
    # The measured jump disproves nesting. An inconclusive probe establishes neither outcome.
    observed_expected_mismatch = (
        coarse_curvatures[0] != coarse_curvatures[1]
        and np.array_equal(coarse_curvatures, [-2, 1])
        and np.array_equal(fine_curvatures, [-8, -8])
    )
    return {
        "status": _status(observed_expected_mismatch),
        "dtype": coarse_curvatures.dtype.name,
        "face_normal_axis": face_normal_axis,
        "nested": False if observed_expected_mismatch else None,
        "coarse_knot_over_width": 0.5,
        "adjacent_fine_knots_over_width": [0.25, 0.75],
        "coarse_one_sided_curvatures_times_width_squared": coarse_curvatures.tolist(),
        "fine_one_sided_curvatures_times_width_squared": fine_curvatures.tolist(),
    }


def _relative_pressure_residual(matrix, pressure):
    dtype = matrix.dtype.type
    absolute_residual = np.linalg.norm(matrix @ pressure, ord=np.inf)
    norm_product = np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(pressure, ord=np.inf)
    denominator = max(norm_product, dtype(RESIDUAL_FLOOR))
    return absolute_residual, denominator, absolute_residual / denominator


def _cell_fluxes_and_divergence(incidence, subface_fluxes, cell_areas):
    cell_fluxes = incidence @ subface_fluxes
    return cell_fluxes, cell_fluxes / cell_areas


def measure_pressure_fixture(dtype, interface_axis):
    """Assemble and solve the sealed three-cell patch with two coarse-to-fine fluxes."""
    rotation = np.eye(2, dtype=dtype) if interface_axis == 0 else np.array(ROTATION_90, dtype=dtype)
    cell_centers = np.array([[-0.5, 0], [0.25, -0.25], [0.25, 0.25]], dtype=dtype) @ rotation.T
    normal = rotation[:, 0]
    predicted_subface_velocities = np.array([[0.75, 0.125], [-0.25, 0.375]], dtype=dtype) @ rotation.T
    cell_areas = np.array([1, 0.25, 0.25], dtype=dtype)
    subface_lengths = np.full(2, 0.5, dtype=dtype)
    normal_center_separations = (cell_centers[1:] - cell_centers[0]) @ normal

    incidence = np.array([[1, 1], [-1, 0], [0, -1]], dtype=dtype)
    predicted_subface_fluxes = subface_lengths * (predicted_subface_velocities @ normal)
    predicted_cell_fluxes, predicted_divergence = _cell_fluxes_and_divergence(
        incidence, predicted_subface_fluxes, cell_areas
    )
    coarse_aggregate_flux = predicted_cell_fluxes[0]

    # Unit inverse density and time step. Pressure acts on each owned flux once.
    conductances = subface_lengths / normal_center_separations
    pressure_matrix = (incidence * conductances) @ incidence.T
    constant_pressure = np.full(3, 1.3, dtype=dtype)
    absolute_residual, denominator, relative_residual = _relative_pressure_residual(pressure_matrix, constant_pressure)
    _, zero_pressure_denominator, _ = _relative_pressure_residual(pressure_matrix, np.zeros(3, dtype=dtype))

    # With the coarse pressure pinned, each independent fine row solves k*p = q*.
    pressure = np.zeros(3, dtype=dtype)
    pressure[1:] = predicted_subface_fluxes / conductances
    corrected_subface_fluxes = predicted_subface_fluxes + conductances * (incidence.T @ pressure)
    _, corrected_divergence = _cell_fluxes_and_divergence(incidence, corrected_subface_fluxes, cell_areas)
    max_corrected_divergence = np.max(np.abs(corrected_divergence))

    passed = (
        incidence.shape == (3, 2)
        and np.all(incidence.sum(axis=0) == 0)
        and np.array_equal(predicted_cell_fluxes, [0.25, -0.375, 0.125])
        and np.array_equal(predicted_divergence, [0.25, -1.5, 0.5])
        and coarse_aggregate_flux == predicted_subface_fluxes.sum()
        and relative_residual <= PRESSURE_TOLERANCE
        and max_corrected_divergence <= PRESSURE_TOLERANCE
    )
    return {
        "status": _status(passed),
        "dtype": corrected_divergence.dtype.name,
        "interface_axis": interface_axis,
        "normal": normal.tolist(),
        "cell_areas_m2": cell_areas.tolist(),
        "subface_lengths_m": subface_lengths.tolist(),
        "normal_center_separations_m": normal_center_separations.tolist(),
        "time_step_s": 1,
        "inverse_density": 1,
        "incidence": incidence.astype(int).tolist(),
        "predicted_subface_fluxes_m2_per_s": predicted_subface_fluxes.tolist(),
        "predicted_cell_fluxes_m2_per_s": predicted_cell_fluxes.tolist(),
        "predicted_divergence_per_s": predicted_divergence.tolist(),
        "coarse_aggregate_flux_m2_per_s": float(coarse_aggregate_flux),
        "integrated_pressure_matrix": pressure_matrix.tolist(),
        "constant_pressure": float(constant_pressure[0]),
        "constant_pressure_absolute_residual_m2_per_s": float(absolute_residual),
        "constant_pressure_relative_residual_denominator_m2_per_s": float(denominator),
        "constant_pressure_relative_residual_floor": RESIDUAL_FLOOR,
        "constant_pressure_relative_residual": float(relative_residual),
        "constant_pressure_relative_residual_limit": PRESSURE_TOLERANCE,
        "zero_pressure_residual_denominator": float(zero_pressure_denominator),
        "gauge_fixed_pressure": pressure.tolist(),
        "corrected_subface_fluxes_m2_per_s": corrected_subface_fluxes.tolist(),
        "corrected_divergence_per_s": corrected_divergence.tolist(),
        "max_corrected_divergence_per_s": float(max_corrected_divergence),
        "corrected_divergence_limit_per_s": PRESSURE_TOLERANCE,
    }


def measure_masked_support_fixtures(dtype):
    """Record raw defects and rank on deliberately incomplete mixed support."""
    query = next(
        query
        for query in fixture_queries(dtype)
        if query.sample_name == "mixed_support" and query.face_normal_axis == query.interface_axis == 0
    )
    stencil = _transition(query, dtype)
    original_level_counts = [int(np.count_nonzero(stencil.levels == level)) for level in (0, 1)]
    single_anchor_missing = np.ones(stencil.weights.size, dtype=bool)
    single_anchor_missing[np.argmax(stencil.weights)] = False
    # The anchors with x at the interface are collinear and retain positive weight.
    collinear_support = stencil.anchor_world_positions[:, 0] == query.origin[0]
    fixtures = []
    for name, retained, expected_rank in (
        ("single_anchor_missing", single_anchor_missing, 2),
        ("collinear_support", collinear_support, 1),
    ):
        retained_positions = stencil.anchor_world_positions[retained]
        retained_weights = stencil.weights[retained]
        removed_weight = stencil.weights[~retained].sum()
        raw_partition_defect = np.abs(dtype(1) - retained_weights.sum())
        affine_rank = np.linalg.matrix_rank(retained_positions - retained_positions[:1]) if retained_weights.size else 0
        disposition = "pic_required_rank_deficient_support" if affine_rank < 2 else "pic_required_incomplete_support"
        passed = (
            all(original_level_counts)
            and retained_weights.size >= 3
            and removed_weight > 0
            and raw_partition_defect > 0
            and affine_rank == expected_rank
        )
        fixtures.append(
            {
                "status": _status(passed),
                "fixture": name,
                "dtype": retained_weights.dtype.name,
                "particle_world_position": query.particle_world_position.tolist(),
                "original_level_counts": original_level_counts,
                "retained_count": int(retained_weights.size),
                "removed_weight": float(removed_weight),
                "raw_partition_defect": float(raw_partition_defect),
                "affine_rank": int(affine_rank),
                "disposition": disposition,
                "affine_claim": False,
            }
        )
    return fixtures


def measure_timings(queries, dtype, repeats):
    """Time warmed batches of identical queries in alternating evaluator order."""
    evaluators = (_uniform_cubic, _transition)
    elapsed_seconds = {evaluate: [] for evaluate in evaluators}
    for evaluate in evaluators:
        for query in queries:
            evaluate(query, dtype)
    for repeat in range(repeats):
        order = evaluators if repeat % 2 == 0 else evaluators[::-1]
        for evaluate in order:
            start = perf_counter()
            for query in queries:
                evaluate(query, dtype)
            elapsed_seconds[evaluate].append(perf_counter() - start)
    uniform_median = median(elapsed_seconds[_uniform_cubic])
    transition_median = median(elapsed_seconds[_transition])
    return {
        "dtype": np.dtype(dtype).name,
        "queries_per_repeat": len(queries),
        "warmup_batches_per_evaluator": 1,
        "repeats": repeats,
        "uniform_cubic_normal_seconds": elapsed_seconds[_uniform_cubic],
        "coarse_fine_seconds": elapsed_seconds[_transition],
        "uniform_cubic_normal_median_seconds": uniform_median,
        "coarse_fine_median_seconds": transition_median,
        "coarse_fine_over_uniform_ratio": transition_median / uniform_median,
    }


def qualify(repeats):
    queries = {dtype: fixture_queries(dtype) for dtype in DTYPES}
    fixtures = {
        "uniform_cubic_normal": [measure_basis(_uniform_cubic, queries[dtype], dtype) for dtype in DTYPES],
        "coarse_fine": [measure_basis(_transition, queries[dtype], dtype, transition=True) for dtype in DTYPES],
        "uniform_quadratic": [measure_basis(_uniform_quadratic, queries[np.float64], np.float64)],
        "quadratic_nesting": [measure_quadratic_nesting(face_normal_axis) for face_normal_axis in (0, 1)],
        "pressure": [measure_pressure_fixture(dtype, axis) for dtype, axis in product(DTYPES, (0, 1))],
        "masked_support": [fixture for dtype in DTYPES for fixture in measure_masked_support_fixtures(dtype)],
    }
    return {
        "status": _status(all(fixture["status"] == "pass" for group in fixtures.values() for fixture in group)),
        **fixtures,
        "timing": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "measurements": [measure_timings(queries[dtype], dtype, repeats) for dtype in DTYPES],
        },
    }


def _replace_nonfinite_metrics(value, path, nonfinite_metrics):
    """Keep strict JSON and record the paths of metrics represented by null."""
    if isinstance(value, dict):
        return {
            key: _replace_nonfinite_metrics(item, f"{path}.{key}" if path else key, nonfinite_metrics)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_nonfinite_metrics(item, f"{path}[{index}]", nonfinite_metrics) for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not isfinite(value):
        nonfinite_metrics.append(path)
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/adaptivity-task-2/qualification.json"))
    parser.add_argument("--repeats", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")

    nonfinite_metrics = []
    evidence = _replace_nonfinite_metrics(qualify(arguments.repeats), "", nonfinite_metrics)
    if nonfinite_metrics:
        evidence["status"] = "fail"
        evidence["nonfinite_metrics"] = nonfinite_metrics
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"2D coarse-fine qualification {evidence['status']} ({arguments.output})")
    return 0 if evidence["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
