"""Research-only CPU experiment with the archived Zhao ONNX weights."""

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from .m2_surface_baseline import (
    DROP_AXES_M,
    extract_surface,
    mesh_metrics,
    peak_rss_bytes,
    read_pinned_frames,
    require_sha256,
)


MODEL_SHA256 = "417263e65280fe6efbdb1cab603c99235b011d85478de5c31641c6f16b461be3"
ARCHIVE_SHA256 = "bc4304606c6b378bfd06dec928025e38d8cf6fd767f70a43d8bdc0f56c3e16ca"
ARCHIVE_ROOT = "surface_reconstruction_fast"
BATCH_SIZE = 1024
PATCH_AXIS = np.arange(-7, 8, 2)
STENCIL_AXIS = np.arange(-4, 6)
DEPENDENCY_CELLS = 18
PARITIES = tuple(product((0, 1), repeat=3))
BOX_CORNERS = tuple(product((0, 1), repeat=3))


def _coordinates(points, queries):
    points = np.asarray(points, dtype=np.float64)
    queries = np.asarray(queries)
    if (points.ndim != 2 or points.shape[1] != 3 or queries.ndim != 2
            or queries.shape[1] != 3 or not np.issubdtype(queries.dtype, np.integer)):
        raise ValueError("expected particle coordinates and integer query coordinates in fine-grid cells")
    if not np.isfinite(points).all():
        raise ValueError("particle coordinates must be finite")
    return points, queries


def _splat(cells, coordinates, lower, shape, particle_weights):
    offsets = cells[:, :, None] + STENCIL_AXIS
    delta = (offsets - coordinates[:, :, None]) / 5
    weights = np.maximum(0, 1 - delta**2) ** 2
    indices = offsets - lower[:, None]
    grid = np.zeros(shape, dtype=np.float64)
    for x, y, z in product(range(len(STENCIL_AXIS)), repeat=3):
        weight = weights[:, 0, x] * weights[:, 1, y] * weights[:, 2, z]
        node_indices = (indices[:, 0, x], indices[:, 1, y], indices[:, 2, z])
        np.add.at(grid, node_indices, weight * particle_weights)
    return grid


def feature_patches(points, queries):
    """Return normalized float32 patches using coordinates in fine-grid cells."""
    points, queries = _coordinates(points, queries)
    if not len(queries) or not len(points):
        return np.zeros((len(queries), 8, 8, 8), dtype=np.float32)
    cells = np.floor(points).astype(np.int64)
    lower = np.minimum(cells.min(axis=0) - 4, queries.min(axis=0) - 7)
    upper = np.maximum(cells.max(axis=0) + 5, queries.max(axis=0) + 7)
    shape = upper - lower + 1
    if math.prod(shape) > 20_000_000:
        raise ValueError("feature lattice exceeds the bounded research workload")
    density_grid = _splat(cells, points, lower, shape, 1)
    local_cells = cells - lower
    fraction = points - cells
    density = np.zeros(len(points))
    for corner in BOX_CORNERS:
        weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
        density += weight * density_grid[tuple((local_cells + corner).T)]
    if np.any(density <= 0):
        raise ValueError("particle has no valid normalization density")
    del density_grid
    features = _splat(cells, points, lower, shape, 1 / density)
    local_queries = queries - lower
    x = local_queries[:, 0, None, None, None] + PATCH_AXIS[None, :, None, None]
    y = local_queries[:, 1, None, None, None] + PATCH_AXIS[None, None, :, None]
    z = local_queries[:, 2, None, None, None] + PATCH_AXIS[None, None, None, :]
    return features[x, y, z].astype(np.float32)


def _window_count(prefix, nodes, radius):
    # The upstream prefix is inclusive. Its differences cover (node-r, node+r].
    lower = nodes - radius + 1
    upper = nodes + radius + 1
    counts = np.zeros(len(nodes), dtype=np.int64)
    for corner in BOX_CORNERS:
        index = np.where(corner, upper, lower)
        sign = (-1) ** (3 - sum(corner))
        counts += sign * prefix[tuple(index.T)]
    return counts


def pruning_classes(points, queries):
    """Return -1 for pruned liquid, 0 for inference and +1 for pruned air."""
    points, queries = _coordinates(points, queries)
    classes = np.ones(len(queries), dtype=np.int8)
    for parity in PARITIES:
        matches = np.all(queries % 2 == parity, axis=1)
        if not np.any(matches):
            continue
        coarse = (queries[matches] - parity) // 2
        lower = coarse.min(axis=0) - 4
        shape = coarse.max(axis=0) + 5 - lower
        if math.prod(shape) > 20_000_000:
            raise ValueError("pruning lattice exceeds the bounded research workload")
        cells = np.floor((points - parity) / 2).astype(np.int64) - lower
        cells = cells[np.all((cells >= 0) & (cells < shape), axis=1)]
        occupied = np.zeros(shape, dtype=bool)
        occupied[tuple(cells.T)] = True
        prefix = np.pad(occupied, ((1, 0),) * 3).cumsum(0).cumsum(1).cumsum(2)
        nodes = coarse - lower
        classes[matches] = np.where(_window_count(prefix, nodes, 4) > 410, -1,
                                    np.where(_window_count(prefix, nodes, 2) == 0, 1, 0))
    return classes


def _halo_particles(points, queries):
    lower = queries.min(axis=0) - DEPENDENCY_CELLS
    upper = queries.max(axis=0) + DEPENDENCY_CELLS
    return points[np.all((points >= lower) & (points <= upper), axis=1)]


def field_values(points, queries, predict, spacing_m, *, frozen_pruning=None, tile_width_cells=None):
    """Evaluate a fixed sequence of parity batches with optional tiled feature gathering."""
    points, queries = _coordinates(points, queries)
    if not math.isfinite(spacing_m) or spacing_m <= 0 or not len(queries):
        raise ValueError("expected positive spacing and at least one query")
    if tile_width_cells is not None and tile_width_cells <= 0:
        raise ValueError("expected positive tile width")
    started = perf_counter()
    classes = pruning_classes(points, queries) if frozen_pruning is None else np.asarray(frozen_pruning)
    if classes.shape != (len(queries),) or not np.isin(classes, (-1, 0, 1)).all():
        raise ValueError("pruning classes must cover the fixed query sequence")
    model_nodes = queries[classes == 0]
    # The published final batch pads with coarse node zero in each parity.
    pad_nodes = np.array(PARITIES)
    feature_nodes = np.concatenate((model_nodes, pad_nodes))
    smallest_support = len(points)
    if tile_width_cells is None:
        patches = feature_patches(points, feature_nodes)
    else:
        patches = np.empty((len(feature_nodes), 8, 8, 8), dtype=np.float32)
        tile_ids = feature_nodes[:, 0] // tile_width_cells
        for tile in np.unique(tile_ids):
            members = tile_ids == tile
            nodes = feature_nodes[members]
            retained = _halo_particles(points, nodes)
            smallest_support = min(smallest_support, len(retained))
            patches[members] = feature_patches(retained, nodes)
    inference_indices = np.flatnonzero(classes == 0)
    values = classes.astype(np.float32) * np.float32(4 * spacing_m)
    batch = np.empty((BATCH_SIZE, 1, 8, 8, 8), dtype=np.float32)
    batch_count = 0
    feature_digest = hashlib.sha256(patches.tobytes()).hexdigest()
    preprocessing_seconds = perf_counter() - started
    started = perf_counter()
    for parity_index, parity in enumerate(PARITIES):
        ordered_indices = np.flatnonzero(np.all(model_nodes % 2 == parity, axis=1))
        for start in range(0, len(ordered_indices), BATCH_SIZE):
            active = ordered_indices[start:start + BATCH_SIZE]
            batch[:len(active), 0] = patches[active]
            batch[len(active):, 0] = patches[len(model_nodes) + parity_index]
            prediction = np.asarray(predict(batch))
            if prediction.shape != (BATCH_SIZE, 1) or not np.isfinite(prediction).all():
                raise ValueError("expected a finite scalar prediction per published batch row")
            values[inference_indices[active]] = prediction[:len(active), 0] * np.float32(2 * spacing_m)
            batch_count += 1
    return {
        "classes": classes,
        "values": values,
        "feature_sha256": feature_digest,
        "smallest_tile_particles": smallest_support,
        "batches": batch_count,
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": perf_counter() - started,
    }


def _field_receipt(field, baseline=None):
    values = field["values"]
    receipt = {key: value for key, value in field.items() if key not in ("classes", "values")}
    receipt.update(
        query_count=len(values),
        inferred_queries=int(np.count_nonzero(field["classes"] == 0)),
        scalar_sha256=hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest(),
    )
    if baseline is not None:
        delta = values.astype(np.float64) - baseline["values"]
        receipt.update(
            field_rms_m=float(np.sqrt(np.mean(delta**2))),
            field_max_m=float(np.max(np.abs(delta))),
            different_pruning_queries=int(np.count_nonzero(field["classes"] != baseline["classes"])),
            different_sign_queries=int(np.count_nonzero((values < 0) != (baseline["values"] < 0))),
        )
    return receipt


def _drop_experiment(frame, predict):
    positions = np.array([point["world_position"] for point in frame["points"]])
    volume = math.fsum(4 * math.pi * point["radius"]**3 / 3 for point in frame["points"])
    spacing = 0.0125
    origin = np.array([-0.25, -0.25, -0.25])
    shape = (41, 41, 41)
    nodes = np.indices(shape).reshape(3, -1).T
    points = (positions - origin) / spacing
    radial_level = np.linalg.norm(positions / DROP_AXES_M, axis=1)
    radial_gap = np.linalg.norm(positions, axis=1) - np.divide(
        np.linalg.norm(positions, axis=1), radial_level,
        out=np.full(len(positions), min(DROP_AXES_M)), where=radial_level > 0,
    )
    full = field_values(points, nodes, predict, spacing)
    variants = [("full", points, full), ("full_tiled", points, field_values(
        points, nodes, predict, spacing, tile_width_cells=8,
    ))]
    for depth in (0.025, 0.05, 0.075):
        band = points[np.abs(radial_gap) <= depth]
        variants.append((f"band_{depth:g}_frozen_pruning", band, field_values(
            band, nodes, predict, spacing, frozen_pruning=full["classes"],
        )))
        variants.append((f"band_{depth:g}_recomputed_pruning", band, field_values(
            band, nodes, predict, spacing,
        )))
    receipts = {}
    for name, particles, field in variants:
        started = perf_counter()
        mesh = extract_surface(field["values"].reshape(shape), origin, spacing)
        receipt = _field_receipt(field, full)
        receipt["particle_count"] = len(particles)
        receipt["mesh"] = mesh_metrics(mesh, volume, DROP_AXES_M)
        receipt["extraction_and_metrics_seconds"] = perf_counter() - started
        receipts[name] = receipt
    return {"source_points_digest": frame["points_digest"], "spacing_m": spacing,
            "origin_m": origin.tolist(), "shape": shape, "variants": receipts}


def _dam_break_crop(positions, predict):
    spacing = 0.005
    # Use the authors' global origin and an interior patch of the initial top surface.
    origin = np.full(3, -0.1)
    start = np.array([44, 114, 114])
    shape = (13, 13, 13)
    nodes = np.indices(shape).reshape(3, -1).T + start
    all_points = (positions - origin) / spacing
    points = _halo_particles(all_points, nodes)
    full = field_values(points, nodes, predict, spacing)
    tiled = field_values(points, nodes, predict, spacing, tile_width_cells=8)
    grid = full["values"].reshape(shape)
    heights = []
    for x, z in product(range(shape[0]), range(shape[2])):
        column = grid[x, :, z]
        crossings = np.flatnonzero((column[:-1] < 0) & (column[1:] >= 0))
        if len(crossings) == 1:
            y = crossings[0]
            fraction = column[y] / (column[y] - column[y + 1])
            heights.append(origin[1] + spacing * (start[1] + y + fraction))
    return {"source_particle_count": len(positions), "halo_particle_count": len(points),
            "spacing_m": spacing, "origin_m": origin.tolist(), "start_index": start.tolist(), "shape": shape,
            "single_crossing_columns": len(heights),
            "top_plane_rms_m": float(np.sqrt(np.mean((np.array(heights) - 0.5)**2))) if heights else None,
            "full": _field_receipt(full), "tiled": _field_receipt(tiled, full)}


def study(archive, reader, run, threads):
    import onnxruntime as ort

    started = perf_counter()
    require_sha256(archive, ARCHIVE_SHA256)
    with ZipFile(archive) as published:
        model = published.read(f"{ARCHIVE_ROOT}/models/test_nnf.onnx")
        if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
            raise ValueError("published ONNX digest mismatch")
        with published.open(f"{ARCHIVE_ROOT}/dambreak/obj/X-000.obj") as source:
            dam_break = np.loadtxt(source, usecols=(1, 2, 3))
    frames = read_pinned_frames(reader, run)
    input_seconds = perf_counter() - started
    started_load = perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    (model_input,) = session.get_inputs()
    (model_output,) = session.get_outputs()
    if (model_input.shape != [BATCH_SIZE, 1, 8, 8, 8] or model_input.type != "tensor(float)"
            or model_output.shape != [BATCH_SIZE, 1]):
        raise ValueError("unexpected published ONNX interface")
    load_seconds = perf_counter() - started_load

    def predict(batch):
        return session.run([model_output.name], {model_input.name: batch})[0]

    dam_break_result = _dam_break_crop(dam_break, predict)
    drop_result = _drop_experiment(frames[0], predict)
    with Path(__file__).open("rb") as script:
        script_digest = hashlib.file_digest(script, "sha256").hexdigest()
    return {"archive_sha256": ARCHIVE_SHA256, "model_sha256": MODEL_SHA256, "script_sha256": script_digest,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "onnxruntime": ort.__version__, "provider": session.get_providers()[0], "threads": threads},
            "input_seconds": input_seconds, "model_load_seconds": load_seconds,
            "dam_break_crop": dam_break_result, "drop": drop_result,
            "peak_process_rss_bytes": peak_rss_bytes(), "wall_seconds": perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--reader", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    result = study(args.archive, args.reader.resolve(), args.run.resolve(), args.threads)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
