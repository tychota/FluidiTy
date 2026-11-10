"""Measure a deterministic geometric reconstruction on the pinned M1 drop."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


SOURCE_REVISION = "3e0526edb9adda66b4be965ab469d74f5042c990"
FINGERPRINT = "f8929d0eac4fc91e745589087773706152691cf51f1bb0e5e114420a7d724afa"
EXECUTION_ID = "328f630737aa672989e6f3cc7be1a542c803efc06411b3b3e4d9bf5542037c77"
READER_SHA256 = "040fec842ea1554a0fee675dd4504569cfef052e120cdc7b3bffec0ca6311da4"
SOLVER_SHA256 = "ee953ed5f397e0f3418b408197c19267752dd68b3f062dc2cab522fa2f28eab8"
RUN_ARTIFACTS = {
    "scene.flrs": "88690db9517077147ab4768b35c416cbff3b7d590bc50d1cc2fe3de062960dd5",
    "scene.flrs.flst": "3e8310a0d0f1a9674bb98d4ac823778dc1e25d51aca82cfbe26ba4347fbb7cf1",
    "frames.jsonl": "f5799408b1ca93b49ea887c9cd2036087691a9e875e321ba39028aede19bd991",
}
SELECTED_FRAMES = (0, 48, 96, 144, 192, 216)
DROP_RADIUS_M = 0.125
DROP_AXES_M = DROP_RADIUS_M * np.array([1 / math.sqrt(1.03), 1.03, 1 / math.sqrt(1.03)])
SUPPORT_SCALE = 1.35
DENSITY_SIGMA_M = DROP_RADIUS_M / 5
DENSITY_THRESHOLD = 0.5
SPACINGS_M = (0.0125, 0.00625)
CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
# Using the same body diagonal in each cube also gives shared face diagonals.
TETRAHEDRA = ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
              (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7))


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray


def require_sha256(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path.name} SHA-256 mismatch: expected {expected}, received {actual}")


def read_pinned_frames(reader: Path, run: Path) -> dict[int, dict]:
    """Revalidate every publication using the checked M1 reader and the frozen export."""
    require_sha256(reader, READER_SHA256)
    require_sha256(reader.with_name("fluidity_simulate" + reader.suffix), SOLVER_SHA256)
    for name, digest in RUN_ARTIFACTS.items():
        require_sha256(run / name, digest)
    selected = {}
    command = [reader, "read-frames", run / "cache", FINGERPRINT, "0", "216"]
    with tempfile.TemporaryFile() as verified, (run / "frames.jsonl").open("rb") as archived:
        subprocess.run(command, stdout=verified, check=True, timeout=120)
        verified.seek(0)
        for ordinal in range(217):
            current = verified.readline().rstrip(b"\r\n")
            if not current or current != archived.readline().rstrip(b"\r\n"):
                raise ValueError(f"committed frame {ordinal} differs from the pinned export")
            frame = json.loads(current)
            if (frame["fixture_version"] != 1 or frame["event"] != "frame" or frame["backend"] != "cuda"
                    or frame["resolved_fingerprint"] != FINGERPRINT or frame["execution_id"] != EXECUTION_ID
                    or frame["frame"] != ordinal or not math.isclose(frame["time_seconds"], ordinal / 24,
                                                                    rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"committed frame {ordinal} has invalid identity or clock")
            if ordinal in SELECTED_FRAMES:
                selected[ordinal] = frame
        if verified.read(1) or archived.read(1):
            raise ValueError("unexpected data after committed frame 216")
    return selected


def extract_surface(field, origin, spacing) -> Mesh:
    """Extract the oriented zero set of a piecewise linear field on a padded lattice."""
    field = np.asarray(field, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if field.ndim != 3 or min(field.shape) < 2 or origin.shape != (3,):
        raise ValueError("expected a three-dimensional lattice and origin")
    if not np.isfinite(field).all() or not np.isfinite(origin).all() or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("expected finite lattice samples, origin and positive spacing")
    for axis in range(3):
        if np.any(field.take([0, -1], axis=axis) <= 0):
            raise ValueError("surface reaches the lattice boundary")
    # Assign exact ties to air and separate their intersections from the shared node.
    field = np.where(field == 0, spacing * 1e-8, field)
    shape = np.array(field.shape)
    inside = field < 0
    corner_signs = np.stack([inside[tuple(slice(c, n - 1 + c) for c, n in zip(corner, shape))]
                             for corner in CORNERS])
    active_cubes = np.argwhere(np.any(corner_signs, axis=0) & ~np.all(corner_signs, axis=0))
    vertices = []
    triangles = []
    edge_vertices = {}
    for cube in active_cubes:
        nodes = cube + CORNERS
        values = field[tuple(nodes.T)]
        node_ids = np.ravel_multi_index(nodes.T, field.shape)
        positions = origin + nodes * spacing

        def crossing(first, second):
            key = tuple(sorted((node_ids[first], node_ids[second])))
            if key not in edge_vertices:
                fraction = values[first] / (values[first] - values[second])
                edge_vertices[key] = len(vertices)
                vertices.append(positions[first] + fraction * (positions[second] - positions[first]))
            return edge_vertices[key]

        for tet in TETRAHEDRA:
            liquid = [node for node in tet if values[node] < 0]
            air = [node for node in tet if values[node] > 0]
            if not liquid or not air:
                continue
            if len(liquid) == 2:
                a, b = liquid
                c, d = air
                ac, ad, bd, bc = crossing(a, c), crossing(a, d), crossing(b, d), crossing(b, c)
                faces = ((ac, ad, bd), (ac, bd, bc))
            else:
                faces = (tuple(crossing(a, b) for a in liquid for b in air),)
            # Every inside-to-outside edge has positive dot product with the PL gradient.
            outward = positions[air[0]] - positions[liquid[0]]
            for a, b, c in faces:
                normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
                triangles.append((a, b, c) if normal @ outward > 0 else (a, c, b))
    return Mesh(np.asarray(vertices, dtype=float).reshape(-1, 3),
                np.asarray(triangles, dtype=np.int64).reshape(-1, 3))


def mesh_metrics(mesh: Mesh, reference_volume: float, axes=None) -> dict:
    vertices, triangles = mesh.vertices, mesh.triangles
    if reference_volume <= 0 or not math.isfinite(reference_volume):
        raise ValueError("expected positive reference volume")
    half_edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges, edge_ids, multiplicity = np.unique(np.sort(half_edges, axis=1), axis=0,
                                             return_inverse=True, return_counts=True)
    orientation = np.bincount(edge_ids, weights=np.where(half_edges[:, 0] < half_edges[:, 1], 1, -1))
    graph = coo_matrix((np.ones(len(edges)), tuple(edges.T)), shape=(len(vertices), len(vertices)))
    components = connected_components(graph, directed=False, return_labels=False)
    a, b, c = vertices[triangles].transpose(1, 0, 2)
    area_twice = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    # Use a local origin for the signed tetrahedron sum.
    offset = vertices[0] if len(vertices) else np.zeros(3)
    signed_volume = float(np.einsum("ij,ij->i", a - offset, np.cross(b - offset, c - offset)).sum() / 6)
    digest = hashlib.sha256()
    for array, dtype in ((vertices, "<f8"), (triangles, "<i8")):
        digest.update(len(array).to_bytes(8, "little"))
        digest.update(array.astype(dtype, copy=False).tobytes())
    result = {
        "mesh_sha256": digest.hexdigest(),
        "vertices": len(vertices),
        "triangles": len(triangles),
        "components": int(components),
        "euler_characteristic": len(vertices) - len(edges) + len(triangles),
        "boundary_edges": int(np.count_nonzero(multiplicity == 1)),
        "nonmanifold_edges": int(np.count_nonzero(multiplicity > 2)),
        "orientation_defects": int(np.count_nonzero((multiplicity == 2) & (orientation != 0))),
        "degenerate_triangles": int(np.count_nonzero(area_twice == 0)),
        "surface_area_m2": float(area_twice.sum() / 2),
        "signed_volume_m3": signed_volume,
        "reference_volume_m3": reference_volume,
        "relative_volume_error": signed_volume / reference_volume - 1,
    }
    if axes is not None and len(vertices):
        # This is a radial discrepancy on extracted vertices about the authored origin.
        radius = np.linalg.norm(vertices, axis=1)
        ellipsoid_level = np.linalg.norm(vertices / np.asarray(axes), axis=1)
        error = radius * (1 - 1 / ellipsoid_level)
        result["vertex_radial_error_rms_m"] = float(np.sqrt(np.mean(error**2)))
        result["vertex_radial_error_max_m"] = float(np.max(np.abs(error)))
    return result


def density_field(positions, volumes, lower, shape, spacing):
    """Splat material volume in eight nodes and smooth to a volume fraction."""
    coordinates = positions / spacing - lower
    cells = np.floor(coordinates).astype(int)
    fraction = coordinates - cells
    deposited = np.zeros(shape)
    for corner in CORNERS:
        weights = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
        np.add.at(deposited, tuple((cells + corner).T), volumes * weights)
    density = gaussian_filter(deposited, DENSITY_SIGMA_M / spacing, mode="constant", truncate=4) / spacing**3
    return DENSITY_THRESHOLD - density


def reconstruct(points: list[dict], spacing: float, method: str) -> tuple[Mesh, float, float]:
    if method not in ("analytic", "isotropic", "covariance", "density"):
        raise ValueError(f"unknown reconstruction method {method}")
    points = sorted(points, key=lambda point: point["id"])
    positions = np.array([point["world_position"] for point in points])
    radii = np.array([point["radius"] for point in points])
    if (len(points) != 552 or [point["id"] for point in points] != list(range(1, 553))
            or not np.isfinite(positions).all() or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError("invalid isolated drop particles")
    volumes = 4 * math.pi / 3 * radii**3
    material_volume = math.fsum(volumes)
    # Equal-radius support is deliberate; changes in individual material radii still enter the volume oracle.
    support_radius = SUPPORT_SCALE * (material_volume / (len(radii) * 4 * math.pi / 3))**(1 / 3)
    if method == "density":
        support_radius = 4 * DENSITY_SIGMA_M
    transform = np.eye(3)
    max_scale = 1.0
    if method == "covariance":
        centered = positions - positions.mean(axis=0)
        variances, basis = np.linalg.eigh(centered.T @ centered / len(positions))
        if np.any(variances <= 0):
            raise ValueError("singular drop covariance")
        scales = np.sqrt(variances / math.exp(np.log(variances).mean()))
        max_scale = float(scales.max())
        transform = (basis / scales) @ basis.T
    padding = support_radius * max_scale + 2 * spacing
    lower = np.floor((np.minimum(positions.min(axis=0), -DROP_AXES_M) - padding) / spacing).astype(int)
    upper = np.ceil((np.maximum(positions.max(axis=0), DROP_AXES_M) + padding) / spacing).astype(int)
    if method == "density":
        field = density_field(positions, volumes, lower, upper - lower + 1, spacing)
    else:
        axes = [np.arange(lo, hi + 1) * spacing for lo, hi in zip(lower, upper)]
        grid = np.meshgrid(*axes, indexing="ij")
        if method == "analytic":
            field = np.sqrt(sum((coord / axis)**2 for coord, axis in zip(grid, DROP_AXES_M))) - 1
        else:
            queries = np.stack(grid, axis=-1).reshape(-1, 3)
            distance, _ = cKDTree(positions @ transform).query(queries @ transform, workers=1)
            field = distance.reshape(grid[0].shape) - support_radius
    return extract_surface(field, lower * spacing, spacing), material_volume, support_radius


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    unit = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit


def benchmark(reader: Path, run: Path) -> dict:
    started = perf_counter()
    frames = read_pinned_frames(reader, run)
    verification_seconds = perf_counter() - started
    samples = []
    timings = {}
    for method in ("analytic", "isotropic", "covariance", "density"):
        for spacing in SPACINGS_M:
            wall_times = []
            for ordinal, frame in frames.items():
                if method == "analytic" and ordinal != 0:
                    continue
                start = perf_counter()
                mesh, material_volume, radius = reconstruct(frame["points"], spacing, method)
                mesh_seconds = perf_counter() - start
                metrics = mesh_metrics(mesh, material_volume, DROP_AXES_M if ordinal == 0 else None)
                samples.append({
                    "frame": ordinal, "method": method, "spacing_m": spacing,
                    "source_points_digest": frame["points_digest"], "support_radius_m": radius,
                    "reconstruction_seconds": mesh_seconds,
                    "metrics_seconds": perf_counter() - start - mesh_seconds, **metrics,
                })
                wall_times.append(mesh_seconds)
            timings[f"{method}:{spacing}"] = {
                "median_seconds": statistics.median(wall_times),
                "p95_seconds": sorted(wall_times)[math.ceil(0.95 * len(wall_times)) - 1],
                "max_seconds": max(wall_times),
            }
    return {
        "source_revision": SOURCE_REVISION,
        "source_execution_id": EXECUTION_ID,
        "source_fingerprint": FINGERPRINT,
        "verified_publications": 217,
        "runtime": {"platform": platform.platform(), "cpu": platform.processor(),
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "verification_seconds": verification_seconds,
        "samples": samples,
        "reconstruction_timing": timings,
        "peak_process_rss_bytes": peak_rss_bytes(),
        "wall_seconds": perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = benchmark(args.reader.resolve(), args.run.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
