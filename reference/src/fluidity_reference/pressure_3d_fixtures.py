"""Dense float64 oracle for the integrated 3D MAC pressure fixture.

For N cases, the three case offset arrays have shape (N + 1,) and give
zero-based slice bounds into concatenated cell, face and gauge arrays.
Step durations have shape (N,) in seconds. Cell indices have shape (C, 3)
and contain (x, y, z) in z/y/x order. Face indices have shape (F, 4) and
contain (normal, x, y, z) in owned slot order. The lattice point names the
head. Normals 0/1/2 are x/y/z. Gauge ordinals have shape (G,) and are local
to each case. All other arrays are one-dimensional with C or F elements.

Cells record fluid volume (m^3), target divergence (s^-1), bulk modulus
and previous/expected pressure (Pa), integrated RHS and residual (m^3/s),
and constraint defect (s^-1). Decode a zero bulk modulus to positive
infinity for incompressible cells. Faces record link kind (0 sealed,
1 internal, 2 Dirichlet), area (m^2), inverse density (m^3/kg), gradient
distance (m), predicted/wall/corrected velocity (m/s), and lift (Pa).
The lift is the oriented jump, exterior pressure, or zero, respectively.
The fixture uses unit-width cells and unit full face area.
"""

from dataclasses import dataclass, replace
from math import fsum, inf

import numpy as np


_SEALED, _INTERNAL, _DIRICHLET = range(3)
_FULL_FACE_AREA = 1.0
_BLOCK_WIDTH = 4


@dataclass(frozen=True)
class _PressureCell:
    index: tuple[int, int, int]
    fluid_volume: float = 1.0
    target_divergence: float = 0.0
    bulk_modulus: float = inf
    previous_pressure: float = 0.0


@dataclass(frozen=True)
class _PressureFace:
    index: tuple[int, int, int, int]
    link_kind: int = _SEALED
    open_area: float = 0.0
    inverse_density: float = 0.0
    pressure_gradient_distance: float = 0.0
    predicted_velocity: float = 0.0
    wall_velocity: float = 0.0
    pressure_lift: float = 0.0


@dataclass(frozen=True)
class _PressureFixtureCase:
    cells: tuple[_PressureCell, ...]
    faces: tuple[_PressureFace, ...]
    step_duration: float


def _shift_cell_index(index, axis, amount):
    return tuple(coordinate + amount if ordinal == axis else coordinate for ordinal, coordinate in enumerate(index))


def _incident_face_pairs(cell_index):
    for normal in range(3):
        yield (normal, *cell_index), (normal, *_shift_cell_index(cell_index, normal, 1))


def _face_endpoints(face_index):
    normal, x, y, z = face_index
    head = (x, y, z)
    return _shift_cell_index(head, normal, -1), head


def _face_slot_key(face_index):
    normal, x, y, z = face_index
    block_z, local_z = divmod(z, _BLOCK_WIDTH)
    block_y, local_y = divmod(y, _BLOCK_WIDTH)
    block_x, local_x = divmod(x, _BLOCK_WIDTH)
    return block_z, block_y, block_x, normal, local_z, local_y, local_x


def _make_case(cells, special_faces=(), step_duration=1.0):
    """Supply every incident face, with a stationary sealed wall by default."""
    incident_faces = {
        index for cell in cells for pair in _incident_face_pairs(cell.index) for index in pair
    }
    overrides = {face.index: face for face in special_faces}
    if len(overrides) != len(special_faces) or not overrides.keys() <= incident_faces:
        raise ValueError("pressure face overrides must be distinct and incident to a pressure cell")
    return _PressureFixtureCase(
        tuple(sorted(cells, key=lambda cell: cell.index[::-1])),
        tuple(overrides.get(index, _PressureFace(index)) for index in sorted(incident_faces, key=_face_slot_key)),
        step_duration,
    )


def _compliance(cell, step_duration):
    return cell.fluid_volume / (cell.bulk_modulus * step_duration)


def _component_gauges(matrix, anchor_diagonal, incidence, source_fluxes, wall_fluxes):
    """Use dense off-diagonals for connectivity and test physical compatibility per component."""
    remaining = set(range(len(matrix)))
    gauges = []
    while remaining:
        lowest = min(remaining)
        remaining.remove(lowest)
        component = []
        pending = [lowest]
        while pending:
            ordinal = pending.pop()
            component.append(ordinal)
            neighbors = remaining.intersection(np.flatnonzero(matrix[ordinal] < 0.0))
            pending.extend(sorted(neighbors))
            remaining.difference_update(neighbors)
        if np.any(anchor_diagonal[component] > 0.0):
            continue

        # Interior face incidences cancel before summation, including blocked wall flux.
        boundary_incidence = incidence[component].sum(axis=0)
        physical_terms = [*source_fluxes[component], *(-boundary_incidence * wall_fluxes)]
        roundoff = 32 * np.finfo(np.float64).eps * fsum(map(abs, physical_terms))
        if abs(fsum(physical_terms)) > roundoff:
            raise ValueError(f"incompatible Neumann component starting at cell {lowest}")
        gauges.append(lowest)
    return gauges


def _assemble_system(case):
    """Assemble all original rows as R diag(k) R.T plus the compliance diagonal."""
    cell_ordinals = {cell.index: ordinal for ordinal, cell in enumerate(case.cells)}
    incidence = np.zeros((len(case.cells), len(case.faces)), dtype=np.float64)
    conductances = np.zeros(len(case.faces), dtype=np.float64)
    jump_fluxes = np.zeros_like(conductances)
    exterior_lifts = np.zeros_like(conductances)
    dirichlet_conductances = np.zeros_like(conductances)
    for column, face in enumerate(case.faces):
        tail, head = _face_endpoints(face.index)
        for index, outward_sign in ((tail, 1), (head, -1)):
            if index in cell_ordinals:
                incidence[cell_ordinals[index], column] = outward_sign
        if face.open_area == 0.0:
            continue
        pressure_to_velocity_scale = case.step_duration * face.inverse_density / face.pressure_gradient_distance
        conductance = face.open_area * pressure_to_velocity_scale
        conductances[column] = conductance
        if face.link_kind == _INTERNAL:
            jump_fluxes[column] = conductance * face.pressure_lift
        elif face.link_kind == _DIRICHLET:
            exterior_lifts[column] = conductance * face.pressure_lift
            dirichlet_conductances[column] = conductance

    compliance = np.array([_compliance(cell, case.step_duration) for cell in case.cells])
    source_fluxes = np.array([cell.fluid_volume * cell.target_divergence for cell in case.cells])
    old_pressures = np.array([cell.previous_pressure for cell in case.cells])
    fluid_fluxes = np.array([face.open_area * face.predicted_velocity for face in case.faces])
    wall_fluxes = np.array([(_FULL_FACE_AREA - face.open_area) * face.wall_velocity for face in case.faces])
    matrix = (incidence * conductances) @ incidence.T + np.diag(compliance)
    rhs = source_fluxes - incidence @ (fluid_fluxes + wall_fluxes + jump_fluxes)
    rhs += np.abs(incidence) @ exterior_lifts + compliance * old_pressures
    anchor_diagonal = compliance + np.abs(incidence) @ dirichlet_conductances
    gauges = _component_gauges(matrix, anchor_diagonal, incidence, source_fluxes, wall_fluxes)
    return matrix, rhs, gauges


def _evaluate_correction(case, pressures):
    """Correct oriented velocities and sum the physical defect from cell-side fluxes."""
    pressure_at = {cell.index: pressure for cell, pressure in zip(case.cells, pressures, strict=True)}
    velocities = []
    flux_at = {}
    for face in case.faces:
        velocity = face.predicted_velocity
        if face.open_area > 0.0:
            tail, head = _face_endpoints(face.index)
            if face.link_kind == _INTERNAL:
                pressure_drop = pressure_at[tail] - pressure_at[head] + face.pressure_lift
            else:
                pressure_drop = pressure_at.get(tail, face.pressure_lift) - pressure_at.get(head, face.pressure_lift)
            pressure_to_velocity_scale = case.step_duration * face.inverse_density / face.pressure_gradient_distance
            velocity += pressure_to_velocity_scale * pressure_drop
        velocities.append(velocity)
        fluid_flux = face.open_area * velocity
        wall_flux = (_FULL_FACE_AREA - face.open_area) * face.wall_velocity
        flux_at[face.index] = fluid_flux + wall_flux

    defects = []
    for cell, pressure in zip(case.cells, pressures, strict=True):
        outward_flux = fsum(flux_at[high] - flux_at[low] for low, high in _incident_face_pairs(cell.index))
        source_flux = cell.fluid_volume * cell.target_divergence
        compliance_flux = _compliance(cell, case.step_duration) * (pressure - cell.previous_pressure)
        defects.append((outward_flux - source_flux + compliance_flux) / cell.fluid_volume)
    return np.array(velocities), np.array(defects)


def _manufactured_cases():
    origin = (0, 0, 0)
    cases = {}
    for normal, axis_name in enumerate("xyz"):
        tail = _shift_cell_index(origin, normal, -1)
        interface = _PressureFace(
            (normal, *origin),
            link_kind=_INTERNAL,
            open_area=1.0,
            inverse_density=1.0,
            pressure_gradient_distance=1.0,
            pressure_lift=1.0,
        )
        cases[f"jump_across_{axis_name}_block"] = _make_case([_PressureCell(tail), _PressureCell(origin)], [interface])

    boundary = _PressureFace(
        (0, *origin),
        link_kind=_DIRICHLET,
        open_area=0.5,
        inverse_density=1.0,
        pressure_gradient_distance=0.5,
        predicted_velocity=1.0,
        wall_velocity=2.0,
        pressure_lift=3.0,
    )
    cases["low_dirichlet"] = _make_case([_PressureCell(origin)], [boundary])
    cases["high_dirichlet"] = _make_case([_PressureCell(origin)], [replace(boundary, index=(0, 1, 0, 0))])
    separator = _PressureFace(
        (0, *origin),
        link_kind=_INTERNAL,
        inverse_density=1.0,
        pressure_gradient_distance=1.0,
        predicted_velocity=0.75,
        wall_velocity=0.25,
    )
    cases["disconnected_neumann"] = _make_case(
        [_PressureCell((-1, 0, 0), target_divergence=0.25), _PressureCell(origin, target_divergence=-0.25)], [separator]
    )
    cases["isolated_compliance"] = _make_case([
        _PressureCell(origin, target_divergence=0.5, bulk_modulus=2.0, previous_pressure=4.0)
    ])

    # Translate each case by a whole block along x, keeping their cell sets separated.
    mixed_cells = []
    mixed_faces = []
    for block_x, case in enumerate(cases.values()):
        shift = block_x * _BLOCK_WIDTH
        mixed_cells.extend(replace(cell, index=_shift_cell_index(cell.index, 0, shift)) for cell in case.cells)
        for face in case.faces:
            normal, x, y, z = face.index
            mixed_faces.append(replace(face, index=(normal, x + shift, y, z)))
    cases["mixed"] = _make_case(mixed_cells, mixed_faces)
    return cases


def uniform_mac_pressure_3d_fixture():
    """Return schema-1 SI arrays and metadata from independently solved tiny systems."""
    units = {
        "case_cell_offsets": "1",
        "case_face_offsets": "1",
        "case_gauge_offsets": "1",
        "step_durations": "s",
        "cell_indices": "1",
        "face_indices": "1",
        "cell_fluid_volumes": "m^3",
        "target_cell_divergences": "s^-1",
        "cell_bulk_moduli": "Pa",
        "previous_cell_pressures": "Pa",
        "face_link_kinds": "1",
        "face_open_areas": "m^2",
        "face_inverse_densities": "m^3/kg",
        "face_pressure_gradient_distances": "m",
        "predicted_face_velocities": "m/s",
        "wall_face_velocities": "m/s",
        "face_pressure_lifts": "Pa",
        "expected_cell_pressures": "Pa",
        "expected_integrated_rhs": "m^3/s",
        "expected_corrected_face_velocities": "m/s",
        "expected_true_residuals": "m^3/s",
        "expected_cell_constraint_defects": "s^-1",
        "expected_gauge_cell_ordinals": "1",
    }
    cases = _manufactured_cases()
    arrays = {name: [] for name in units}
    for offset_array in ("case_cell_offsets", "case_face_offsets", "case_gauge_offsets"):
        arrays[offset_array].append(0)
    for case in cases.values():
        matrix, rhs, gauges = _assemble_system(case)
        free_cells = [ordinal for ordinal in range(len(case.cells)) if ordinal not in gauges]
        pressures = np.zeros(len(case.cells), dtype=np.float64)
        if free_cells:
            reduced = np.ix_(free_cells, free_cells)
            pressures[free_cells] = np.linalg.solve(matrix[reduced], rhs[free_cells])
        corrected, defects = _evaluate_correction(case, pressures)

        arrays["step_durations"].append(case.step_duration)
        arrays["cell_indices"].extend(cell.index for cell in case.cells)
        arrays["face_indices"].extend(face.index for face in case.faces)
        arrays["cell_fluid_volumes"].extend(cell.fluid_volume for cell in case.cells)
        arrays["target_cell_divergences"].extend(cell.target_divergence for cell in case.cells)
        arrays["cell_bulk_moduli"].extend(0.0 if cell.bulk_modulus == inf else cell.bulk_modulus for cell in case.cells)
        arrays["previous_cell_pressures"].extend(cell.previous_pressure for cell in case.cells)
        arrays["face_link_kinds"].extend(face.link_kind for face in case.faces)
        arrays["face_open_areas"].extend(face.open_area for face in case.faces)
        arrays["face_inverse_densities"].extend(face.inverse_density for face in case.faces)
        arrays["face_pressure_gradient_distances"].extend(face.pressure_gradient_distance for face in case.faces)
        arrays["predicted_face_velocities"].extend(face.predicted_velocity for face in case.faces)
        arrays["wall_face_velocities"].extend(face.wall_velocity for face in case.faces)
        arrays["face_pressure_lifts"].extend(face.pressure_lift for face in case.faces)
        arrays["expected_cell_pressures"].extend(pressures)
        arrays["expected_integrated_rhs"].extend(rhs)
        arrays["expected_corrected_face_velocities"].extend(corrected)
        arrays["expected_true_residuals"].extend(rhs - matrix @ pressures)
        arrays["expected_cell_constraint_defects"].extend(defects)
        arrays["expected_gauge_cell_ordinals"].extend(gauges)
        arrays["case_cell_offsets"].append(len(arrays["cell_indices"]))
        arrays["case_face_offsets"].append(len(arrays["face_indices"]))
        arrays["case_gauge_offsets"].append(len(arrays["expected_gauge_cell_ordinals"]))

    metadata = {
        "case_names": list(cases),
        "units": units,
        "unit_system": "SI",
        "provenance": "Dense float64 NumPy incidence solve and separate cell-side corrected-flux sums.",
        "default_policy": (
            "Cell width is 1 m. Zero bulk modulus encodes infinity. "
            "Pin the first cell in each compatible unanchored component."
        ),
        "tolerances": {"rtol": 2e-12, "atol": 2e-12},
    }
    return {name: np.asarray(values) for name, values in arrays.items()}, metadata
