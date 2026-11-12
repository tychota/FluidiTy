"""Fixed-domain low-Mach gas reference operators on a uniform MAC grid."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import connected_components

from .pressure import PressureInputs, solve_pressure


GAMMA = 1.4


@dataclass(frozen=True)
class HeatingProjection:
    p0: np.ndarray
    divergence: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class DensityTransport:
    densities: np.ndarray
    outward_rate: np.ndarray


def _finite_array(values, shape, name):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid {name}")
    return array


def _positive_scalar(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {name}")


def project_heating(grid, dt, heat_rate, previous_p0, permeable, open_faces, ambient_p0):
    """Project from rest with volumetric heat in W/m³ and absolute pressure in Pa.

    The 2D fixture assumes unit depth, constant mobility, fixed walls and
    frozen old-time p0 in the divergence denominator. A permeable exterior
    face must be open to ambient gauge pressure. Net conductive heat, when
    prescribed, is included in heat_rate.
    """
    _positive_scalar(dt, "dt")
    _positive_scalar(ambient_p0, "ambient p0")
    heat = _finite_array(heat_rate, (grid.cell_count,), "heat rate")
    old_p0 = _finite_array(previous_p0, heat.shape, "previous p0")
    permeable = np.asarray(permeable)
    open_faces = np.asarray(open_faces)
    for mask in (permeable, open_faces):
        if mask.dtype.kind != "b" or mask.shape != (grid.face_count,):
            raise ValueError("invalid face mask")
    incidence = grid.incidence
    interior = np.asarray(abs(incidence).sum(axis=0)).ravel() == 2
    if np.any(open_faces != (permeable & ~interior)) or np.any(old_p0 <= 0):
        raise ValueError("invalid open face or absolute pressure")
    links = abs(incidence[:, interior & permeable])
    count, labels = connected_components(links @ links.T, directed=False)
    ambient_cells = np.asarray(abs(incidence) @ open_faces).ravel() > 0
    p0 = np.empty_like(heat)
    divergence = np.empty_like(heat)
    for component in range(count):
        cells = labels == component
        component_heat = heat[cells]
        if np.any(ambient_cells[cells]):
            p0[cells] = ambient_p0
            divergence[cells] = (GAMMA - 1) * component_heat / (GAMMA * ambient_p0)
            continue
        pressure = old_p0[cells][0]
        if np.any(old_p0[cells] != pressure):
            raise ValueError("sealed component requires uniform previous p0")
        mean_heat = np.mean(component_heat)
        p0[cells] = pressure + dt * (GAMMA - 1) * mean_heat
        divergence[cells] = (GAMMA - 1) * (component_heat - mean_heat) / (GAMMA * pressure)
    if np.any(p0 <= 0) or not np.all(np.isfinite(p0)):
        raise ValueError("updated p0 must be finite and positive")
    projection = solve_pressure(PressureInputs(
        grid, dt, np.zeros(grid.face_count), aperture=permeable.astype(float),
        dirichlet_mask=open_faces, source=divergence,
    ))
    return HeatingProjection(p0, divergence, projection.velocity)


def transport_densities(grid, dt, densities, source_rates, face_velocity, exterior_densities):
    """Advance nonnegative cell densities with donor-cell MAC fluxes and sources.

    Each column is one extensive quantity per unit volume. The returned
    outward rate is net across all exterior faces and positive for outflow.
    Closed faces have zero velocity. Reject a transport CFL above one or a
    source update that would leave a negative or nonfinite density.
    """
    _positive_scalar(dt, "dt")
    densities = np.asarray(densities, dtype=float)
    if densities.ndim != 2 or densities.shape[0] != grid.cell_count or densities.shape[1] == 0:
        raise ValueError("invalid density shape")
    rates = _finite_array(source_rates, densities.shape, "source rates")
    velocity = _finite_array(face_velocity, (grid.face_count,), "face velocity")
    exterior = _finite_array(exterior_densities, (densities.shape[1],), "exterior densities")
    if not np.all(np.isfinite(densities)) or np.any(densities < 0) or np.any(exterior < 0):
        raise ValueError("invalid densities")

    incidence = grid.incidence
    volume_flux = velocity * grid.face_areas
    # Positive incidence means positive face velocity leaves that cell.
    donors = incidence @ diags(np.sign(volume_flux))
    donors.data = np.maximum(donors.data, 0)
    has_donor = np.asarray(donors.sum(axis=0)).ravel() > 0
    outflow = donors @ np.abs(volume_flux)
    volume = grid.cell_volumes
    if np.any(dt * outflow > volume):
        raise ValueError("transport CFL exceeds one")
    face_densities = donors.T @ densities
    face_densities[~has_donor] = exterior
    flux = volume_flux[:, None] * face_densities
    updated = densities + dt * (rates - (incidence @ flux) / volume[:, None])
    outward_rate = np.asarray(incidence.sum(axis=0)).ravel() @ flux
    if np.any(updated < 0) or not np.all(np.isfinite(updated)) or not np.all(np.isfinite(outward_rate)):
        raise ValueError("invalid transport result")
    return DensityTransport(updated, outward_rate)
