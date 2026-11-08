"""Analytic SI cases for uniform 3D liquid boundaries."""

import numpy as np


def uniform_mac_liquid_boundary_3d_fixture():
    """Return schema-1 arrays and metadata for analytic boundary cases.

    Geometry uses the 1 m cube [0, 1]^3 and its z=0 face, which spans x and y.
    The three halfspace rows have solid x <= d for d = 0, 1/4 and 1-2^-20 m.
    The single slab row has center x = 1/2 m and half-width 1/32 m.

    Free-surface rows pair (liquid, air) signed center distances. The first
    pair needs lower regularization and the second is already in range.
    The single capillary row is a 0.5 m sphere with zero ambient gauge pressure.

    Sweep rows are a particle crossing a stationary slab, a moving slab crossing
    a stationary particle, and a particle co-moving outside the slab. Coordinates
    are along x and vary linearly over the normalized step. Incident side is -1
    on the low side, +1 on the high side, and 0 for no contact. A first contact
    fraction of -1 also denotes no contact.

    Each array has shape (N,) for its group except the signed center distances,
    which have shape (N, 2).
    """
    plane_offsets = np.array([0.0, 0.25, 1.0 - 2.0**-20])
    # The open strip and prism both have x width 1-d and unit transverse extents.
    open_widths = 1.0 - plane_offsets
    slab_half_widths = np.array([1.0 / 32.0])
    # Both slab boundaries lie between the face's x endpoints.
    slab_blocked_face_areas = 2.0 * slab_half_widths

    center_distances = np.array([[-0.01, 0.99], [-0.25, 0.75]])
    liquid_distances, air_distances = center_distances.T
    interface_fractions = liquid_distances / (liquid_distances - air_distances)
    regularized_theta = np.clip(interface_fractions, 0.05, 1.0)

    sphere_radii = np.array([0.5])
    surface_tensions = np.array([0.072])
    sphere_curvatures = 2.0 / sphere_radii
    sphere_pressure_lifts = surface_tensions * sphere_curvatures

    sweep_start_coordinates = np.array([-1.0, 0.0, 0.0])
    sweep_end_coordinates = np.array([1.0, 0.0, 1.0])
    sweep_wall_start_centers = np.array([0.0, -1.0, -1.0])
    sweep_wall_end_centers = np.array([0.0, 1.0, 0.0])
    sweep_half_widths = np.full(3, slab_half_widths[0])
    relative_starts = sweep_start_coordinates - sweep_wall_start_centers
    relative_ends = sweep_end_coordinates - sweep_wall_end_centers
    incident_sides = np.sign(relative_starts)
    # All paths start outside the slab. Find the zero of the signed distance
    # to the incident plane, which stays linear even after crossing the far plane.
    start_plane_distances = np.abs(relative_starts) - sweep_half_widths
    end_plane_distances = incident_sides * relative_ends - sweep_half_widths
    hits_slab = end_plane_distances <= 0.0
    first_contact_fractions = np.full(3, -1.0)
    np.divide(
        start_plane_distances, start_plane_distances - end_plane_distances,
        out=first_contact_fractions, where=hits_slab,
    )

    arrays = {
        "unit_cell_plane_offsets": plane_offsets,
        "expected_open_face_areas": open_widths.copy(),
        "expected_open_cell_volumes": open_widths,
        "slab_half_widths": slab_half_widths,
        "expected_slab_blocked_face_areas": slab_blocked_face_areas,
        "free_surface_center_distances": center_distances,
        "expected_regularized_theta": regularized_theta,
        "sphere_radii": sphere_radii,
        "surface_tensions": surface_tensions,
        "expected_sphere_curvatures": sphere_curvatures,
        "expected_sphere_pressure_lifts": sphere_pressure_lifts,
        "sweep_start_coordinates": sweep_start_coordinates,
        "sweep_end_coordinates": sweep_end_coordinates,
        "sweep_wall_start_centers": sweep_wall_start_centers,
        "sweep_wall_end_centers": sweep_wall_end_centers,
        "sweep_half_widths": sweep_half_widths,
        "expected_first_contact_fractions": first_contact_fractions,
    }
    metadata = {
        "units": {
            "unit_cell_plane_offsets": "m",
            "expected_open_face_areas": "m^2",
            "expected_open_cell_volumes": "m^3",
            "slab_half_widths": "m",
            "expected_slab_blocked_face_areas": "m^2",
            "free_surface_center_distances": "m",
            "expected_regularized_theta": "1",
            "sphere_radii": "m",
            "surface_tensions": "N/m",
            "expected_sphere_curvatures": "m^-1",
            "expected_sphere_pressure_lifts": "Pa",
            "sweep_start_coordinates": "m",
            "sweep_end_coordinates": "m",
            "sweep_wall_start_centers": "m",
            "sweep_wall_end_centers": "m",
            "sweep_half_widths": "m",
            "expected_first_contact_fractions": "1",
        },
        "unit_system": "SI",
        "provenance": "Analytic strip and prism measures, linear interface fraction, sphere curvature, and relative sweep.",
        "default_policy": (
            "Solid is x <= d or a centered slab. Liquid signed distance is negative inside. "
            "Clamp theta to [0.05, 1]. A contact fraction of -1 denotes no contact."
        ),
        "sweep_incident_sides": np.where(hits_slab, incident_sides, 0).astype(int).tolist(),
        "tolerances": {"rtol": 2e-12, "atol": 2e-12},
    }
    return arrays, metadata
