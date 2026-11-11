#pragma once

#include "fluidity/host/liquid_substep_3d.h"
#include "fluidity/host/whitewater_3d.h"

namespace fluidity {
// True for a stationary, finite unit-normal halfspace usable by the beach
// source and transport. Moving and curved obstacles require a different adapter.
bool IsStationaryWhitewaterBeach3D(const LiquidObstacle3D& obstacle);
// Samples an accepted analytic sphere or particle sphere union. A stationary
// halfspace clips the occupancy probes and excludes birth centers in solid.
// Uses right-end quadrature: both the occupied volume and the carrier
// come from this accepted version, with duration measured from the previous
// accepted time. The fixed world source grid uses eight midpoint occupancy
// probes per cell, in a surface layer of two source spacings. The sphere radius
// must exceed two source spacings so the centre has no singular normal query.
// Uses exact analytic outward normals and curvature 2/R and the checked frozen
// carrier. Empty material returns no events. Missing carrier propagates;
// unsupported scene/side returns Unimplemented, invalid controls InvalidArgument,
// unrepresentable coordinates OutOfRange and >4096 candidate cells or excess
// births ResourceExhausted. Borrows all input and never changes Simulate state.
absl::StatusOr<std::vector<WhitewaterCellBirth3D>> SampleAcceptedWhitewaterStep3D(
    const LiquidState3D& accepted,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    Vec3 world_source_grid_origin,
    double previous_accepted_time_seconds,
    const WhitewaterSourceOptions3D& source_options,
    std::size_t maximum_births = 4096
);
}  // namespace fluidity
