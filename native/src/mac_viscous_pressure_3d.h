#pragma once

#include <span>

#include "fluidity/host/viscosity.h"
#include "mac_pressure_3d_system.h"

namespace fluidity::pressure_internal {

// Solve the same coupled system on a checked explicit link graph. Strain terms
// use the graph's face slots. All links are open; wall strain offsets are
// supplied directly in the rows. Cell and face output order matches the inputs.
// Propagates explicit-graph validation and the coupled CPU solve status.
absl::StatusOr<MacViscousPressureProjection3D> ProjectExplicitMacViscousPressure3D(
    double step_duration,
    std::span<const MacPressureCell3D> cells,
    std::span<const ExplicitMacPressureFace3D> faces,
    std::span<const MacStrainRow3D> strain_rows
);

}  // namespace fluidity::pressure_internal
