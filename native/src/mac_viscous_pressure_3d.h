#pragma once

#include <span>

#include "fluidity/host/viscosity.h"
#include "mac_pressure_3d_system.h"

namespace fluidity::pressure_internal {

using SolveViscousVelocity3D = absl::StatusOr<MacViscousVelocityBlockSolution3D> (*)(
    const MacViscousVelocityBlockInputs3D&,
    std::span<const double>,
    MacViscousVelocityBlockOptions3D
);

// Runs the uniform coupled pressure solve with the supplied velocity inverse.
// The callback must satisfy the checked velocity-block contract. Its failures
// propagate without a projection. Cell and face output order matches inputs.
absl::StatusOr<MacViscousPressureProjection3D> ProjectMacViscousPressureWithVelocitySolver3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    SolveViscousVelocity3D solve_velocity_block
);

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
