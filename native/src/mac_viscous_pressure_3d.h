#pragma once

#include <span>

#include "absl/functional/function_ref.h"
#include "fluidity/host/viscosity.h"
#include "mac_pressure_3d_system.h"
#include "viscosity_velocity_block_3d.h"

namespace fluidity::pressure_internal {

class MacPressureScratch3D;

// The returned velocity is valid until the next call. The callback may use
// temporary scratch above the current mark and must restore that mark.
using SolveViscousVelocityView3D = absl::FunctionRef<absl::StatusOr<
    viscosity_internal::
        VelocityBlockViewSolution3D>(const MacViscousVelocityBlockInputs3D&, std::span<const double>, MacViscousVelocityBlockOptions3D, MacPressureScratch3D&)>;

using SolveViscousVelocity3D = absl::FunctionRef<absl::StatusOr<MacViscousVelocityBlockSolution3D>(
    const MacViscousVelocityBlockInputs3D&,
    std::span<const double>,
    MacViscousVelocityBlockOptions3D
)>;

// Runs the uniform coupled pressure solve with the supplied velocity inverse.
// The callback must satisfy the checked velocity-block contract. Its failures
// propagate without a projection. Cell and face output order matches inputs.
absl::StatusOr<MacViscousPressureProjection3D> ProjectMacViscousPressureWithVelocitySolver3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    SolveViscousVelocity3D solve_velocity_block
);

struct UniformViscousPressureInputs3D {
  SparseMacGridView3D mac_grid;
  double step_duration;
  std::span<const MacPressureCell3D> pressure_cells;
  std::span<const MacPressureFace3D> pressure_faces;
};

// Borrows the reserved pressure arena. Consume the projection before any
// other pressure scratch use. Earlier spans must remain disjoint from the appended solver scratch.
// Failures leave any published inviscid projection bank intact.
absl::StatusOr<MacViscousPressureProjectionView3D> ProjectScratchMacViscousPressure3D(
    const UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    MacPressureScratch3D& scratch
);
absl::StatusOr<MacViscousPressureProjectionView3D> ProjectScratchMacViscousPressureWithVelocitySolver3D(
    const UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    MacPressureScratch3D& scratch,
    SolveViscousVelocityView3D solve_velocity_block
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

// Runs the bounded coupled CPU solve on M3's canonical adaptive subface graph.
// Strain terms reference open owned slots. Supply sealed velocity contributions
// through prescribed strain and blocked pressure flux through the subfaces.
// Returns in cell and canonical subface order, or propagates graph/solve errors.
absl::StatusOr<MacViscousPressureProjection3D> ProjectAdaptiveMacViscousPressure3D(
    const AdaptiveMacTopology3D& topology,
    double step_duration,
    std::span<const MacPressureCell3D> cells,
    std::span<const AdaptiveMacPressureSubface3D> subfaces,
    std::span<const MacStrainRow3D> strain_rows
);

// Uses the same canonical graph and row validation with an owning velocity
// block callback. Callback failures propagate without a projection.
absl::StatusOr<MacViscousPressureProjection3D> ProjectAdaptiveMacViscousPressureWithVelocitySolver3D(
    const AdaptiveMacTopology3D& topology,
    double step_duration,
    std::span<const MacPressureCell3D> cells,
    std::span<const AdaptiveMacPressureSubface3D> subfaces,
    std::span<const MacStrainRow3D> strain_rows,
    SolveViscousVelocity3D solve_velocity_block
);

}  // namespace fluidity::pressure_internal
