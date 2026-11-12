#pragma once

#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/viscosity.h"

namespace fluidity {

struct CudaMacViscousVelocityBlockSolution3D {
  MacViscousVelocityBlockSolution3D solution;
  CudaOperatorMeasurement measurement;  // CUDA event timing for the solve, excluding transfers.
};

// Solves on CUDA device 0 with the units, eliminated slots, true residual gate, and
// failure contract of SolveMacViscousVelocityBlock3D. Unavailable hardware returns
// FailedPrecondition. On failure no velocity or measurement is published.
absl::StatusOr<CudaMacViscousVelocityBlockSolution3D> CudaSolveMacViscousVelocityBlock3D(
    const MacViscousVelocityBlockInputs3D& inputs,
    std::span<const double> rhs,
    MacViscousVelocityBlockOptions3D options = {}
);

// Projects the coupled viscous momentum and pressure equations. The pressure
// iteration is host-orchestrated and every velocity inverse executes on CUDA
// device 0. Inputs, ordering, units and failure semantics match the CPU API.
absl::StatusOr<MacViscousPressureProjection3D> CudaProjectMacViscousPressure3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options = {}
);

}  // namespace fluidity
