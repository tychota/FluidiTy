#pragma once

#include <span>
#include <vector>

#include "fluidity/host/viscosity.h"

namespace fluidity::viscosity_internal {

struct PreparedVelocityBlock3D {
  std::vector<MacStrainRow3D> weighted_rows;  // Original order, with eliminated terms set to zero.
  std::vector<double> diagonal;  // Jacobi diagonal in original face order.
  double max_rhs;  // Infinity norm of the momentum RHS, in kg m/s.
};

// Checks the inputs and builds square-root weighted rows and their Jacobi diagonal.
absl::StatusOr<PreparedVelocityBlock3D> PrepareVelocityBlock3D(
    const MacViscousVelocityBlockInputs3D& inputs,
    std::span<const double> rhs,
    MacViscousVelocityBlockOptions3D options
);

struct VelocityBlockScratch3D {
  std::span<MacStrainRow3D> weighted_rows;
  std::span<double> diagonal;
  std::span<double> velocity;
  std::span<double> residual;
  std::span<double> preconditioned;
  std::span<double> direction;
  std::span<double> applied;
};

struct VelocityBlockViewSolution3D {
  std::span<const double> velocity;  // m/s, borrowed from scratch.
  std::size_t iterations;
  double max_residual;  // kg m/s.
};

// Runs the same checked Jacobi PCG as the owning API using disjoint scratch.
// Supply one weighted row per input row and one entry per face in every
// double buffer. Errors have empty messages and carry no solution. Scratch
// may be changed on failure; no input is changed and no allocation occurs.
absl::StatusOr<VelocityBlockViewSolution3D> SolveVelocityBlockWithScratch3D(
    const MacViscousVelocityBlockInputs3D& inputs,
    std::span<const double> rhs,
    MacViscousVelocityBlockOptions3D options,
    VelocityBlockScratch3D scratch
);

}  // namespace fluidity::viscosity_internal
