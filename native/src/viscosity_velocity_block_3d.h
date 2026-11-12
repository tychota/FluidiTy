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

}  // namespace fluidity::viscosity_internal
