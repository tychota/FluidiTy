#pragma once

#include <array>
#include <span>

#include "fluidity/host/viscosity.h"

namespace fluidity::pressure_internal {
class MacPressureScratch3D;
}

namespace fluidity::viscosity_internal {
// Builds ordered strain in reserved scratch. Empty octants select full cells;
// otherwise pressure faces must be in owned order. Results borrow scratch.
absl::StatusOr<std::span<MacStrainRow3D>> AssembleUniformStrainInScratch3D(
    const SparseMacGridView3D& grid,
    std::span<const MacViscosityCell3D> cells,
    std::span<const std::array<double, 8>> octant_volumes,
    std::span<const MacPressureFace3D> pressure_faces,
    pressure_internal::MacPressureScratch3D& scratch
);
}  // namespace fluidity::viscosity_internal
