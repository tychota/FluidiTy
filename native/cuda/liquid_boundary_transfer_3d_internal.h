#pragma once
#include "../src/liquid_boundary_transfer_3d_internal.h"
namespace fluidity::cuda_internal {
// Internal numerical qualification seam: same QR used by the device ghost kernel.
// Matches CPU fit bounds and errors; owning result publishes after cleanup only.
absl::StatusOr<internal::LiquidGhostAffineFit3D>
    CudaFitLiquidGhostAffine3D(std::span<const internal::LiquidGhostFitSample3D>);
}  // namespace fluidity::cuda_internal
