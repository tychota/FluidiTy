#pragma once

#include <span>

#include "absl/status/statusor.h"
#include "fluidity/base/liquid_boundary_transfer_math_3d.h"
#include "fluidity/host/liquid_boundary_transfer_3d.h"

namespace fluidity::internal {

// Internal host numerical seam for the bounded ghost kernel. Fits [1,x,y,z]
// in supplied donor order using deterministic pivoted QR. At most 125 finite
// rows; rank cutoff is 1e-10 of the largest pivot. Rank loss is a successful
// full_rank=false result, never a division by a deficient pivot. Invalid rows
// return InvalidArgument; nonfinite intermediate arithmetic returns OutOfRange.
absl::StatusOr<LiquidGhostAffineFit3D> FitLiquidGhostAffine3D(std::span<const LiquidGhostFitSample3D> samples);

// Stage times are checked separately. The continuation samples can be reused
// when both validated histories borrow the same immutable arrays.
inline bool SameBoundaryVelocitySamples3D(
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history
) {
  return old_history.face_velocities.data() == new_history.face_velocities.data()
      && old_history.validity.data() == new_history.validity.data()
      && old_history.face_tags.data() == new_history.face_tags.data()
      && old_history.wall_samples.data() == new_history.wall_samples.data();
}

// Adds reference-volume Kirchhoff stress forces in newtons using the same
// normalized support as compatible mass and momentum deposition. Stencils
// are in Cartesian order and must come from the checked deposit for this
// owned grid. The caller owns the initialized force scratch and discards it
// on error. Nonfinite force accumulation returns OutOfRange.
absl::Status AccumulateCompatibleSolidStress3D(
    const std::array<CompatibleMacFaceStencil3D, 3>& stencils,
    double reference_volume,
    Mat3 kirchhoff_stress,
    std::span<double> face_force
);

struct StationaryFloorGhostTerm3D {
  std::size_t donor_slot;
  double coefficient;
};

struct StationaryFloorGhostRow3D {
  std::size_t target_slot;
  std::vector<StationaryFloorGhostTerm3D> terms;
  bool fallback = false;
};

// Applies the transpose of the single-side stationary floor continuation to
// a full-stencil force field. Below-floor normal ghost velocities are zero.
// All other missing samples use the radius-two affine fit or its nearest
// donor fallback. On success, only compatible live donor slots carry force.
absl::StatusOr<double> ScatterStationaryFloorGhostTranspose3D(
    SparseMacGridView3D grid,
    std::span<const LiquidFaceValidity3D> validity,
    std::span<const LiquidBoundaryFaceTag3D> tags,
    std::span<double> force,
    std::span<const MacFaceDeposit3D> masses = {},
    std::span<const std::uint8_t> needed = {},
    std::vector<StationaryFloorGhostRow3D>* ghost_rows = nullptr,
    bool reflect_normal = false
);

}  // namespace fluidity::internal
