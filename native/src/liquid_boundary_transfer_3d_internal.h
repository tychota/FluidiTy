#pragma once

#include <span>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity::internal {

struct LiquidGhostFitSample3D {
  Vec3 offset_in_cell_widths;
  double velocity;
};
struct LiquidGhostAffineFit3D {
  bool full_rank;
  double velocity_at_target;
  Vec3 velocity_slopes_in_cell_widths;
};

// Internal host numerical seam for the bounded ghost kernel. Fits [1,x,y,z]
// in supplied donor order using deterministic pivoted QR. At most 125 finite
// rows; rank cutoff is 1e-10 of the largest pivot. Rank loss is a successful
// full_rank=false result, never a division by a deficient pivot. Invalid rows
// return InvalidArgument; nonfinite intermediate arithmetic returns OutOfRange.
absl::StatusOr<LiquidGhostAffineFit3D> FitLiquidGhostAffine3D(std::span<const LiquidGhostFitSample3D> samples);

}  // namespace fluidity::internal
