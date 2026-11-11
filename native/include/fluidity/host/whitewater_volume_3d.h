#pragma once

#include <array>
#include <span>

#include "fluidity/host/whitewater_3d.h"

namespace fluidity {
struct WhitewaterVolumeGrid3D {
  Vec3 origin;  // World centre of voxel (0,0,0), metres.
  double spacing_m;
  std::array<std::size_t, 3> extent;
};
struct WhitewaterVolumePoint3D {
  WhitewaterKind3D kind;
  Vec3 position;
  double represented_count;  // Positive physical-particle count represented by this point.
  double gas_volume_m3;  // Total represented gas volume; zero except for bubbles.
};
struct WhitewaterVolume3D {
  WhitewaterVolumeGrid3D grid;
  std::vector<double> foam_count_per_m3;
  std::vector<double> bubble_count_per_m3;
  std::vector<double> gas_fraction;  // Additive dispersed gas volume / voxel volume.
  std::vector<std::uint8_t> validity;  // Caller-supplied known-domain mask, preserved verbatim.
};

// Deposits foam/bubble represented counts and bubble gas volume conservatively
// onto eight voxel centres. Spray contributes neither density. Arrays are
// x-major, then y, then z. Input validity must contain exactly extent product
// bytes in {0,1}; positive-weight support must be inside known voxels. No clipped
// or renormalized deposits. Empty frames retain the full grid and validity with
// zero density. Invalid input returns InvalidArgument, missing/clipped support
// FailedPrecondition, node budgets ResourceExhausted, numerical overflow
// OutOfRange. Inputs are borrowed; failure returns no partial volume.
absl::StatusOr<WhitewaterVolume3D> RasterizeWhitewaterVolume3D(
    std::span<const WhitewaterVolumePoint3D> points,
    const WhitewaterVolumeGrid3D& grid,
    std::span<const std::uint8_t> validity,
    std::size_t maximum_voxels
);
}  // namespace fluidity
