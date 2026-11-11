#pragma once

#include "fluidity/host/whitewater_frame_3d.h"
#include "fluidity/host/whitewater_volume_3d.h"

namespace fluidity {
struct WhitewaterVolumeSettings3D {
  WhitewaterVolumeGrid3D grid{};  // Fixed declared known box, identical across frames.
  double physical_bubble_radius_m = 0.001;  // Radius of one represented gas bubble.
  std::size_t maximum_voxels = 2000000;
};
struct WhitewaterVolumeFrame3D {
  LiquidClock3D clock{};
  WhitewaterVolume3D volume{};
};

// Deposits checked Whitewater points into the declared box. Each bubble adds
// represented_count times 4*pi*r³/3 gas volume; foam/spray add no gas and spray
// adds no density. The known mask is one throughout the box; positive support
// outside fails rather than clipping. Invalid settings/records return
// InvalidArgument, numerical overflow OutOfRange, bounds ResourceExhausted.
// Borrows inputs, publishes no partial volume.
absl::StatusOr<WhitewaterVolumeFrame3D> BuildWhitewaterVolumeFrame3D(
    const WhitewaterFrame3D& points,
    const WhitewaterVolumeSettings3D& settings,
    double represented_count_per_point
);

// Encodes the full validity mask and only nonzero voxels in increasing x-major
// index order. Requires finite nonnegative fields and no values on unknown
// voxels. Both codecs enforce up to two million voxels before dense allocation.
// Invalid inputs return InvalidArgument; malformed payload DataLoss; limits
// ResourceExhausted. Decoding owns all arrays and preserves empty known frames.
absl::StatusOr<LiquidCacheBytes3D> EncodeWhitewaterVolumeFrame3D(
    const WhitewaterVolumeFrame3D& frame,
    std::size_t maximum_voxels = 2000000,
    LiquidCacheLimits3D limits = {}
);
absl::StatusOr<WhitewaterVolumeFrame3D> DecodeWhitewaterVolumeFrame3D(
    std::span<const std::uint8_t> bytes,
    std::size_t maximum_voxels = 2000000,
    LiquidCacheLimits3D limits = {}
);

// Canonical physical box and bubble controls, excluding the voxel work limit.
// Returns InvalidArgument for invalid controls, ResourceExhausted for limits.
absl::StatusOr<LiquidCacheBytes3D>
EncodeWhitewaterVolumeSettings3D(const WhitewaterVolumeSettings3D& settings, LiquidCacheLimits3D limits = {});
// Restores physical settings and the default voxel limit. Invalid bytes return
// DataLoss, unknown algorithm FailedPrecondition, excess capacity ResourceExhausted.
absl::StatusOr<WhitewaterVolumeSettings3D>
DecodeWhitewaterVolumeSettings3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
}  // namespace fluidity
