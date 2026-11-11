#pragma once

#include "fluidity/host/whitewater_frame_3d.h"

namespace fluidity {
struct WhitewaterCacheLimits3D {
  std::size_t maximum_points = 65536;
  std::size_t maximum_source_cells = 4096;
  LiquidCacheLimits3D cache{};
};

// Encodes a completed Whitewater point checkpoint. IDs and signed cell keys
// must be unique and sorted; birth times cannot exceed the frame time. The
// emission cursor must equal the frame's accepted step. Invalid records return
// InvalidArgument and record/byte limits ResourceExhausted. No input mutation.
absl::StatusOr<LiquidCacheBytes3D>
EncodeWhitewaterFrame3D(const WhitewaterFrame3D& frame, WhitewaterCacheLimits3D limits = {});

// Validates the FLWW envelope and bounded records before returning owned state.
// Invalid payload data returns DataLoss, and exceeded limits ResourceExhausted.
absl::StatusOr<WhitewaterFrame3D>
DecodeWhitewaterFrame3D(std::span<const std::uint8_t> bytes, WhitewaterCacheLimits3D limits = {});

// Encodes physical emission/transport controls. Work limits and the Simulate
// cell width and gravity are excluded; the latter are bound by source identity.
// Invalid physical settings return InvalidArgument, byte limits ResourceExhausted.
absl::StatusOr<LiquidCacheBytes3D>
EncodeWhitewaterSettings3D(const WhitewaterFrameOptions3D& options, LiquidCacheLimits3D limits = {});

// Restores canonical physical settings with default work limits. Malformed
// data returns DataLoss, unknown algorithm version FailedPrecondition, and
// excessive byte counts ResourceExhausted. Retains no input aliases.
absl::StatusOr<WhitewaterFrameOptions3D>
DecodeWhitewaterSettings3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
}  // namespace fluidity
