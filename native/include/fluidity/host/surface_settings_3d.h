#pragma once

#include "fluidity/host/surface_reconstruction_3d.h"

namespace fluidity {
// Encodes canonical physical Mesh settings, including the reconstruction
// algorithm version. Work/record limits are operational and excluded: increasing
// them can permit previously rejected work but cannot alter successful output.
// Invalid physical controls return InvalidArgument; insufficient byte capacity
// returns ResourceExhausted. Borrows options and publishes no bytes on failure.
absl::StatusOr<LiquidCacheBytes3D>
EncodeSurfaceSettings3D(const SurfaceReconstructionOptions3D& options, LiquidCacheLimits3D limits = {});

// Authenticates the bounded FLMC buffer and restores physical controls with
// default operational limits. Unsupported algorithm versions return
// FailedPrecondition, malformed data DataLoss, and byte limits ResourceExhausted.
// Input bytes are borrowed; no partial settings are returned on failure.
absl::StatusOr<SurfaceReconstructionOptions3D>
DecodeSurfaceSettings3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
}  // namespace fluidity
