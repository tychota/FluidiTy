#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "fluidity/host/gas_substep_3d.h"
#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity {

struct GasCheckpoint3D {
  std::uint64_t frame;
  double time;  // Completed-frame time in seconds.
  std::uint64_t accepted_steps;
  GasFrame3D state;
};

// Encodes a completed gas frame in canonical FLGC v1 bytes bound to the
// resolved-scene digest. Returns InvalidArgument for invalid state and
// ResourceExhausted when the bounded format capacity is exceeded.
absl::StatusOr<std::vector<std::uint8_t>>
EncodeGasCheckpoint3D(const GasCheckpoint3D& checkpoint, const LiquidDigest3D& scene_digest);

// Restores dynamic state in the fixed domain's cell and face order.
// The caller supplies the authenticated scene and verifies the payload
// digest before decode. Truncation or malformed state returns InvalidArgument;
// a different scene or layout returns FailedPrecondition.
absl::StatusOr<GasCheckpoint3D> DecodeGasCheckpoint3D(
    std::span<const std::uint8_t> bytes,
    const GasUniformDomain3D& domain,
    const LiquidDigest3D& scene_digest
);

}  // namespace fluidity
