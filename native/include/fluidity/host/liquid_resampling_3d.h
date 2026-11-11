#pragma once

#include <array>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {

enum class LiquidUniformResamplingBasis3D {
  kQuadratic,
  kCubicNormal,  // Quadratic transverse axes.
};

// Certifies complete, uniform, unmasked support and admissibility of this particle
// at the supplied spacing in meters. The caller checks geometry, phase/body/side,
// and material/EOS rules against its snapshot. The callback borrows the record
// for the call and must not mutate the input snapshot. A failed status refuses
// the candidate and is propagated unchanged.
using LiquidResamplingAdmissibility3D = absl::Status (*)(
    const LiquidParticleState3D& particle,
    double cell_width,
    LiquidUniformResamplingBasis3D basis,
    void* context
);

struct LiquidUniformSplit3D {
  // Octants in ascending ordinal, low side first, with x changing fastest.
  std::array<LiquidParticleState3D, 8> children;
  std::uint64_t next_particle_id;  // First unused ID after the eight new children.
  double kinetic_affine_energy_deficit;  // Analytical F3 loss in joules, rounded to binary64.
};

// Proposes a 2:1 symmetric split of an affine-enabled parent. Each child uses
// half the parent's cell width, one eighth of its mass and reference volume,
// the same affine rows and volume ratio, and velocity v + C * world_offset.
// The first child ID must be the next unused ID in the caller's snapshot and
// greater than the parent ID. Children inherit all other parent metadata.
//
// The required callback certifies the parent first, then the children in
// result order. The proposal neither modifies the input nor consumes any ID.
// InvalidArgument indicates malformed input, FailedPrecondition a disabled
// affine row set, and OutOfRange an unrepresentable inventory, ID, offset,
// velocity, or deficit. Callback errors publish no proposal.
absl::StatusOr<LiquidUniformSplit3D> SplitUniformLiquidParticle3D(
    const LiquidParticleState3D& parent,
    std::uint64_t first_child_id,
    double parent_cell_width,
    LiquidUniformResamplingBasis3D basis,
    LiquidResamplingAdmissibility3D certify,
    void* context
);

}  // namespace fluidity
