#pragma once

#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/adaptive_family_change_3d.h"
#include "fluidity/host/liquid_remap_3d.h"

namespace fluidity {

struct AdaptiveLiquidRemapInput3D {
  std::span<const LiquidRemapInventory3D> inventories;  // Committed cell order.
  std::span<const double> previous_pressures_pa;  // Committed cell order.
  std::span<const double> interior_fluxes_m3_per_s;  // Committed face order, positive along its normal.
};

struct AdaptiveLiquidRemapProposal3D {
  AdaptiveFamilyChange3D family;
  std::vector<LiquidRemapInventory3D> inventories;  // Candidate cell order.
  std::vector<double> interior_fluxes_m3_per_s;  // Candidate face order, positive along its normal.
};

// Propose one aligned refinement on a sealed geometric topology. Input spans
// cover the committed topology exactly. The caller provides same-width parent
// inventory samples in x/y/z low/high order. Preserves retained inventories
// and warm starts, remaps outer faces, and redistributes new sibling fluxes
// to preserve the parent's volume-scaled divergence within binary64 roundoff.
absl::StatusOr<AdaptiveLiquidRemapProposal3D> PrepareAdaptiveLiquidRefinement3D(
    const AdaptiveMacTopology3D& committed_topology,
    const AdaptiveLiquidRemapInput3D& committed,
    AdaptiveMacCellKey3D parent,
    const LiquidRemapStencil3D& parent_stencil
);

// Propose coarsening of the eight siblings. Sums their inventories and outer
// subface fluxes. Internal sibling fluxes cancel from the combined balance.
// Both operations return new state and never modify inputs. InvalidArgument
// denotes malformed input, FailedPrecondition missing or incompatible family
// coverage, and OutOfRange unrepresentable arithmetic. Topology errors propagate.
absl::StatusOr<AdaptiveLiquidRemapProposal3D> PrepareAdaptiveLiquidCoarsening3D(
    const AdaptiveMacTopology3D& committed_topology,
    const AdaptiveLiquidRemapInput3D& committed,
    AdaptiveMacCellKey3D parent
);

}  // namespace fluidity
