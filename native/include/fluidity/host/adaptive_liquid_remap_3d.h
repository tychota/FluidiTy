#pragma once

#include <optional>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/adaptive_family_change_3d.h"
#include "fluidity/host/liquid_remap_3d.h"

namespace fluidity {

struct AdaptiveLiquidExteriorRemapInput3D {
  // All spans follow BuildAdaptiveMacFaceOwnership3D exterior order for
  // their respective topologies. Areas are in m². Positive flux is outward
  // from the owning cell on either side of the world-axis normal.
  std::span<const double> committed_open_areas_m2;
  std::span<const double> committed_outward_fluxes_m3_per_s;
  std::span<const double> candidate_open_areas_m2;
};

struct AdaptiveLiquidRemapInput3D {
  std::span<const LiquidRemapInventory3D> inventories;  // Committed cell order.
  std::span<const double> previous_pressures_pa;  // Committed cell order.
  std::span<const double> interior_fluxes_m3_per_s;  // Committed face order, positive along its normal.
  std::optional<AdaptiveLiquidExteriorRemapInput3D> exterior = std::nullopt;
};

struct AdaptiveLiquidRemapProposal3D {
  AdaptiveFamilyChange3D family;
  std::vector<LiquidRemapInventory3D> inventories;  // Candidate cell order.
  std::vector<double> interior_fluxes_m3_per_s;  // Candidate face order, positive along its normal.
  std::vector<double> exterior_outward_fluxes_m3_per_s;  // Candidate exterior order. Empty in sealed mode.
};

// Propose one aligned refinement. Input spans cover their respective
// topologies exactly. Omit exterior fields for a sealed geometric topology.
// Otherwise provide finite apertures in [0, geometric area] with zero flux on
// sealed faces. Aperture must match across unchanged faces and partitions.
// The caller provides same-width parent inventory samples in x/y/z low/high
// order. Retained inventories and warm starts are copied. Outer flux is
// conserved, with exterior splits weighted by aperture within binary64
// roundoff. New sibling fluxes preserve the parent's volume-scaled divergence
// within binary64 roundoff, including exterior outflow.
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
