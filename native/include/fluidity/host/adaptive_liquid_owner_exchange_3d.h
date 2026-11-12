#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {

// One projected sample in AdaptiveMacTopology3D::interior_faces() order.
// The carrier is positive in the Cartesian normal, from tail to head.
struct AdaptiveLiquidOwnerExchangeFace3D {
  LiquidAdaptiveFaceState3D projected_carrier;
  double open_area_m2;
};

// A single donor debit and matching receiver credit. Mass and volumes in
// transferred_inventory are nonnegative. Momentum and enthalpy retain their
// physical signs independent of the direction of the face flux.
struct AdaptiveLiquidOwnerTransfer3D {
  std::size_t interior_face_ordinal;
  std::size_t donor_leaf_ordinal;
  std::size_t receiver_leaf_ordinal;
  double signed_flux_m3_per_s;
  LiquidRemapInventory3D transferred_inventory;
};

struct AdaptiveLiquidOwnerExchangePlan3D {
  // The step-start roster with staged inventory updates and unchanged owners.
  std::vector<LiquidAdaptiveCellState3D> candidate_cells;
  // One entry per nonzero participating interior face, in canonical order.
  std::vector<AdaptiveLiquidOwnerTransfer3D> transfers;
  // Sorted canonical ordinals of particle-owned leaves requiring application.
  std::vector<std::size_t> particle_reconciliation_leaf_ordinals;
};

// Proposes transport across every nonzero interior face incident to Eulerian
// ownership, including core-to-core faces. Particle-to-particle and exterior
// transport are outside this plan. Supply canonical step-start cells, one
// projected sample per interior face, and a finite positive duration in
// seconds. Active carriers must be valid, structurally eligible, and on the
// implicit liquid side. Participating endpoints must be implicit liquid.
// Gross outgoing volume is limited by step-start donor volume and the final
// volume by geometric capacity. Incoming material cannot fund outgoing flux.
// Each candidate must exactly represent its stored binary64 transfer balance.
//
// The caller must reconcile actual particle records to the candidate mass,
// reference and current volumes, and momentum in every listed particle leaf.
// Signed enthalpy requires a separate ledger because particle records lack
// that lane. This call prepares an inventory proposal without applying it to
// particles or publishing state. All inputs remain unchanged on failure.
//
// InvalidArgument denotes malformed extents, keys, or samples.
// FailedPrecondition denotes unsupported ownership, inventory, carrier, or
// volume balance. OutOfRange denotes unrepresentable transport arithmetic.
absl::StatusOr<AdaptiveLiquidOwnerExchangePlan3D> PrepareAdaptiveLiquidOwnerExchange3D(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const AdaptiveLiquidOwnerExchangeFace3D> projected_interior_faces,
    double step_duration_s
);

}  // namespace fluidity
