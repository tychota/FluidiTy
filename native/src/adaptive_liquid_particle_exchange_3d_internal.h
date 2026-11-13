#pragma once

#include <span>

#include "absl/status/status.h"
#include "fluidity/host/adaptive_liquid_owner_exchange_3d.h"
#include "fluidity/host/graded_mac_transfer_3d.h"

namespace fluidity::liquid_step_internal {

// The caller has certified same-leaf kinematics and unchanged particle
// mass, V0 and J. Apply the trusted pre-motion mixed deltas to the actual
// post-kinematic particle inventories. Changes neither output on failure.
absl::Status ReconcileProjectedAdaptiveOwnerExchange3D(
    std::span<const LiquidAdaptiveCellState3D> step_start_cells,
    const AdaptiveLiquidOwnerExchangePlan3D& exchange,
    const GradedMacTransfer3D& transfer,
    std::span<const LiquidAdaptiveFaceState3D> projected_carriers,
    double reference_density_kg_per_m3,
    std::span<LiquidParticleState3D> particles,
    std::span<LiquidRemapInventory3D> inventories
);

}  // namespace fluidity::liquid_step_internal
