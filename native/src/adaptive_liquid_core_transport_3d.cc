#include "fluidity/host/adaptive_liquid_core_transport_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "liquid_remap_3d_internal.h"

namespace fluidity {
namespace {

using pressure_internal::arithmetic::CompensatedSum;
using pressure_internal::arithmetic::IsRepresentablePressureProduct;

enum InventoryLane : std::size_t {
  kMass,
  kReferenceVolume,
  kCurrentVolume,
  kMomentumX,
  kMomentumY,
  kMomentumZ,
  kEnthalpy,
  kInventoryLaneCount,
};

using InventoryLanes = std::array<double, kInventoryLaneCount>;

InventoryLanes InventoryValues(const LiquidRemapInventory3D& inventory) {
  return {
      inventory.mass_kg,
      inventory.reference_volume_m3,
      inventory.current_volume_m3,
      inventory.momentum_kg_m_per_s.x,
      inventory.momentum_kg_m_per_s.y,
      inventory.momentum_kg_m_per_s.z,
      inventory.enthalpy_j,
  };
}

bool IsEulerianLiquid(const LiquidAdaptiveCellState3D& cell) {
  return cell.material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian
      && cell.occupancy == LiquidCellOccupancy3D::kLiquid;
}

struct TransportBalance {
  std::array<CompensatedSum, kInventoryLaneCount> remaining_inventory;
  std::array<CompensatedSum, kInventoryLaneCount> received_inventory;
  // Track volume relative to each bound before rounding the candidate inventory.
  CompensatedSum outgoing_volume_excess_m3;
  CompensatedSum capacity_excess_m3;
  bool has_transport = false;
};

}  // namespace

absl::StatusOr<std::vector<LiquidAdaptiveCellState3D>> PrepareAdaptiveLiquidCoreTransport3D(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const double> interior_fluxes_m3_per_s,
    double step_duration_s
) {
  const auto leaves = topology.cells();
  const auto faces = topology.interior_faces();
  if (leaves.empty() || committed_cells.size() != leaves.size() || interior_fluxes_m3_per_s.size() != faces.size()) {
    return absl::InvalidArgumentError("adaptive core transport fields must cover the topology");
  }
  if (!std::isfinite(step_duration_s) || step_duration_s <= 0.0) {
    return absl::InvalidArgumentError("adaptive core transport requires a finite positive step duration");
  }

  std::vector<TransportBalance> balances(leaves.size());
  for (std::size_t ordinal = 0; ordinal < leaves.size(); ++ordinal) {
    const auto& cell = committed_cells[ordinal];
    const auto& leaf = leaves[ordinal];
    if (pressure_internal::arithmetic::CellSortKey(cell.key.lower_corner)
            != pressure_internal::arithmetic::CellSortKey(leaf.key.lower_corner)
        || cell.key.width_exponent != leaf.key.width_exponent
        || cell.material_owner > AdaptiveLiquidMaterialOwner3D::kEulerian
        || cell.occupancy > LiquidCellOccupancy3D::kSolid) {
      return absl::InvalidArgumentError("adaptive core transport cell metadata must match the topology");
    }
    const auto amounts = InventoryValues(cell.inventory);
    const bool finite_amounts = std::ranges::all_of(amounts, [](double amount) {
      return std::isfinite(amount);
    });
    if (!finite_amounts || amounts[kMass] < 0.0 || amounts[kReferenceVolume] < 0.0 || amounts[kCurrentVolume] < 0.0
        || amounts[kCurrentVolume] > leaf.cell_volume
        || !liquid_remap_internal::ValidConstituentInventory(cell.inventory)) {
      return absl::InvalidArgumentError("adaptive core transport input inventory is invalid");
    }
    if (!IsEulerianLiquid(cell)) {
      continue;
    }
    if (cell.inventory.phase_id != 0) {
      return absl::FailedPreconditionError("adaptive core transport requires implicit single-phase inventory");
    }
    auto& balance = balances[ordinal];
    for (std::size_t lane = 0; lane < amounts.size(); ++lane) {
      balance.remaining_inventory[lane].Add(amounts[lane]);
    }
    balance.outgoing_volume_excess_m3.Add(-amounts[kCurrentVolume]);
    balance.capacity_excess_m3.Add(-leaf.cell_volume);
    balance.capacity_excess_m3.Add(amounts[kCurrentVolume]);
  }

  for (std::size_t ordinal = 0; ordinal < faces.size(); ++ordinal) {
    const auto& face = faces[ordinal];
    const double flux_m3_per_s = interior_fluxes_m3_per_s[ordinal];
    if (!std::isfinite(flux_m3_per_s)) {
      return absl::InvalidArgumentError("adaptive core transport requires finite subface flux");
    }
    if (flux_m3_per_s == 0.0) {
      continue;
    }
    const bool tail_in_core = IsEulerianLiquid(committed_cells[face.tail_cell_ordinal]);
    const bool head_in_core = IsEulerianLiquid(committed_cells[face.head_cell_ordinal]);
    if (!tail_in_core && !head_in_core) {
      continue;
    }
    if (tail_in_core != head_in_core) {
      return absl::FailedPreconditionError("adaptive core transport cannot exchange material with a non-core leaf");
    }
    const double flux_magnitude_m3_per_s = std::abs(flux_m3_per_s);
    const double swept_volume_m3 = flux_magnitude_m3_per_s * step_duration_s;
    if (!IsRepresentablePressureProduct(flux_magnitude_m3_per_s, step_duration_s, swept_volume_m3)) {
      return absl::OutOfRangeError("adaptive core transport swept volume is unrepresentable");
    }
    const bool flows_to_head = flux_m3_per_s > 0.0;
    const std::size_t donor_ordinal = flows_to_head ? face.tail_cell_ordinal : face.head_cell_ordinal;
    const std::size_t receiver_ordinal = flows_to_head ? face.head_cell_ordinal : face.tail_cell_ordinal;
    auto& donor = balances[donor_ordinal];
    auto& receiver = balances[receiver_ordinal];
    donor.outgoing_volume_excess_m3.Add(swept_volume_m3);
    const double outgoing_volume_excess_m3 = donor.outgoing_volume_excess_m3.Total();
    if (!std::isfinite(outgoing_volume_excess_m3)) {
      return absl::OutOfRangeError("adaptive core transport outgoing volume is unrepresentable");
    }
    if (outgoing_volume_excess_m3 > 0.0) {
      return absl::FailedPreconditionError("adaptive core transport outflow exceeds step-start donor volume");
    }

    // Every outgoing subface uses the same step-start donor inventory.
    // The current-volume lane uses the swept volume without a second product.
    const auto donor_amounts = InventoryValues(committed_cells[donor_ordinal].inventory);
    const double swept_fraction = swept_volume_m3 / donor_amounts[kCurrentVolume];
    if (swept_fraction < std::numeric_limits<double>::min()) {
      const double reconstructed_swept_volume_m3 = swept_fraction * donor_amounts[kCurrentVolume];
      const bool exact_swept_volume =
          IsRepresentablePressureProduct(swept_fraction, donor_amounts[kCurrentVolume], reconstructed_swept_volume_m3);
      if (!exact_swept_volume || reconstructed_swept_volume_m3 != swept_volume_m3) {
        return absl::OutOfRangeError("adaptive core transport donor fraction is unrepresentable");
      }
    }
    const bool drains_donor = outgoing_volume_excess_m3 == 0.0;
    for (std::size_t lane = 0; lane < donor_amounts.size(); ++lane) {
      auto& remaining_amount = donor.remaining_inventory[lane];
      double transferred_amount = swept_volume_m3;
      if (lane != kCurrentVolume && drains_donor) {
        // Send the rounded remainder on the last subface of an exact drain.
        // Independent products can strand inventory in a zero-volume donor.
        transferred_amount = remaining_amount.Total();
        const bool reversed_sign =
            transferred_amount != 0.0 && std::signbit(transferred_amount) != std::signbit(donor_amounts[lane]);
        if (!std::isfinite(transferred_amount) || reversed_sign) {
          return absl::OutOfRangeError("adaptive core transport donor remainder is unrepresentable");
        }
      } else if (lane != kCurrentVolume) {
        transferred_amount = swept_fraction * donor_amounts[lane];
        if (!IsRepresentablePressureProduct(swept_fraction, donor_amounts[lane], transferred_amount)) {
          return absl::OutOfRangeError("adaptive core transport inventory transfer is unrepresentable");
        }
      }
      if (drains_donor) {
        remaining_amount = {};
      } else {
        remaining_amount.Add(-transferred_amount);
      }
      receiver.received_inventory[lane].Add(transferred_amount);
    }
    donor.capacity_excess_m3.Add(-swept_volume_m3);
    receiver.capacity_excess_m3.Add(swept_volume_m3);
    donor.has_transport = true;
    receiver.has_transport = true;
  }

  std::vector<LiquidAdaptiveCellState3D> candidate(committed_cells.begin(), committed_cells.end());
  for (std::size_t ordinal = 0; ordinal < balances.size(); ++ordinal) {
    const auto& balance = balances[ordinal];
    if (!balance.has_transport) {
      continue;
    }
    const double capacity_excess_m3 = balance.capacity_excess_m3.Total();
    if (!std::isfinite(capacity_excess_m3)) {
      return absl::OutOfRangeError("adaptive core transport capacity balance is unrepresentable");
    }
    if (capacity_excess_m3 > 0.0) {
      return absl::FailedPreconditionError("adaptive core transport candidate volume exceeds leaf capacity");
    }
    if (balance.remaining_inventory[kMass].Total() < 0.0
        || balance.remaining_inventory[kReferenceVolume].Total() < 0.0) {
      return absl::OutOfRangeError("adaptive core transport rounded outflow exceeds donor inventory");
    }
    InventoryLanes amounts;
    for (std::size_t lane = 0; lane < amounts.size(); ++lane) {
      auto candidate_amount = balance.received_inventory[lane];
      candidate_amount.Add(balance.remaining_inventory[lane].sum);
      candidate_amount.Add(balance.remaining_inventory[lane].correction);
      amounts[lane] = candidate_amount.Total();
      if (!std::isfinite(amounts[lane])) {
        return absl::OutOfRangeError("adaptive core transport candidate inventory is unrepresentable");
      }
    }
    if (amounts[kMass] < 0.0 || amounts[kReferenceVolume] < 0.0 || amounts[kCurrentVolume] < 0.0
        || amounts[kCurrentVolume] > leaves[ordinal].cell_volume) {
      return absl::OutOfRangeError("adaptive core transport candidate inventory rounded outside its bounds");
    }
    auto& inventory = candidate[ordinal].inventory;
    inventory.mass_kg = amounts[kMass];
    inventory.reference_volume_m3 = amounts[kReferenceVolume];
    inventory.current_volume_m3 = amounts[kCurrentVolume];
    inventory.momentum_kg_m_per_s = {amounts[kMomentumX], amounts[kMomentumY], amounts[kMomentumZ]};
    inventory.enthalpy_j = amounts[kEnthalpy];
  }
  return candidate;
}

}  // namespace fluidity
