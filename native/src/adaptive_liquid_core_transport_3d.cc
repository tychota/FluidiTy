#include "fluidity/host/adaptive_liquid_core_transport_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <new>
#include <utility>

#include "absl/status/status.h"
#include "fluidity/host/adaptive_liquid_owner_exchange_3d.h"
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

bool RepresentsImplicitLiquid(const LiquidAdaptiveCellState3D& cell) {
  if (cell.occupancy != LiquidCellOccupancy3D::kLiquid || cell.inventory.phase_id != 0) {
    return false;
  }
  const auto amounts = InventoryValues(cell.inventory);
  if (amounts[kCurrentVolume] > 0.0) {
    return amounts[kMass] > 0.0 && amounts[kReferenceVolume] > 0.0;
  }
  return std::ranges::all_of(amounts, [](double amount) {
    return amount == 0.0;
  });
}

LiquidRemapInventory3D TransferInventory(const InventoryLanes& amounts) {
  return {
      .mass_kg = amounts[kMass],
      .reference_volume_m3 = amounts[kReferenceVolume],
      .current_volume_m3 = amounts[kCurrentVolume],
      .momentum_kg_m_per_s = {amounts[kMomentumX], amounts[kMomentumY], amounts[kMomentumZ]},
      .enthalpy_j = amounts[kEnthalpy],
  };
}

struct TransportBalance {
  std::array<CompensatedSum, kInventoryLaneCount> remaining_inventory;
  std::array<CompensatedSum, kInventoryLaneCount> received_inventory;
  // Track volume relative to each bound before rounding the candidate inventory.
  CompensatedSum outgoing_volume_excess_m3;
  CompensatedSum capacity_excess_m3;
  InventoryLanes candidate_inventory{};
  bool has_transport = false;
};

enum class TransportDiagnostics {
  kCodeOnly,
  kExplain
};

struct TransportErrors {
  TransportDiagnostics diagnostics;

  absl::Status InvalidArgument(const char* message) const {
    return Make(absl::StatusCode::kInvalidArgument, message);
  }
  absl::Status FailedPrecondition(const char* message) const {
    return Make(absl::StatusCode::kFailedPrecondition, message);
  }
  absl::Status OutOfRange(const char* message) const {
    return Make(absl::StatusCode::kOutOfRange, message);
  }

 private:
  absl::Status Make(absl::StatusCode code, const char* message) const {
    return absl::Status(code, diagnostics == TransportDiagnostics::kExplain ? message : "");
  }
};

absl::Status CheckTransportInputs(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::size_t interior_face_count,
    double step_duration_s,
    TransportErrors errors
) {
  const auto leaves = topology.cells();
  const auto faces = topology.interior_faces();
  if (leaves.empty() || committed_cells.size() != leaves.size() || interior_face_count != faces.size()) {
    return errors.InvalidArgument("adaptive core transport fields must cover the topology");
  }
  if (!std::isfinite(step_duration_s) || step_duration_s <= 0.0) {
    return errors.InvalidArgument("adaptive core transport requires a finite positive step duration");
  }
  return absl::OkStatus();
}

// Accumulate in canonical subface order, then stage every changed inventory
// before the caller publishes any output records.
absl::Status StageTransport(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const double> interior_fluxes_m3_per_s,
    double step_duration_s,
    std::span<TransportBalance> balances,
    TransportErrors errors,
    std::vector<AdaptiveLiquidOwnerTransfer3D>* owner_transfers = nullptr
) {
  const bool allow_owner_exchange = owner_transfers != nullptr;
  const auto leaves = topology.cells();
  const auto faces = topology.interior_faces();
  for (std::size_t ordinal = 0; ordinal < leaves.size(); ++ordinal) {
    const auto& cell = committed_cells[ordinal];
    const auto& leaf = leaves[ordinal];
    if (pressure_internal::arithmetic::CellSortKey(cell.key.lower_corner)
            != pressure_internal::arithmetic::CellSortKey(leaf.key.lower_corner)
        || cell.key.width_exponent != leaf.key.width_exponent
        || cell.material_owner > AdaptiveLiquidMaterialOwner3D::kEulerian
        || cell.occupancy > LiquidCellOccupancy3D::kSolid) {
      return errors.InvalidArgument("adaptive core transport cell metadata must match the topology");
    }
    const auto amounts = InventoryValues(cell.inventory);
    const bool finite_amounts = std::ranges::all_of(amounts, [](double amount) {
      return std::isfinite(amount);
    });
    if (!finite_amounts || amounts[kMass] < 0.0 || amounts[kReferenceVolume] < 0.0 || amounts[kCurrentVolume] < 0.0
        || amounts[kCurrentVolume] > leaf.cell_volume
        || !liquid_remap_internal::ValidConstituentInventory(cell.inventory)) {
      return errors.InvalidArgument("adaptive core transport input inventory is invalid");
    }
    const bool is_core = IsEulerianLiquid(cell);
    if (!is_core && (!allow_owner_exchange || cell.occupancy != LiquidCellOccupancy3D::kLiquid)) {
      continue;
    }
    if (is_core && cell.inventory.phase_id != 0) {
      return errors.FailedPrecondition("adaptive core transport requires implicit single-phase inventory");
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
      return errors.InvalidArgument("adaptive core transport requires finite subface flux");
    }
    if (flux_m3_per_s == 0.0) {
      continue;
    }
    const auto& tail = committed_cells[face.tail_cell_ordinal];
    const auto& head = committed_cells[face.head_cell_ordinal];
    const bool tail_in_core = IsEulerianLiquid(tail)
        || (allow_owner_exchange && tail.material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian);
    const bool head_in_core = IsEulerianLiquid(head)
        || (allow_owner_exchange && head.material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian);
    if (!tail_in_core && !head_in_core) {
      continue;
    }
    if (!allow_owner_exchange && tail_in_core != head_in_core) {
      return errors.FailedPrecondition("adaptive core transport cannot exchange material with a non-core leaf");
    }
    if (allow_owner_exchange && (!RepresentsImplicitLiquid(tail) || !RepresentsImplicitLiquid(head))) {
      return errors.FailedPrecondition("owner exchange requires liquid endpoints with implicit material inventories");
    }
    const double flux_magnitude_m3_per_s = std::abs(flux_m3_per_s);
    const double swept_volume_m3 = flux_magnitude_m3_per_s * step_duration_s;
    if (!IsRepresentablePressureProduct(flux_magnitude_m3_per_s, step_duration_s, swept_volume_m3)) {
      return errors.OutOfRange("adaptive core transport swept volume is unrepresentable");
    }
    const bool flows_to_head = flux_m3_per_s > 0.0;
    const std::size_t donor_ordinal = flows_to_head ? face.tail_cell_ordinal : face.head_cell_ordinal;
    const std::size_t receiver_ordinal = flows_to_head ? face.head_cell_ordinal : face.tail_cell_ordinal;
    auto& donor = balances[donor_ordinal];
    auto& receiver = balances[receiver_ordinal];
    donor.outgoing_volume_excess_m3.Add(swept_volume_m3);
    const double outgoing_volume_excess_m3 = donor.outgoing_volume_excess_m3.Total();
    if (!std::isfinite(outgoing_volume_excess_m3)) {
      return errors.OutOfRange("adaptive core transport outgoing volume is unrepresentable");
    }
    if (outgoing_volume_excess_m3 > 0.0) {
      return errors.FailedPrecondition("adaptive core transport outflow exceeds step-start donor volume");
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
        return errors.OutOfRange("adaptive core transport donor fraction is unrepresentable");
      }
    }
    const bool drains_donor = outgoing_volume_excess_m3 == 0.0;
    InventoryLanes transferred_inventory{};
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
          return errors.OutOfRange("adaptive core transport donor remainder is unrepresentable");
        }
      } else if (lane != kCurrentVolume) {
        transferred_amount = swept_fraction * donor_amounts[lane];
        if (!IsRepresentablePressureProduct(swept_fraction, donor_amounts[lane], transferred_amount)) {
          return errors.OutOfRange("adaptive core transport inventory transfer is unrepresentable");
        }
      }
      if (drains_donor) {
        remaining_amount = {};
      } else {
        remaining_amount.Add(-transferred_amount);
      }
      receiver.received_inventory[lane].Add(transferred_amount);
      if (allow_owner_exchange) {
        transferred_inventory[lane] = transferred_amount;
      }
    }
    if (allow_owner_exchange) {
      owner_transfers->push_back({
          .interior_face_ordinal = ordinal,
          .donor_leaf_ordinal = donor_ordinal,
          .receiver_leaf_ordinal = receiver_ordinal,
          .signed_flux_m3_per_s = flux_m3_per_s,
          .transferred_inventory = TransferInventory(transferred_inventory),
      });
    }
    donor.capacity_excess_m3.Add(-swept_volume_m3);
    receiver.capacity_excess_m3.Add(swept_volume_m3);
    donor.has_transport = true;
    receiver.has_transport = true;
  }

  for (std::size_t ordinal = 0; ordinal < balances.size(); ++ordinal) {
    auto& balance = balances[ordinal];
    if (!balance.has_transport) {
      continue;
    }
    const double capacity_excess_m3 = balance.capacity_excess_m3.Total();
    if (!std::isfinite(capacity_excess_m3)) {
      return errors.OutOfRange("adaptive core transport capacity balance is unrepresentable");
    }
    if (capacity_excess_m3 > 0.0) {
      return errors.FailedPrecondition("adaptive core transport candidate volume exceeds leaf capacity");
    }
    if (balance.remaining_inventory[kMass].Total() < 0.0
        || balance.remaining_inventory[kReferenceVolume].Total() < 0.0) {
      return errors.OutOfRange("adaptive core transport rounded outflow exceeds donor inventory");
    }
    InventoryLanes amounts;
    for (std::size_t lane = 0; lane < amounts.size(); ++lane) {
      auto candidate_amount = balance.received_inventory[lane];
      candidate_amount.Add(balance.remaining_inventory[lane].sum);
      candidate_amount.Add(balance.remaining_inventory[lane].correction);
      amounts[lane] = candidate_amount.Total();
      if (!std::isfinite(amounts[lane])) {
        return errors.OutOfRange("adaptive core transport candidate inventory is unrepresentable");
      }
    }
    if (amounts[kMass] < 0.0 || amounts[kReferenceVolume] < 0.0 || amounts[kCurrentVolume] < 0.0
        || amounts[kCurrentVolume] > leaves[ordinal].cell_volume) {
      return errors.OutOfRange("adaptive core transport candidate inventory rounded outside its bounds");
    }
    balance.candidate_inventory = amounts;
  }
  return absl::OkStatus();
}

void WriteTransportInventories(
    std::span<const TransportBalance> balances,
    std::span<LiquidAdaptiveCellState3D> candidate
) {
  for (std::size_t ordinal = 0; ordinal < balances.size(); ++ordinal) {
    const auto& balance = balances[ordinal];
    if (!balance.has_transport) {
      continue;
    }
    const auto& amounts = balance.candidate_inventory;
    auto& inventory = candidate[ordinal].inventory;
    inventory.mass_kg = amounts[kMass];
    inventory.reference_volume_m3 = amounts[kReferenceVolume];
    inventory.current_volume_m3 = amounts[kCurrentVolume];
    inventory.momentum_kg_m_per_s = {amounts[kMomentumX], amounts[kMomentumY], amounts[kMomentumZ]};
    inventory.enthalpy_j = amounts[kEnthalpy];
  }
}

// At most four fine subfaces touch each of the six sides of a 2:1 leaf.
constexpr std::size_t kMaximumIncidentTransfers = 24;

struct IncidentTransfers {
  std::array<std::size_t, kMaximumIncidentTransfers> ordinals{};
  std::size_t count = 0;

  bool Add(std::size_t ordinal) {
    if (count == ordinals.size()) {
      return false;
    }
    ordinals[count++] = ordinal;
    return true;
  }

  std::span<const std::size_t> values() const {
    return std::span(ordinals).first(count);
  }
};

// Ordered FastTwoSum retains each residual in a nonoverlapping expansion.
// The largest nonzero partial determines the sign of the exact binary64 sum.
class ExchangeBalanceSum {
 public:
  bool Add(double term) {
    std::size_t retained = 0;
    for (std::size_t ordinal = 0; ordinal < count_; ++ordinal) {
      double partial = partials_[ordinal];
      if (std::abs(term) < std::abs(partial)) {
        std::swap(term, partial);
      }
      const double sum = term + partial;
      if (!std::isfinite(sum)) {
        return false;
      }
      const double remainder = partial - (sum - term);
      if (remainder != 0.0) {
        partials_[retained++] = remainder;
      }
      term = sum;
    }
    if (retained == partials_.size()) {
      return false;
    }
    partials_[retained] = term;
    count_ = retained + 1;
    return true;
  }

  int sign() const {
    for (std::size_t ordinal = count_; ordinal > 0; --ordinal) {
      const double partial = partials_[ordinal - 1];
      if (partial != 0.0) {
        return partial < 0.0 ? -1 : 1;
      }
    }
    return 0;
  }

 private:
  std::array<double, kMaximumIncidentTransfers + 2> partials_{};
  std::size_t count_ = 0;
};

absl::Status CheckExactExchangeBalances(
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    AdaptiveLiquidOwnerExchangePlan3D& plan
) {
  if (plan.transfers.empty()) {
    return absl::OkStatus();
  }
  std::vector<IncidentTransfers> incident(committed_cells.size());
  for (std::size_t ordinal = 0; ordinal < plan.transfers.size(); ++ordinal) {
    const auto& transfer = plan.transfers[ordinal];
    if (!incident[transfer.donor_leaf_ordinal].Add(ordinal) || !incident[transfer.receiver_leaf_ordinal].Add(ordinal)) {
      return absl::InvalidArgumentError("owner exchange exceeds the graded incidence bound");
    }
  }

  // Reconcile the actual represented debits and credits. An independently
  // rounded leaf cannot silently absorb a transfer while its neighbor pays it.
  for (std::size_t leaf = 0; leaf < committed_cells.size(); ++leaf) {
    const auto transfer_ordinals = incident[leaf].values();
    if (transfer_ordinals.empty()) {
      continue;
    }
    const auto start = InventoryValues(committed_cells[leaf].inventory);
    const auto candidate = InventoryValues(plan.candidate_cells[leaf].inventory);
    for (std::size_t lane = 0; lane < start.size(); ++lane) {
      ExchangeBalanceSum candidate_residual;
      ExchangeBalanceSum donor_remaining;
      if (!candidate_residual.Add(start[lane]) || !candidate_residual.Add(-candidate[lane])
          || !donor_remaining.Add(start[lane])) {
        return absl::OutOfRangeError("owner exchange balance is unrepresentable");
      }
      for (const auto ordinal : transfer_ordinals) {
        const auto& transfer = plan.transfers[ordinal];
        const auto amount = InventoryValues(transfer.transferred_inventory)[lane];
        const bool outgoing = transfer.donor_leaf_ordinal == leaf;
        if (!candidate_residual.Add(outgoing ? -amount : amount)
            || (outgoing && lane <= kCurrentVolume && !donor_remaining.Add(-amount))) {
          return absl::OutOfRangeError("owner exchange balance is unrepresentable");
        }
      }
      if (lane == kCurrentVolume && donor_remaining.sign() < 0) {
        return absl::FailedPreconditionError("owner exchange outflow exceeds step-start donor volume");
      }
      const bool donor_inventory_overdrawn = lane < kCurrentVolume && donor_remaining.sign() < 0;
      if (donor_inventory_overdrawn || candidate_residual.sign() != 0) {
        return absl::OutOfRangeError("owner exchange candidate cannot represent its exact inventory balance");
      }
    }
    if (committed_cells[leaf].material_owner == AdaptiveLiquidMaterialOwner3D::kParticles) {
      plan.particle_reconciliation_leaf_ordinals.push_back(leaf);
    }
  }
  return absl::OkStatus();
}

template <typename Left, typename Right>
bool SpansOverlap(std::span<Left> left, std::span<Right> right) {
  const std::less<const void*> before;
  return !left.empty() && !right.empty() && before(left.data(), right.data() + right.size())
      && before(right.data(), left.data() + left.size());
}

}  // namespace

struct AdaptiveLiquidCoreTransportWorkspace3D::Storage {
  LiquidRecords3D<TransportBalance> balances;
  LiquidRecords3D<LiquidAdaptiveCellState3D> output;
};

AdaptiveLiquidCoreTransportWorkspace3D::AdaptiveLiquidCoreTransportWorkspace3D(std::unique_ptr<Storage> storage
) noexcept
    : storage_(std::move(storage)) {
}

AdaptiveLiquidCoreTransportWorkspace3D::
    AdaptiveLiquidCoreTransportWorkspace3D(AdaptiveLiquidCoreTransportWorkspace3D&&) noexcept = default;

AdaptiveLiquidCoreTransportWorkspace3D&
AdaptiveLiquidCoreTransportWorkspace3D::operator=(AdaptiveLiquidCoreTransportWorkspace3D&&) noexcept = default;

AdaptiveLiquidCoreTransportWorkspace3D::~AdaptiveLiquidCoreTransportWorkspace3D() = default;

absl::StatusOr<AdaptiveLiquidCoreTransportWorkspace3D> AdaptiveLiquidCoreTransportWorkspace3D::Create(
    std::size_t cell_capacity
) {
  if (cell_capacity > kMaximumAdaptiveLiquidCells3D) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  auto balances = LiquidRecords3D<TransportBalance>::Create(cell_capacity);
  if (!balances.ok()) {
    return balances.status();
  }
  auto output = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(cell_capacity);
  if (!output.ok()) {
    return output.status();
  }
  const auto balance_extent = balances->Resize(cell_capacity);
  if (!balance_extent.ok()) {
    return balance_extent;
  }
  const auto output_extent = output->Resize(cell_capacity);
  if (!output_extent.ok()) {
    return output_extent;
  }
  std::unique_ptr<Storage> storage(new (std::nothrow) Storage{
      .balances = std::move(*balances),
      .output = std::move(*output),
  });
  if (!storage) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  return AdaptiveLiquidCoreTransportWorkspace3D(std::move(storage));
}

std::size_t AdaptiveLiquidCoreTransportWorkspace3D::reserved_bytes() const {
  if (!storage_) {
    return 0;
  }
  return sizeof(Storage) + storage_->balances.capacity() * sizeof(TransportBalance)
      + storage_->output.capacity() * sizeof(LiquidAdaptiveCellState3D);
}

absl::StatusOr<std::span<const LiquidAdaptiveCellState3D>> AdaptiveLiquidCoreTransportWorkspace3D::Prepare(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const double> interior_fluxes_m3_per_s,
    double step_duration_s
) {
  constexpr TransportErrors errors{.diagnostics = TransportDiagnostics::kCodeOnly};
  if (!storage_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  const auto input_status =
      CheckTransportInputs(topology, committed_cells, interior_fluxes_m3_per_s.size(), step_duration_s, errors);
  if (!input_status.ok()) {
    return input_status;
  }
  if (committed_cells.size() > storage_->balances.capacity()) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const auto output_storage = storage_->output.values();
  if (SpansOverlap(committed_cells, output_storage) || SpansOverlap(interior_fluxes_m3_per_s, output_storage)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto balances = storage_->balances.values().first(committed_cells.size());
  std::ranges::fill(balances, TransportBalance{});
  const auto transport_status =
      StageTransport(topology, committed_cells, interior_fluxes_m3_per_s, step_duration_s, balances, errors);
  if (!transport_status.ok()) {
    return transport_status;
  }
  const auto output = output_storage.first(committed_cells.size());
  std::ranges::copy(committed_cells, output.begin());
  WriteTransportInventories(balances, output);
  return std::span<const LiquidAdaptiveCellState3D>(output);
}

absl::StatusOr<std::vector<LiquidAdaptiveCellState3D>> PrepareAdaptiveLiquidCoreTransport3D(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const double> interior_fluxes_m3_per_s,
    double step_duration_s
) {
  constexpr TransportErrors errors{.diagnostics = TransportDiagnostics::kExplain};
  const auto input_status =
      CheckTransportInputs(topology, committed_cells, interior_fluxes_m3_per_s.size(), step_duration_s, errors);
  if (!input_status.ok()) {
    return input_status;
  }
  std::vector<TransportBalance> balances(committed_cells.size());
  const auto transport_status =
      StageTransport(topology, committed_cells, interior_fluxes_m3_per_s, step_duration_s, balances, errors);
  if (!transport_status.ok()) {
    return transport_status;
  }
  std::vector<LiquidAdaptiveCellState3D> candidate(committed_cells.begin(), committed_cells.end());
  WriteTransportInventories(balances, candidate);
  return candidate;
}

absl::StatusOr<AdaptiveLiquidOwnerExchangePlan3D> PrepareAdaptiveLiquidOwnerExchange3D(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const AdaptiveLiquidOwnerExchangeFace3D> projected_interior_faces,
    double step_duration_s
) {
  constexpr TransportErrors errors{.diagnostics = TransportDiagnostics::kExplain};
  const auto input_status =
      CheckTransportInputs(topology, committed_cells, projected_interior_faces.size(), step_duration_s, errors);
  if (!input_status.ok()) {
    return input_status;
  }

  const auto interior_faces = topology.interior_faces();
  std::vector<double> fluxes_m3_per_s(interior_faces.size());
  for (std::size_t ordinal = 0; ordinal < interior_faces.size(); ++ordinal) {
    const auto& face = interior_faces[ordinal];
    const auto& sample = projected_interior_faces[ordinal];
    const auto& carrier = sample.projected_carrier;
    if (!std::isfinite(sample.open_area_m2) || sample.open_area_m2 < 0.0 || sample.open_area_m2 > face.face_area
        || !std::isfinite(carrier.carrier_velocity) || carrier.validity > LiquidFaceValidity3D::kLowMass) {
      return absl::InvalidArgumentError("owner exchange projected sample is invalid");
    }
    const bool touches_eulerian =
        committed_cells[face.tail_cell_ordinal].material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian
        || committed_cells[face.head_cell_ordinal].material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian;
    if (!touches_eulerian || sample.open_area_m2 == 0.0 || carrier.carrier_velocity == 0.0) {
      continue;
    }
    if (carrier.validity != LiquidFaceValidity3D::kValid || !carrier.structurally_eligible || carrier.side_tag != 0) {
      return absl::FailedPreconditionError("owner exchange requires a valid implicit-liquid projected carrier");
    }
    const double flux_m3_per_s = sample.open_area_m2 * carrier.carrier_velocity;
    if (!IsRepresentablePressureProduct(sample.open_area_m2, carrier.carrier_velocity, flux_m3_per_s)) {
      return absl::OutOfRangeError("owner exchange projected flux is unrepresentable");
    }
    fluxes_m3_per_s[ordinal] = flux_m3_per_s;
  }

  AdaptiveLiquidOwnerExchangePlan3D plan;
  std::vector<TransportBalance> balances(committed_cells.size());
  const auto transport_status =
      StageTransport(topology, committed_cells, fluxes_m3_per_s, step_duration_s, balances, errors, &plan.transfers);
  if (!transport_status.ok()) {
    return transport_status;
  }
  plan.candidate_cells.assign(committed_cells.begin(), committed_cells.end());
  WriteTransportInventories(balances, plan.candidate_cells);
  const auto exact_balance = CheckExactExchangeBalances(committed_cells, plan);
  if (!exact_balance.ok()) {
    return exact_balance;
  }
  return plan;
}

}  // namespace fluidity
