#include "fluidity/host/adaptive_liquid_particle_exchange_3d.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "adaptive_liquid_particle_exchange_3d_internal.h"
#include "fluidity/host/graded_mac_transfer_3d.h"
#include "liquid_exact_inventory_sum_internal.h"
#include "mac_pressure_3d_arithmetic.h"

namespace fluidity {
namespace {

using pressure_internal::arithmetic::CompensatedSum;
using pressure_internal::arithmetic::IsRepresentablePressureProduct;

enum ParticleInventoryLane : std::size_t {
  kMass,
  kReferenceVolume,
  kCurrentVolume,
  kMomentumX,
  kMomentumY,
  kMomentumZ,
  kParticleInventoryLaneCount,
};

using ParticleAmounts = std::array<double, kParticleInventoryLaneCount>;

ParticleAmounts InventoryAmounts(const LiquidRemapInventory3D& inventory) {
  return {
      inventory.mass_kg,
      inventory.reference_volume_m3,
      inventory.current_volume_m3,
      inventory.momentum_kg_m_per_s.x,
      inventory.momentum_kg_m_per_s.y,
      inventory.momentum_kg_m_per_s.z,
  };
}

absl::StatusOr<ParticleAmounts> StoredParticleAmounts(const LiquidParticleState3D& particle) {
  const double current_volume_m3 = particle.reference_volume * particle.volume_ratio;
  if (!IsRepresentablePressureProduct(particle.reference_volume, particle.volume_ratio, current_volume_m3)) {
    return absl::OutOfRangeError("particle exchange current volume is unrepresentable");
  }
  const std::array velocity{particle.world_velocity.x, particle.world_velocity.y, particle.world_velocity.z};
  std::array<double, 3> momentum{};
  for (std::size_t axis = 0; axis < velocity.size(); ++axis) {
    momentum[axis] = particle.mass * velocity[axis];
    if (!IsRepresentablePressureProduct(particle.mass, velocity[axis], momentum[axis])) {
      return absl::OutOfRangeError("particle exchange momentum is unrepresentable");
    }
  }
  return ParticleAmounts{
      particle.mass,
      particle.reference_volume,
      current_volume_m3,
      momentum[0],
      momentum[1],
      momentum[2]
  };
}

struct LeafParticles {
  ParticleAmounts totals{};
  ParticleAmounts before_carrier{};
  std::optional<std::size_t> carrier_ordinal;
};

absl::Status BinParticles(
    std::span<const LiquidAdaptiveCellState3D> cells,
    std::span<const LiquidParticleState3D> particles,
    const GradedMacTransfer3D& transfer,
    std::span<const std::uint8_t> affected,
    std::span<LeafParticles> binned
) {
  for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
    const auto& particle = particles[ordinal];
    const auto leaf = transfer.ParticleLeafOrdinal(particle.world_position);
    if (!leaf.ok()) {
      return leaf.status();
    }
    if (cells[*leaf].material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian) {
      return absl::FailedPreconditionError("particle exchange source overlaps Eulerian ownership");
    }
    if (affected[*leaf] == 0) {
      continue;
    }
    if (particle.phase_id != 0 || particle.constituent_count != 0 || particle.side_tag != 0) {
      return absl::FailedPreconditionError("particle exchange requires implicit liquid on the supported side");
    }
    const auto amounts = StoredParticleAmounts(particle);
    if (!amounts.ok()) {
      return amounts.status();
    }
    auto& entry = binned[*leaf];
    entry.before_carrier = entry.totals;
    entry.carrier_ordinal = ordinal;
    // Preserve the scheduler's native record order and binary64 addition.
    for (std::size_t lane = 0; lane < amounts->size(); ++lane) {
      entry.totals[lane] += (*amounts)[lane];
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<double> ApplyLeafDelta(double carrier_amount, double source_amount, double target_amount) {
  CompensatedSum candidate;
  candidate.Add(target_amount);
  candidate.Add(-source_amount);
  candidate.Add(carrier_amount);
  const double candidate_amount = candidate.Total();
  liquid_exchange_internal::ExactInventorySum<4> residual;
  if (!std::isfinite(candidate_amount) || !residual.Add(target_amount) || !residual.Add(-source_amount)
      || !residual.Add(carrier_amount) || !residual.Add(-candidate_amount) || residual.sign() != 0) {
    return absl::OutOfRangeError("particle exchange carrier cannot represent the exact inventory correction");
  }
  return candidate_amount;
}

absl::StatusOr<AdaptiveLiquidParticlePrescribedState3D> ResolveConstitutiveState(
    const AdaptiveLiquidParticleConstitutiveState3D& constitutive
) {
  if (const auto* eos = std::get_if<AdaptiveLiquidParticleEosState3D>(&constitutive)) {
    const auto state = EvaluateLiquidEosAtPressure3D(eos->eos, eos->gauge_pressure_pa);
    if (!state.ok()) {
      return state.status();
    }
    return AdaptiveLiquidParticlePrescribedState3D{eos->eos.reference_density, state->volume_ratio};
  }
  const auto& prescribed = std::get<AdaptiveLiquidParticlePrescribedState3D>(constitutive);
  if (!std::isfinite(prescribed.reference_density_kg_per_m3) || prescribed.reference_density_kg_per_m3 <= 0.0
      || !std::isfinite(prescribed.volume_ratio) || prescribed.volume_ratio <= 0.0) {
    return absl::InvalidArgumentError("particle exchange prescribed constitutive state is invalid");
  }
  return prescribed;
}

absl::StatusOr<LiquidParticleState3D> ReconcileCarrier(
    const LiquidParticleState3D& source_particle,
    const LeafParticles& binned,
    const ParticleAmounts& source_inventory,
    const ParticleAmounts& target_inventory,
    const AdaptiveLiquidParticleConstitutiveState3D& constitutive
) {
  const auto source_amounts = StoredParticleAmounts(source_particle);
  if (!source_amounts.ok()) {
    return source_amounts.status();
  }
  ParticleAmounts desired{};
  for (std::size_t lane = 0; lane < desired.size(); ++lane) {
    const auto corrected = ApplyLeafDelta((*source_amounts)[lane], source_inventory[lane], target_inventory[lane]);
    if (!corrected.ok()) {
      return corrected.status();
    }
    desired[lane] = *corrected;
    if (binned.before_carrier[lane] + desired[lane] != target_inventory[lane]) {
      return absl::OutOfRangeError("particle exchange leaf does not match native accumulation order");
    }
  }
  if (desired[kMass] <= 0.0 || desired[kReferenceVolume] <= 0.0 || desired[kCurrentVolume] <= 0.0) {
    return absl::FailedPreconditionError("particle exchange requires a positive surviving resident");
  }

  const auto state = ResolveConstitutiveState(constitutive);
  if (!state.ok()) {
    return state.status();
  }
  const double eos_mass_kg = state->reference_density_kg_per_m3 * desired[kReferenceVolume];
  const double eos_volume_m3 = state->volume_ratio * desired[kReferenceVolume];
  if (!IsRepresentablePressureProduct(state->reference_density_kg_per_m3, desired[kReferenceVolume], eos_mass_kg)
      || !IsRepresentablePressureProduct(state->volume_ratio, desired[kReferenceVolume], eos_volume_m3)) {
    return absl::OutOfRangeError("particle exchange constitutive products are unrepresentable");
  }
  if (eos_mass_kg != desired[kMass] || eos_volume_m3 != desired[kCurrentVolume]) {
    return absl::FailedPreconditionError("particle exchange correction conflicts with the constitutive state");
  }

  auto candidate = source_particle;
  candidate.mass = desired[kMass];
  candidate.reference_volume = desired[kReferenceVolume];
  candidate.volume_ratio = state->volume_ratio;
  const std::array source_velocity{
      source_particle.world_velocity.x,
      source_particle.world_velocity.y,
      source_particle.world_velocity.z
  };
  std::array<double, 3> velocity{};
  for (std::size_t axis = 0; axis < velocity.size(); ++axis) {
    const auto lane = kMomentumX + axis;
    const bool unchanged_momentum = candidate.mass == source_particle.mass && desired[lane] == (*source_amounts)[lane];
    velocity[axis] = unchanged_momentum ? source_velocity[axis] : desired[lane] / candidate.mass;
  }
  candidate.world_velocity = {velocity[0], velocity[1], velocity[2]};
  const auto represented = StoredParticleAmounts(candidate);
  if (!represented.ok()) {
    return represented.status();
  }
  if (*represented != desired) {
    return absl::OutOfRangeError("particle exchange stored carrier does not match the exact correction");
  }
  return candidate;
}

bool HasParticleEnthalpy(
    std::span<const LiquidAdaptiveCellState3D> source_cells,
    const AdaptiveLiquidOwnerExchangePlan3D& plan
) {
  for (const auto& transfer : plan.transfers) {
    const auto donor = source_cells[transfer.donor_leaf_ordinal].material_owner;
    const auto receiver = source_cells[transfer.receiver_leaf_ordinal].material_owner;
    if (donor != receiver && transfer.transferred_inventory.enthalpy_j != 0.0) {
      return true;
    }
  }
  for (const auto leaf : plan.particle_reconciliation_leaf_ordinals) {
    if (source_cells[leaf].inventory.enthalpy_j != 0.0 || plan.candidate_cells[leaf].inventory.enthalpy_j != 0.0) {
      return true;
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<AdaptiveLiquidParticleExchangeCandidate3D> PrepareAdaptiveLiquidParticleExchange3D(
    const LiquidState3D& step_start,
    const AdaptiveLiquidParticleExchangeInputs3D& inputs
) {
  if (step_start.clock != inputs.projected_source_clock) {
    return absl::FailedPreconditionError("particle exchange projected samples have a stale source clock");
  }
  const auto source_status = ValidateLiquidState3D(step_start);
  if (!source_status.ok()) {
    return source_status;
  }
  const auto source_cells = step_start.adaptive_cells.values();
  if (source_cells.empty() || inputs.constitutive_states.size() != source_cells.size()) {
    return absl::InvalidArgumentError("particle exchange constitutive records must cover the adaptive leaves");
  }
  std::vector<AdaptiveMacCellKey3D> keys;
  keys.reserve(source_cells.size());
  for (const auto& cell : source_cells) {
    keys.push_back(cell.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, step_start.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  auto exchange = PrepareAdaptiveLiquidOwnerExchange3D(
      *topology,
      source_cells,
      inputs.projected_interior_faces,
      inputs.step_duration_s
  );
  if (!exchange.ok()) {
    return exchange.status();
  }
  if (HasParticleEnthalpy(source_cells, *exchange)) {
    return absl::FailedPreconditionError("particle exchange cannot represent signed enthalpy");
  }
  const auto& affected_leaves = exchange->particle_reconciliation_leaf_ordinals;
  if (!affected_leaves.empty() && inputs.certify_support == nullptr) {
    return absl::FailedPreconditionError("particle exchange requires a support certificate");
  }
  const auto transfer = GradedMacTransfer3D::Create(*topology, step_start.geometry.world_origin, inputs.solid_upper);
  if (!transfer.ok()) {
    return transfer.status();
  }
  std::vector<std::uint8_t> affected(source_cells.size());
  for (const auto leaf : affected_leaves) {
    affected[leaf] = 1;
  }
  std::vector<LeafParticles> binned(source_cells.size());
  const auto bin_status = BinParticles(source_cells, step_start.particles.values(), *transfer, affected, binned);
  if (!bin_status.ok()) {
    return bin_status;
  }

  const auto source_particles = step_start.particles.values();
  std::vector<LiquidParticleState3D> candidate_particles(source_particles.begin(), source_particles.end());
  for (const auto leaf : affected_leaves) {
    const auto& entry = binned[leaf];
    if (!entry.carrier_ordinal) {
      return absl::FailedPreconditionError("particle exchange has no resident carrier");
    }
    const auto source_inventory = InventoryAmounts(source_cells[leaf].inventory);
    if (entry.totals != source_inventory) {
      return absl::FailedPreconditionError("particle exchange source particles do not match their inventory");
    }
    const auto& source_particle = source_particles[*entry.carrier_ordinal];
    const auto target_inventory = InventoryAmounts(exchange->candidate_cells[leaf].inventory);
    const auto candidate =
        ReconcileCarrier(source_particle, entry, source_inventory, target_inventory, inputs.constitutive_states[leaf]);
    if (!candidate.ok()) {
      return candidate.status();
    }
    const auto support = inputs.certify_support(inputs.support_context, leaf, source_particle, *candidate);
    if (!support.ok()) {
      return support;
    }
    candidate_particles[*entry.carrier_ordinal] = *candidate;
  }
  return AdaptiveLiquidParticleExchangeCandidate3D{
      .source_clock = step_start.clock,
      .next_particle_id = step_start.next_particle_id,
      .exchange = std::move(*exchange),
      .candidate_particles = std::move(candidate_particles),
  };
}

absl::Status liquid_step_internal::ReconcileProjectedAdaptiveOwnerExchange3D(
    std::span<const LiquidAdaptiveCellState3D> step_start_cells,
    const AdaptiveLiquidOwnerExchangePlan3D& exchange,
    const GradedMacTransfer3D& transfer,
    std::span<const LiquidAdaptiveFaceState3D> projected_carriers,
    double reference_density_kg_per_m3,
    std::span<LiquidParticleState3D> particles,
    std::span<LiquidRemapInventory3D> inventories
) {
  if (HasParticleEnthalpy(step_start_cells, exchange)) {
    return absl::FailedPreconditionError("particle exchange cannot represent signed enthalpy");
  }
  std::vector<std::uint8_t> affected(step_start_cells.size());
  for (const auto leaf : exchange.particle_reconciliation_leaf_ordinals) {
    affected[leaf] = 1;
  }
  std::vector<LeafParticles> binned(step_start_cells.size());
  const auto bin_status = BinParticles(step_start_cells, particles, transfer, affected, binned);
  if (!bin_status.ok()) {
    return bin_status;
  }

  struct Revision {
    std::size_t leaf;
    std::size_t particle;
    LiquidParticleState3D state;
    LiquidRemapInventory3D inventory;
  };
  std::vector<Revision> revisions;
  revisions.reserve(exchange.particle_reconciliation_leaf_ordinals.size());
  for (const auto leaf : exchange.particle_reconciliation_leaf_ordinals) {
    const auto& entry = binned[leaf];
    if (!entry.carrier_ordinal) {
      return absl::FailedPreconditionError("projected owner exchange has no resident carrier");
    }
    const auto start = InventoryAmounts(step_start_cells[leaf].inventory);
    const auto planned = InventoryAmounts(exchange.candidate_cells[leaf].inventory);
    const auto baseline = InventoryAmounts(inventories[leaf]);
    if (entry.totals != baseline || start[kMass] != baseline[kMass]
        || start[kReferenceVolume] != baseline[kReferenceVolume] || start[kCurrentVolume] != baseline[kCurrentVolume]) {
      return absl::FailedPreconditionError("projected owner exchange kinematic baseline changed material ownership");
    }
    ParticleAmounts target{};
    for (std::size_t lane = 0; lane < target.size(); ++lane) {
      const auto amount = ApplyLeafDelta(baseline[lane], start[lane], planned[lane]);
      if (!amount.ok()) {
        return amount.status();
      }
      target[lane] = *amount;
    }

    const auto& source_particle = particles[*entry.carrier_ordinal];
    const auto candidate = ReconcileCarrier(
        source_particle,
        entry,
        baseline,
        target,
        AdaptiveLiquidParticlePrescribedState3D{reference_density_kg_per_m3, source_particle.volume_ratio}
    );
    if (!candidate.ok()) {
      return candidate.status();
    }
    const auto support = transfer.Gather(*candidate, projected_carriers);
    if (!support.ok()) {
      return support.status();
    }
    revisions.push_back({
        .leaf = leaf,
        .particle = *entry.carrier_ordinal,
        .state = *candidate,
        .inventory =
            {
                .mass_kg = target[kMass],
                .reference_volume_m3 = target[kReferenceVolume],
                .current_volume_m3 = target[kCurrentVolume],
                .momentum_kg_m_per_s = {target[kMomentumX], target[kMomentumY], target[kMomentumZ]},
            },
    });
  }
  for (const auto& revision : revisions) {
    particles[revision.particle] = revision.state;
    inventories[revision.leaf] = revision.inventory;
  }
  return absl::OkStatus();
}

}  // namespace fluidity
