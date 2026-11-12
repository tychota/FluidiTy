#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/adaptive_liquid_owner_exchange_3d.h"
#include "fluidity/host/liquid_eos.h"

namespace fluidity {

struct AdaptiveLiquidParticlePrescribedState3D {
  double reference_density_kg_per_m3 = 0;
  double volume_ratio = 0;
};

struct AdaptiveLiquidParticleEosState3D {
  LiquidEos3D eos{};
  double gauge_pressure_pa = 0;
};

using AdaptiveLiquidParticleConstitutiveState3D =
    std::variant<AdaptiveLiquidParticlePrescribedState3D, AdaptiveLiquidParticleEosState3D>;

// Certifies that the unchanged position and revised particle are supported by
// the caller's material, side and transfer rules. The callback must not modify
// the source snapshot or publish externally visible changes.
using AdaptiveLiquidParticleExchangeSupport3D = absl::Status (*)(
    void* context,
    std::size_t leaf_ordinal,
    const LiquidParticleState3D& source,
    const LiquidParticleState3D& candidate
);

struct AdaptiveLiquidParticleExchangeInputs3D {
  LiquidClock3D projected_source_clock;
  std::span<const AdaptiveLiquidOwnerExchangeFace3D> projected_interior_faces;
  double step_duration_s = 0;
  // One state per canonical leaf. Only affected particle leaves are evaluated.
  std::span<const AdaptiveLiquidParticleConstitutiveState3D> constitutive_states;
  std::optional<Vec3> solid_upper;
  void* support_context = nullptr;
  AdaptiveLiquidParticleExchangeSupport3D certify_support = nullptr;
};

struct AdaptiveLiquidParticleExchangeCandidate3D {
  LiquidClock3D source_clock;
  std::uint64_t next_particle_id;
  AdaptiveLiquidOwnerExchangePlan3D exchange;
  std::vector<LiquidParticleState3D> candidate_particles;
};

// Prepare a material-only reconciliation from the actual step-start snapshot.
// The caller certifies that projected samples belong to the supplied source
// clock and that the particles have not applied motion or exchange from that
// trial. Regenerates the exchange plan from the supplied roster and samples.
// One supported resident, last in physical record order, carries each leaf's
// exact mass, reference-volume, current-volume and momentum correction.
// Preserves particle IDs, order, positions and metadata. The optional solid
// upper coordinates use the same X/Z ownership convention as graded transfer.
//
// Refuses empty and unsupported leaves, nonzero particle-side enthalpy,
// unsupported constitutive states, and ledgers that cannot be represented
// exactly by the stored particles in native accumulation order. A refusal
// changes neither the source snapshot nor any published inventory. The
// proposal does not perform advection or publish state.
absl::StatusOr<AdaptiveLiquidParticleExchangeCandidate3D> PrepareAdaptiveLiquidParticleExchange3D(
    const LiquidState3D& step_start,
    const AdaptiveLiquidParticleExchangeInputs3D& inputs
);

}  // namespace fluidity
