#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/dry_solid_step_3d.h"

namespace fluidity {

// Accepted plastic-particle payload for a coupled checkpoint. The resolved
// scene owns the material parameters keyed by material ID.
struct PlasticSolidCheckpointParticle3D {
  std::uint64_t particle_id;
  std::uint64_t body_id;
  std::uint64_t material_id;
  MacTransferParticle3D transfer;
  double reference_volume;  // Cubic meters.
  Mat3 elastic_deformation;  // Dimensionless accepted elastic map.
  double log_plastic_jacobian;  // Natural logarithm of the plastic volume ratio.
};

// Captures an admissible accepted dry particle with nonzero identities. The
// dry step rebins each attempt, so both transfer positions are the accepted
// position in world meters. Invalid records or material state return a status.
absl::StatusOr<PlasticSolidCheckpointParticle3D> CapturePlasticSolidCheckpointParticle3D(
    std::uint64_t particle_id,
    std::uint64_t body_id,
    std::uint64_t material_id,
    const DrySolidMaterial3D& material,
    const DrySolidParticle3D& particle
);

// Reconstructs a single dry body's particles in strictly increasing particle
// ID order. The input IDs must match the supplied nonzero body/material binding.
// Sorted positions must be finite. The next dry attempt bins from the accepted
// position. Invalid IDs or transfer data return InvalidArgument. Inadmissible
// material state returns FailedPrecondition. Failure publishes no batch.
absl::StatusOr<std::vector<DrySolidParticle3D>> RestoreDrySolidCheckpointParticles3D(
    std::span<const PlasticSolidCheckpointParticle3D> records,
    std::uint64_t body_id,
    std::uint64_t material_id,
    const DrySolidMaterial3D& material
);

}  // namespace fluidity
