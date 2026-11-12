#pragma once

#include "fluidity/host/plastic_solid_checkpoint_3d.h"
#include "liquid_cache_3d_internal.h"
#include "plastic_solid_particle_generated.h"

namespace fluidity::internal {

// Packs the fixed FLSI particle payload without changing its units or ordering.
// The owning codec validates the accepted records before writing the envelope.
inline solid::v1::PlasticParticle PackPlasticSolidCheckpointParticle3D(const PlasticSolidCheckpointParticle3D& record) {
  using liquid_cache_internal::Pack;
  const auto pack_matrix = [](Mat3 matrix) {
    return cache::v1::Matrix3(Pack(matrix.x_row), Pack(matrix.y_row), Pack(matrix.z_row));
  };
  return {
      record.particle_id,
      record.body_id,
      record.material_id,
      Pack(record.transfer.sorted_world_position),
      Pack(record.transfer.trial_world_position),
      record.transfer.particle_mass,
      Pack(record.transfer.particle_velocity),
      pack_matrix(record.transfer.particle_velocity_slopes),
      record.reference_volume,
      pack_matrix(record.elastic_deformation),
      record.log_plastic_jacobian,
  };
}

// Copies a verified wire struct. Material-domain validation happens after
// resolving the material ID against the matching scene.
inline PlasticSolidCheckpointParticle3D UnpackPlasticSolidCheckpointParticle3D(const solid::v1::PlasticParticle& wire) {
  using liquid_cache_internal::Unpack;
  const auto unpack_matrix = [](const cache::v1::Matrix3& matrix) {
    return Mat3{Unpack(&matrix.x_row()), Unpack(&matrix.y_row()), Unpack(&matrix.z_row())};
  };
  return {
      .particle_id = wire.particle_id(),
      .body_id = wire.body_id(),
      .material_id = wire.material_id(),
      .transfer =
          {
              .sorted_world_position = Unpack(&wire.sorted_position()),
              .trial_world_position = Unpack(&wire.position()),
              .particle_mass = wire.mass(),
              .particle_velocity = Unpack(&wire.velocity()),
              .particle_velocity_slopes = unpack_matrix(wire.velocity_slopes()),
          },
      .reference_volume = wire.reference_volume(),
      .elastic_deformation = unpack_matrix(wire.elastic_deformation()),
      .log_plastic_jacobian = wire.log_plastic_jacobian(),
  };
}

}  // namespace fluidity::internal
