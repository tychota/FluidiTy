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

using PlasticCheckpointWireVector3D = flatbuffers::Vector<const solid::v1::PlasticParticle*>;

// Preserve FLSI v1 for an absent plastic payload. A nonempty field needs v2
// so a legacy decoder rejects the envelope instead of dropping the records.
inline constexpr std::uint32_t InteractionCheckpointVersionForPlasticPayload3D(std::size_t particle_count) {
  return particle_count == 0 ? 1 : 2;
}

// Validate a payload before adding its optional vector to the owning builder.
// An empty payload yields a null offset. Capacity or validation errors leave
// the builder unchanged. The owner limits the size of the complete envelope.
inline absl::StatusOr<flatbuffers::Offset<PlasticCheckpointWireVector3D>> EncodePlasticCheckpointWireVector3D(
    flatbuffers::FlatBufferBuilder& builder,
    std::span<const PlasticSolidCheckpointParticle3D> records,
    std::span<const PlasticSolidCheckpointBinding3D> bindings,
    std::size_t maximum_particle_count
) {
  if (records.size() > maximum_particle_count
      || records.size() >= FLATBUFFERS_MAX_BUFFER_SIZE / sizeof(solid::v1::PlasticParticle)) {
    return absl::ResourceExhaustedError("plastic checkpoint particle count exceeds capacity");
  }
  const auto status = ValidatePlasticSolidCheckpointParticles3D(records, bindings);
  if (!status.ok()) {
    return status;
  }
  if (records.empty()) {
    return flatbuffers::Offset<PlasticCheckpointWireVector3D>{};
  }
  std::vector<solid::v1::PlasticParticle> wire_records;
  wire_records.reserve(records.size());
  for (const auto& record : records) {
    wire_records.push_back(PackPlasticSolidCheckpointParticle3D(record));
  }
  return builder.CreateVectorOfStructs(wire_records.data(), wire_records.size());
}

// Decode a structurally verified optional vector from the owning FLSI envelope.
// Check the version and particle limit before allocating. Validate all resolved
// body bindings and material states before returning any decoded records.
inline absl::StatusOr<std::vector<PlasticSolidCheckpointParticle3D>> DecodePlasticCheckpointWireVector3D(
    std::uint32_t schema_version,
    const PlasticCheckpointWireVector3D* wire_records,
    std::span<const PlasticSolidCheckpointBinding3D> bindings,
    std::size_t maximum_particle_count
) {
  if (schema_version != 1 && schema_version != 2) {
    return absl::FailedPreconditionError("unsupported solid checkpoint version");
  }
  const std::size_t particle_count = wire_records == nullptr ? 0 : wire_records->size();
  if (schema_version < InteractionCheckpointVersionForPlasticPayload3D(particle_count)) {
    return absl::InvalidArgumentError("plastic checkpoint payload requires FLSI version 2");
  }
  if (particle_count > maximum_particle_count) {
    return absl::ResourceExhaustedError("plastic checkpoint particle count exceeds capacity");
  }
  std::vector<PlasticSolidCheckpointParticle3D> records;
  if (wire_records != nullptr) {
    records.reserve(particle_count);
    for (const auto* wire_record : *wire_records) {
      records.push_back(UnpackPlasticSolidCheckpointParticle3D(*wire_record));
    }
  }
  const auto status = ValidatePlasticSolidCheckpointParticles3D(records, bindings);
  if (!status.ok()) {
    return status;
  }
  return records;
}

}  // namespace fluidity::internal
