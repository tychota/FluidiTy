#include "fluidity/host/plastic_solid_checkpoint_3d.h"

#include <algorithm>

#include "dry_solid_step_3d_internal.h"

namespace fluidity {
namespace {

absl::Status ValidateCheckpointParticle(const DrySolidParticle3D& particle, const DrySolidMaterial3D& material) {
  if (!solid_detail::Finite(particle.world_position) || !solid_detail::Finite(particle.world_velocity)
      || !solid_detail::Finite(particle.velocity_slopes.x_row) || !solid_detail::Finite(particle.velocity_slopes.y_row)
      || !solid_detail::Finite(particle.velocity_slopes.z_row)) {
    return absl::InvalidArgumentError("nonfinite plastic particle transfer state");
  }
  return internal::EvaluateAcceptedDrySolid3D(particle, material).status();
}

absl::StatusOr<DrySolidParticle3D>
RestoreCheckpointParticle(const PlasticSolidCheckpointParticle3D& record, const DrySolidMaterial3D& material) {
  if (!solid_detail::Finite(record.transfer.sorted_world_position)) {
    return absl::InvalidArgumentError("nonfinite plastic particle sort position");
  }
  const DrySolidParticle3D particle{
      .world_position = record.transfer.trial_world_position,
      .world_velocity = record.transfer.particle_velocity,
      .velocity_slopes = record.transfer.particle_velocity_slopes,
      .mass = record.transfer.particle_mass,
      .reference_volume = record.reference_volume,
      .elastic_deformation = record.elastic_deformation,
      .log_plastic_jacobian = record.log_plastic_jacobian,
  };
  const auto status = ValidateCheckpointParticle(particle, material);
  if (!status.ok()) {
    return status;
  }
  return particle;
}

}  // namespace

absl::Status ValidatePlasticSolidCheckpointParticles3D(
    std::span<const PlasticSolidCheckpointParticle3D> records,
    std::span<const PlasticSolidCheckpointBinding3D> bindings
) {
  std::uint64_t previous_body_id = 0;
  for (const auto& binding : bindings) {
    if (binding.body_id <= previous_body_id || binding.material_id == 0) {
      return absl::InvalidArgumentError("invalid plastic checkpoint body binding");
    }
    previous_body_id = binding.body_id;
  }

  std::uint64_t previous_particle_id = 0;
  for (const auto& record : records) {
    if (record.particle_id <= previous_particle_id) {
      return absl::InvalidArgumentError("plastic checkpoint particle IDs must increase");
    }
    const auto binding =
        std::ranges::lower_bound(bindings, record.body_id, {}, &PlasticSolidCheckpointBinding3D::body_id);
    if (binding == bindings.end() || binding->body_id != record.body_id || binding->material_id != record.material_id) {
      return absl::InvalidArgumentError("plastic checkpoint particle has no matching body and material binding");
    }
    const auto particle = RestoreCheckpointParticle(record, binding->material);
    if (!particle.ok()) {
      return particle.status();
    }
    previous_particle_id = record.particle_id;
  }
  return absl::OkStatus();
}

absl::StatusOr<PlasticSolidCheckpointParticle3D> CapturePlasticSolidCheckpointParticle3D(
    std::uint64_t particle_id,
    std::uint64_t body_id,
    std::uint64_t material_id,
    const DrySolidMaterial3D& material,
    const DrySolidParticle3D& particle
) {
  if (particle_id == 0 || body_id == 0 || material_id == 0) {
    return absl::InvalidArgumentError("zero plastic particle identity");
  }
  const auto status = ValidateCheckpointParticle(particle, material);
  if (!status.ok()) {
    return status;
  }
  return PlasticSolidCheckpointParticle3D{
      .particle_id = particle_id,
      .body_id = body_id,
      .material_id = material_id,
      .transfer =
          {
              .sorted_world_position = particle.world_position,
              .trial_world_position = particle.world_position,
              .particle_mass = particle.mass,
              .particle_velocity = particle.world_velocity,
              .particle_velocity_slopes = particle.velocity_slopes,
          },
      .reference_volume = particle.reference_volume,
      .elastic_deformation = particle.elastic_deformation,
      .log_plastic_jacobian = particle.log_plastic_jacobian,
  };
}

absl::StatusOr<std::vector<DrySolidParticle3D>> RestoreDrySolidCheckpointParticles3D(
    std::span<const PlasticSolidCheckpointParticle3D> records,
    std::uint64_t body_id,
    std::uint64_t material_id,
    const DrySolidMaterial3D& material
) {
  if (body_id == 0 || material_id == 0) {
    return absl::InvalidArgumentError("zero plastic body or material identity");
  }
  std::vector<DrySolidParticle3D> particles;
  particles.reserve(records.size());
  std::uint64_t previous_particle_id = 0;
  for (const auto& record : records) {
    if (record.particle_id <= previous_particle_id || record.body_id != body_id || record.material_id != material_id) {
      return absl::InvalidArgumentError("plastic particle identity or checkpoint order mismatch");
    }
    const auto particle = RestoreCheckpointParticle(record, material);
    if (!particle.ok()) {
      return particle.status();
    }
    particles.push_back(*particle);
    previous_particle_id = record.particle_id;
  }
  return particles;
}

}  // namespace fluidity
