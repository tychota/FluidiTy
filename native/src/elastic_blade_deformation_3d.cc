#include "fluidity/host/elastic_blade_deformation_3d.h"

#include <array>
#include <cmath>
#include <limits>

namespace fluidity {
namespace {

// The row of L multiplies the old deformation gradient on its left.
Vec3 AdvanceDeformationRow(Vec3 old_row, Vec3 gradient_row, Mat3 old_deformation, double dt) {
  const Vec3 rate{
      gradient_row.x * old_deformation.x_row.x + gradient_row.y * old_deformation.y_row.x
          + gradient_row.z * old_deformation.z_row.x,
      gradient_row.x * old_deformation.x_row.y + gradient_row.y * old_deformation.y_row.y
          + gradient_row.z * old_deformation.z_row.y,
      gradient_row.x * old_deformation.x_row.z + gradient_row.y * old_deformation.y_row.z
          + gradient_row.z * old_deformation.z_row.z,
  };
  return {std::fma(dt, rate.x, old_row.x), std::fma(dt, rate.y, old_row.y), std::fma(dt, rate.z, old_row.z)};
}

}  // namespace

static absl::StatusOr<ElasticBladeDeformation3D> GatherElasticBladeDeformation(
    const SparseMacGrid3D& mac_grid,
    const MacTransferParticle3D& particle,
    ElasticBladeSide3D blade_side,
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history,
    Mat3 old_deformation_gradient,
    std::vector<double>* completed_new_face_velocities
) {
  const bool assigned_sign = blade_side.sign == -1 || blade_side.sign == 1;
  if (blade_side.blade_id == 0 || !assigned_sign) {
    return absl::InvalidArgumentError("elastic deformation requires an assigned blade side");
  }
  const auto old_domain = NeoHookeanKirchhoffStress3D(old_deformation_gradient, 1.0, 0.0);
  if (!old_domain.ok()) {
    return old_domain.status();
  }
  const double dt = new_history.velocity_sample_time - old_history.velocity_sample_time;
  if (!std::isfinite(dt) || dt <= 0.0) {
    return absl::InvalidArgumentError("invalid elastic deformation time interval");
  }
  const LiquidBoundaryParticle3D boundary_particle{
      .transfer = particle,
      .side_tag = blade_side.sign,
      .affine_enabled = true,
  };
  const LiquidBoundaryGatherInputs3D inputs{
      .mac_grid = mac_grid,
      .particles = {&boundary_particle, 1},
      .old_time = old_history.velocity_sample_time,
      .new_time = new_history.velocity_sample_time,
      .old_history = old_history,
      .new_history = new_history,
  };
  const auto gathered = GatherElasticBoundaryParticle3D(inputs, completed_new_face_velocities);
  if (!gathered.ok()) {
    return gathered.status();
  }
  const auto& gather = gathered->particles.front();
  if (!gather.affine_enabled) {
    return absl::FailedPreconditionError("elastic deformation requires a full-rank velocity continuation");
  }

  const Mat3 gradient = gather.velocity_gradient;
  const Mat3 candidate{
      .x_row = AdvanceDeformationRow(old_deformation_gradient.x_row, gradient.x_row, old_deformation_gradient, dt),
      .y_row = AdvanceDeformationRow(old_deformation_gradient.y_row, gradient.y_row, old_deformation_gradient, dt),
      .z_row = AdvanceDeformationRow(old_deformation_gradient.z_row, gradient.z_row, old_deformation_gradient, dt),
  };
  if (!std::isfinite(candidate.x_row.x) || !std::isfinite(candidate.x_row.y) || !std::isfinite(candidate.x_row.z)
      || !std::isfinite(candidate.y_row.x) || !std::isfinite(candidate.y_row.y) || !std::isfinite(candidate.y_row.z)
      || !std::isfinite(candidate.z_row.x) || !std::isfinite(candidate.z_row.y) || !std::isfinite(candidate.z_row.z)) {
    return absl::OutOfRangeError("nonfinite elastic deformation");
  }
  const auto candidate_domain = NeoHookeanKirchhoffStress3D(candidate, 1.0, 0.0);
  if (absl::IsInvalidArgument(candidate_domain.status())) {
    return absl::FailedPreconditionError("elastic deformation loses positive volume");
  }
  if (!candidate_domain.ok()) {
    return candidate_domain.status();
  }
  return ElasticBladeDeformation3D{.gather = gather, .deformation_gradient = candidate};
}

absl::StatusOr<ElasticBladeDeformation3D> GatherElasticBladeDeformation3D(
    const SparseMacGrid3D& mac_grid,
    const MacTransferParticle3D& particle,
    ElasticBladeSide3D blade_side,
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history,
    Mat3 old_deformation_gradient
) {
  return GatherElasticBladeDeformation(
      mac_grid,
      particle,
      blade_side,
      old_history,
      new_history,
      old_deformation_gradient,
      nullptr
  );
}

namespace {

double Dot(Vec3 first, Vec3 second) {
  return (first.x * second.x + first.y * second.y) + first.z * second.z;
}

struct ElasticStepWork {
  std::span<const Mat3> stresses;
  std::span<const std::array<CompatibleMacFaceStencil3D, 3>> stencils;
  ElasticBladeGridStepResult3D& result;
};

}  // namespace

static absl::StatusOr<std::vector<ElasticBladeContact3D>> AdvanceElasticBladeParticles(
    const SparseMacGrid3D& mac_grid,
    const TranslatingSlab3D& blade,
    std::uint64_t blade_id,
    double retained_velocity_fraction,
    double friction_coefficient,
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history,
    SolidInteractionCheckpoint3D& checkpoint,
    ElasticStepWork* work
) {
  const double dt = new_history.velocity_sample_time - old_history.velocity_sample_time;
  const bool valid_retention = std::isfinite(retained_velocity_fraction) && retained_velocity_fraction >= 0.0
      && retained_velocity_fraction <= 1.0;
  const bool valid_friction = std::isfinite(friction_coefficient) && friction_coefficient >= 0.0;
  if (!valid_retention || !valid_friction || blade_id == 0 || !std::isfinite(dt) || dt <= 0.0
      || checkpoint.elastic_particles.empty() || checkpoint.elastic_particles.size() != checkpoint.sides.size()) {
    return absl::InvalidArgumentError("invalid elastic blade step inputs");
  }
  if (checkpoint.accepted_step == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("elastic accepted-step ordinal exhausted");
  }
  auto candidates = checkpoint.elastic_particles;
  std::vector<ElasticBladeContact3D> reactions;
  reactions.reserve(candidates.size());
  std::vector<double> completed_new_face_velocities;
  double compatible_stress_work = 0.0;
  std::uint64_t previous_particle_id = 0;
  for (std::size_t particle_index = 0; particle_index < candidates.size(); ++particle_index) {
    auto& candidate = candidates[particle_index];
    const auto& side = checkpoint.sides[particle_index];
    if (candidate.particle_id <= previous_particle_id || candidate.particle_id != side.particle_id
        || side.side.blade_id != blade_id || candidate.body_id == 0 || !std::isfinite(candidate.reference_volume)
        || candidate.reference_volume <= 0.0) {
      return absl::InvalidArgumentError("invalid elastic blade step identity or reference volume");
    }
    previous_particle_id = candidate.particle_id;
    auto& transfer = candidate.transfer;
    const auto deformation = GatherElasticBladeDeformation(
        mac_grid,
        transfer,
        side.side,
        old_history,
        new_history,
        candidate.deformation_gradient,
        work == nullptr ? nullptr : &completed_new_face_velocities
    );
    if (!deformation.ok()) {
      return deformation.status();
    }
    if (work != nullptr) {
      const Mat3 stress = work->stresses[particle_index];
      const Mat3 gradient = deformation->gather.velocity_gradient;
      const double volume_time = dt * candidate.reference_volume;
      work->result.deformation_work += volume_time
          * (Dot(stress.x_row, gradient.x_row) + Dot(stress.y_row, gradient.y_row) + Dot(stress.z_row, gradient.z_row));
      const std::array stress_rows{stress.x_row, stress.y_row, stress.z_row};
      for (std::size_t axis = 0; axis < stress_rows.size(); ++axis) {
        for (const auto& sample : work->stencils[particle_index][axis]) {
          if (sample.compatible) {
            const double completed_velocity = completed_new_face_velocities[sample.face_slot];
            compatible_stress_work -=
                volume_time * Dot(stress_rows[axis], sample.normalized_weight_gradient) * completed_velocity;
          }
        }
      }
    }
    const Vec3 pic_velocity = deformation->gather.new_pic_velocity;
    const Vec3 residual = deformation->gather.particle_minus_old_pic_velocity;
    const Vec3 stored_velocity{
        std::fma(retained_velocity_fraction, residual.x, pic_velocity.x),
        std::fma(retained_velocity_fraction, residual.y, pic_velocity.y),
        std::fma(retained_velocity_fraction, residual.z, pic_velocity.z),
    };
    const auto contact = ResolveElasticBladeContact3D(
        {
            .start_world_position = transfer.trial_world_position,
            .stored_world_velocity = stored_velocity,
            .advecting_world_velocity = pic_velocity,
            .obstacle = blade,
            .step_duration = dt,
            .friction_coefficient = friction_coefficient,
        },
        transfer.particle_mass,
        blade_id,
        side.side
    );
    if (!contact.ok()) {
      return contact.status();
    }
    for (const auto axis : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
      const auto end_support =
          BuildOwnedMacFaceStencil3D(mac_grid, axis, transfer.trial_world_position, contact->contact.world_position);
      if (!end_support.ok()) {
        return end_support.status();
      }
    }
    transfer.sorted_world_position = transfer.trial_world_position;
    transfer.trial_world_position = contact->contact.world_position;
    transfer.particle_velocity = contact->contact.stored_world_velocity;
    transfer.particle_velocity_slopes = deformation->gather.fitted_velocity_slopes;
    candidate.deformation_gradient = deformation->deformation_gradient;
    reactions.push_back(*contact);
  }
  if (work != nullptr) {
    work->result.cut_work = work->result.deformation_work + compatible_stress_work;
    work->result.omitted_stress_work = compatible_stress_work - work->result.grid_stress_work;
    if (!std::isfinite(work->result.cut_work) || !std::isfinite(work->result.omitted_stress_work)) {
      return absl::OutOfRangeError("nonfinite elastic work budget");
    }
  }
  checkpoint.elastic_particles.swap(candidates);
  ++checkpoint.accepted_step;
  return reactions;
}

absl::StatusOr<std::vector<ElasticBladeContact3D>> AdvanceElasticBladeParticles3D(
    const SparseMacGrid3D& mac_grid,
    const TranslatingSlab3D& blade,
    std::uint64_t blade_id,
    double retained_velocity_fraction,
    double friction_coefficient,
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history,
    SolidInteractionCheckpoint3D& checkpoint
) {
  return AdvanceElasticBladeParticles(
      mac_grid,
      blade,
      blade_id,
      retained_velocity_fraction,
      friction_coefficient,
      old_history,
      new_history,
      checkpoint,
      nullptr
  );
}

absl::StatusOr<ElasticBladeGridStepResult3D> StepElasticBladeGrid3D(
    const SparseMacGrid3D& mac_grid,
    const ElasticBladeGridStep3D& settings,
    SolidInteractionCheckpoint3D& checkpoint
) {
  const bool valid_acoustic_cfl =
      std::isfinite(settings.acoustic_cfl) && settings.acoustic_cfl > 0.0 && settings.acoustic_cfl <= 1.0;
  if (!valid_acoustic_cfl || !std::isfinite(settings.duration) || settings.duration <= 0.0
      || checkpoint.elastic_particles.empty() || checkpoint.elastic_particles.size() != checkpoint.sides.size()) {
    return absl::InvalidArgumentError("invalid elastic grid step duration or particle state");
  }
  if (checkpoint.elastic_particles.size() > kLiquidBoundaryParticleCapacity3D
      || mac_grid.face_count() > kLiquidBoundaryFaceCapacity3D) {
    return absl::ResourceExhaustedError("elastic blade grid capacity exceeded");
  }
  const auto old_tags = BuildElasticBladeFaceTags3D(mac_grid.view(), settings.blade, 0.0);
  if (!old_tags.ok()) {
    return old_tags.status();
  }
  const auto new_tags = BuildElasticBladeFaceTags3D(mac_grid.view(), settings.blade, settings.duration);
  if (!new_tags.ok()) {
    return new_tags.status();
  }
  std::vector<LiquidBoundaryParticle3D> particles;
  std::vector<Mat3> stresses;
  particles.reserve(checkpoint.elastic_particles.size());
  stresses.reserve(checkpoint.elastic_particles.size());
  for (std::size_t particle_index = 0; particle_index < checkpoint.elastic_particles.size(); ++particle_index) {
    const auto& particle = checkpoint.elastic_particles[particle_index];
    const auto& side = checkpoint.sides[particle_index];
    const bool valid_side = side.side.sign == -1 || side.side.sign == 1;
    if (particle.particle_id != side.particle_id || side.side.blade_id != settings.blade_id || !valid_side
        || !std::isfinite(particle.reference_volume) || particle.reference_volume <= 0.0) {
      return absl::InvalidArgumentError("invalid elastic particle affinity or reference volume");
    }
    const auto stress = NeoHookeanKirchhoffStress3D(
        particle.deformation_gradient,
        settings.shear_modulus,
        settings.first_lame_parameter
    );
    if (!stress.ok()) {
      return stress.status();
    }
    const double reference_density = particle.transfer.particle_mass / particle.reference_volume;
    const auto wave_speed_bound = NeoHookeanAcousticSpeedBound3D(
        particle.deformation_gradient,
        settings.shear_modulus,
        settings.first_lame_parameter,
        reference_density
    );
    if (!wave_speed_bound.ok()) {
      return wave_speed_bound.status();
    }
    const double acoustic_duration_limit = settings.acoustic_cfl * mac_grid.geometry().cell_width / *wave_speed_bound;
    if (settings.duration > acoustic_duration_limit) {
      return absl::FailedPreconditionError("elastic grid step exceeds the acoustic duration limit");
    }
    particles.push_back({.transfer = particle.transfer, .side_tag = side.side.sign, .affine_enabled = true});
    stresses.push_back(*stress);
  }
  const auto deposits = DepositCompatibleMacQuanta3D({
      .mac_grid = mac_grid,
      .particles = particles,
      .face_tags = *old_tags,
      .quantization = settings.quantization,
  });
  if (!deposits.ok()) {
    return deposits.status();
  }
  const auto old_grid =
      NormalizeLiquidMacFaces3D(deposits->face_quanta, settings.quantization, settings.minimum_face_mass);
  if (!old_grid.ok()) {
    return old_grid.status();
  }
  std::vector<double> stress_force_sums(mac_grid.face_count());
  for (std::size_t particle_index = 0; particle_index < particles.size(); ++particle_index) {
    const auto& stress = stresses[particle_index];
    const std::array stress_rows{stress.x_row, stress.y_row, stress.z_row};
    const double reference_volume = checkpoint.elastic_particles[particle_index].reference_volume;
    for (std::size_t axis = 0; axis < stress_rows.size(); ++axis) {
      const Vec3 stress_row = stress_rows[axis];
      for (const auto& sample : deposits->compatible_stencils[particle_index][axis]) {
        if (!sample.compatible) {
          continue;
        }
        const auto gradient = sample.normalized_weight_gradient;
        const double stress_force_contribution =
            -reference_volume * ((stress_row.x * gradient.x + stress_row.y * gradient.y) + stress_row.z * gradient.z);
        auto& stress_force_sum = stress_force_sums[sample.face_slot];
        stress_force_sum += stress_force_contribution;
        if (!std::isfinite(stress_force_sum)) {
          return absl::OutOfRangeError("nonfinite elastic face force");
        }
      }
    }
  }
  ElasticBladeGridStepResult3D result;
  auto new_velocities = old_grid->face_velocities;
  auto new_validity = old_grid->validity;
  for (std::size_t face_slot = 0; face_slot < mac_grid.face_count(); ++face_slot) {
    const auto old_tag = (*old_tags)[face_slot];
    const auto new_tag = (*new_tags)[face_slot];
    if (!new_tag.structurally_eligible || new_tag.side_tag != old_tag.side_tag) {
      new_validity[face_slot] = LiquidFaceValidity3D::kMissing;
      continue;
    }
    if (new_validity[face_slot] == LiquidFaceValidity3D::kValid) {
      const double acceleration = stress_force_sums[face_slot] / old_grid->face_masses[face_slot];
      new_velocities[face_slot] = std::fma(settings.duration, acceleration, new_velocities[face_slot]);
      result.grid_stress_work += settings.duration * stress_force_sums[face_slot] * new_velocities[face_slot];
      if (!std::isfinite(new_velocities[face_slot]) || !std::isfinite(result.grid_stress_work)) {
        return absl::OutOfRangeError("nonfinite elastic face velocity");
      }
    }
  }
  const std::vector<LiquidPrescribedWallSample3D> wall_samples(mac_grid.face_count());
  ElasticStepWork work{stresses, deposits->compatible_stencils, result};
  auto contacts = AdvanceElasticBladeParticles(
      mac_grid,
      settings.blade,
      settings.blade_id,
      settings.retained_velocity_fraction,
      settings.friction_coefficient,
      {0.0, 0.0, old_grid->face_velocities, old_grid->validity, *old_tags, wall_samples},
      {settings.duration, settings.duration, new_velocities, new_validity, *new_tags, wall_samples},
      checkpoint,
      &work
  );
  if (!contacts.ok()) {
    return contacts.status();
  }
  result.contacts = std::move(*contacts);
  return result;
}

}  // namespace fluidity
