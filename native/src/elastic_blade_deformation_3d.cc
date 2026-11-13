#include "fluidity/host/elastic_blade_deformation_3d.h"

#include <array>
#include <cmath>
#include <limits>

#include "liquid_boundary_transfer_3d_internal.h"

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
    std::span<const ElasticParticleBladeSide3D> sides,
    ElasticStepWork* work
) {
  const double dt = new_history.velocity_sample_time - old_history.velocity_sample_time;
  const bool valid_retention = std::isfinite(retained_velocity_fraction) && retained_velocity_fraction >= 0.0
      && retained_velocity_fraction <= 1.0;
  const bool valid_friction = std::isfinite(friction_coefficient) && friction_coefficient >= 0.0;
  if (!valid_retention || !valid_friction || blade_id == 0 || !std::isfinite(dt) || dt <= 0.0
      || checkpoint.elastic_particles.empty() || checkpoint.elastic_particles.size() != sides.size()) {
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
    const auto& side = sides[particle_index];
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
      checkpoint.sides,
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
      || checkpoint.elastic_particles.empty()
      || (!checkpoint.sides.empty() && checkpoint.elastic_particles.size() != checkpoint.sides.size())) {
    return absl::InvalidArgumentError("invalid elastic grid step duration or particle state");
  }
  if (checkpoint.elastic_particles.size() > kLiquidBoundaryParticleCapacity3D
      || mac_grid.face_count() > kLiquidBoundaryFaceCapacity3D) {
    return absl::ResourceExhaustedError("elastic blade grid capacity exceeded");
  }
  std::vector<ElasticParticleBladeSide3D> initial_sides;
  if (checkpoint.sides.empty()) {
    initial_sides.reserve(checkpoint.elastic_particles.size());
    for (const auto& particle : checkpoint.elastic_particles) {
      const auto side =
          ClassifyInitialElasticBladeSide3D(settings.blade, settings.blade_id, particle.transfer.trial_world_position);
      if (!side.ok()) {
        return side.status();
      }
      initial_sides.push_back({.particle_id = particle.particle_id, .side = *side});
    }
  }
  const auto& sides = checkpoint.sides.empty() ? initial_sides : checkpoint.sides;
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
    const auto& side = sides[particle_index];
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
    const auto status = internal::AccumulateCompatibleSolidStress3D(
        deposits->compatible_stencils[particle_index],
        checkpoint.elastic_particles[particle_index].reference_volume,
        stresses[particle_index],
        stress_force_sums
    );
    if (!status.ok()) {
      return status;
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
      sides,
      &work
  );
  if (!contacts.ok()) {
    return contacts.status();
  }
  if (!initial_sides.empty()) {
    checkpoint.sides.swap(initial_sides);
  }
  result.contacts = std::move(*contacts);
  return result;
}

absl::StatusOr<ResolvedElasticCutterScene3D> ResolveElasticCutterScene3D(const ElasticCutterSceneInputs3D& inputs) {
  if (inputs.particle_positions.empty() || inputs.body_id == 0 || inputs.blade_id == 0
      || inputs.body_id == inputs.blade_id) {
    return absl::InvalidArgumentError("elastic Cutter requires seeds and distinct nonzero body and blade IDs");
  }
  if (inputs.particle_positions.size() > kLiquidBoundaryParticleCapacity3D) {
    return absl::ResourceExhaustedError("elastic Cutter seed capacity exceeded");
  }
  const double mass = inputs.density * inputs.particle_reference_volume;
  if (!std::isfinite(mass) || mass <= 0.0 || inputs.density <= 0.0 || inputs.particle_reference_volume <= 0.0) {
    return absl::InvalidArgumentError("invalid elastic seed density or reference volume");
  }

  const double cell_width = inputs.geometry.cell_width;
  if (!std::isfinite(cell_width) || cell_width <= 0.0) {
    return absl::InvalidArgumentError("invalid elastic Cutter grid spacing");
  }
  const std::array lower{inputs.domain_low.x, inputs.domain_low.y, inputs.domain_low.z};
  const std::array upper{inputs.domain_high.x, inputs.domain_high.y, inputs.domain_high.z};
  const auto origin = inputs.geometry.world_origin;
  const std::array origin_coordinates{origin.x, origin.y, origin.z};
  std::array<std::int32_t, 3> first_cell{};
  std::array<std::int32_t, 3> last_cell{};
  for (std::size_t axis = 0; axis < lower.size(); ++axis) {
    if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || !std::isfinite(origin_coordinates[axis])
        || lower[axis] >= upper[axis]) {
      return absl::InvalidArgumentError("invalid elastic Cutter domain or grid origin");
    }
    const double first = std::floor((lower[axis] - origin_coordinates[axis]) / cell_width) - 3.0;
    const double last = std::ceil((upper[axis] - origin_coordinates[axis]) / cell_width) + 3.0;
    if (first < std::numeric_limits<std::int32_t>::min() || last > std::numeric_limits<std::int32_t>::max()) {
      return absl::OutOfRangeError("elastic Cutter domain exceeds the signed lattice");
    }
    first_cell[axis] = static_cast<std::int32_t>(first);
    last_cell[axis] = static_cast<std::int32_t>(last);
  }
  const auto first_block = MacOwnerBlock3D({first_cell[0], first_cell[1], first_cell[2]});
  const auto last_block = MacOwnerBlock3D({last_cell[0], last_cell[1], last_cell[2]});
  constexpr std::size_t kFacesPerBlock = 192;
  constexpr std::size_t kBlockCapacity = kLiquidBoundaryFaceCapacity3D / kFacesPerBlock;
  const std::array<std::int64_t, 3> block_extents{
      static_cast<std::int64_t>(last_block.x) - first_block.x + 1,
      static_cast<std::int64_t>(last_block.y) - first_block.y + 1,
      static_cast<std::int64_t>(last_block.z) - first_block.z + 1,
  };
  std::size_t block_count = 1;
  for (const auto extent : block_extents) {
    if (extent > static_cast<std::int64_t>(kBlockCapacity / block_count)) {
      return absl::ResourceExhaustedError("elastic Cutter owned grid capacity exceeded");
    }
    block_count *= static_cast<std::size_t>(extent);
  }
  std::vector<BlockIndex3D> blocks;
  blocks.reserve(block_count);
  for (auto z = first_block.z; z <= last_block.z; ++z) {
    for (auto y = first_block.y; y <= last_block.y; ++y) {
      for (auto x = first_block.x; x <= last_block.x; ++x) {
        blocks.push_back({x, y, z});
      }
    }
  }
  auto grid = SparseMacGrid3D::Create(inputs.geometry, blocks, kBlockCapacity);
  if (!grid.ok()) {
    return grid.status();
  }

  constexpr Mat3 kRestDeformation{
      .x_row = {1.0, 0.0, 0.0},
      .y_row = {0.0, 1.0, 0.0},
      .z_row = {0.0, 0.0, 1.0},
  };
  SolidInteractionCheckpoint3D checkpoint{};
  checkpoint.elastic_particles.reserve(inputs.particle_positions.size());
  checkpoint.sides.reserve(inputs.particle_positions.size());
  for (const Vec3 position : inputs.particle_positions) {
    if (!(position.x >= lower[0] && position.x <= upper[0] && position.y >= lower[1] && position.y <= upper[1]
          && position.z >= lower[2] && position.z <= upper[2])) {
      return absl::InvalidArgumentError("elastic Cutter seed is outside the authored domain");
    }
    const auto side = ClassifyInitialElasticBladeSide3D(inputs.blade, inputs.blade_id, position);
    if (!side.ok()) {
      return side.status();
    }
    const std::uint64_t particle_id = checkpoint.elastic_particles.size() + 1;
    checkpoint.sides.push_back({.particle_id = particle_id, .side = *side});
    checkpoint.elastic_particles.push_back({
        .particle_id = particle_id,
        .body_id = inputs.body_id,
        .transfer =
            {
                .sorted_world_position = position,
                .trial_world_position = position,
                .particle_mass = mass,
                .particle_velocity = {},
                .particle_velocity_slopes = {},
            },
        .reference_volume = inputs.particle_reference_volume,
        .deformation_gradient = kRestDeformation,
    });
  }
  return ResolvedElasticCutterScene3D{.grid = std::move(*grid), .checkpoint = std::move(checkpoint)};
}
absl::StatusOr<ElasticBladeFrameResult3D> AdvanceElasticBladeFrame3D(
    const SparseMacGrid3D& mac_grid,
    const ElasticBladeGridStep3D& frame_settings,
    std::uint32_t substeps_per_frame,
    SolidInteractionCheckpoint3D& checkpoint
) {
  if (substeps_per_frame == 0 || checkpoint.accepted_step % substeps_per_frame != 0) {
    return absl::InvalidArgumentError("elastic frame requires a positive substep count and a frame-aligned checkpoint");
  }
  const double substep_duration = frame_settings.duration / substeps_per_frame;
  if (!std::isfinite(substep_duration) || substep_duration <= 0.0) {
    return absl::InvalidArgumentError("invalid elastic frame duration");
  }
  constexpr std::uint64_t kLargestExactOrdinal = std::uint64_t{1} << std::numeric_limits<double>::digits;
  if (checkpoint.accepted_step > kLargestExactOrdinal - substeps_per_frame) {
    return absl::OutOfRangeError("elastic frame step ordinal cannot be represented exactly");
  }

  // Compute every pose from the authored time-zero slab and accepted ordinal
  // so replay never depends on a separately accumulated clock or pose.
  const double normal_speed = Dot(frame_settings.blade.unit_normal, frame_settings.blade.wall_velocity);
  auto step_settings = frame_settings;
  step_settings.duration = substep_duration;
  auto trial = checkpoint;
  ElasticBladeFrameResult3D result;
  for (std::uint32_t substep = 0; substep < substeps_per_frame; ++substep) {
    const double start_time = static_cast<double>(trial.accepted_step) * substep_duration;
    const double end_time = static_cast<double>(trial.accepted_step + 1) * substep_duration;
    if (!std::isfinite(end_time) || end_time <= start_time) {
      return absl::OutOfRangeError("elastic substep time cannot be resolved");
    }
    step_settings.blade.start_center_coordinate =
        std::fma(start_time, normal_speed, frame_settings.blade.start_center_coordinate);
    const auto step = StepElasticBladeGrid3D(mac_grid, step_settings, trial);
    if (!step.ok()) {
      return step.status();
    }

    for (const auto& contact : step->contacts) {
      result.particle_impulse.x += contact.particle_impulse.x;
      result.particle_impulse.y += contact.particle_impulse.y;
      result.particle_impulse.z += contact.particle_impulse.z;
      result.blade_work += contact.blade_work;
    }
    result.grid_stress_work += step->grid_stress_work;
    result.deformation_work += step->deformation_work;
    result.cut_work += step->cut_work;
    result.omitted_stress_work += step->omitted_stress_work;
  }
  if (!std::isfinite(result.particle_impulse.x) || !std::isfinite(result.particle_impulse.y)
      || !std::isfinite(result.particle_impulse.z) || !std::isfinite(result.blade_work)
      || !std::isfinite(result.grid_stress_work) || !std::isfinite(result.deformation_work)
      || !std::isfinite(result.cut_work) || !std::isfinite(result.omitted_stress_work)) {
    return absl::OutOfRangeError("elastic frame budget exceeds the finite range");
  }
  checkpoint = std::move(trial);
  return result;
}
}  // namespace fluidity
