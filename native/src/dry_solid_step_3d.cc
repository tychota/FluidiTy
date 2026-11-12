#include "fluidity/host/dry_solid_step_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "dry_solid_step_3d_internal.h"
#include "fluidity/base/solid_acoustic_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
#include "fluidity/host/transfers_3d.h"
#include "fluidity/host/views.h"
#include "liquid_boundary_transfer_3d_internal.h"

namespace fluidity {
namespace {

constexpr double Vec3::* kComponents[]{&Vec3::x, &Vec3::y, &Vec3::z};
constexpr Vec3 Mat3::* kRows[]{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
constexpr FaceNormalAxis kAxes[]{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};

Mat3 TrialDeformation(Mat3 accepted, Mat3 velocity_gradient, double duration) {
  Mat3 trial = accepted;
  for (int axis = 0; axis < 3; ++axis) {
    const Vec3 gradient = velocity_gradient.*kRows[axis];
    Vec3& row = trial.*kRows[axis];
    for (const auto component : kComponents) {
      const Vec3 column{accepted.x_row.*component, accepted.y_row.*component, accepted.z_row.*component};
      row.*component += duration * solid_detail::Dot(gradient, column);
    }
  }
  return trial;
}

bool ReturnState(
    Mat3 trial,
    double previous_log_plastic_jacobian,
    const DrySolidMaterial3D& material,
    SolidDeformationState3D& state
) {
  if (const auto* granular = std::get_if<GranularParameters3D>(&material)) {
    return ReturnGranularDeformation3D(trial, previous_log_plastic_jacobian, *granular, state);
  }
  return ReturnSnowDeformation3D(trial, previous_log_plastic_jacobian, std::get<SnowParameters3D>(material), state);
}

constexpr double kReturnTolerance = 1e-12;

bool MatchesPlasticVolume(double log_plastic_jacobian, const SolidDeformationState3D& returned) {
  const double plastic_scale = std::max(1.0, std::abs(log_plastic_jacobian));
  return std::abs(log_plastic_jacobian - returned.log_plastic_jacobian) <= kReturnTolerance * plastic_scale;
}

bool MatchesDeformationState(
    Mat3 elastic_deformation,
    double log_plastic_jacobian,
    const SolidDeformationState3D& returned
) {
  double map_scale = 0;
  double maximum_difference = 0;
  for (const auto row : kRows) {
    for (const auto component : kComponents) {
      const double input = (elastic_deformation.*row).*component;
      const double output = (returned.elastic_deformation.*row).*component;
      map_scale = std::max(map_scale, std::abs(input));
      maximum_difference = std::max(maximum_difference, std::abs(input - output));
    }
  }
  return maximum_difference <= kReturnTolerance * map_scale && MatchesPlasticVolume(log_plastic_jacobian, returned);
}

bool AcousticBound(
    const SolidDeformationState3D& state,
    double reference_density,
    const DrySolidMaterial3D& material,
    double& speed_squared
) {
  if (const auto* granular = std::get_if<GranularParameters3D>(&material)) {
    return GranularAcousticBoundSquared3D(state.elastic_stretches, *granular, reference_density, speed_squared);
  }
  return SnowAcousticBoundSquared3D(
      state.elastic_stretches,
      state.log_plastic_jacobian,
      std::get<SnowParameters3D>(material),
      reference_density,
      speed_squared
  );
}

// The gap and motion bound ensure the path cannot meet both planes during a step.
absl::StatusOr<LiquidWallContactResult3D> SweepDryWalls(
    const DrySolidParticle3D& particle,
    Vec3 stored_velocity,
    Vec3 advecting_velocity,
    const DrySolidStep3D& settings
) {
  LiquidWallContactInputs3D input{
      .start_world_position = particle.world_position,
      .stored_world_velocity = stored_velocity,
      .advecting_world_velocity = advecting_velocity,
      .obstacle = TranslatingHalfspace3D{{0, 1, 0}, 0, {}},
      .step_duration = settings.duration,
      .friction_coefficient = settings.floor_friction,
  };
  auto floor = ResolveLiquidWallContact3D(input);
  if (!floor.ok() || !settings.platen.has_value()) {
    return floor;
  }
  input.obstacle = TranslatingHalfspace3D{{0, -1, 0}, -settings.platen->start_y, {0, settings.platen->velocity_y, 0}};
  input.friction_coefficient = 0;
  auto platen = ResolveLiquidWallContact3D(input);
  if (!platen.ok() || platen->contact_fraction.has_value()) {
    return platen;
  }
  return floor;
}

constexpr char kDrySolidDurationBound[] = "type.fluidity.dev/DrySolidDurationBound";

absl::Status DurationExceeded(const char* message) {
  auto status = absl::FailedPreconditionError(message);
  status.SetPayload(kDrySolidDurationBound, absl::Cord{});
  return status;
}

}  // namespace

bool internal::DrySolidDurationExceeded3D(const absl::Status& status) {
  return status.GetPayload(kDrySolidDurationBound).has_value();
}

Mat3 internal::ProjectStationaryFloorVelocityGradient3D(Vec3 velocity, Mat3 gradient, double friction_coefficient) {
  if (velocity.y >= 0) {
    return gradient;
  }
  const Vec3 incoming_normal_gradient = gradient.y_row;
  gradient.y_row = {};
  if (friction_coefficient == 0) {
    return gradient;
  }
  const double tangent_speed = std::hypot(velocity.x, velocity.z);
  if (tangent_speed == 0 || friction_coefficient >= tangent_speed / -velocity.y) {
    return {};
  }
  const double tangent_x = velocity.x / tangent_speed;
  const double tangent_z = velocity.z / tangent_speed;
  const double reduction_fraction = friction_coefficient * -velocity.y / tangent_speed;
  const Vec3 tangent_speed_gradient = solid_detail::Combine(gradient.x_row, tangent_x, gradient.z_row, tangent_z);
  const Vec3 radial_correction =
      solid_detail::Combine(incoming_normal_gradient, friction_coefficient, tangent_speed_gradient, reduction_fraction);
  gradient.x_row = solid_detail::Combine(gradient.x_row, 1 - reduction_fraction, radial_correction, tangent_x);
  gradient.z_row = solid_detail::Combine(gradient.z_row, 1 - reduction_fraction, radial_correction, tangent_z);
  return gradient;
}

absl::StatusOr<internal::AcceptedDrySolid3D>
internal::EvaluateAcceptedDrySolid3D(const DrySolidParticle3D& particle, const DrySolidMaterial3D& material) {
  if (!solid_detail::PositiveFinite(particle.mass) || !solid_detail::PositiveFinite(particle.reference_volume)) {
    return absl::InvalidArgumentError("invalid dry particle mass or volume");
  }
  AcceptedDrySolid3D accepted{};
  const double reference_density = particle.mass / particle.reference_volume;
  if (!ReturnState(particle.elastic_deformation, particle.log_plastic_jacobian, material, accepted.state)
      || !MatchesDeformationState(particle.elastic_deformation, particle.log_plastic_jacobian, accepted.state)
      || !AcousticBound(accepted.state, reference_density, material, accepted.wave_speed_squared)) {
    return absl::FailedPreconditionError("inadmissible dry material state");
  }
  return accepted;
}

absl::Status internal::AccumulateDrySolidStress3D(
    const SparseMacGrid3D& grid,
    Vec3 world_position,
    double reference_volume,
    Mat3 kirchhoff_stress,
    std::span<double> face_force
) {
  for (int axis = 0; axis < 3; ++axis) {
    const auto stencil = BuildOwnedMacFaceStencil3D(grid, kAxes[axis], world_position, world_position);
    if (!stencil.ok()) {
      return stencil.status();
    }
    const Vec3 stress_row = kirchhoff_stress.*kRows[axis];
    for (const auto& sample : *stencil) {
      const double force_contribution = -reference_volume * solid_detail::Dot(stress_row, sample.weight_gradient);
      face_force[sample.face_slot] += force_contribution;
    }
  }
  return absl::OkStatus();
}

absl::Status internal::AdvanceDrySolidWithTransfers3D(
    const SparseMacGrid3D& grid,
    const DrySolidMaterial3D& material,
    const DrySolidStep3D& settings,
    std::span<DrySolidParticle3D> particles,
    DrySolidDeposit3D deposit,
    DrySolidGather3D gather,
    DrySolidStepBudget3D* budget,
    const DrySolidFloorTransfers3D* floor_transfers
) {
  if (budget != nullptr) {
    const auto disjoint = CheckWritableViews(
        {std::as_bytes(particles), std::as_bytes(std::span(budget, 1))},
        {std::as_bytes(std::span(&grid, 1)),
         std::as_bytes(grid.active_block_indices()),
         std::as_bytes(std::span(&material, 1)),
         std::as_bytes(std::span(&settings, 1))}
    );
    if (!disjoint.ok()) {
      return disjoint;
    }
  }
  if (!solid_detail::PositiveFinite(settings.duration) || !std::isfinite(settings.gravity_y)
      || !solid_detail::NonnegativeFinite(settings.floor_friction)
      || !solid_detail::PositiveFinite(settings.acoustic_cfl) || settings.acoustic_cfl > 1) {
    return absl::InvalidArgumentError("invalid dry step settings");
  }
  if (!solid_detail::NonnegativeFinite(settings.particle_energy_blend) || settings.particle_energy_blend > 1) {
    return absl::InvalidArgumentError("invalid dry particle energy blend");
  }
  const bool compatible_floor = settings.compatible_floor;
  if (compatible_floor
      && (floor_transfers == nullptr || !std::holds_alternative<GranularParameters3D>(material)
          || settings.platen.has_value() || settings.particle_energy_blend != 0)) {
    return absl::InvalidArgumentError("compatible floor requires granular PIC storage without a platen");
  }
  const double cell_width = grid.geometry().cell_width;
  const double maximum_displacement = 0.45 * cell_width;
  if (settings.platen.has_value()) {
    const auto platen = *settings.platen;
    const double platen_displacement = settings.duration * platen.velocity_y;
    const double final_height = platen.start_y + platen_displacement;
    if (!std::isfinite(platen.start_y) || !std::isfinite(final_height) || platen.start_y <= cell_width
        || final_height <= cell_width || std::abs(platen_displacement) > maximum_displacement) {
      return absl::FailedPreconditionError("platen gap or motion outside dry step support");
    }
  }

  std::vector<MacTransferParticle3D> transfer_particles;
  transfer_particles.reserve(particles.size());
  std::vector<double> face_force(grid.face_count());
  std::vector<Mat3> particle_stresses;
  if (compatible_floor) {
    particle_stresses.reserve(particles.size());
  }
  double minimum_particle_mass = std::numeric_limits<double>::infinity();
  for (const auto& particle : particles) {
    const auto accepted = internal::EvaluateAcceptedDrySolid3D(particle, material);
    if (!accepted.ok()) {
      return accepted.status();
    }
    minimum_particle_mass = std::min(minimum_particle_mass, particle.mass);
    if (settings.duration * std::sqrt(accepted->wave_speed_squared) > settings.acoustic_cfl * cell_width) {
      return DurationExceeded("dry acoustic step bound");
    }
    if (compatible_floor) {
      particle_stresses.push_back(accepted->state.kirchhoff_stress);
    } else {
      const auto stress = internal::AccumulateDrySolidStress3D(
          grid,
          particle.world_position,
          particle.reference_volume,
          accepted->state.kirchhoff_stress,
          face_force
      );
      if (!stress.ok()) {
        return stress;
      }
    }
    transfer_particles.push_back({
        .sorted_world_position = particle.world_position,
        .trial_world_position = particle.world_position,
        .particle_mass = particle.mass,
        .particle_velocity = particle.world_velocity,
        .particle_velocity_slopes = particle.velocity_slopes,
    });
  }

  std::vector<MacFaceDeposit3D> deposits(grid.face_count());
  std::vector<LiquidBoundaryFaceTag3D> floor_tags;
  std::vector<LiquidBoundaryParticle3D> boundary_particles;
  LiquidBoundaryFaceDeposits3D boundary_deposits;
  double ghost_wall_force_y = 0;
  if (compatible_floor) {
    floor_tags.resize(grid.face_count());
    const auto layout = grid.view();
    for (std::size_t slot = 0; slot < grid.face_count(); ++slot) {
      const auto face = layout.FaceIndex(slot);
      if (!face.ok()) {
        return face.status();
      }
      const auto anchor = layout.FaceWorldAnchor(*face);
      if (!anchor.ok()) {
        return anchor.status();
      }
      floor_tags[slot].structurally_eligible = anchor->y >= 0;
    }
    boundary_particles.reserve(particles.size());
    for (const auto& particle : transfer_particles) {
      boundary_particles.push_back({.transfer = particle});
    }
    const auto quantization = floor_transfers->quantization;
    auto deposited = floor_transfers->deposit({grid, boundary_particles, floor_tags, quantization});
    if (!deposited.ok()) {
      return deposited.status();
    }
    boundary_deposits = std::move(*deposited);
    for (std::size_t slot = 0; slot < deposits.size(); ++slot) {
      const auto& quanta = boundary_deposits.face_quanta[slot];
      deposits[slot] = {
          .deposited_mass = static_cast<double>(quanta.mass_quanta_sum) * quantization.mass_per_quantum,
          .deposited_pure_momentum =
              static_cast<double>(quanta.pure_momentum_quanta_sum) * quantization.pure_momentum_per_quantum,
      };
    }
    std::vector<LiquidFaceValidity3D> validity(grid.face_count(), LiquidFaceValidity3D::kMissing);
    for (std::size_t slot = 0; slot < deposits.size(); ++slot) {
      if (deposits[slot].deposited_mass > 0) {
        validity[slot] = LiquidFaceValidity3D::kValid;
      }
    }
    for (std::size_t index = 0; index < particles.size(); ++index) {
      const auto status = AccumulateDrySolidStress3D(
          grid,
          particles[index].world_position,
          particles[index].reference_volume,
          particle_stresses[index],
          face_force
      );
      if (!status.ok()) {
        return status;
      }
    }
    const auto wall_force = ScatterStationaryFloorGhostTranspose3D(grid.view(), validity, floor_tags, face_force);
    if (!wall_force.ok()) {
      return wall_force.status();
    }
    ghost_wall_force_y = *wall_force;
  } else {
    const auto deposited = deposit(grid, transfer_particles, deposits);
    if (!deposited.ok()) {
      return deposited;
    }
  }
  std::vector<double> velocities(grid.face_count());
  const bool blend_stored_velocity = settings.particle_energy_blend > 0;
  std::vector<double> old_velocities(blend_stored_velocity ? grid.face_count() : 0);
  const double minimum_force_mass = 1e-4 * minimum_particle_mass;
  const auto layout = grid.view();
  DrySolidStepBudget3D trial_budget{};
  trial_budget.grid_wall_impulse_y = settings.duration * ghost_wall_force_y;
  for (std::size_t slot = 0; slot < grid.face_count(); ++slot) {
    const double mass = deposits[slot].deposited_mass;
    const double stress_impulse = budget == nullptr ? 0 : settings.duration * face_force[slot];
    trial_budget.proposed_stress_impulse_l1 += std::abs(stress_impulse);
    if (mass == 0) {
      trial_budget.omitted_stress_impulse_l1 += std::abs(stress_impulse);
      continue;
    }
    const auto face = layout.FaceIndex(slot);
    if (!face.ok()) {
      return face.status();
    }
    // Quadratic weight vanishes faster than its derivative at the support edge.
    // Regularize force acceleration there while retaining the normalized APIC deposit.
    const double force_mass = std::max(mass, minimum_force_mass);
    const double pre_force_velocity = deposits[slot].deposited_pure_momentum / mass;
    if (blend_stored_velocity) {
      old_velocities[slot] = pre_force_velocity;
    }
    double velocity = pre_force_velocity + settings.duration * (face_force[slot] / force_mass);
    if (face->axis == FaceNormalAxis::kY) {
      velocity += settings.duration * settings.gravity_y;
      const auto anchor = layout.FaceWorldAnchor(*face);
      if (!anchor.ok()) {
        return anchor.status();
      }
      // Sand's particle sweep must see its incoming normal speed for Coulomb friction.
      // The platen fixture also constrains the grid to transmit load across the snow bed.
      if (!std::isfinite(velocity)) {
        return absl::OutOfRangeError("nonfinite dry normal face velocity");
      }
      const double velocity_before_wall = velocity;
      if (anchor->y <= 0 && std::holds_alternative<SnowParameters3D>(material)) {
        velocity = std::max(0.0, velocity);
      }
      if (settings.platen.has_value() && anchor->y >= settings.platen->start_y) {
        velocity = std::min(settings.platen->velocity_y, velocity);
      }
      if (budget != nullptr) {
        trial_budget.grid_wall_impulse_y += mass * (velocity - velocity_before_wall);
      }
    }
    if (!std::isfinite(velocity)) {
      return absl::OutOfRangeError("nonfinite dry face velocity");
    }
    if (std::abs(velocity) * settings.duration > maximum_displacement) {
      return DurationExceeded("dry face motion bound");
    }
    if (budget != nullptr) {
      const double applied_impulse = stress_impulse * (mass / force_mass);
      const double omitted_impulse = stress_impulse - applied_impulse;
      trial_budget.omitted_stress_impulse_l1 += std::abs(omitted_impulse);
      trial_budget.applied_stress_work += applied_impulse * velocity;
      trial_budget.omitted_stress_work += omitted_impulse * velocity;
    }
    velocities[slot] = velocity;
  }

  std::vector<MacParticleGatherResult3D> gathers(particles.size());
  const auto& history = blend_stored_velocity ? old_velocities : velocities;
  if (compatible_floor) {
    std::vector<LiquidFaceValidity3D> validity(grid.face_count(), LiquidFaceValidity3D::kMissing);
    std::vector<LiquidPrescribedWallSample3D> walls(grid.face_count());
    for (std::size_t slot = 0; slot < deposits.size(); ++slot) {
      if (deposits[slot].deposited_mass > 0) {
        validity[slot] = LiquidFaceValidity3D::kValid;
      }
      if (!floor_tags[slot].structurally_eligible) {
        // The continuation may extrapolate tangential velocity inside the
        // stationary floor. It cannot extrapolate normal flow there.
        walls[slot] = {.available = true, .unit_normal = {0, 1, 0}};
      }
    }
    const auto gathered = floor_transfers->gather({
        grid,
        boundary_particles,
        0,
        settings.duration,
        {0, 0, history, validity, floor_tags, walls},
        {settings.duration, settings.duration, velocities, validity, floor_tags, walls},
    });
    if (!gathered.ok()) {
      return gathered.status();
    }
    trial_budget.cut_stress_work = trial_budget.applied_stress_work + trial_budget.omitted_stress_work;
    for (std::size_t index = 0; index < particles.size(); ++index) {
      auto& result = gathers[index];
      const auto& complete = gathered->particles[index];
      result.velocity_gradient = complete.velocity_gradient;
      result.fitted_velocity_slopes = complete.affine_enabled ? complete.fitted_velocity_slopes : Mat3{};
      // Compatible PIC is the transpose of the mass deposit. Use the complete
      // ghost field for the affine fit and the true deformation derivative.
      for (std::size_t axis = 0; axis < 3; ++axis) {
        for (const auto& sample : boundary_deposits.compatible_stencils[index][axis]) {
          if (sample.compatible) {
            result.new_pic_velocity.*kComponents[axis] += sample.normalized_weight * velocities[sample.face_slot];
          }
        }
        if (budget != nullptr) {
          trial_budget.cut_stress_work += settings.duration * particles[index].reference_volume
              * solid_detail::Dot(particle_stresses[index].*kRows[axis], complete.velocity_gradient.*kRows[axis]);
        }
      }
    }
  } else {
    const auto gathered = gather(grid, transfer_particles, history, velocities, gathers);
    if (!gathered.ok()) {
      return gathered;
    }
  }
  struct RetainedVelocity {
    Vec3 correction{};
    bool eligible = false;
  };
  std::vector<RetainedVelocity> retained_velocities;
  if (blend_stored_velocity) {
    retained_velocities.resize(particles.size());
    Vec3 correction_momentum{};
    double eligible_mass = 0;
    for (std::size_t index = 0; index < particles.size(); ++index) {
      const auto& particle = particles[index];
      bool usable_history = true;
      for (const auto axis : kAxes) {
        const auto support = BuildOwnedMacFaceStencil3D(grid, axis, particle.world_position, particle.world_position);
        if (!support.ok()) {
          return support.status();
        }
        for (const auto& sample : *support) {
          if (sample.weight > 0 && deposits[sample.face_slot].deposited_mass < minimum_force_mass) {
            usable_history = false;
          }
        }
      }
      if (!usable_history) {
        continue;
      }
      const auto& particle_gather = gathers[index];
      const Vec3 old_pic =
          solid_detail::Combine(particle_gather.new_pic_velocity, 1, particle_gather.grid_velocity_increment, -1);
      const Vec3 residual = solid_detail::Combine(particle.world_velocity, 1, old_pic, -1);
      const Vec3 correction = solid_detail::Combine({}, 0, residual, 0.99 * settings.particle_energy_blend);
      retained_velocities[index] = {.correction = correction, .eligible = true};
      eligible_mass += particle.mass;
      correction_momentum = solid_detail::Combine(correction_momentum, 1, correction, particle.mass);
    }
    if (eligible_mass > 0) {
      if (!std::isfinite(eligible_mass) || !solid_detail::Finite(correction_momentum)) {
        return absl::OutOfRangeError("nonfinite retained-velocity momentum");
      }
      const Vec3 mean_correction{
          correction_momentum.x / eligible_mass,
          correction_momentum.y / eligible_mass,
          correction_momentum.z / eligible_mass
      };
      for (auto& retained : retained_velocities) {
        if (retained.eligible) {
          retained.correction = solid_detail::Combine(retained.correction, 1, mean_correction, -1);
        }
      }
    }
  }
  std::vector<DrySolidParticle3D> trial_particles(particles.begin(), particles.end());
  for (std::size_t index = 0; index < trial_particles.size(); ++index) {
    auto& particle = trial_particles[index];
    const auto& particle_gather = gathers[index];
    const Vec3 velocity = particle_gather.new_pic_velocity;
    if (std::hypot(velocity.x, velocity.y, velocity.z) * settings.duration > maximum_displacement) {
      return DurationExceeded("dry particle motion bound");
    }
    Vec3 stored_velocity = velocity;
    if (!retained_velocities.empty() && retained_velocities[index].eligible) {
      stored_velocity = solid_detail::Combine(velocity, 1, retained_velocities[index].correction, 1);
    }
    const auto contact = SweepDryWalls(particle, stored_velocity, velocity, settings);
    if (!contact.ok()) {
      return contact.status();
    }
    const bool granular_floor_impact = std::holds_alternative<GranularParameters3D>(material)
        && !settings.platen.has_value() && contact->contact_fraction.has_value();
    const double incoming_duration =
        granular_floor_impact ? settings.duration * *contact->contact_fraction : settings.duration;
    Mat3 trial = TrialDeformation(particle.elastic_deformation, particle_gather.velocity_gradient, incoming_duration);
    if (granular_floor_impact) {
      // Compose the two deformation increments in physical order across impact.
      const double projected_duration = settings.duration - incoming_duration;
      const Mat3 projected_gradient = ProjectStationaryFloorVelocityGradient3D(
          velocity,
          particle_gather.velocity_gradient,
          settings.floor_friction
      );
      trial = TrialDeformation(trial, projected_gradient, projected_duration);
    }
    SolidDeformationState3D state{};
    double speed_squared = 0;
    if (!ReturnState(trial, particle.log_plastic_jacobian, material, state)
        || !AcousticBound(state, particle.mass / particle.reference_volume, material, speed_squared)) {
      return absl::FailedPreconditionError("inadmissible dry trial state");
    }
    const Vec3 displacement = solid_detail::Combine(contact->world_position, 1, particle.world_position, -1);
    if (std::hypot(displacement.x, displacement.y, displacement.z) > maximum_displacement
        || std::hypot(
               contact->advecting_world_velocity.x,
               contact->advecting_world_velocity.y,
               contact->advecting_world_velocity.z
           ) * settings.duration
            > maximum_displacement) {
      return DurationExceeded("dry post-contact motion bound");
    }
    for (const auto axis : kAxes) {
      const auto next_support =
          BuildOwnedMacFaceStencil3D(grid, axis, contact->world_position, contact->world_position);
      if (!next_support.ok()) {
        return next_support.status();
      }
    }
    if (budget != nullptr && contact->contact_fraction.has_value()) {
      const Vec3 wall_velocity_change = solid_detail::Combine(contact->stored_world_velocity, 1, stored_velocity, -1);
      trial_budget.wall_impulse_on_particles =
          solid_detail::Combine(trial_budget.wall_impulse_on_particles, 1, wall_velocity_change, particle.mass);
      const Vec3 midpoint_velocity = solid_detail::Combine(stored_velocity, 0.5, contact->stored_world_velocity, 0.5);
      trial_budget.particle_wall_kinetic_energy_change +=
          particle.mass * solid_detail::Dot(midpoint_velocity, wall_velocity_change);
      ++trial_budget.particle_wall_contacts;
    }
    if (budget != nullptr && std::holds_alternative<GranularParameters3D>(material)) {
      if (!MatchesDeformationState(trial, particle.log_plastic_jacobian, state)) {
        ++trial_budget.granular_projected_particles;
      }
      if (!MatchesPlasticVolume(particle.log_plastic_jacobian, state)) {
        ++trial_budget.granular_plastic_volume_changes;
      }
    }
    particle.world_position = contact->world_position;
    particle.world_velocity = contact->stored_world_velocity;
    particle.velocity_slopes = contact->contact_fraction.has_value() ? Mat3{} : particle_gather.fitted_velocity_slopes;
    particle.elastic_deformation = state.elastic_deformation;
    particle.log_plastic_jacobian = state.log_plastic_jacobian;
  }
  if (budget != nullptr
      && (!solid_detail::Finite(trial_budget.wall_impulse_on_particles)
          || !std::isfinite(trial_budget.grid_wall_impulse_y) || !std::isfinite(trial_budget.proposed_stress_impulse_l1)
          || !std::isfinite(trial_budget.omitted_stress_impulse_l1) || !std::isfinite(trial_budget.applied_stress_work)
          || !std::isfinite(trial_budget.omitted_stress_work) || !std::isfinite(trial_budget.cut_stress_work)
          || !std::isfinite(trial_budget.particle_wall_kinetic_energy_change))) {
    return absl::OutOfRangeError("nonfinite dry step budget");
  }
  std::ranges::copy(trial_particles, particles.begin());
  if (budget != nullptr) {
    *budget = trial_budget;
  }
  return absl::OkStatus();
}

absl::Status AdvanceDrySolid3D(
    const SparseMacGrid3D& grid,
    const DrySolidMaterial3D& material,
    const DrySolidStep3D& settings,
    std::span<DrySolidParticle3D> particles,
    DrySolidStepBudget3D* budget
) {
  const internal::DrySolidFloorTransfers3D floor_transfers{
      {0x1p-42, 0x1p-38, particles.size()},
      DepositCompatibleMacQuanta3D,
      GatherLiquidBoundaryParticles3D,
  };
  return internal::AdvanceDrySolidWithTransfers3D(
      grid,
      material,
      settings,
      particles,
      DepositMac3D,
      GatherMac3D,
      budget,
      &floor_transfers
  );
}

}  // namespace fluidity
