#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <string_view>
#include <vector>

#include "../src/dry_solid_step_3d_internal.h"
#include "../src/plastic_solid_checkpoint_wire_3d.h"
#include "fluidity/host/contact_face_bodies_3d.h"
#include "fluidity/host/dry_solid_step_3d.h"
#include "fluidity/host/mixed_contact_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"
#if defined(FLUIDITY_M5_CUDA_SCENES)
#include "fluidity/host/cuda_dry_solid_step_3d.h"
#include "fluidity/host/cuda_transfers_3d.h"
#endif

namespace {

using fluidity::BlockIndex3D;
using fluidity::DrySolidMaterial3D;
using fluidity::DrySolidParticle3D;
using fluidity::DrySolidStep3D;
using fluidity::Mat3;
using fluidity::SparseMacGrid3D;

constexpr Mat3 kIdentity{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

struct SandSurfaceSlopes {
  double minimum_degrees = 90;
  double maximum_degrees = -90;
  int size = 0;
};

// Fit each quadrant's free surface independently. Use actual sample radii to
// avoid a bin-center bias. The quadrant spread describes sampling uncertainty.
SandSurfaceSlopes FitSandSurface(std::span<const DrySolidParticle3D> particles, double maximum_radius) {
  struct SurfaceSample {
    double radius = 0;
    double height = -1;
  };
  constexpr int kBinCount = 6;
  constexpr double kStart = 0.15;
  constexpr double kEnd = 0.85;
  constexpr double kDegreesPerRadian = 57.29577951308232;
  std::array<std::array<SurfaceSample, kBinCount>, 4> quadrants{};
  for (const auto& particle : particles) {
    const auto position = particle.world_position;
    const double radius = std::hypot(position.x, position.z);
    const double fraction = radius / maximum_radius;
    if (fraction < kStart || fraction >= kEnd) {
      continue;
    }
    const int quadrant = (position.x < 0 ? 1 : 0) + (position.z < 0 ? 2 : 0);
    const int bin = std::min(kBinCount - 1, static_cast<int>((fraction - kStart) * kBinCount / (kEnd - kStart)));
    auto& sample = quadrants[quadrant][bin];
    if (position.y > sample.height) {
      sample = {.radius = radius, .height = position.y};
    }
  }

  SandSurfaceSlopes slopes{};
  for (const auto& quadrant : quadrants) {
    int sample_count = 0;
    double radius_sum = 0;
    double height_sum = 0;
    double radius_squared_sum = 0;
    double radius_height_sum = 0;
    for (const auto& sample : quadrant) {
      if (sample.height >= 0) {
        ++sample_count;
        radius_sum += sample.radius;
        height_sum += sample.height;
        radius_squared_sum += sample.radius * sample.radius;
        radius_height_sum += sample.radius * sample.height;
      }
    }
    if (sample_count < 4) {
      continue;
    }
    const double radius_variation = sample_count * radius_squared_sum - radius_sum * radius_sum;
    if (radius_variation > 0) {
      const double slope = (sample_count * radius_height_sum - radius_sum * height_sum) / radius_variation;
      const double degrees = -kDegreesPerRadian * std::atan(slope);
      slopes.minimum_degrees = std::min(slopes.minimum_degrees, degrees);
      slopes.maximum_degrees = std::max(slopes.maximum_degrees, degrees);
      ++slopes.size;
    }
  }
  return slopes;
}

bool TestScene(
    double width,
    double duration,
    bool snow,
    double particle_energy_blend = 0,
    int particles_per_axis_multiplier = 1,
    double sand_release_height = 0.10,
    bool release_cone = false,
    double sand_end_time = 0.75,
    int quantized_mass_bits = 0
) {
  std::vector<BlockIndex3D> blocks;
  for (int z = -2; z <= 2; ++z) {
    for (int y = -1; y <= 4; ++y) {
      for (int x = -3; x <= 3; ++x) {
        blocks.push_back({x, y, z});
      }
    }
  }
  const auto grid = SparseMacGrid3D::Create({{}, width}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  const DrySolidMaterial3D material = snow
      ? DrySolidMaterial3D(fluidity::SnowParameters3D{1200, 700, 0.025, 0.01, 4})
      : DrySolidMaterial3D(fluidity::GranularParameters3D{700, 450, 0.5235987755982989, 0});
  std::vector<DrySolidParticle3D> particles;
  constexpr double kConeRadius = 0.175;
  const int horizontal_samples = release_cone ? 15 : 3 * particles_per_axis_multiplier;
  const int vertical_samples = (snow ? 4 : 7) * particles_per_axis_multiplier;
  const double spacing = 0.05 / particles_per_axis_multiplier;
  const double horizontal_center = 0.5 * (horizontal_samples - 1);
  const double volume = spacing * spacing * spacing;
  for (int z = 0; z < horizontal_samples; ++z) {
    for (int y = 0; y < (release_cone ? 7 : vertical_samples); ++y) {
      for (int x = 0; x < horizontal_samples; ++x) {
        const double world_x = (x - horizontal_center) * spacing;
        const double world_z = (z - horizontal_center) * spacing;
        const double height_above_base = (y + 0.5) * spacing;
        if (release_cone && height_above_base + std::hypot(world_x, world_z) > kConeRadius) {
          continue;
        }
        particles.push_back(
            {.world_position = {world_x, height_above_base + (snow ? 0.0 : sand_release_height), world_z},
             .mass = 300 * volume,
             .reference_volume = volume,
             .elastic_deformation = kIdentity}
        );
      }
    }
  }
  if (release_cone) {
    std::printf("45-degree cone samples=%zu mass_kg=%.9f\n", particles.size(), particles.size() * 300 * volume);
  }
  const double time = snow ? 0.45 : sand_end_time;
  const double settling_sample_time = time == 0.75 ? 0.70 : time - 0.05;
  constexpr double kDegreesPerRadian = 57.29577951308232;
  double early_envelope_angle = 0;
  fluidity::DrySolidStepBudget3D total_budget{};
  int step = 0;
  for (double t = 0; t < time; ++step) {
    const double phase_end =
        snow ? (t < 0.23 ? 0.23 : (t < 0.35 ? 0.35 : time)) : (t < settling_sample_time ? settling_sample_time : time);
    const double next_time = t + duration >= phase_end - 1e-12 ? phase_end : t + duration;
    DrySolidStep3D settings{
        .duration = next_time - t,
        .gravity_y = snow ? 0.0 : -9.81,
        .floor_friction = 0.7,
        .acoustic_cfl = 0.25,
        .particle_energy_blend = particle_energy_blend,
    };
    if (snow && t < 0.35) {
      const double speed = t < 0.23 ? -0.4 : 0.5;
      const double position = t < 0.23 ? 0.21 - 0.4 * t : 0.118 + 0.5 * (t - 0.23);
      settings.platen = fluidity::DrySolidPlaten3D{.start_y = position, .velocity_y = speed};
    }
    fluidity::DrySolidStepBudget3D budget{};
    const int mass_bits = quantized_mass_bits == 0 ? 42 : quantized_mass_bits;
    const fluidity::MacDepositQuantization3D quantization{
        std::ldexp(1.0, -mass_bits),
        std::ldexp(1.0, 4 - mass_bits),
        std::max<std::size_t>(128, particles.size()),
    };
#if defined(FLUIDITY_M5_CUDA_SCENES)
    const auto status = fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, quantization, particles, &budget);
#else
    const auto quantized_deposit = [quantization](
                                       const SparseMacGrid3D& layout,
                                       std::span<const fluidity::MacTransferParticle3D> points,
                                       std::span<fluidity::MacFaceDeposit3D> faces
                                   ) {
      std::vector<fluidity::MacFaceDepositQuanta3D> quanta(faces.size());
      const auto deposited = fluidity::DepositMacQuanta3D(layout, points, quantization, quanta);
      if (!deposited.ok()) {
        return deposited;
      }
      for (std::size_t slot = 0; slot < faces.size(); ++slot) {
        faces[slot] = {
            .deposited_mass = static_cast<double>(quanta[slot].mass_quanta_sum) * quantization.mass_per_quantum,
            .deposited_pure_momentum =
                static_cast<double>(quanta[slot].pure_momentum_quanta_sum) * quantization.pure_momentum_per_quantum,
        };
      }
      return absl::OkStatus();
    };
    const auto status = quantized_mass_bits == 0
        ? fluidity::AdvanceDrySolid3D(*grid, material, settings, particles, &budget)
        : fluidity::internal::AdvanceDrySolidWithTransfers3D(
              *grid,
              material,
              settings,
              particles,
              quantized_deposit,
              fluidity::GatherMac3D,
              &budget
          );
#endif
    if (!status.ok()) {
      std::fprintf(
          stderr,
          "dry scene %s h=%.3f step %d: %s\n",
          snow ? "snow" : "sand",
          width,
          step,
          status.ToString().c_str()
      );
      return false;
    }
    if (!snow && t < settling_sample_time && next_time == settling_sample_time) {
      double height = 0;
      double radius = 0;
      for (const auto& particle : particles) {
        height = std::fmax(height, particle.world_position.y);
        radius = std::fmax(radius, std::hypot(particle.world_position.x, particle.world_position.z));
      }
      early_envelope_angle = kDegreesPerRadian * std::atan2(height, radius);
    }
    total_budget.proposed_stress_impulse_l1 += budget.proposed_stress_impulse_l1;
    total_budget.omitted_stress_impulse_l1 += budget.omitted_stress_impulse_l1;
    total_budget.omitted_stress_work += budget.omitted_stress_work;
    total_budget.grid_wall_impulse_y += budget.grid_wall_impulse_y;
    total_budget.wall_impulse_on_particles.y += budget.wall_impulse_on_particles.y;
    total_budget.particle_wall_contacts += budget.particle_wall_contacts;
    t = next_time;
  }
  double maximum_height = 0;
  double maximum_radius = 0;
  double log_plastic_sum = 0;
  double kinetic_energy = 0;
  double final_momentum_y = 0;
  for (const auto& particle : particles) {
    if (particle.world_position.y < 0 || particle.mass != 300 * volume || particle.reference_volume != volume
        || !std::isfinite(particle.log_plastic_jacobian)) {
      return false;
    }
    maximum_height = std::fmax(maximum_height, particle.world_position.y);
    maximum_radius = std::fmax(maximum_radius, std::hypot(particle.world_position.x, particle.world_position.z));
    log_plastic_sum += particle.log_plastic_jacobian;
    const auto v = particle.world_velocity;
    final_momentum_y += particle.mass * v.y;
    kinetic_energy += 0.5 * particle.mass * (v.x * v.x + v.y * v.y + v.z * v.z);
  }
  const double mean_log_plastic = log_plastic_sum / particles.size();
  std::printf(
      "%s h=%.4f dt=%.8f height=%.6f radius=%.6f mean_log_Jpl=%.6f kinetic_J=%.6f\n",
      snow ? "snow" : "sand",
      width,
      duration,
      maximum_height,
      maximum_radius,
      mean_log_plastic,
      kinetic_energy
  );
  std::printf(
      "budget proposed_stress_Ns=%.9f omitted_stress_Ns=%.9f omitted_work_J=%.9g "
      "grid_wall_y_Ns=%.9f particle_wall_y_Ns=%.9f contacts=%zu\n",
      total_budget.proposed_stress_impulse_l1,
      total_budget.omitted_stress_impulse_l1,
      total_budget.omitted_stress_work,
      total_budget.grid_wall_impulse_y,
      total_budget.wall_impulse_on_particles.y,
      total_budget.particle_wall_contacts
  );
  const double gravity_impulse_y = (snow ? 0.0 : -9.81) * (300 * volume * particles.size()) * time;
  const double vertical_residual = final_momentum_y - gravity_impulse_y - total_budget.grid_wall_impulse_y
      - total_budget.wall_impulse_on_particles.y;
  std::printf("vertical momentum residual_Ns=%.9g\n", vertical_residual);
  constexpr double kMomentumRoundoffTolerance = 1e-6;
  if (std::abs(vertical_residual) > total_budget.omitted_stress_impulse_l1 + kMomentumRoundoffTolerance) {
    return false;
  }
  if (total_budget.omitted_stress_impulse_l1 > total_budget.proposed_stress_impulse_l1) {
    return false;
  }
  if (snow) {
    return mean_log_plastic > -0.48 && mean_log_plastic < -0.34 && maximum_height > 0.11 && maximum_height < 0.15;
  }
  const double envelope_angle = kDegreesPerRadian * std::atan2(maximum_height, maximum_radius);
  const double settling_drift = std::abs(envelope_angle - early_envelope_angle);
  std::printf("sand envelope_degrees=%.6f drift_degrees=%.6f\n", envelope_angle, settling_drift);
  if (particles_per_axis_multiplier > 1 && maximum_radius > 0) {
    const auto slopes = FitSandSurface(particles, maximum_radius);
    if (slopes.size > 0) {
      std::printf(
          "sand surface slope range_degrees=[%.6f,%.6f] quadrants=%d\n",
          slopes.minimum_degrees,
          slopes.maximum_degrees,
          slopes.size
      );
    } else {
      std::printf("sand surface slope unresolved: fewer than four occupied radial bins per quadrant\n");
    }
  }
  if (release_cone) {
    return envelope_angle >= 30 && envelope_angle <= 35 && settling_drift <= 2 && kinetic_energy < 0.01;
  }
  return envelope_angle >= 30 && envelope_angle <= 35 && settling_drift <= 2 && maximum_radius > 0.10
      && maximum_radius < 0.16 && kinetic_energy < 0.01;
}

bool TestSandSurfaceFit() {
  std::vector<DrySolidParticle3D> particles;
  for (double sign_x : {-1.0, 1.0}) {
    for (double sign_z : {-1.0, 1.0}) {
      for (int bin = 0; bin < 6; ++bin) {
        const double radius = 0.15 + (bin + 0.5) * 0.70 / 6;
        const double component = radius / std::sqrt(2.0);
        const double height = (1 - radius) / std::sqrt(3.0);
        particles.push_back({.world_position = {sign_x * component, height, sign_z * component}});
        particles.push_back({.world_position = {sign_x * component, 0, sign_z * component}});
      }
    }
  }
  const auto slopes = FitSandSurface(particles, 1);
  return slopes.size == 4 && std::abs(slopes.minimum_degrees - 30) < 1e-11
      && std::abs(slopes.maximum_degrees - 30) < 1e-11;
}

bool TestMixedContactAcceptance() {
  fluidity::LiquidParticleState3D water{
      .id = 7,
      .world_position = {2, 3, 4},
      .world_velocity = {2, 0, 0},
      .mass = 6,
      .volume_ratio = 0.95,
  };
  fluidity::PlasticSolidCheckpointParticle3D snow{
      .particle_id = 3,
      .body_id = 9,
      .material_id = 1,
      .transfer =
          {
              .trial_world_position = {1, 2, 3},
              .particle_mass = 2,
              .particle_velocity = {-2, 3, 0},
          },
      .reference_volume = 0.001,
      .elastic_deformation = kIdentity,
      .log_plastic_jacobian = std::log(0.8),
  };
  constexpr std::array contributors{
      fluidity::ContactFaceParticle3D{3, 9, 2, 0.5, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{7, 11, 6, 0.5, {2, 0, 0}},
  };
  std::array faces{
      fluidity::DetectedContactFace3D{3, contributors, 9, 11, {1, 0, 0}, 0.25},
      fluidity::DetectedContactFace3D{6, contributors, 9, 99, {1, 0, 0}, 0.25},
  };
  water.world_velocity.z = -0.0;
  if (!fluidity::ApplyMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, {}).ok()
      || !std::signbit(water.world_velocity.z)) {
    return false;
  }
  auto extreme_water = water;
  extreme_water.world_velocity.x = -1.7e308;
  auto extreme_snow = snow;
  auto extreme_contributors = contributors;
  extreme_contributors[0].world_face_velocity = {-4e307, 0, 0};
  extreme_contributors[1].world_face_velocity = {4e307, 0, 0};
  const fluidity::DetectedContactFace3D extreme_face{3, extreme_contributors, 9, 11, {1, 0, 0}, 0};
  const auto overflow =
      fluidity::ApplyMixedContactFaces3D({&extreme_water, 1}, 11, {&extreme_snow, 1}, {&extreme_face, 1});
  if (overflow.status().code() != absl::StatusCode::kOutOfRange
      || overflow.status().message() != "unrepresentable stored mixed-contact velocity"
      || extreme_water.world_velocity.x != -1.7e308
      || extreme_snow.transfer.particle_velocity.x != snow.transfer.particle_velocity.x) {
    return false;
  }
  const auto old_water = water;
  const auto old_snow = snow;
  if (fluidity::ApplyMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, faces).ok() || !(water == old_water)
      || snow.transfer.particle_velocity.x != old_snow.transfer.particle_velocity.x
      || snow.transfer.particle_velocity.y != old_snow.transfer.particle_velocity.y) {
    return false;
  }
  faces[1].second_body_id = 11;
  const auto accepted = fluidity::ApplyMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, faces);
  return accepted.ok() && accepted->size() == 2 && std::abs(water.world_velocity.x - 1) < 1e-12
      && std::abs(snow.transfer.particle_velocity.x - 1) < 1e-12
      && std::abs(6 * water.world_velocity.y + 2 * snow.transfer.particle_velocity.y - 6) < 1e-12
      && water.world_position.x == old_water.world_position.x && water.volume_ratio == old_water.volume_ratio
      && snow.log_plastic_jacobian == old_snow.log_plastic_jacobian
      && snow.transfer.trial_world_position.x == old_snow.transfer.trial_world_position.x;
}

bool TestContactFaceBatch() {
  constexpr std::array particles{
      fluidity::ContactParticleIdentity3D{1, 9, 2},
      fluidity::ContactParticleIdentity3D{2, 11, 6},
  };
  constexpr std::array contributors{
      fluidity::ContactFaceParticle3D{1, 9, 2, 0.5, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{2, 11, 6, 0.5, {2, 0, 0}},
  };
  const std::array faces{
      fluidity::DetectedContactFace3D{3, contributors, 9, 11, {1, 0, 0}, 0.25},
      fluidity::DetectedContactFace3D{6, contributors, 9, 11, {1, 0, 0}, 0.25},
  };
  const auto batch = fluidity::StageContactFaceBatch3D(particles, faces);
  if (!batch.ok() || batch->face_contacts.size() != 2 || batch->velocity_deltas.size() != 2
      || std::abs(batch->velocity_deltas[0].x - 3) > 1e-12 || std::abs(batch->velocity_deltas[1].x + 1) > 1e-12
      || std::abs(2 * batch->velocity_deltas[0].y + 6 * batch->velocity_deltas[1].y) > 1e-12) {
    return false;
  }
  auto invalid = faces;
  invalid[1].second_body_id = 99;
  if (fluidity::StageContactFaceBatch3D(particles, invalid).ok()) {
    return false;
  }
  invalid = faces;
  invalid[1].face_slot = invalid[0].face_slot;
  if (fluidity::StageContactFaceBatch3D(particles, invalid).ok()) {
    return false;
  }
  auto excessive = contributors;
  excessive[0].weight = 0.75;
  invalid = faces;
  invalid[1].contributors = excessive;
  return !fluidity::StageContactFaceBatch3D(particles, invalid).ok();
}

bool TestContactFaceParticleScatter() {
  constexpr std::array particles{
      fluidity::ContactFaceParticle3D{1, 9, 2, 0.5, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{2, 9, 4, 0.25, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{3, 11, 6, 1, {2, 0, 0}},
      fluidity::ContactFaceParticle3D{4, 13, 1, 1, {9, 0, 0}},
  };
  const auto scattered = fluidity::ResolveContactFaceParticles3D(particles, 9, 11, {1, 0, 0}, 0.25);
  if (!scattered.ok() || scattered->velocity_deltas.size() != particles.size()) {
    return false;
  }
  const auto& deltas = scattered->velocity_deltas;
  const double impulse_x = 2 * deltas[0].x + 4 * deltas[1].x;
  const double impulse_y = 2 * deltas[0].y + 4 * deltas[1].y;
  if (std::abs(impulse_x - 6) > 1e-12 || std::abs(impulse_y + 1.5) > 1e-12
      || std::abs(impulse_x + 6 * deltas[2].x) > 1e-12 || std::abs(impulse_y + 6 * deltas[2].y) > 1e-12
      || deltas[3].x != 0 || deltas[3].y != 0) {
    return false;
  }
  auto invalid = particles;
  invalid.back().particle_id = 3;
  return !fluidity::ResolveContactFaceParticles3D(invalid, 9, 11, {1, 0, 0}, 0.25).ok();
}

bool TestPlasticCheckpointHandoff() {
  const BlockIndex3D block{0, 0, 0};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&block, 1}, 1);
  if (!grid.ok()) {
    return false;
  }
  const DrySolidMaterial3D snow = fluidity::SnowParameters3D{2, 3, 0.025, 0.01, 4};
  const DrySolidParticle3D accepted{
      .world_position = {2.25, 2.25, 2.25},
      .world_velocity = {0.1, 0, 0},
      .velocity_slopes = {{0.1, -0.02, 0}, {0, 0.05, 0}, {0, 0, -0.1}},
      .mass = 1,
      .reference_volume = 0.001,
      .elastic_deformation = {{1, 0, 0}, {0, 0.975, 0}, {0, 0, 1}},
      .log_plastic_jacobian = std::log(0.8),
  };
  const auto same_vector = [](fluidity::Vec3 first, fluidity::Vec3 second) {
    return first.x == second.x && first.y == second.y && first.z == second.z;
  };
  const auto same_matrix = [same_vector](Mat3 first, Mat3 second) {
    return same_vector(first.x_row, second.x_row) && same_vector(first.y_row, second.y_row)
        && same_vector(first.z_row, second.z_row);
  };
  const auto same_particle = [same_vector,
                              same_matrix](const DrySolidParticle3D& first, const DrySolidParticle3D& second) {
    return same_vector(first.world_position, second.world_position)
        && same_vector(first.world_velocity, second.world_velocity)
        && same_matrix(first.velocity_slopes, second.velocity_slopes) && first.mass == second.mass
        && first.reference_volume == second.reference_volume
        && same_matrix(first.elastic_deformation, second.elastic_deformation)
        && first.log_plastic_jacobian == second.log_plastic_jacobian;
  };
  const auto captured = fluidity::CapturePlasticSolidCheckpointParticle3D(7, 9, 11, snow, accepted);
  if (!captured.ok()) {
    return false;
  }
  const auto packed = fluidity::internal::PackPlasticSolidCheckpointParticle3D(*captured);
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(builder.CreateVectorOfStructs(&packed, 1));
  using WireVector = flatbuffers::Vector<const fluidity::solid::v1::PlasticParticle*>;
  const auto* wire_records = flatbuffers::GetRoot<WireVector>(builder.GetBufferPointer());
  if (wire_records->size() != 1) {
    return false;
  }
  const auto decoded = fluidity::internal::UnpackPlasticSolidCheckpointParticle3D(*wire_records->Get(0));
  if (decoded.particle_id != 7 || decoded.body_id != 9 || decoded.material_id != 11
      || !same_vector(decoded.transfer.sorted_world_position, accepted.world_position)) {
    return false;
  }
  auto restored = fluidity::RestoreDrySolidCheckpointParticles3D({&decoded, 1}, 9, 11, snow);
  if (!restored.ok() || restored->size() != 1 || !same_particle(accepted, restored->front())) {
    return false;
  }
  auto corrupt = *captured;
  corrupt.log_plastic_jacobian = 1000;
  if (fluidity::RestoreDrySolidCheckpointParticles3D({&corrupt, 1}, 9, 11, snow).ok()
      || fluidity::RestoreDrySolidCheckpointParticles3D({&*captured, 1}, 9, 12, snow).ok()) {
    return false;
  }
  const std::array duplicate{*captured, *captured};
  if (fluidity::RestoreDrySolidCheckpointParticles3D(duplicate, 9, 11, snow).ok()) {
    return false;
  }
  auto uninterrupted = accepted;
  const DrySolidStep3D step{.duration = 0.01, .acoustic_cfl = 0.25};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto first = fluidity::CudaAdvanceDrySolid3D(*grid, snow, step, {0x1p-42, 0x1p-38, 128}, {&uninterrupted, 1});
  const auto second = fluidity::CudaAdvanceDrySolid3D(*grid, snow, step, {0x1p-42, 0x1p-38, 128}, *restored);
#else
  const auto first = fluidity::AdvanceDrySolid3D(*grid, snow, step, {&uninterrupted, 1});
  const auto second = fluidity::AdvanceDrySolid3D(*grid, snow, step, *restored);
#endif
  return first.ok() && second.ok() && same_particle(uninterrupted, restored->front());
}

bool TestWallImpulseBudget() {
  constexpr std::array blocks{BlockIndex3D{0, -1, 0}, BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  DrySolidParticle3D particle{
      .world_position = {2.25, 0, 2.25},
      .world_velocity = {1, -1, 0},
      .mass = 1,
      .reference_volume = 0.001,
      .elastic_deformation = kIdentity,
  };
  const DrySolidMaterial3D material = fluidity::GranularParameters3D{2, 3, 0.5, 0};
  const DrySolidStep3D settings{.duration = 0.01, .floor_friction = 0.5, .acoustic_cfl = 0.25};
  fluidity::DrySolidStepBudget3D budget{};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1}, &budget);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1}, &budget);
#endif
  return status.ok() && budget.particle_wall_contacts == 1 && std::abs(budget.wall_impulse_on_particles.x + 0.5) < 1e-9
      && std::abs(budget.wall_impulse_on_particles.y - 1) < 1e-9 && budget.omitted_stress_impulse_l1 == 0
      && budget.grid_wall_impulse_y == 0;
}

bool TestStoredVelocityResidual() {
  const BlockIndex3D block{0, 0, 0};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&block, 1}, 1);
  if (!grid.ok()) {
    return false;
  }
  const DrySolidParticle3D first{
      .world_position = {2.25, 2.25, 2.25},
      .world_velocity = {1, 0, 0},
      .mass = 1000,
      .reference_volume = 1,
      .elastic_deformation = kIdentity,
  };
  std::array particles{first, first, first, first};
  particles[1].world_velocity.x = -1;
  particles[3].world_velocity.x = -1;
  const DrySolidMaterial3D material = fluidity::GranularParameters3D{2, 3, 0.5, 0};
  const DrySolidStep3D settings{.duration = 0.01, .acoustic_cfl = 0.25, .particle_energy_blend = 0.6};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status = fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, particles);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, particles);
#endif
  return status.ok() && std::abs(particles[0].world_velocity.x - 0.594) < 1e-12
      && std::abs(particles[1].world_velocity.x + 0.594) < 1e-12 && particles[0].world_position.x == 2.25
      && particles[1].world_position.x == 2.25;
}

bool TestStoredResidualMomentum() {
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}, BlockIndex3D{1, 0, 0}};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  const DrySolidParticle3D first{
      .world_position = {1.75, 2.25, 2.25},
      .world_velocity = {1, 0, 0},
      .mass = 1,
      .reference_volume = 0.001,
      .elastic_deformation = kIdentity,
  };
  std::array particles{first, first, first, first, first};
  particles.back().world_position.x = 2.5001;
  particles.back().world_velocity.x = -1;
  const DrySolidMaterial3D material = fluidity::GranularParameters3D{2, 3, 0.5, 0};
  const DrySolidStep3D settings{.duration = 0.01, .acoustic_cfl = 0.25, .particle_energy_blend = 0.6};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status = fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, particles);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, particles);
#endif
  double momentum_x = 0;
  for (const auto& particle : particles) {
    momentum_x += particle.mass * particle.world_velocity.x;
  }
  if (!status.ok() || std::abs(momentum_x - 3) > 1e-9) {
    std::fprintf(stderr, "stored residual momentum %.12g N s, %s\n", momentum_x, status.ToString().c_str());
    return false;
  }
  return true;
}

bool TestRollbackOnUnsupportedDestination() {
  const BlockIndex3D block{0, 0, 0};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&block, 1}, 1);
  if (!grid.ok()) {
    return false;
  }
  DrySolidParticle3D particle{
      .world_position = {2.4, 2, 2},
      .world_velocity = {1, 0, 0},
      .mass = 1000,
      .reference_volume = 1,
      .elastic_deformation = kIdentity
  };
  const DrySolidMaterial3D material = fluidity::GranularParameters3D{2, 3, 0.5, 0};
  const DrySolidStep3D settings{.duration = 0.4, .acoustic_cfl = 0.25};
  fluidity::DrySolidStepBudget3D budget{.particle_wall_contacts = 42};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1}, &budget);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1}, &budget);
#endif
  if (budget.particle_wall_contacts != 42 || status.code() != absl::StatusCode::kFailedPrecondition
      || particle.world_position.x != 2.4 || particle.world_velocity.x != 1 || particle.log_plastic_jacobian != 0) {
    std::fprintf(
        stderr,
        "unsupported destination rollback failed: %s x=%.6f\n",
        status.ToString().c_str(),
        particle.world_position.x
    );
    return false;
  }
  return true;
}

bool TestRejectsInadmissibleInitialState() {
  const BlockIndex3D block{0, 0, 0};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&block, 1}, 1);
  if (!grid.ok()) {
    return false;
  }
  const Mat3 overcompressed{{0.8, 0, 0}, {0, 0.8, 0}, {0, 0, 0.8}};
  DrySolidParticle3D particle{
      .world_position = {2, 2, 2},
      .mass = 1000,
      .reference_volume = 1,
      .elastic_deformation = overcompressed
  };
  const DrySolidMaterial3D material = fluidity::SnowParameters3D{2, 3, 0.025, 0.01, 4};
  const DrySolidStep3D settings{.duration = 0.01, .acoustic_cfl = 0.25};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1});
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1});
#endif
  if (status.code() != absl::StatusCode::kFailedPrecondition || particle.elastic_deformation.y_row.y != 0.8
      || particle.log_plastic_jacobian != 0) {
    std::fprintf(stderr, "inadmissible initial state was not preserved: %s\n", status.ToString().c_str());
    return false;
  }
  return true;
}

bool TestUncutStressWork() {
  const BlockIndex3D block{0, 0, 0};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&block, 1}, 1);
  if (!grid.ok()) {
    return false;
  }
  const fluidity::Vec3 position{2.2, 2.3, 2.4};
  constexpr Mat3 stress{{10, 2, 3}, {2, 20, 4}, {3, 4, -30}};
  constexpr Mat3 gradient{{3, 2, 1}, {1, 4, 2}, {2, 3, -1}};
  constexpr double volume = 0.5;
  const auto view = grid->view();
  std::vector<double> forces(view.face_count());
  const auto deposited = fluidity::internal::AccumulateDrySolidStress3D(*grid, position, volume, stress, forces);
  if (!deposited.ok()) {
    return false;
  }
  std::vector<double> velocities(view.face_count());
  double grid_power = 0;
  const fluidity::Vec3 rows[]{gradient.x_row, gradient.y_row, gradient.z_row};
  for (std::size_t slot = 0; slot < view.face_count(); ++slot) {
    const auto face = view.FaceIndex(slot);
    if (!face.ok()) {
      return false;
    }
    const auto anchor = view.FaceWorldAnchor(*face);
    if (!anchor.ok()) {
      return false;
    }
    const auto row = rows[static_cast<int>(face->axis)];
    velocities[slot] = row.x * anchor->x + row.y * anchor->y + row.z * anchor->z;
    grid_power += velocities[slot] * forces[slot];
  }
  const fluidity::MacTransferParticle3D particle{.sorted_world_position = position, .trial_world_position = position};
  fluidity::MacParticleGatherResult3D result{};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  fluidity::CudaOperatorMeasurement measurement{};
  const auto gathered =
      fluidity::CudaGatherMac3D(*grid, {&particle, 1}, velocities, velocities, {&result, 1}, measurement);
#else
  const auto gathered = fluidity::GatherMac3D(*grid, {&particle, 1}, velocities, velocities, {&result, 1});
#endif
  if (!gathered.ok()) {
    return false;
  }
  const auto measured = result.velocity_gradient;
  const double particle_power = volume
      * (10 * measured.x_row.x + 2 * measured.x_row.y + 3 * measured.x_row.z + 2 * measured.y_row.x
         + 20 * measured.y_row.y + 4 * measured.y_row.z + 3 * measured.z_row.x + 4 * measured.z_row.y
         - 30 * measured.z_row.z);
  constexpr double kExpectedPower = 87.5;
  const double scaled_error = std::abs(grid_power + particle_power) / kExpectedPower;
  std::printf(
      "uncut stress work grid_W=%.12f particle_W=%.12f scaled_residual=%.3g\n",
      grid_power,
      particle_power,
      scaled_error
  );
  return scaled_error <= 1e-6 && std::abs(particle_power - kExpectedPower) <= 1e-6 * kExpectedPower;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-refined") {
    return TestScene(0.0625, 0.0015625, false) ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-half-step") {
    const bool reference = TestScene(0.075, 0.0009375, false);
    const bool refined = TestScene(0.0625, 0.00078125, false);
    return reference && refined ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-stored-blend") {
    const bool reference = TestScene(0.075, 0.001875, false, 0.6);
    const bool refined = TestScene(0.0625, 0.0015625, false, 0.6);
    const bool reference_half = TestScene(0.075, 0.0009375, false, 0.6);
    const bool refined_half = TestScene(0.0625, 0.00078125, false, 0.6);
    return reference && refined && reference_half && refined_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-dense-particles") {
    const bool reference = TestScene(0.075, 0.001875, false, 0, 2);
    const bool refined = TestScene(0.0625, 0.0015625, false, 0, 2);
    const bool reference_half = TestScene(0.075, 0.0009375, false, 0, 2);
    const bool refined_half = TestScene(0.0625, 0.00078125, false, 0, 2);
    return reference && refined && reference_half && refined_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-dense-refined-grid") {
    const bool coarse = TestScene(0.0375, 0.00078125, false, 0, 2);
    const bool fine = TestScene(0.03125, 0.00078125, false, 0, 2);
    const bool coarse_half = TestScene(0.0375, 0.000390625, false, 0, 2);
    const bool fine_half = TestScene(0.03125, 0.000390625, false, 0, 2);
    return coarse && fine && coarse_half && fine_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-floor-release") {
    const bool coarse = TestScene(0.0375, 0.00078125, false, 0, 2, 0);
    const bool fine = TestScene(0.03125, 0.00078125, false, 0, 2, 0);
    const bool coarse_half = TestScene(0.0375, 0.000390625, false, 0, 2, 0);
    const bool fine_half = TestScene(0.03125, 0.000390625, false, 0, 2, 0);
    return coarse && fine && coarse_half && fine_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-cone-release") {
    const bool coarse = TestScene(0.0375, 0.00078125, false, 0, 2, 0, true);
    const bool fine = TestScene(0.03125, 0.00078125, false, 0, 2, 0, true);
    const bool coarse_half = TestScene(0.0375, 0.000390625, false, 0, 2, 0, true);
    const bool fine_half = TestScene(0.03125, 0.000390625, false, 0, 2, 0, true);
    return coarse && fine && coarse_half && fine_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-cone-long-settle") {
    const bool coarse = TestScene(0.0375, 0.00078125, false, 0, 2, 0, true, 1.5);
    const bool fine = TestScene(0.03125, 0.00078125, false, 0, 2, 0, true, 1.5);
    const bool coarse_half = TestScene(0.0375, 0.000390625, false, 0, 2, 0, true, 1.5);
    const bool fine_half = TestScene(0.03125, 0.000390625, false, 0, 2, 0, true, 1.5);
    return coarse && fine && coarse_half && fine_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-cone-stored-only") {
    const bool coarse = TestScene(0.0375, 0.00078125, false, 0.6, 2, 0, true);
    const bool fine = TestScene(0.03125, 0.00078125, false, 0.6, 2, 0, true);
    const bool coarse_half = TestScene(0.0375, 0.000390625, false, 0.6, 2, 0, true);
    const bool fine_half = TestScene(0.03125, 0.000390625, false, 0.6, 2, 0, true);
    return coarse && fine && coarse_half && fine_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-cone-quantization") {
    const bool standard = TestScene(0.03125, 0.00078125, false, 0.6, 2, 0, true, 0.75, 42);
    const bool fine_quanta = TestScene(0.03125, 0.00078125, false, 0.6, 2, 0, true, 0.75, 50);
    return standard && fine_quanta ? 0 : 1;
  }
  if (argc != 1) {
    return 2;
  }
  const auto capacity = fluidity::ContactFaceDepositStatus(fluidity::ContactFaceDepositStatus3D::kBodyCapacity);
  if (capacity.code() != absl::StatusCode::kResourceExhausted || capacity.message() != "E-CONTACT-BODY-CAPACITY") {
    return 1;
  }
  if (!TestMixedContactAcceptance() || !TestContactFaceBatch() || !TestContactFaceParticleScatter()
      || !TestPlasticCheckpointHandoff() || !TestSandSurfaceFit() || !TestWallImpulseBudget()
      || !TestStoredVelocityResidual() || !TestStoredResidualMomentum() || !TestUncutStressWork()
      || !TestRollbackOnUnsupportedDestination() || !TestRejectsInadmissibleInitialState()) {
    return 1;
  }
  const bool coarse_sand = TestScene(0.075, 0.001875, false);
  const bool coarse_snow = TestScene(0.1, 0.0025, true);
  const bool fine_snow = TestScene(0.075, 0.001875, true);
  return coarse_sand && coarse_snow && fine_snow ? 0 : 1;
}
