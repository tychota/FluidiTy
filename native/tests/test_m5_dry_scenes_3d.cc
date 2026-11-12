#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>
#include <vector>

#include "../src/dry_solid_step_3d_internal.h"
#include "../src/liquid_boundary_transfer_3d_internal.h"
#include "fluidity/host/contact_face_bodies_3d.h"
#include "fluidity/host/dry_solid_step_3d.h"
#if defined(FLUIDITY_M5_CUDA_SCENES)
#include "fluidity/host/cuda_dry_solid_step_3d.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
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
    int quantized_mass_bits = 0,
    fluidity::Vec3 grid_origin = {},
    double floor_friction = 0.7,
    bool reset_column_velocity = false,
    double granular_modulus_scale = 1,
    double stored_velocity_decay_rate = 0,
    double granular_lambda = 700,
    bool compatible_floor = false
) {
  std::vector<BlockIndex3D> blocks;
  const int maximum_block_y = compatible_floor ? 2 : 4;
  const int maximum_block_x = compatible_floor ? 2 : 3;
  for (int z = -2; z <= 2; ++z) {
    for (int y = -1; y <= maximum_block_y; ++y) {
      for (int x = -maximum_block_x; x <= maximum_block_x; ++x) {
        blocks.push_back({x, y, z});
      }
    }
  }
  const auto grid = SparseMacGrid3D::Create({grid_origin, width}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  const DrySolidMaterial3D material = snow ? DrySolidMaterial3D(fluidity::SnowParameters3D{1200, 700, 0.025, 0.01, 4})
                                           : DrySolidMaterial3D(fluidity::GranularParameters3D{
                                                 granular_lambda * granular_modulus_scale,
                                                 ((1600 - granular_lambda) / 2) * granular_modulus_scale,
                                                 0.5235987755982989,
                                                 0
                                             });
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
  double late_projection_particle_seconds = 0;
  double late_plastic_volume_particle_seconds = 0;
  double late_particle_seconds = 0;
  double final_projection_particle_seconds = 0;
  double final_plastic_volume_particle_seconds = 0;
  double final_particle_seconds = 0;
  std::vector<double> previous_log_plastic(particles.size());
  double final_log_plastic_change = 0;
  double final_absolute_log_plastic_change = 0;
  double floor_layer_absolute_log_plastic_change = 0;
  double reset_impulse_y = 0;
  const bool extended_column = !snow && !release_cone && sand_end_time > 0.75;
  fluidity::DrySolidStepBudget3D total_budget{};
  int step = 0;
  for (double t = 0; t < time; ++step) {
    double phase_end =
        snow ? (t < 0.23 ? 0.23 : (t < 0.35 ? 0.35 : time)) : (t < settling_sample_time ? settling_sample_time : time);
    if (extended_column && t < 0.75) {
      // Replay the original two sampling boundaries before extending the run.
      phase_end = t < 0.70 ? 0.70 : 0.75;
    }
    const double next_time = t + duration >= phase_end - 1e-12 ? phase_end : t + duration;
    const double step_duration = next_time - t;
    const double step_energy_blend = stored_velocity_decay_rate > 0
        ? std::min(1.0, std::exp(-stored_velocity_decay_rate * step_duration) / 0.99)
        : particle_energy_blend;
    DrySolidStep3D settings{
        .duration = step_duration,
        .gravity_y = snow ? 0.0 : -9.81,
        .floor_friction = floor_friction,
        .acoustic_cfl = 0.25,
        .particle_energy_blend = step_energy_blend,
        .compatible_floor = compatible_floor,
    };
    if (snow && t < 0.35) {
      const double speed = t < 0.23 ? -0.4 : 0.5;
      const double position = t < 0.23 ? 0.21 - 0.4 * t : 0.118 + 0.5 * (t - 0.23);
      settings.platen = fluidity::DrySolidPlaten3D{.start_y = position, .velocity_y = speed};
    }
    const bool in_final_window = extended_column && t >= settling_sample_time;
    if (in_final_window) {
      for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
        previous_log_plastic[ordinal] = particles[ordinal].log_plastic_jacobian;
      }
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
    const bool late_drift_sample = !snow && t < settling_sample_time && next_time == settling_sample_time;
    const bool original_endpoint = extended_column && t < 0.75 && next_time == 0.75;
    if (late_drift_sample || original_endpoint) {
      double height = 0;
      double radius = 0;
      for (const auto& particle : particles) {
        height = std::fmax(height, particle.world_position.y);
        radius = std::fmax(radius, std::hypot(particle.world_position.x, particle.world_position.z));
      }
      const double angle = kDegreesPerRadian * std::atan2(height, radius);
      if (late_drift_sample) {
        early_envelope_angle = angle;
      }
      if (original_endpoint) {
        std::printf("sand envelope at 0.75 s=%.6f\n", angle);
      }
    }
    if (original_endpoint && reset_column_velocity) {
      double removed_kinetic_energy = 0;
      for (auto& particle : particles) {
        const auto velocity = particle.world_velocity;
        reset_impulse_y -= particle.mass * velocity.y;
        removed_kinetic_energy +=
            0.5 * particle.mass * (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
        particle.world_velocity = {};
        particle.velocity_slopes = {};
      }
      std::printf(
          "sand reset at 0.75 s removed kinetic_J=%.9g applied vertical impulse_Ns=%.9g\n",
          removed_kinetic_energy,
          reset_impulse_y
      );
    }
    if (extended_column && t >= 0.75) {
      late_projection_particle_seconds += step_duration * budget.granular_projected_particles;
      late_plastic_volume_particle_seconds += step_duration * budget.granular_plastic_volume_changes;
      late_particle_seconds += step_duration * particles.size();
      if (in_final_window) {
        for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
          const auto& particle = particles[ordinal];
          const double change = particle.log_plastic_jacobian - previous_log_plastic[ordinal];
          final_log_plastic_change += change;
          final_absolute_log_plastic_change += std::abs(change);
          if (particle.world_position.y <= spacing) {
            floor_layer_absolute_log_plastic_change += std::abs(change);
          }
        }
        final_projection_particle_seconds += step_duration * budget.granular_projected_particles;
        final_plastic_volume_particle_seconds += step_duration * budget.granular_plastic_volume_changes;
        final_particle_seconds += step_duration * particles.size();
      }
    }
    total_budget.proposed_stress_impulse_l1 += budget.proposed_stress_impulse_l1;
    total_budget.omitted_stress_impulse_l1 += budget.omitted_stress_impulse_l1;
    total_budget.omitted_stress_work += budget.omitted_stress_work;
    total_budget.cut_stress_work += budget.cut_stress_work;
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
  if (compatible_floor) {
    std::printf("compatible floor total cut stress work_J=%.9g\n", total_budget.cut_stress_work);
  }
  const double gravity_impulse_y = (snow ? 0.0 : -9.81) * (300 * volume * particles.size()) * time;
  const double vertical_residual = final_momentum_y - reset_impulse_y - gravity_impulse_y
      - total_budget.grid_wall_impulse_y - total_budget.wall_impulse_on_particles.y;
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
  if (late_particle_seconds > 0) {
    std::printf(
        "sand late projected fraction=%.9f plastic volume change fraction=%.9f\n",
        late_projection_particle_seconds / late_particle_seconds,
        late_plastic_volume_particle_seconds / late_particle_seconds
    );
  }
  if (final_particle_seconds > 0) {
    std::printf(
        "sand final 0.05 s projected fraction=%.9f plastic volume change fraction=%.9f\n",
        final_projection_particle_seconds / final_particle_seconds,
        final_plastic_volume_particle_seconds / final_particle_seconds
    );
  }
  if (final_particle_seconds > 0) {
    std::printf(
        "sand final 0.05 s log_Jpl mean signed=%.9g absolute path=%.9g floor layer absolute path=%.9g\n",
        final_log_plastic_change / particles.size(),
        final_absolute_log_plastic_change / particles.size(),
        floor_layer_absolute_log_plastic_change / particles.size()
    );
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

bool TestWallImpulseBudget(bool compatible_floor = false) {
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
  const DrySolidStep3D settings{
      .duration = 0.01,
      .floor_friction = 0.5,
      .acoustic_cfl = 0.25,
      .compatible_floor = compatible_floor,
  };
  fluidity::DrySolidStepBudget3D budget{};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1}, &budget);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1}, &budget);
#endif
  return status.ok() && budget.particle_wall_contacts == 1 && std::abs(budget.wall_impulse_on_particles.x + 0.5) < 1e-9
      && std::abs(budget.wall_impulse_on_particles.y - 1) < 1e-9 && budget.omitted_stress_impulse_l1 == 0
      && budget.grid_wall_impulse_y == 0 && budget.cut_stress_work == 0 && budget.granular_projected_particles == 0
      && budget.granular_plastic_volume_changes == 0;
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
  fluidity::DrySolidStepBudget3D budget{
      .particle_wall_contacts = 42,
      .granular_projected_particles = 17,
      .granular_plastic_volume_changes = 9,
  };
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1}, &budget);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1}, &budget);
#endif
  if (budget.particle_wall_contacts != 42 || budget.granular_projected_particles != 17
      || budget.granular_plastic_volume_changes != 9 || status.code() != absl::StatusCode::kFailedPrecondition
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

bool TestCompatibleFloorStressWork() {
  constexpr std::array blocks{BlockIndex3D{0, -1, 0}, BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  constexpr fluidity::Vec3 position{2.2, 0.2, 2.3};
  constexpr Mat3 stress{{10, 2, 3}, {2, 20, 4}, {3, 4, -30}};
  constexpr double volume = 0.5;
  constexpr double duration = 0.125;
  const fluidity::LiquidBoundaryParticle3D particle{
      .transfer = {.sorted_world_position = position, .trial_world_position = position, .particle_mass = 1},
  };
  const auto view = grid->view();
  std::vector<fluidity::LiquidBoundaryFaceTag3D> tags(view.face_count());
  std::vector<std::uint8_t> mask(view.face_count());
  std::vector<double> velocities(view.face_count());
  std::vector<fluidity::LiquidFaceValidity3D> validity(view.face_count());
  const std::vector<fluidity::LiquidPrescribedWallSample3D> walls(view.face_count());
  for (std::size_t slot = 0; slot < view.face_count(); ++slot) {
    const auto face = view.FaceIndex(slot);
    if (!face.ok()) {
      return false;
    }
    const auto anchor = view.FaceWorldAnchor(*face);
    if (!anchor.ok()) {
      return false;
    }
    const bool compatible = anchor->y >= 0;
    tags[slot].structurally_eligible = compatible;
    mask[slot] = compatible;
    validity[slot] = compatible ? fluidity::LiquidFaceValidity3D::kValid : fluidity::LiquidFaceValidity3D::kMissing;
    const std::array components{2 * anchor->x + anchor->z, 3 * anchor->y, anchor->x + 4 * anchor->z};
    velocities[slot] = components[static_cast<std::size_t>(face->axis)];
  }
  const fluidity::LiquidBoundaryDepositInputs3D deposit_inputs{*grid, {&particle, 1}, tags, {0x1p-42, 0x1p-38, 1}};
  const fluidity::LiquidBoundaryGatherInputs3D gather_inputs{
      *grid,
      {&particle, 1},
      0,
      duration,
      {0, 0, velocities, validity, tags, walls},
      {duration, duration, velocities, validity, tags, walls},
  };
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto device_deposits = fluidity::CudaDepositCompatibleMacQuanta3D(deposit_inputs);
  const auto device_gather = fluidity::CudaGatherLiquidBoundaryParticles3D(gather_inputs);
  if (!device_deposits.ok() || !device_gather.ok()) {
    return false;
  }
  const auto& deposits = device_deposits->deposits;
  const auto& gathered = device_gather->gathered;
#else
  const auto host_deposits = fluidity::DepositCompatibleMacQuanta3D(deposit_inputs);
  const auto host_gather = fluidity::GatherLiquidBoundaryParticles3D(gather_inputs);
  if (!host_deposits.ok() || !host_gather.ok()) {
    return false;
  }
  const auto& deposits = *host_deposits;
  const auto& gathered = *host_gather;
#endif
  std::vector<double> force(view.face_count());
  const auto assembled =
      fluidity::internal::AccumulateCompatibleSolidStress3D(deposits.compatible_stencils[0], volume, stress, force);
  if (!assembled.ok() || gathered.new_pic_fallback_particle_count != 0 || !gathered.particles[0].affine_enabled) {
    return false;
  }
  std::vector<double> oracle_force(view.face_count());
  const auto oracle = fluidity::DepositCutElasticStress3D(
      *grid,
      position,
      position,
      volume,
      stress,
      duration,
      mask,
      velocities,
      oracle_force
  );
  if (!oracle.ok()) {
    return false;
  }
  double grid_work = 0;
  for (std::size_t slot = 0; slot < force.size(); ++slot) {
    if (std::abs(force[slot] - oracle_force[slot]) > 1e-11 || (mask[slot] == 0 && force[slot] != 0)) {
      return false;
    }
    grid_work += duration * force[slot] * velocities[slot];
  }
  const auto& gradient = gathered.particles[0].velocity_gradient;
  const double particle_work = duration * volume
      * ((10 * gradient.x_row.x + 2 * gradient.x_row.y + 3 * gradient.x_row.z)
         + (2 * gradient.y_row.x + 20 * gradient.y_row.y + 4 * gradient.y_row.z)
         + (3 * gradient.z_row.x + 4 * gradient.z_row.y - 30 * gradient.z_row.z));
  std::printf(
      "cut floor stress work grid_J=%.9g particle_J=%.9g cut_J=%.9g\n",
      grid_work,
      particle_work,
      oracle->cut_work
  );
  constexpr double expected_particle_work = -2.125;
  return std::abs(particle_work - expected_particle_work) < 1e-11
      && std::abs(grid_work - oracle->grid_stress_work) < 1e-11
      && std::abs(particle_work + grid_work - oracle->cut_work) < 1e-11 && std::abs(oracle->cut_work) > 1e-3
      && std::hypot(oracle->stress_impulse.x, oracle->stress_impulse.y, oracle->stress_impulse.z) < 1e-11;
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
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-reset-column-velocity") {
    const bool reference = TestScene(0.075, 0.001875, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, true);
    const bool refined = TestScene(0.0625, 0.0015625, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, true);
    const bool reference_half = TestScene(0.075, 0.0009375, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, true);
    const bool refined_half = TestScene(0.0625, 0.00078125, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, true);
    return reference && refined && reference_half && refined_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-floor-friction") {
    bool all_pass = true;
    for (const double floor_friction : {0.0, 10.0}) {
      std::printf("sand floor friction coefficient=%.1f\n", floor_friction);
      const bool reference = TestScene(0.075, 0.001875, false, 0, 1, 0.10, false, 1.5, 0, {}, floor_friction);
      const bool refined = TestScene(0.0625, 0.0015625, false, 0, 1, 0.10, false, 1.5, 0, {}, floor_friction);
      all_pass = all_pass && reference && refined;
    }
    return all_pass ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-plane-strain-moduli") {
    bool all_pass = true;
    for (const double width : {0.075, 0.0625}) {
      for (const double duration : {width / 40, width / 80}) {
        const bool pass = TestScene(width, duration, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, false, 1, 0, 1280);
        all_pass = all_pass && pass;
      }
    }
    return all_pass ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-time-scaled-retention") {
    const double decay_rate = -std::log(0.594) / 0.001875;
    bool all_pass = true;
    for (const double width : {0.075, 0.0625}) {
      for (const double duration : {width / 40, width / 80}) {
        const bool pass = TestScene(width, duration, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, false, 1, decay_rate);
        all_pass = all_pass && pass;
      }
    }
    return all_pass ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-stiffer-column") {
    constexpr double kModulusScale = 4;
    bool all_pass = true;
    for (const double width : {0.075, 0.0625}) {
      for (const double duration : {width / 40, width / 80}) {
        const bool pass = TestScene(width, duration, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, false, kModulusScale);
        all_pass = all_pass && pass;
      }
    }
    return all_pass ? 0 : 1;
  }
  if (argc == 2
      && (std::string_view(argv[1]) == "--q-sand-column-long-settle"
          || std::string_view(argv[1]) == "--q-sand-dense-column-long-settle")) {
    const int sampling_multiplier = std::string_view(argv[1]) == "--q-sand-dense-column-long-settle" ? 2 : 1;
    const bool reference = TestScene(0.075, 0.001875, false, 0, sampling_multiplier, 0.10, false, 1.5);
    const bool refined = TestScene(0.0625, 0.0015625, false, 0, sampling_multiplier, 0.10, false, 1.5);
    const bool reference_half = TestScene(0.075, 0.0009375, false, 0, sampling_multiplier, 0.10, false, 1.5);
    const bool refined_half = TestScene(0.0625, 0.00078125, false, 0, sampling_multiplier, 0.10, false, 1.5);
    return reference && refined && reference_half && refined_half ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-dense-three-second") {
    const bool reference = TestScene(0.075, 0.0009375, false, 0, 2, 0.10, false, 3);
    const bool refined = TestScene(0.0625, 0.00078125, false, 0, 2, 0.10, false, 3);
    return reference && refined ? 0 : 1;
  }
  if (argc == 2
      && (std::string_view(argv[1]) == "--q-sand-compatible-floor"
          || std::string_view(argv[1]) == "--q-sand-compatible-floor-long-settle")) {
    const bool extended = std::string_view(argv[1]) == "--q-sand-compatible-floor-long-settle";
    const double end_time = extended ? 1.5 : 0.75;
    bool all_pass = true;
    for (const double width : {0.075, 0.0625}) {
      const bool nominal =
          TestScene(width, width / 40, false, 0, 1, 0.10, false, end_time, 0, {}, 0.7, false, 1, 0, 700, true);
      all_pass = all_pass && nominal;
      if (extended) {
        const bool half_step =
            TestScene(width, width / 80, false, 0, 1, 0.10, false, end_time, 0, {}, 0.7, false, 1, 0, 700, true);
        all_pass = all_pass && half_step;
      }
    }
    return all_pass ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-floor-grid-phase") {
    bool all_pass = true;
    for (const double width : {0.075, 0.0625}) {
      for (const double phase : {0.0, 0.25, 0.5}) {
        std::printf("sand floor grid offset in cell widths y=%.2f\n", phase);
        const bool pass = TestScene(width, width / 80, false, 0, 2, 0.10, false, 1.5, 0, {0, phase * width, 0});
        all_pass = all_pass && pass;
      }
    }
    return all_pass ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-grid-phase") {
    bool all_pass = true;
    for (const double width : {0.075, 0.0625}) {
      for (const fluidity::Vec3 phase : {fluidity::Vec3{}, {0.5, 0, 0}, {0.5, 0, 0.5}}) {
        std::printf("sand grid origin in cell widths x=%.1f z=%.1f\n", phase.x, phase.z);
        const bool pass =
            TestScene(width, width / 40, false, 0, 1, 0.10, false, 0.75, 0, {phase.x * width, 0, phase.z * width});
        all_pass = all_pass && pass;
      }
    }
    return all_pass ? 0 : 1;
  }
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
  if (!TestSandSurfaceFit() || !TestWallImpulseBudget() || !TestWallImpulseBudget(true) || !TestStoredVelocityResidual()
      || !TestStoredResidualMomentum() || !TestCompatibleFloorStressWork() || !TestUncutStressWork()
      || !TestRollbackOnUnsupportedDestination() || !TestRejectsInadmissibleInitialState()) {
    return 1;
  }
  const bool coarse_sand = TestScene(0.075, 0.001875, false);
  const bool coarse_snow = TestScene(0.1, 0.0025, true);
  const bool fine_snow = TestScene(0.075, 0.001875, true);
  return coarse_sand && coarse_snow && fine_snow ? 0 : 1;
}
