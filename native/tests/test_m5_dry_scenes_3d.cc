#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
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
    bool compatible_floor = false,
    double symmetric_affine_decay_per_second = 0,
    std::string_view affine_decay_region = "all",
    bool trace_early_contact = false,
    bool retry_motion = false
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
  std::vector<double> previous_height(particles.size());
  double floor_resident_plastic_path = 0;
  double floor_arrival_plastic_path = 0;
  double floor_departure_plastic_path = 0;
  double above_floor_plastic_path = 0;
  constexpr double kFloorPositionToleranceInCells = 1e-12;
  const double floor_tolerance = kFloorPositionToleranceInCells * width;
  double final_log_plastic_change = 0;
  double final_absolute_log_plastic_change = 0;
  double floor_layer_absolute_log_plastic_change = 0;
  double reset_impulse_y = 0;
  const bool extended_column = !snow && !release_cone && sand_end_time > 0.75;
  fluidity::DrySolidStepBudget3D total_budget{};
  int step = 0;
  int retries = 0;
  constexpr std::array early_sample_times{0.10, 0.15, 0.20, 0.25, 0.35, 0.50};
  std::size_t early_sample = 0;
  for (double t = 0; t < time; ++step) {
    double phase_end =
        snow ? (t < 0.23 ? 0.23 : (t < 0.35 ? 0.35 : time)) : (t < settling_sample_time ? settling_sample_time : time);
    if (extended_column && t < 0.75) {
      // Replay the original two sampling boundaries before extending the run.
      phase_end = t < 0.70 ? 0.70 : 0.75;
    }
    double next_time = t + duration >= phase_end - 1e-12 ? phase_end : t + duration;
    double step_duration = next_time - t;
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
        previous_height[ordinal] = particles[ordinal].world_position.y;
      }
    }
    fluidity::DrySolidStepBudget3D budget{};
    const int mass_bits = quantized_mass_bits == 0 ? 42 : quantized_mass_bits;
    const fluidity::MacDepositQuantization3D quantization{
        std::ldexp(1.0, -mass_bits),
        std::ldexp(1.0, 4 - mass_bits),
        std::max<std::size_t>(128, particles.size()),
    };
    const auto attempt = [&]() {
#if defined(FLUIDITY_M5_CUDA_SCENES)
      return fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, quantization, particles, &budget);
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
      return quantized_mass_bits == 0 ? fluidity::AdvanceDrySolid3D(*grid, material, settings, particles, &budget)
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
    };
    auto status = attempt();
    for (int retry = 0; retry_motion && retry < 8 && fluidity::internal::DrySolidDurationExceeded3D(status); ++retry) {
      ++retries;
      std::printf("sand retry t=%.9g dt=%.9g status=%s\n", t, step_duration, status.ToString().c_str());
      next_time = std::midpoint(t, next_time);
      step_duration = next_time - t;
      settings.duration = step_duration;
      status = attempt();
    }
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
    if (symmetric_affine_decay_per_second > 0) {
      // This diagnostic decays only the symmetric part of the next deposit's APIC fit.
      const double decay = -0.5 * std::expm1(-symmetric_affine_decay_per_second * step_duration);
      constexpr std::array rows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
      constexpr std::array coordinates{&fluidity::Vec3::x, &fluidity::Vec3::y, &fluidity::Vec3::z};
      for (auto& particle : particles) {
        const bool floor_layer = particle.world_position.y <= spacing;
        if ((affine_decay_region == "floor" && !floor_layer) || (affine_decay_region == "interior" && floor_layer)) {
          continue;
        }
        const Mat3 previous = particle.velocity_slopes;
        for (std::size_t row = 0; row < rows.size(); ++row) {
          for (std::size_t column = 0; column < coordinates.size(); ++column) {
            const double symmetric_sum =
                (previous.*rows[row]).*coordinates[column] + (previous.*rows[column]).*coordinates[row];
            (particle.velocity_slopes.*rows[row]).*coordinates[column] -= decay * symmetric_sum;
          }
        }
      }
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
          const double path = std::abs(change);
          final_absolute_log_plastic_change += path;
          const bool started_on_floor = previous_height[ordinal] <= floor_tolerance;
          const bool ended_on_floor = particle.world_position.y <= floor_tolerance;
          if (started_on_floor && ended_on_floor) {
            floor_resident_plastic_path += path;
          } else if (ended_on_floor) {
            floor_arrival_plastic_path += path;
          } else if (started_on_floor) {
            floor_departure_plastic_path += path;
          } else {
            above_floor_plastic_path += path;
          }
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
    total_budget.particle_wall_kinetic_energy_change += budget.particle_wall_kinetic_energy_change;
    if (trace_early_contact && early_sample < early_sample_times.size()
        && next_time >= early_sample_times[early_sample]) {
      double height = 0;
      double radius = 0;
      double energy = 0;
      for (const auto& particle : particles) {
        height = std::max(height, particle.world_position.y);
        radius = std::max(radius, std::hypot(particle.world_position.x, particle.world_position.z));
        const auto velocity = particle.world_velocity;
        energy += 0.5 * particle.mass * (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
      }
      std::printf(
          "sand trace t=%.8f height=%.8f radius=%.8f kinetic_J=%.8f grid_wall_Ns=%.8f swept_wall_Ns=%.8f "
          "contacts=%zu\n",
          next_time,
          height,
          radius,
          energy,
          total_budget.grid_wall_impulse_y,
          total_budget.wall_impulse_on_particles.y,
          total_budget.particle_wall_contacts
      );
      ++early_sample;
    }
    t = next_time;
  }
  if (retry_motion) {
    std::printf("sand accepted_steps=%d retries=%d\n", step, retries);
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
    std::printf(
        "compatible floor cut stress work_J=%.9g wall particle kinetic change_J=%.9g\n",
        total_budget.cut_stress_work,
        total_budget.particle_wall_kinetic_energy_change
    );
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
  if (extended_column) {
    const double particle_count = static_cast<double>(particles.size());
    std::printf(
        "sand final plastic path resident=%.9g arrival=%.9g departure=%.9g above=%.9g\n",
        floor_resident_plastic_path / particle_count,
        floor_arrival_plastic_path / particle_count,
        floor_departure_plastic_path / particle_count,
        above_floor_plastic_path / particle_count
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
  std::fflush(stdout);
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

bool TestDryGridWallFriction() {
  using fluidity::internal::SolveDryGridWallFriction3D;
  constexpr fluidity::Vec3 velocity{3, 0, 4};
  constexpr fluidity::Vec3 mobility{2, 0, 4};
  const auto sticking = SolveDryGridWallFriction3D(velocity, mobility, 2);
  const auto frictionless = SolveDryGridWallFriction3D(velocity, mobility, 0);
  const double cap = std::hypot(0.75, 2.0 / 3);
  const auto sliding = SolveDryGridWallFriction3D(velocity, mobility, cap);
  if (!sticking.ok() || !frictionless.ok() || !sliding.ok()) {
    return false;
  }
  const double work = sliding->x * (velocity.x + 0.5 * mobility.x * sliding->x)
      + sliding->z * (velocity.z + 0.5 * mobility.z * sliding->z);
  return sticking->x == -1.5 && sticking->z == -1 && frictionless->x == 0 && frictionless->z == 0 && sliding->y == 0
      && std::abs(sliding->x + 0.75) < 1e-12 && std::abs(sliding->z + 2.0 / 3) < 1e-12
      && std::abs(std::hypot(sliding->x, sliding->z) - cap) < 1e-12 && std::abs(work + 3.465277777777778) < 1e-12
      && !SolveDryGridWallFriction3D(velocity, {0, 0, 4}, cap).ok();
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
  // The reflected field has L_yy = -1/s and zero shear at this particle.
  // This compression stays elastic at the chosen friction angle. The
  // swept velocity still removes inward normal motion.
  const auto& elastic = particle.elastic_deformation;
  const double determinant = elastic.x_row.x * elastic.y_row.y * elastic.z_row.z;
  const double expected_determinant = compatible_floor ? 0.99 : 1.0;
  return status.ok() && budget.particle_wall_contacts == 1 && std::abs(budget.wall_impulse_on_particles.x + 0.5) < 1e-9
      && std::abs(budget.wall_impulse_on_particles.y - 1) < 1e-9 && budget.omitted_stress_impulse_l1 == 0
      && std::abs(budget.particle_wall_kinetic_energy_change + 0.875) < 1e-9 && budget.grid_wall_impulse_y == 0
      && budget.cut_stress_work == 0 && budget.granular_projected_particles == 0
      && std::abs(determinant - expected_determinant) < 1e-10 && std::abs(elastic.x_row.y) < 1e-10
      && budget.granular_plastic_volume_changes == 0;
}

bool TestGridWallTractionBudget() {
  std::vector<BlockIndex3D> blocks;
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 0; ++y) {
      for (int x = -1; x <= 1; ++x) {
        blocks.push_back({x, y, z});
      }
    }
  }
  const auto grid = SparseMacGrid3D::Create({{}, 1}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  DrySolidParticle3D particle{
      .world_position = {2.25, 0, 2.25},
      .world_velocity = {1, 0, 0},
      .mass = 1,
      .reference_volume = 0.001,
      .elastic_deformation = {{0.95, 0, 0}, {0, 0.95, 0}, {0, 0, 0.95}},
  };
  const DrySolidMaterial3D material = fluidity::GranularParameters3D{2, 3, 0.5, 0};
  const DrySolidStep3D settings{
      .duration = 0.001,
      .floor_friction = 0.5,
      .acoustic_cfl = 0.25,
      .compatible_floor = true,
  };
  fluidity::DrySolidStepBudget3D budget{};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1}, &budget);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1}, &budget);
#endif
  if (!status.ok() || budget.grid_wall_impulse_y <= 0 || budget.grid_wall_friction_work >= 0) {
    std::fprintf(
        stderr,
        "grid traction status=%s normal=%.9g work=%.9g\n",
        status.ToString().c_str(),
        budget.grid_wall_impulse_y,
        budget.grid_wall_friction_work
    );
    return false;
  }
  const auto impulse = budget.grid_wall_tangent_impulse;
  const double expected_impulse = settings.floor_friction * budget.grid_wall_impulse_y;
  return impulse.x < 0 && impulse.y == 0 && std::abs(impulse.z) < 1e-10
      && std::abs(std::hypot(impulse.x, impulse.z) - expected_impulse) < 1e-10
      && std::abs(budget.cut_stress_work) < 1e-12;
}

bool TestUnsupportedFloorHasNoReaction() {
  constexpr std::array blocks{BlockIndex3D{0, -1, 0}, BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  DrySolidParticle3D particle{
      .world_position = {2.25, 0.2, 2.25},
      .mass = 1,
      .reference_volume = 0.001,
      .elastic_deformation = {{0.95, 0, 0}, {0, 0.95, 0}, {0, 0, 0.95}},
  };
  const DrySolidMaterial3D material = fluidity::GranularParameters3D{2, 3, 0.5, 0};
  const DrySolidStep3D settings{.duration = 0.001, .acoustic_cfl = 0.25, .compatible_floor = true};
  fluidity::DrySolidStepBudget3D budget{};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1}, &budget);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1}, &budget);
#endif
  return status.ok() && budget.proposed_stress_impulse_l1 > 0 && budget.particle_wall_contacts == 0
      && std::abs(budget.grid_wall_impulse_y) < 1e-12 && std::abs(particle.world_velocity.y) < 1e-10;
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
      .particle_wall_kinetic_energy_change = -3,
  };
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto status =
      fluidity::CudaAdvanceDrySolid3D(*grid, material, settings, {0x1p-42, 0x1p-38, 128}, {&particle, 1}, &budget);
#else
  const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, {&particle, 1}, &budget);
#endif
  if (budget.particle_wall_contacts != 42 || budget.granular_projected_particles != 17
      || budget.granular_plastic_volume_changes != 9 || budget.particle_wall_kinetic_energy_change != -3
      || status.code() != absl::StatusCode::kFailedPrecondition || particle.world_position.x != 2.4
      || particle.world_velocity.x != 1 || particle.log_plastic_jacobian != 0) {
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
  const auto independent_velocities = velocities;
  auto independent_histories = gather_inputs;
  independent_histories.new_history.face_velocities = independent_velocities;
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto device_deposits = fluidity::CudaDepositCompatibleMacQuanta3D(deposit_inputs);
  const auto device_gather = fluidity::CudaGatherLiquidBoundaryParticles3D(gather_inputs);
  const auto independent_device_gather = fluidity::CudaGatherLiquidBoundaryParticles3D(independent_histories);
  if (!device_deposits.ok() || !device_gather.ok() || !independent_device_gather.ok()) {
    return false;
  }
  const auto& deposits = device_deposits->deposits;
  const auto& gathered = device_gather->gathered;
  const auto& independent = independent_device_gather->gathered;
#else
  const auto host_deposits = fluidity::DepositCompatibleMacQuanta3D(deposit_inputs);
  const auto host_gather = fluidity::GatherLiquidBoundaryParticles3D(gather_inputs);
  const auto independent_host_gather = fluidity::GatherLiquidBoundaryParticles3D(independent_histories);
  if (!host_deposits.ok() || !host_gather.ok() || !independent_host_gather.ok()) {
    return false;
  }
  const auto& deposits = *host_deposits;
  const auto& gathered = *host_gather;
  const auto& independent = *independent_host_gather;
#endif
  const auto same_vector = [](fluidity::Vec3 first, fluidity::Vec3 second) {
    return first.x == second.x && first.y == second.y && first.z == second.z;
  };
  const auto same_matrix = [&](const Mat3& first, const Mat3& second) {
    return same_vector(first.x_row, second.x_row) && same_vector(first.y_row, second.y_row)
        && same_vector(first.z_row, second.z_row);
  };
  const auto& aliased_particle = gathered.particles[0];
  const auto& independent_particle = independent.particles[0];
  if (!same_vector(aliased_particle.new_pic_velocity, independent_particle.new_pic_velocity)
      || !same_vector(aliased_particle.grid_velocity_increment, independent_particle.grid_velocity_increment)
      || !same_vector(
          aliased_particle.particle_minus_old_pic_velocity,
          independent_particle.particle_minus_old_pic_velocity
      )
      || !same_matrix(aliased_particle.fitted_velocity_slopes, independent_particle.fitted_velocity_slopes)
      || !same_matrix(aliased_particle.velocity_gradient, independent_particle.velocity_gradient)
      || aliased_particle.affine_enabled != independent_particle.affine_enabled
      || aliased_particle.old_pic_fallback != independent_particle.old_pic_fallback
      || aliased_particle.new_pic_fallback != independent_particle.new_pic_fallback
      || aliased_particle.low_mass_fallback != independent_particle.low_mass_fallback) {
    return false;
  }
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
  std::vector<fluidity::LiquidPrescribedWallSample3D> stationary_walls(view.face_count());
  for (std::size_t slot = 0; slot < view.face_count(); ++slot) {
    if (!tags[slot].structurally_eligible) {
      stationary_walls[slot] = {.available = true, .unit_normal = {0, 1, 0}};
    }
    if (view.FaceIndex(slot)->axis == fluidity::FaceNormalAxis::kY) {
      velocities[slot] += 0.8;
    }
  }
  auto stationary_inputs = gather_inputs;
  stationary_inputs.old_history.wall_samples = stationary_walls;
  stationary_inputs.new_history.wall_samples = stationary_walls;
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto stationary = fluidity::CudaGatherLiquidBoundaryParticles3D(stationary_inputs);
  if (!stationary.ok()) {
    return false;
  }
  const auto& stationary_gradient = stationary->gathered.particles.front().velocity_gradient;
#else
  const auto stationary = fluidity::GatherLiquidBoundaryParticles3D(stationary_inputs);
  if (!stationary.ok()) {
    return false;
  }
  const auto& stationary_gradient = stationary->particles.front().velocity_gradient;
#endif
  std::vector<double> paired_force(view.face_count());
  const auto full_stress =
      fluidity::internal::AccumulateDrySolidStress3D(*grid, position, volume, stress, paired_force);
  if (!full_stress.ok()) {
    return false;
  }
  const auto wall_force =
      fluidity::internal::ScatterStationaryFloorGhostTranspose3D(view, validity, tags, paired_force);
  if (!wall_force.ok()) {
    return false;
  }
  double paired_grid_work = 0;
  for (std::size_t slot = 0; slot < paired_force.size(); ++slot) {
    paired_grid_work += duration * paired_force[slot] * velocities[slot];
  }
  const double paired_particle_work = duration * volume
      * ((10 * stationary_gradient.x_row.x + 2 * stationary_gradient.x_row.y + 3 * stationary_gradient.x_row.z)
         + (2 * stationary_gradient.y_row.x + 20 * stationary_gradient.y_row.y + 4 * stationary_gradient.y_row.z)
         + (3 * stationary_gradient.z_row.x + 4 * stationary_gradient.z_row.y - 30 * stationary_gradient.z_row.z));
  const double transpose_work_residual = paired_grid_work + paired_particle_work;
  std::printf("stationary ghost transpose work residual_J=%.9g\n", transpose_work_residual);
  if (std::abs(transpose_work_residual) >= 1e-11 || std::abs(*wall_force) < 1e-3) {
    return false;
  }
  std::vector<double> weighted_force(view.face_count());
  const auto weighted_full =
      fluidity::internal::AccumulateDrySolidStress3D(*grid, position, volume, stress, weighted_force);
  if (!weighted_full.ok()) {
    return false;
  }
  std::vector<fluidity::MacFaceDeposit3D> masses(view.face_count());
  std::vector<std::uint8_t> needed(view.face_count());
  for (std::size_t slot = 0; slot < view.face_count(); ++slot) {
    masses[slot].deposited_mass = mask[slot] ? (slot % 3 == 0 ? 1e-8 : 1.0) : 0.0;
    needed[slot] = weighted_force[slot] != 0;
  }
  std::vector<fluidity::internal::StationaryFloorGhostRow3D> weighted_rows;
  const auto weighted_wall = fluidity::internal::ScatterStationaryFloorGhostTranspose3D(
      view,
      validity,
      tags,
      weighted_force,
      masses,
      needed,
      &weighted_rows
  );
  if (!weighted_wall.ok() || weighted_rows.empty()) {
    return false;
  }
  for (const auto& row : weighted_rows) {
    double reconstructed = 0;
    for (const auto& term : row.terms) {
      reconstructed = std::fma(term.coefficient, velocities[term.donor_slot], reconstructed);
    }
    const auto face = view.FaceIndex(row.target_slot);
    if (!face.ok()) {
      return false;
    }
    const double expected = face->axis == fluidity::FaceNormalAxis::kY ? 0.0 : velocities[row.target_slot];
    if (row.fallback || std::abs(reconstructed - expected) > 1e-10) {
      return false;
    }
  }
  double weighted_grid_work = 0;
  for (std::size_t slot = 0; slot < weighted_force.size(); ++slot) {
    weighted_grid_work += duration * weighted_force[slot] * velocities[slot];
  }
  const double weighted_work_residual = weighted_grid_work + paired_particle_work;
  std::printf("mass-weighted floor transpose work residual_J=%.9g\n", weighted_work_residual);
  if (std::abs(weighted_work_residual) >= 1e-10) {
    return false;
  }
  constexpr double expected_particle_work = -2.125;
  return std::abs(particle_work - expected_particle_work) < 1e-11
      && std::abs(grid_work - oracle->grid_stress_work) < 1e-11
      && std::abs(particle_work + grid_work - oracle->cut_work) < 1e-11 && std::abs(oracle->cut_work) > 1e-3
      && std::hypot(oracle->stress_impulse.x, oracle->stress_impulse.y, oracle->stress_impulse.z) < 1e-11;
}

bool TestBoundedFloorDonorFallback() {
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}, BlockIndex3D{1, 0, 0}};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  const auto target = grid->FaceSlot({fluidity::FaceNormalAxis::kX, {0, 0, 0}});
  const auto donor = grid->FaceSlot({fluidity::FaceNormalAxis::kX, {3, 0, 0}});
  const auto distant = grid->FaceSlot({fluidity::FaceNormalAxis::kX, {5, 0, 0}});
  if (!target.ok() || !donor.ok() || !distant.ok()) {
    return false;
  }
  const auto count = grid->face_count();
  std::vector<fluidity::LiquidFaceValidity3D> validity(count, fluidity::LiquidFaceValidity3D::kMissing);
  const std::vector<fluidity::LiquidBoundaryFaceTag3D> tags(count);
  std::vector<double> forces(count);
  std::vector<std::uint8_t> needed(count);
  std::vector<fluidity::internal::StationaryFloorGhostRow3D> rows;
  validity[*donor] = fluidity::LiquidFaceValidity3D::kValid;
  forces[*target] = 2;
  needed[*target] = 1;
  const auto reaction = fluidity::internal::ScatterStationaryFloorGhostTranspose3D(
      grid->view(),
      validity,
      tags,
      forces,
      {},
      needed,
      &rows
  );
  if (!reaction.ok() || *reaction != 0 || forces[*target] != 0 || forces[*donor] != 2 || rows.size() != 1
      || !rows.front().fallback || rows.front().terms.size() != 1 || rows.front().terms.front().donor_slot != *donor
      || rows.front().terms.front().coefficient != 1) {
    return false;
  }
  validity[*donor] = fluidity::LiquidFaceValidity3D::kMissing;
  validity[*distant] = fluidity::LiquidFaceValidity3D::kValid;
  forces[*donor] = 0;
  forces[*target] = 2;
  const auto prior = forces;
  const auto rejected = fluidity::internal::ScatterStationaryFloorGhostTranspose3D(
      grid->view(),
      validity,
      tags,
      forces,
      {},
      needed,
      &rows
  );
  return rejected.status().code() == absl::StatusCode::kFailedPrecondition && forces == prior;
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
  if (argc == 2 && std::string_view(argv[1]) == "--q-sand-motion-retry") {
    return TestScene(
               0.0625,
               0.0625 / 40,
               false,
               0,
               1,
               0.10,
               false,
               3.0,
               0,
               {},
               0.7,
               false,
               1,
               0,
               700,
               true,
               0,
               "all",
               false,
               true
           )
        ? 0
        : 1;
  }
  if (argc == 4 && std::string_view(argv[1]) == "--q-sand-contact-trace") {
    const std::string_view grid_name = argv[2];
    const std::string_view step_name = argv[3];
    if ((grid_name != "coarse" && grid_name != "fine") || (step_name != "nominal" && step_name != "half")) {
      return 2;
    }
    const double width = grid_name == "coarse" ? 0.075 : 0.0625;
    const double duration = width / (step_name == "half" ? 80 : 40);
    return TestScene(width, duration, false, 0, 1, 0.10, false, 1.5, 0, {}, 0.7, false, 1, 0, 700, true, 0, "all", true)
        ? 0
        : 1;
  }
  if (argc == 4 && std::string_view(argv[1]) == "--q-sand-reflected-timestep") {
    const std::string_view grid_name = argv[2];
    const std::string_view timestep_name = argv[3];
    if ((grid_name != "coarse" && grid_name != "fine") || (timestep_name != "quarter" && timestep_name != "eighth")) {
      return 2;
    }
    const double width = grid_name == "coarse" ? 0.075 : 0.0625;
    const double duration = width / (timestep_name == "quarter" ? 160 : 320);
    return TestScene(width, duration, false, 0, 1, 0.10, false, 3.0, 0, {}, 0.7, false, 1, 0, 700, true) ? 0 : 1;
  }
  if (argc >= 4 && argc <= 6 && std::string_view(argv[1]) == "--q-sand-symmetric-affine-decay") {
    const std::string_view rate_name = argv[2];
    if (rate_name != "0" && rate_name != "2" && rate_name != "8" && rate_name != "32") {
      return 2;
    }
    double rate = 0;
    if (rate_name == "2") {
      rate = 2;
    } else if (rate_name == "8") {
      rate = 8;
    } else if (rate_name == "32") {
      rate = 32;
    }
    const std::string_view region = argc >= 5 ? argv[4] : "all";
    if (region != "all" && region != "floor" && region != "interior") {
      return 2;
    }
    if (argc == 6 && std::string_view(argv[5]) != "3") {
      return 2;
    }
    const double end_time = argc == 6 ? 3.0 : 1.5;
    const std::string_view selection = argv[3];
    const bool coarse = selection == "coarse" || selection == "coarse-half";
    const bool fine = selection == "fine" || selection == "fine-half";
    if (!coarse && !fine) {
      return 2;
    }
    const double width = coarse ? 0.075 : 0.0625;
    const double duration = width / (selection.ends_with("-half") ? 80 : 40);
    std::printf(
        "symmetric APIC decay in inverse seconds=%.0f region=%.*s\n",
        rate,
        static_cast<int>(region.size()),
        region.data()
    );
    return TestScene(
               width,
               duration,
               false,
               0,
               1,
               0.10,
               false,
               end_time,
               0,
               {},
               0.7,
               false,
               1,
               0,
               700,
               true,
               rate,
               region
           )
        ? 0
        : 1;
  }
  if (argc >= 2
      && (std::string_view(argv[1]) == "--q-sand-compatible-floor"
          || std::string_view(argv[1]) == "--q-sand-compatible-floor-long-settle")) {
    const bool extended = std::string_view(argv[1]) == "--q-sand-compatible-floor-long-settle";
    if (argc > 3 || (argc == 3 && !extended)) {
      return 2;
    }
    const std::string_view selected = argc == 3 ? argv[2] : "";
    struct FloorCase {
      std::string_view name;
      double cell_width;
      int steps_per_cell_width;
    };
    constexpr std::array cases{
        FloorCase{"coarse", 0.075, 40},
        FloorCase{"coarse-half", 0.075, 80},
        FloorCase{"fine", 0.0625, 40},
        FloorCase{"fine-half", 0.0625, 80},
    };
    bool ran_case = false;
    bool all_pass = true;
    for (const auto& floor_case : cases) {
      if ((!extended && floor_case.steps_per_cell_width == 80) || (!selected.empty() && selected != floor_case.name)) {
        continue;
      }
      ran_case = true;
      const double width = floor_case.cell_width;
      const bool pass = TestScene(
          width,
          width / floor_case.steps_per_cell_width,
          false,
          0,
          1,
          0.10,
          false,
          extended ? 1.5 : 0.75,
          0,
          {},
          0.7,
          false,
          1,
          0,
          700,
          true
      );
      all_pass = all_pass && pass;
    }
    return ran_case ? (all_pass ? 0 : 1) : 2;
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
  if (!TestSandSurfaceFit() || !TestDryGridWallFriction() || !TestWallImpulseBudget() || !TestWallImpulseBudget(true)
      || !TestGridWallTractionBudget() || !TestUnsupportedFloorHasNoReaction() || !TestStoredVelocityResidual()
      || !TestStoredResidualMomentum() || !TestCompatibleFloorStressWork() || !TestBoundedFloorDonorFallback()
      || !TestUncutStressWork() || !TestRollbackOnUnsupportedDestination() || !TestRejectsInadmissibleInitialState()) {
    return 1;
  }
  const bool coarse_sand = TestScene(0.075, 0.001875, false);
  const bool coarse_snow = TestScene(0.1, 0.0025, true);
  const bool fine_snow = TestScene(0.075, 0.001875, true);
  return coarse_sand && coarse_snow && fine_snow ? 0 : 1;
}
