#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <string_view>
#include <vector>

#include "../src/dry_solid_step_3d_internal.h"
#include "fluidity/host/contact_face_bodies_3d.h"
#include "fluidity/host/dry_solid_step_3d.h"
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
    double sand_end_time = 0.75
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
#if defined(FLUIDITY_M5_CUDA_SCENES)
    const auto status = fluidity::CudaAdvanceDrySolid3D(
        *grid,
        material,
        settings,
        {0x1p-42, 0x1p-38, std::max<std::size_t>(128, particles.size())},
        particles,
        &budget
    );
#else
    const auto status = fluidity::AdvanceDrySolid3D(*grid, material, settings, particles, &budget);
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
  if (particle_energy_blend == 0
      && std::abs(vertical_residual) > total_budget.omitted_stress_impulse_l1 + kMomentumRoundoffTolerance) {
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
  if (argc != 1) {
    return 2;
  }
  const auto capacity = fluidity::ContactFaceDepositStatus(fluidity::ContactFaceDepositStatus3D::kBodyCapacity);
  if (capacity.code() != absl::StatusCode::kResourceExhausted || capacity.message() != "E-CONTACT-BODY-CAPACITY") {
    return 1;
  }
  if (!TestSandSurfaceFit() || !TestWallImpulseBudget() || !TestStoredVelocityResidual() || !TestUncutStressWork()
      || !TestRollbackOnUnsupportedDestination() || !TestRejectsInadmissibleInitialState()) {
    return 1;
  }
  const bool coarse_sand = TestScene(0.075, 0.001875, false);
  const bool coarse_snow = TestScene(0.1, 0.0025, true);
  const bool fine_snow = TestScene(0.075, 0.001875, true);
  return coarse_sand && coarse_snow && fine_snow ? 0 : 1;
}
