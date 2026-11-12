#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/dry_solid_step_3d_internal.h"
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

bool TestSnowCompaction(double width, double nominal_duration) {
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
  const DrySolidMaterial3D snow = fluidity::SnowParameters3D{1200, 700, 0.025, 0.01, 4};
  constexpr double kSpacing = 0.05;
  constexpr double kVolume = kSpacing * kSpacing * kSpacing;
  constexpr double kMass = 300 * kVolume;
  std::vector<DrySolidParticle3D> particles;
  for (int z = -1; z <= 1; ++z) {
    for (int y = 0; y < 4; ++y) {
      for (int x = -1; x <= 1; ++x) {
        particles.push_back({
            .world_position = {x * kSpacing, (y + 0.5) * kSpacing, z * kSpacing},
            .mass = kMass,
            .reference_volume = kVolume,
            .elastic_deformation = kIdentity,
        });
      }
    }
  }

  double wall_impulse_y = 0;
  double omitted_stress_impulse = 0;
  for (double time = 0; time < 0.45;) {
    const double phase_end = time < 0.23 ? 0.23 : (time < 0.35 ? 0.35 : 0.45);
    const double next_time = time + nominal_duration >= phase_end - 1e-12 ? phase_end : time + nominal_duration;
    DrySolidStep3D settings{
        .duration = next_time - time,
        .floor_friction = 0.7,
        .acoustic_cfl = 0.25,
    };
    if (time < 0.35) {
      const double speed = time < 0.23 ? -0.4 : 0.5;
      const double position = time < 0.23 ? 0.21 - 0.4 * time : 0.118 + 0.5 * (time - 0.23);
      settings.platen = fluidity::DrySolidPlaten3D{.start_y = position, .velocity_y = speed};
    }
    fluidity::DrySolidStepBudget3D budget{};
#if defined(FLUIDITY_M5_CUDA_SCENES)
    const auto status =
        fluidity::CudaAdvanceDrySolid3D(*grid, snow, settings, {0x1p-42, 0x1p-38, 128}, particles, &budget);
#else
    const auto status = fluidity::AdvanceDrySolid3D(*grid, snow, settings, particles, &budget);
#endif
    if (!status.ok()) {
      std::fprintf(stderr, "snow h=%.4f t=%.6f: %s\n", width, time, status.ToString().c_str());
      return false;
    }
    wall_impulse_y += budget.grid_wall_impulse_y + budget.wall_impulse_on_particles.y;
    omitted_stress_impulse += budget.omitted_stress_impulse_l1;
    time = next_time;
  }
  double maximum_height = 0;
  double log_plastic_sum = 0;
  double momentum_y = 0;
  for (const auto& particle : particles) {
    if (particle.world_position.y < 0 || particle.mass != kMass || particle.reference_volume != kVolume
        || !std::isfinite(particle.log_plastic_jacobian)) {
      return false;
    }
    maximum_height = std::max(maximum_height, particle.world_position.y);
    log_plastic_sum += particle.log_plastic_jacobian;
    momentum_y += particle.mass * particle.world_velocity.y;
  }
  const double mean_log_plastic = log_plastic_sum / particles.size();
  const double momentum_residual = momentum_y - wall_impulse_y;
  std::printf(
      "snow h=%.4f height=%.6f mean_log_Jpl=%.6f momentum_residual_Ns=%.9g\n",
      width,
      maximum_height,
      mean_log_plastic,
      momentum_residual
  );
  return std::abs(momentum_residual) <= omitted_stress_impulse + 1e-6 && mean_log_plastic > -0.48
      && mean_log_plastic < -0.34 && maximum_height > 0.11 && maximum_height < 0.15;
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

int main() {
  if (!TestUncutStressWork() || !TestRollbackOnUnsupportedDestination() || !TestRejectsInadmissibleInitialState()
      || !TestWallImpulseBudget() || !TestStoredVelocityResidual() || !TestStoredResidualMomentum()) {
    return 1;
  }
  const bool coarse = TestSnowCompaction(0.1, 0.0025);
  const bool fine = TestSnowCompaction(0.075, 0.001875);
  return coarse && fine ? 0 : 1;
}
