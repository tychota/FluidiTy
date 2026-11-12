#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <utility>

#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity::curved_phase_test {

inline constexpr double kRadius = 0.8;
inline constexpr double kTension = 0.031;
inline constexpr double kExpectedCurvature = 2.0 / kRadius;
inline constexpr double kStepDuration = 0.001;

struct Fixture {
  LiquidState3D initial;
  LiquidSceneInputs3D scene;
  LiquidStepOptions3D options;
};

inline absl::StatusOr<Fixture> MakeFixture() {
  constexpr std::size_t kShellDonors = 1000;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(kShellDonors + 1);
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resized = particles->Resize(kShellDonors + 1);
  if (!resized.ok()) {
    return resized;
  }
  LiquidState3D initial;
  initial.particles = std::move(*particles);
  initial.next_particle_id = kShellDonors + 2;
  for (std::size_t ordinal = 0; ordinal < initial.particles.size(); ++ordinal) {
    initial.particles[ordinal].id = ordinal + 1;
    initial.particles[ordinal].pic_initialized = true;
  }
  initial.geometry.cell_width = 0.5;
  initial.particles[0].world_position = {};
  initial.particles[0].mass = 1000;
  initial.particles[0].phase_id = 4;
  initial.particles[0].constituent_count = 1;
  initial.particles[0].constituents[0] = {8, 1};
  constexpr double kGoldenAngle = std::numbers::pi * (3.0 - 2.2360679774997896964);
  for (std::size_t index = 0; index < kShellDonors; ++index) {
    const double latitude = 1.0 - (2.0 * static_cast<double>(index) + 1.0) / kShellDonors;
    const double azimuth = static_cast<double>(index) * kGoldenAngle;
    const double radial = std::sqrt(1.0 - latitude * latitude);
    auto& particle = initial.particles[index + 1];
    particle.world_position = {
        2 * kRadius * radial * std::cos(azimuth),
        2 * kRadius * latitude,
        2 * kRadius * radial * std::sin(azimuth),
    };
    particle.reference_volume = 0.01;
    particle.mass = 820 * particle.reference_volume;
    particle.phase_id = 5;
    particle.constituent_count = 1;
    particle.constituents[0] = {25, 1};
  }
  static constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000}, {25, 5, 820}}};
  static constexpr std::array<LiquidPhasePair3D, 1> kPairs{{{4, 5, kTension}}};
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{2};
  scene.obstacle = AxisAlignedLiquidTank3D{-2, 2, -2, -2, 2, 0};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  scene.phase_pairs = kPairs;
  LiquidStepOptions3D options;
  options.capacities = {kShellDonors + 1, 64, 1024, 12288, 1};
  return Fixture{std::move(initial), scene, options};
}

inline std::pair<double, std::size_t> CorePressureSum(const LiquidState3D& state) {
  double pressure_sum = 0.0;
  std::size_t cell_count = 0;
  for (const auto& cell : state.cells.values()) {
    const auto index = cell.cell_index;
    if (index.x >= -1 && index.x <= 0 && index.y >= -1 && index.y <= 0 && index.z >= -1 && index.z <= 0) {
      pressure_sum += cell.previous_pressure;
      ++cell_count;
    }
  }
  return {pressure_sum, cell_count};
}

}  // namespace fluidity::curved_phase_test
