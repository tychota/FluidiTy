#pragma once

#include <cmath>
#include <utility>

#include "fluidity/host/liquid_substep_3d.h"
#include "gtest/gtest.h"

namespace fluidity::viscous_tank_test {

inline constexpr double kFillHeight = 2.65;
inline constexpr double kGravity = 9.81;
inline constexpr double kDensity = 1000.0;
inline constexpr double kStepDuration = 0.01;

struct FractionalFillFixture {
  LiquidState3D initial;
  LiquidSceneInputs3D scene;
  LiquidStepOptions3D options;
};

inline absl::StatusOr<FractionalFillFixture> FractionalFill() {
  FractionalFillFixture fixture;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(27);
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resize = particles->Resize(27);
  if (!resize.ok()) {
    return resize;
  }
  std::size_t ordinal = 0;
  for (int z = 0; z < 3; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        const double volume = y == 2 ? kFillHeight - 2.0 : 1.0;
        (*particles)[ordinal] = {
            .id = ordinal + 1,
            .world_position = {x + 0.5, y + 0.5, z + 0.5},
            .mass = kDensity * volume,
            .reference_volume = volume,
        };
        ++ordinal;
      }
    }
  }
  fixture.initial.next_particle_id = ordinal + 1;
  fixture.initial.particles = std::move(*particles);
  fixture.scene.field = LiquidHorizontalPlaneSurface3D{kFillHeight};
  fixture.scene.obstacle = AxisAlignedLiquidTank3D{0.0, 3.0, 0.0, 0.0, 3.0, 0.0};
  fixture.scene.world_acceleration = {0.0, -kGravity, 0.0};
  fixture.scene.liquid_density = kDensity;
  fixture.scene.dynamic_viscosity = 10.0;
  fixture.options.capacities = {64, 64, 128, 12288, 1};
  return fixture;
}

inline absl::StatusOr<FractionalFillFixture> CartesianCut() {
  auto fixture = FractionalFill();
  if (!fixture.ok()) {
    return fixture.status();
  }
  auto& tank = std::get<AxisAlignedLiquidTank3D>(fixture->scene.obstacle);
  tank.start_low_x = 0.25;
  for (auto& particle : fixture->initial.particles.values()) {
    if (particle.world_position.x < 1.0) {
      particle.reference_volume *= 0.75;
      particle.mass *= 0.75;
    }
  }
  return fixture;
}

inline void ExpectHydrostaticRest(const LiquidSubstepScheduler3D& scheduler) {
  ASSERT_EQ(scheduler.committed().cells.size(), 27);
  for (const auto& cell : scheduler.committed().cells.values()) {
    const double height = cell.cell_index.y + 0.5;
    EXPECT_NEAR(cell.previous_pressure, kDensity * kGravity * (kFillHeight - height), 0.03);
  }
  for (const auto& particle : scheduler.committed().particles.values()) {
    const auto velocity = particle.world_velocity;
    EXPECT_LT(std::hypot(velocity.x, velocity.y, velocity.z), 1e-6);
  }
  const auto& accepted = scheduler.diagnostics().accepted_measurement;
  ASSERT_TRUE(accepted.has_value());
  EXPECT_LT(accepted->viscous_max_momentum_residual_kg_m_per_s.value_or(1.0), 1e-5);
}

}  // namespace fluidity::viscous_tank_test
