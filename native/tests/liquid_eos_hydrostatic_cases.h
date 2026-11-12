#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "fluidity/host/liquid_substep_3d.h"
#include "gtest/gtest.h"

namespace fluidity::pressure_test {
namespace hydrostatic {

constexpr double kGravity = 9.81;
constexpr double kDensity = 1000.0;
constexpr double kSoundSpeed = 10.0;
constexpr double kBulkModulus = kDensity * kSoundSpeed * kSoundSpeed;

struct HydrostaticState {
  double pressure;
  double volume_ratio;
};

// Integrate dp/d(depth) = rho0*g/J(p), with p=0 at the open surface.
inline HydrostaticState AtDepth(LiquidEosLaw law, double depth) {
  const double compression = kGravity * depth / (kSoundSpeed * kSoundSpeed);
  if (law == LiquidEosLaw::kHencky) {
    return {-kBulkModulus * std::log1p(-compression), 1.0 - compression};
  }
  const double log_shifted_pressure = std::log1p(6.0 * compression);
  return {(kBulkModulus / 7.0) * std::expm1((7.0 / 6.0) * log_shifted_pressure), std::exp(-log_shifted_pressure / 6.0)};
}

}  // namespace hydrostatic

struct HydrostaticRun {
  LiquidState3D state;
  std::size_t pressure_iterations;
};

inline absl::StatusOr<HydrostaticRun>
AdvanceHydrostaticColumn(LiquidSubstepScheduler3D& scheduler, LiquidStateCapacities3D capacities) {
  std::size_t iterations = 0;
  const auto advanced = scheduler.AdvanceFrame(
      0.02,
      {
          .context = &iterations,
          .observe =
              [](void* context, const LiquidState3D&, const LiquidStepDiagnostics3D& diagnostics, bool rollback) {
                if (!rollback && diagnostics.accepted_measurement) {
                  *static_cast<std::size_t*>(context) += diagnostics.accepted_measurement->pressure.iteration_count;
                }
                return absl::OkStatus();
              },
      }
  );
  if (!advanced.ok()) {
    ADD_FAILURE() << "stage=" << static_cast<int>(scheduler.diagnostics().failed_stage);
    return advanced;
  }
  auto state = CloneLiquidState3D(scheduler.committed(), capacities);
  if (!state.ok()) {
    return state.status();
  }
  return HydrostaticRun{std::move(*state), iterations};
}

template <typename Run>
void ExpectEosHydrostaticColumn(Run run) {
  using hydrostatic::AtDepth;
  using hydrostatic::kBulkModulus;
  using hydrostatic::kDensity;
  using hydrostatic::kGravity;
  using hydrostatic::kSoundSpeed;
  const auto start = std::chrono::steady_clock::now();
  std::size_t pressure_iterations = 0;
  double max_pressure_error = 0.0;
  double max_volume_ratio_error = 0.0;
  double max_particle_speed = 0.0;
  for (const auto law : {LiquidEosLaw::kHencky, LiquidEosLaw::kTait}) {
    for (const std::int32_t rows : {4, 8}) {
      SCOPED_TRACE(::testing::Message() << "law=" << static_cast<int>(law) << " rows=" << rows);
      const double h = 1.0 / rows;
      const std::int32_t width = rows / 2;
      const auto count = static_cast<std::size_t>(rows * width * width);
      const auto block_count = static_cast<std::size_t>(rows / 4);
      LiquidState3D initial;
      initial.geometry.cell_width = h;
      auto blocks = LiquidRecords3D<LiquidBlockState3D>::Create(block_count);
      auto cells = LiquidRecords3D<LiquidCellState3D>::Create(count);
      auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(count);
      ASSERT_TRUE(blocks.ok() && cells.ok() && particles.ok());
      initial.blocks = std::move(*blocks);
      initial.cells = std::move(*cells);
      initial.particles = std::move(*particles);
      ASSERT_TRUE(initial.blocks.Resize(block_count).ok());
      ASSERT_TRUE(initial.cells.Resize(count).ok());
      ASSERT_TRUE(initial.particles.Resize(count).ok());
      for (std::size_t block = 0; block < block_count; ++block) {
        initial.blocks[block].block_index = {0, static_cast<std::int32_t>(block), 0};
      }
      std::size_t index = 0;
      for (std::int32_t z = 0; z < width; ++z) {
        for (std::int32_t y = 0; y < rows; ++y) {
          for (std::int32_t x = 0; x < width; ++x) {
            const Vec3 center{(x + 0.5) * h, (y + 0.5) * h, (z + 0.5) * h};
            const auto reference = AtDepth(law, 1.0 - center.y);
            const double reference_volume = h * h * h / reference.volume_ratio;
            initial.cells[index] = {{x, y, z}, LiquidCellOccupancy3D::kLiquid, reference.pressure};
            initial.particles[index] = {
                .id = index + 1,
                .world_position = center,
                .mass = kDensity * reference_volume,
                .reference_volume = reference_volume,
                .volume_ratio = reference.volume_ratio,
                .pic_initialized = true,
            };
            ++index;
          }
        }
      }
      initial.next_particle_id = count + 1;
      LiquidSceneInputs3D scene;
      scene.field = LiquidHorizontalPlaneSurface3D{1.0};
      scene.obstacle = AxisAlignedLiquidTank3D{0.0, 0.5, 0.0, 0.0, 0.5, 0.0};
      scene.world_acceleration = {0.0, -kGravity, 0.0};
      scene.eos = LiquidSceneEos3D{.reference_sound_speed = kSoundSpeed, .law = law};
      LiquidStepOptions3D options;
      options.capacities = {256, 128, 256, 24576, 1};
      const auto result = run(initial, scene, options);
      ASSERT_TRUE(result.ok()) << result.status();
      pressure_iterations += result->pressure_iterations;
      ASSERT_EQ(result->state.cells.size(), count);
      ASSERT_EQ(result->state.particles.size(), count);
      // The liquid-side free-surface mobility is one-sided. Its leading
      // pressure-drop error scales as rho0*g^2*h^2/(8*c^2).
      const double pressure_limit = kDensity * kGravity * kGravity * h * h / (4.0 * kSoundSpeed * kSoundSpeed) + 0.1;
      for (const auto& cell : result->state.cells.values()) {
        const auto reference = AtDepth(law, 1.0 - (cell.cell_index.y + 0.5) * h);
        const double error = std::abs(cell.previous_pressure - reference.pressure);
        EXPECT_LE(error, pressure_limit);
        max_pressure_error = std::max(max_pressure_error, error);
      }
      for (const auto& particle : result->state.particles.values()) {
        const auto reference = AtDepth(law, 1.0 - particle.world_position.y);
        const double error = std::abs(particle.volume_ratio - reference.volume_ratio);
        EXPECT_LE(error, 2.0 * pressure_limit / kBulkModulus);
        max_volume_ratio_error = std::max(max_volume_ratio_error, error);
        const auto velocity = particle.world_velocity;
        const double speed = std::hypot(velocity.x, velocity.y, velocity.z);
        EXPECT_LE(speed, 0.02 * kGravity * kGravity * h / (kSoundSpeed * kSoundSpeed));
        max_particle_speed = std::max(max_particle_speed, speed);
      }
    }
  }
  const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
  ::testing::Test::RecordProperty("hydrostatic_pressure_iterations", static_cast<int>(pressure_iterations));
  ::testing::Test::RecordProperty("hydrostatic_wall_milliseconds", std::to_string(elapsed.count()));
  ::testing::Test::RecordProperty("hydrostatic_max_particle_speed_m_s", std::to_string(max_particle_speed));
  ::testing::Test::RecordProperty("hydrostatic_max_pressure_error_pa", std::to_string(max_pressure_error));
  ::testing::Test::RecordProperty("hydrostatic_max_volume_ratio_error", std::to_string(max_volume_ratio_error));
}

}  // namespace fluidity::pressure_test
