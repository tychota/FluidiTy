#include <cmath>
#include <limits>
#include <numbers>

#include "fluidity/host/liquid_cavity.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(LiquidCavity3D, SubmergedSphereBalancesHydrostaticPressureAndSignedCurvature) {
  constexpr double kRadius = 0.25;
  constexpr double kVolume = 4.0 * std::numbers::pi * kRadius * kRadius * kRadius / 3.0;
  constexpr double kAtmosphere = 101325.0;
  constexpr double kHydrostatic = 9810.0;
  constexpr double kSurfaceTension = 0.072;
  const auto inventory = InitializeSubmergedLiquidCavity3D(kVolume, kAtmosphere, kHydrostatic, kSurfaceTension);
  ASSERT_TRUE(inventory.ok()) << inventory.status();
  const double expected_pressure = kAtmosphere + kHydrostatic + 2.0 * kSurfaceTension / kRadius;
  EXPECT_NEAR(inventory->initial_absolute_pressure, expected_pressure, 1e-10);
  const auto state =
      EvaluateLiquidCavity3D(*inventory, kVolume, kAtmosphere, 1000.0, kSurfaceTension, -2.0 / kRadius, 0.12);
  ASSERT_TRUE(state.ok()) << state.status();
  EXPECT_NEAR(state->boundary_gauge_pressure, kHydrostatic, 1e-10);
  EXPECT_NEAR(state->equivalent_radius, kRadius, 1e-15);
  EXPECT_TRUE(state->resolved);
  EXPECT_NEAR(state->maximum_step_duration, 0.5 * kRadius / std::sqrt(4.2 * expected_pressure / 1000.0), 1e-15);
}

TEST(LiquidCavity3D, FixedInventoryCompressesAndFallsBelowResolution) {
  const LiquidCavityInventory3D inventory{1.0, 100000.0};
  const auto compressed = EvaluateLiquidCavity3D(inventory, 0.5, 100000.0, 1000.0, 0.0, 0.0, 0.3);
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  const double expected_absolute = 100000.0 * std::pow(2.0, 1.4);
  EXPECT_NEAR(compressed->absolute_pressure, expected_absolute, 1e-8);
  EXPECT_NEAR(compressed->boundary_gauge_pressure, expected_absolute - 100000.0, 1e-8);
  EXPECT_FALSE(compressed->resolved);
  const auto expanded = EvaluateLiquidCavity3D(inventory, 2.0, 100000.0, 1000.0, 0.0, 0.0, 0.3);
  ASSERT_TRUE(expanded.ok()) << expanded.status();
  EXPECT_LT(expanded->boundary_gauge_pressure, 0.0);
  EXPECT_GT(expanded->maximum_step_duration, compressed->maximum_step_duration);
  EXPECT_TRUE(expanded->resolved);
  const auto threshold =
      EvaluateLiquidCavity3D(inventory, 2.0, 100000.0, 1000.0, 0.0, 0.0, expanded->equivalent_radius / 2.0);
  ASSERT_TRUE(threshold.ok()) << threshold.status();
  EXPECT_TRUE(threshold->resolved);
}

TEST(LiquidCavity3D, PreservesSmallGaugeIncrementAndChecksRange) {
  const double volume = std::nextafter(1.0, 2.0);
  const auto expanded = EvaluateLiquidCavity3D({1.0, 101325.0}, volume, 101325.0, 1000.0, 0.0, 0.0, 0.25);
  ASSERT_TRUE(expanded.ok()) << expanded.status();
  const double linearized_gauge = -1.4 * 101325.0 * (volume - 1.0);
  EXPECT_NEAR(expanded->boundary_gauge_pressure, linearized_gauge, 1e-24);
  EXPECT_EQ(InitializeSubmergedLiquidCavity3D(0.0, 1.0, 0.0, 0.0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      EvaluateLiquidCavity3D({1.0, 1.0}, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      EvaluateLiquidCavity3D({1.0, 1e300}, 1e-30, 1.0, 1.0, 0.0, 0.0, 1.0).status().code(),
      absl::StatusCode::kOutOfRange
  );
  const auto scaled = EvaluateLiquidCavity3D({1e-300, 1e-320}, 1e-300, 1e-320, 1e308, 0.0, 0.0, 1.0);
  ASSERT_TRUE(scaled.ok()) << scaled.status();
  EXPECT_TRUE(std::isfinite(scaled->maximum_step_duration));
  EXPECT_GT(scaled->maximum_step_duration, 1e210);
}

}  // namespace
}  // namespace fluidity
