#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "../src/liquid_eos_sensor_3d.h"
#include "fluidity/host/liquid_eos.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(LiquidEos3D, CellSensorUsesReferenceVolumeAndSameSideTentWeights) {
  const std::array cells{
      FrozenLiquidCell3D{.cell_index = {-1, 0, 0}},
      FrozenLiquidCell3D{.cell_index = {0, 0, 0}},
      FrozenLiquidCell3D{.cell_index = {2, 0, 0}},
  };
  std::array<LiquidParticleState3D, 3> particles;
  particles[0].world_position = {-0.5, 0.5, 0.5};
  particles[0].reference_volume = 1.0;
  particles[0].volume_ratio = 2.0;
  particles[1].world_position = {0.0, 0.5, 0.5};
  particles[1].reference_volume = 6.0;
  particles[1].volume_ratio = 0.5;
  particles[2] = particles[1];
  particles[2].side_tag = 1;
  particles[2].volume_ratio = 100.0;
  std::array<liquid_step_internal::CellVolumeRatioSensor3D, 3> sensors;
  const auto status =
      liquid_step_internal::SampleLiquidCellVolumeRatios3D({{0.0, 0.0, 0.0}, 1.0}, cells, particles, 0, sensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(sensors[0].sampled_reference_volume, 4.0);
  EXPECT_EQ(sensors[0].mean_volume_ratio, 0.875);
  EXPECT_EQ(sensors[1].sampled_reference_volume, 3.0);
  EXPECT_EQ(sensors[1].mean_volume_ratio, 0.5);
  EXPECT_EQ(sensors[2].sampled_reference_volume, 0.0);
  for (auto& particle : particles) {
    particle.world_position.x *= 0.25;
    particle.world_position.y *= 0.25;
    particle.world_position.z *= 0.25;
  }
  const auto refined =
      liquid_step_internal::SampleLiquidCellVolumeRatios3D({{0.0, 0.0, 0.0}, 0.25}, cells, particles, 0, sensors);
  ASSERT_TRUE(refined.ok()) << refined;
  EXPECT_EQ(sensors[0].mean_volume_ratio, 0.875);
  EXPECT_EQ(sensors[1].mean_volume_ratio, 0.5);
  EXPECT_EQ(sensors[2].sampled_reference_volume, 0.0);
}

TEST(LiquidEos3D, CellSensorRetainsLightDonorWithLargeVolumeRatio) {
  const std::array cells{FrozenLiquidCell3D{.cell_index = {0, 0, 0}}};
  std::array<LiquidParticleState3D, 2> particles;
  particles[0].world_position = particles[1].world_position = {0.5, 0.5, 0.5};
  particles[0].reference_volume = 1.0;
  particles[0].volume_ratio = 1e20;
  particles[1].reference_volume = 1e20;
  particles[1].volume_ratio = 1.0;
  std::array<liquid_step_internal::CellVolumeRatioSensor3D, 1> sensors;
  for (int order = 0; order < 2; ++order) {
    const auto status =
        liquid_step_internal::SampleLiquidCellVolumeRatios3D({{0.0, 0.0, 0.0}, 1.0}, cells, particles, 0, sensors);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_NEAR(sensors[0].mean_volume_ratio, 2.0, 1e-15);
    std::ranges::reverse(particles);
  }
}

TEST(LiquidEos3D, DriftTracksEosStateAndClampsOnlySensedCells) {
  using liquid_step_internal::LiquidVolumeDriftRate3D;
  const auto compressed = EvaluateLiquidEosAtPressure3D({1000.0, 2.0}, 4000.0 * std::log(2.0));
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  const auto matched = LiquidVolumeDriftRate3D({1.0, 0.5}, compressed->log_volume_ratio, 0.2, 0.3);
  ASSERT_TRUE(matched.ok()) << matched.status();
  EXPECT_NEAR(*matched, 0.0, 1e-14);
  const auto expanded = LiquidVolumeDriftRate3D({1.0, 2.0}, 0.0, 0.2, 0.3);
  ASSERT_TRUE(expanded.ok()) << expanded.status();
  EXPECT_EQ(*expanded, -0.3);
  const auto compressed_drift = LiquidVolumeDriftRate3D({1.0, 0.5}, 0.0, 2.0, 1.0);
  ASSERT_TRUE(compressed_drift.ok()) << compressed_drift.status();
  EXPECT_NEAR(*compressed_drift, std::log(2.0) / 2.0, 1e-14);
  const auto core = LiquidVolumeDriftRate3D({}, compressed->log_volume_ratio, 0.2, 0.3);
  ASSERT_TRUE(core.ok()) << core.status();
  EXPECT_EQ(*core, 0.0);
  EXPECT_EQ(LiquidVolumeDriftRate3D({1.0, 1.0}, 0.0, 0.0, 0.3).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidEos3D, HenckyAndExactIncompressibleStates) {
  const auto finite = EvaluateLiquidEosAtPressure3D({1000.0, 2.0}, 4000.0 * std::log(2.0));
  ASSERT_TRUE(finite.ok()) << finite.status();
  EXPECT_NEAR(finite->volume_ratio, 0.5, 1e-15);
  EXPECT_NEAR(finite->density, 2000.0, 1e-12);
  EXPECT_EQ(finite->bulk_modulus, 4000.0);
  EXPECT_NEAR(finite->sound_speed, std::sqrt(2.0), 1e-14);
  const auto infinite = EvaluateLiquidEosAtPressure3D({1000.0, std::numeric_limits<double>::infinity()}, 1e30);
  ASSERT_TRUE(infinite.ok()) << infinite.status();
  EXPECT_EQ(infinite->volume_ratio, 1.0);
  EXPECT_EQ(infinite->log_volume_ratio, 0.0);
  EXPECT_EQ(infinite->density, 1000.0);
  EXPECT_TRUE(std::isinf(infinite->bulk_modulus));
  EXPECT_TRUE(std::isinf(infinite->sound_speed));
}

TEST(LiquidEos3D, TaitTangentAndTensileLimit) {
  const LiquidEos3D eos{7.0, 2.0, LiquidEosLaw::kTait};
  const auto compressed = EvaluateLiquidEosAtPressure3D(eos, 508.0);
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  EXPECT_NEAR(compressed->volume_ratio, 0.5, 1e-15);
  EXPECT_EQ(compressed->bulk_modulus, 3584.0);
  EXPECT_NEAR(compressed->sound_speed, 16.0, 1e-13);
  const auto small = EvaluateLiquidEosAtPressure3D(eos, 1e-18);
  ASSERT_TRUE(small.ok()) << small.status();
  EXPECT_NEAR(small->log_volume_ratio, -1e-18 / 28.0, 1e-34);
  const auto just_inside = EvaluateLiquidEosAtPressure3D(eos, std::nextafter(-4.0, 0.0));
  ASSERT_TRUE(just_inside.ok()) << just_inside.status();
  EXPECT_GT(just_inside->volume_ratio, 1.0);
  EXPECT_EQ(EvaluateLiquidEosAtPressure3D(eos, -4.0).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidEos3D, TaitRestTangentRetainsSubnormalReferenceModulus) {
  const double reference_density = 8.0 * std::numeric_limits<double>::denorm_min();
  const auto rest = EvaluateLiquidEosAtPressure3D({reference_density, 1.0, LiquidEosLaw::kTait}, 0.0);
  ASSERT_TRUE(rest.ok()) << rest.status();
  EXPECT_EQ(rest->density, reference_density);
  EXPECT_EQ(rest->bulk_modulus, reference_density);
  EXPECT_EQ(rest->sound_speed, 1.0);
}

TEST(LiquidEos3D, ChecksRangeAndInput) {
  EXPECT_EQ(EvaluateLiquidEosAtPressure3D({0.0, 1.0}, 0.0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(EvaluateLiquidEosAtPressure3D({1.0, 1e200}, 0.0).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(EvaluateLiquidEosAtPressure3D({1.0, 1.0}, 1000.0).status().code(), absl::StatusCode::kOutOfRange);
  const auto compensated = EvaluateLiquidEosAtPressure3D({1e-300, 1e200}, 0.0);
  ASSERT_TRUE(compensated.ok()) << compensated.status();
  EXPECT_NEAR(compensated->bulk_modulus / 1e100, 1.0, 1e-15);
  EXPECT_NEAR(compensated->sound_speed / 1e200, 1.0, 1e-15);
}

}  // namespace
}  // namespace fluidity
