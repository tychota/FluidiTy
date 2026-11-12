#include <cmath>
#include <limits>

#include "fluidity/host/liquid_eos.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

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
