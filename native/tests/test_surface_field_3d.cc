#include <cmath>
#include <numbers>

#include "fluidity/host/surface_field_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(SurfaceField3D, ConservativeSplatAndGaussianRecoverMaterialVolume) {
  const std::array particles{
      SurfaceMaterialPoint3D{.position = {-0.15, 0.025, 0.075}, .volume_m3 = 0.002},
      SurfaceMaterialPoint3D{.position = {0.15, -0.025, -0.075}, .volume_m3 = 0.003},
  };
  const SurfaceFieldOptions3D options{
      .origin = {-0.6, -0.6, -0.6},
      .spacing_m = 0.025,
      .extent = {49, 49, 49},
      .sigma_m = 0.08,
  };
  auto field = BuildSurfaceDensity3D(particles, options, 200000);
  ASSERT_TRUE(field.ok()) << field.status();
  double volume = 0;
  for (const double value : field->values) {
    volume += (0.5 - value) * std::pow(options.spacing_m, 3);
  }
  EXPECT_NEAR(volume, 0.005, 1e-14);
  auto sample = SampleSurfaceField3D(*field, {-0.15, 0.025, 0.075});
  ASSERT_TRUE(sample.ok()) << sample.status();
  EXPECT_LT(sample->value, 0.5);
  EXPECT_EQ(SampleSurfaceField3D(*field, {2, 0, 0}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(SurfaceField3D, RejectsClippedSupportInvalidVolumeAndBudget) {
  const std::array particles{SurfaceMaterialPoint3D{.position = {}, .volume_m3 = 0.001}};
  SurfaceFieldOptions3D options{
      .origin = {-0.2, -0.2, -0.2},
      .spacing_m = 0.1,
      .extent = {5, 5, 5},
      .sigma_m = 0.1,
  };
  EXPECT_EQ(BuildSurfaceDensity3D(particles, options, 1000).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(BuildSurfaceDensity3D(particles, options, 10).status().code(), absl::StatusCode::kResourceExhausted);
  auto invalid = particles;
  invalid[0].volume_m3 = -1;
  EXPECT_EQ(BuildSurfaceDensity3D(invalid, options, 1000).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SurfaceField3D, MatchesIndependentSingleGaussianAndLinearGradient) {
  const std::array particles{SurfaceMaterialPoint3D{.position = {}, .volume_m3 = 0.001}};
  const SurfaceFieldOptions3D options{
      .origin = {-1, -1, -1},
      .spacing_m = 0.05,
      .extent = {41, 41, 41},
      .sigma_m = 0.1,
  };
  auto field = BuildSurfaceDensity3D(particles, options, 100000);
  ASSERT_TRUE(field.ok());
  double sum = 0;
  for (int i = -8; i <= 8; ++i) {
    sum += std::exp(-0.125 * i * i);
  }
  const double expected = 0.001 * std::exp(-0.125 * 5) / (std::pow(sum, 3) * std::pow(0.05, 3));
  EXPECT_NEAR(0.5 - field->values[(21 * 41 + 22) * 41 + 20], expected, 1e-14);
  SurfaceLattice3D linear{.origin = {-1, -1, -1}, .spacing = 1, .extent = {3, 3, 3}};
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        linear.values.push_back(x + 2 * y - 0.5 * z);
      }
    }
  }
  const auto sample = SampleSurfaceField3D(linear, {-0.3, 0.4, 0.2});
  ASSERT_TRUE(sample.ok());
  EXPECT_NEAR(sample->value, 0.4, 1e-14);
  EXPECT_NEAR(sample->gradient.x, 1, 1e-14);
  EXPECT_NEAR(sample->gradient.y, 2, 1e-14);
  EXPECT_NEAR(sample->gradient.z, -0.5, 1e-14);
}
}  // namespace
}  // namespace fluidity
