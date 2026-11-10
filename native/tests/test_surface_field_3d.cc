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
}  // namespace
}  // namespace fluidity
