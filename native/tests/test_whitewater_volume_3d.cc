#include <numeric>

#include "fluidity/host/whitewater_volume_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(WhitewaterVolume3D, ConservesRepresentedFoamBubbleCountsAndGasVolume) {
  const WhitewaterVolumeGrid3D grid{
      .origin = {-0.3, -0.3, -0.3},
      .spacing_m = 0.1,
      .extent = {7, 7, 7},
  };
  const std::vector<std::uint8_t> validity(343, 1);
  const std::array points{
      WhitewaterVolumePoint3D{
          .kind = WhitewaterKind3D::kFoam,
          .position = {0.015, -0.035, 0.055},
          .represented_count = 7,
          .gas_volume_m3 = 0
      },
      WhitewaterVolumePoint3D{
          .kind = WhitewaterKind3D::kBubble,
          .position = {-0.025, 0.045, -0.015},
          .represented_count = 11,
          .gas_volume_m3 = 0.0002
      },
  };
  const auto result = RasterizeWhitewaterVolume3D(points, grid, validity, 1000);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto integral = [](const auto& values) {
    return 0.001 * std::accumulate(values.begin(), values.end(), 0.0);
  };
  EXPECT_NEAR(integral(result->foam_count_per_m3), 7, 1e-13);
  EXPECT_NEAR(integral(result->bubble_count_per_m3), 11, 1e-13);
  EXPECT_NEAR(integral(result->gas_fraction), 0.0002, 1e-17);
  EXPECT_EQ(result->validity, validity);
}
TEST(WhitewaterVolume3D, EmptyFramesPreserveValidityAndUnknownDonorsFail) {
  const WhitewaterVolumeGrid3D grid{.origin = {}, .spacing_m = 1, .extent = {3, 3, 3}};
  std::vector<std::uint8_t> validity(27, 1);
  validity[0] = 0;
  auto result = RasterizeWhitewaterVolume3D({}, grid, validity, 100);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->foam_count_per_m3.size(), 27);
  EXPECT_EQ(result->validity[0], 0);
  EXPECT_EQ(std::accumulate(result->foam_count_per_m3.begin(), result->foam_count_per_m3.end(), 0.0), 0);
  const std::array points{WhitewaterVolumePoint3D{
      .kind = WhitewaterKind3D::kFoam,
      .position = {0.5, 0.5, 0.5},
      .represented_count = 1,
      .gas_volume_m3 = 0
  }};
  EXPECT_EQ(
      RasterizeWhitewaterVolume3D(points, grid, validity, 100).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(RasterizeWhitewaterVolume3D({}, grid, validity, 1).status().code(), absl::StatusCode::kResourceExhausted);
}
}  // namespace
}  // namespace fluidity
