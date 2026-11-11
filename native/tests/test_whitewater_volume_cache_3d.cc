#include <numbers>

#include "fluidity/host/whitewater_volume_cache_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(WhitewaterVolumeCache3D, DepositsPhysicalInventoryAndRoundTripsSparseVoxels) {
  WhitewaterFrame3D points{.clock = {.time = 0.125, .frame_ordinal = 3, .accepted_step_ordinal = 4, .epoch = 4}};
  for (auto kind : {WhitewaterKind3D::kFoam, WhitewaterKind3D::kBubble, WhitewaterKind3D::kSpray}) {
    points.points.push_back({
        .id = points.points.size() + 1,
        .point = {.kinematics = {.position = {0.5, 0.5, 0.5}, .velocity = {}}, .kind = kind},
    });
  }
  const WhitewaterVolumeSettings3D settings{
      .grid = {.origin = {}, .spacing_m = 1, .extent = {2, 2, 2}},
      .physical_bubble_radius_m = 0.1,
  };
  auto volume = BuildWhitewaterVolumeFrame3D(points, settings, 2);
  ASSERT_TRUE(volume.ok()) << volume.status();
  ASSERT_EQ(volume->volume.foam_count_per_m3.size(), 8);
  const double represented_gas_volume_m3 = 2 * 4 * std::numbers::pi / 3 * 0.001;
  for (std::size_t index = 0; index < 8; ++index) {
    EXPECT_DOUBLE_EQ(volume->volume.foam_count_per_m3[index], 0.25);
    EXPECT_DOUBLE_EQ(volume->volume.bubble_count_per_m3[index], 0.25);
    EXPECT_NEAR(volume->volume.gas_fraction[index], represented_gas_volume_m3 / 8, 1e-18);
    EXPECT_EQ(volume->volume.validity[index], 1);
  }
  auto bytes = EncodeWhitewaterVolumeFrame3D(*volume);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto decoded = DecodeWhitewaterVolumeFrame3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->clock, points.clock);
  EXPECT_EQ(decoded->volume.foam_count_per_m3, volume->volume.foam_count_per_m3);
  EXPECT_EQ(decoded->volume.bubble_count_per_m3, volume->volume.bubble_count_per_m3);
  EXPECT_EQ(decoded->volume.gas_fraction, volume->volume.gas_fraction);
  volume->volume.validity[0] = 0;
  EXPECT_EQ(EncodeWhitewaterVolumeFrame3D(*volume).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(WhitewaterVolumeCache3D, EmptyFramePreservesKnownDomainAndRejectsMalformedBudget) {
  const WhitewaterVolumeSettings3D settings{
      .grid = {.origin = {-1, 0, 1}, .spacing_m = 0.5, .extent = {2, 3, 4}},
      .physical_bubble_radius_m = 0.05,
  };
  auto volume = BuildWhitewaterVolumeFrame3D({}, settings, 1);
  ASSERT_TRUE(volume.ok());
  auto bytes = EncodeWhitewaterVolumeFrame3D(*volume);
  ASSERT_TRUE(bytes.ok());
  auto decoded = DecodeWhitewaterVolumeFrame3D(bytes->values());
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ(decoded->volume.validity, std::vector<std::uint8_t>(24, 1));
  EXPECT_EQ(decoded->volume.foam_count_per_m3, std::vector<double>(24));
  EXPECT_EQ(DecodeWhitewaterVolumeFrame3D(bytes->values(), 23).status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(DecodeWhitewaterVolumeFrame3D(bytes->values().first(12)).status().code(), absl::StatusCode::kDataLoss);
}

TEST(WhitewaterVolumeCache3D, SettingsIdentityBindsGridAndBubbleRadiusButNotCapacity) {
  WhitewaterVolumeSettings3D settings{
      .grid = {.origin = {-1, 0, 1}, .spacing_m = 0.5, .extent = {2, 3, 4}},
      .physical_bubble_radius_m = 0.05,
  };
  auto first = EncodeWhitewaterVolumeSettings3D(settings);
  ASSERT_TRUE(first.ok());
  settings.maximum_voxels *= 2;
  auto capacity = EncodeWhitewaterVolumeSettings3D(settings);
  ASSERT_TRUE(capacity.ok());
  EXPECT_TRUE(std::ranges::equal(first->values(), capacity->values()));
  settings.physical_bubble_radius_m *= 2;
  auto changed = EncodeWhitewaterVolumeSettings3D(settings);
  ASSERT_TRUE(changed.ok());
  EXPECT_NE(LiquidContentDigest3D(first->values()), LiquidContentDigest3D(changed->values()));
  auto decoded = DecodeWhitewaterVolumeSettings3D(changed->values());
  ASSERT_TRUE(decoded.ok());
  EXPECT_DOUBLE_EQ(decoded->physical_bubble_radius_m, 0.1);
  EXPECT_EQ(decoded->grid.extent, settings.grid.extent);
}
}  // namespace
}  // namespace fluidity
