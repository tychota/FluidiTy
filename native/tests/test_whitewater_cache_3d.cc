#include "fluidity/host/whitewater_cache_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
WhitewaterFrame3D Frame() {
  return {
      .clock = {.time = 0.125, .frame_ordinal = 3, .accepted_step_ordinal = 5, .epoch = 5},
      .emission =
          {
              .next_point_id = 4,
              .last_accepted_step_ordinal = 5,
              .remainders = {{.source_cell = {-1, 3, 0}, .fractional_count = 0.25}},
          },
      .points = {{
          .id = 3,
          .point =
              {
                  .kinematics = {.position = {1, 2, 3}, .velocity = {-1, 0, 1}},
                  .kind = WhitewaterKind3D::kFoam,
                  .birth_time_seconds = 0.0625,
              },
      }},
  };
}

TEST(WhitewaterCache3D, RoundTripsPointIdentityKindTimeAndSignedCellRemainder) {
  const auto frame = Frame();
  const auto bytes = EncodeWhitewaterFrame3D(frame);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto decoded = DecodeWhitewaterFrame3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->clock, frame.clock);
  EXPECT_EQ(decoded->emission.next_point_id, 4);
  EXPECT_EQ(decoded->emission.last_accepted_step_ordinal, 5);
  ASSERT_EQ(decoded->emission.remainders.size(), 1);
  EXPECT_EQ(decoded->emission.remainders[0].source_cell.x, -1);
  EXPECT_DOUBLE_EQ(decoded->emission.remainders[0].fractional_count, 0.25);
  ASSERT_EQ(decoded->points.size(), 1);
  EXPECT_EQ(decoded->points[0].id, 3);
  EXPECT_EQ(decoded->points[0].point.kind, WhitewaterKind3D::kFoam);
  EXPECT_DOUBLE_EQ(decoded->points[0].point.birth_time_seconds, 0.0625);
  EXPECT_DOUBLE_EQ(decoded->points[0].point.kinematics.velocity.x, -1);
  auto reencoded = EncodeWhitewaterFrame3D(*decoded);
  ASSERT_TRUE(reencoded.ok());
  EXPECT_TRUE(std::ranges::equal(bytes->values(), reencoded->values()));
}

TEST(WhitewaterCache3D, RejectsFutureBirthBadCursorDuplicateCellAndCapacity) {
  auto frame = Frame();
  EXPECT_EQ(
      EncodeWhitewaterFrame3D(frame, {.maximum_points = 0}).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  auto encoded = EncodeWhitewaterFrame3D(frame);
  ASSERT_TRUE(encoded.ok());
  EXPECT_EQ(DecodeWhitewaterFrame3D(encoded->values().first(12)).status().code(), absl::StatusCode::kDataLoss);
  frame.points[0].point.birth_time_seconds = 1;
  EXPECT_EQ(EncodeWhitewaterFrame3D(frame).status().code(), absl::StatusCode::kInvalidArgument);
  frame = Frame();
  frame.emission.last_accepted_step_ordinal = 6;
  EXPECT_EQ(EncodeWhitewaterFrame3D(frame).status().code(), absl::StatusCode::kInvalidArgument);
  frame = Frame();
  frame.emission.remainders.push_back(frame.emission.remainders.front());
  EXPECT_EQ(EncodeWhitewaterFrame3D(frame).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(WhitewaterCache3D, SettingsBindPhysicalControlsAndExcludeWorkLimitsAndSourceGeometry) {
  WhitewaterFrameOptions3D options;
  auto encoded = EncodeWhitewaterSettings3D(options);
  ASSERT_TRUE(encoded.ok());
  options.maximum_points = 4;
  options.transport.maximum_substeps = 10;
  options.transport.cell_width_m = 0.25;
  options.transport.gravity = {0, -9.81, 0};
  auto operational = EncodeWhitewaterSettings3D(options);
  ASSERT_TRUE(operational.ok());
  EXPECT_TRUE(std::ranges::equal(encoded->values(), operational->values()));
  options.transport.foam_lifetime_seconds = 3;
  auto changed = EncodeWhitewaterSettings3D(options);
  ASSERT_TRUE(changed.ok());
  EXPECT_NE(LiquidContentDigest3D(encoded->values()), LiquidContentDigest3D(changed->values()));
  auto decoded = DecodeWhitewaterSettings3D(changed->values());
  ASSERT_TRUE(decoded.ok());
  EXPECT_DOUBLE_EQ(decoded->transport.foam_lifetime_seconds, 3);
  EXPECT_EQ(decoded->maximum_points, WhitewaterFrameOptions3D{}.maximum_points);
}
}  // namespace
}  // namespace fluidity
