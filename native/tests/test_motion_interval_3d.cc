#include <array>
#include <limits>

#include "absl/status/status.h"
#include "fluidity/host/motion_interval_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr LiquidClock3D kFirstFrame{.time = 1.0, .frame_ordinal = 1, .accepted_step_ordinal = 2, .epoch = 2};
constexpr LiquidClock3D kLastFrame{.time = 2.0, .frame_ordinal = 2, .accepted_step_ordinal = 8, .epoch = 8};
constexpr std::array kStamps{MotionStamp3D{1.0, 2}, MotionStamp3D{1.5, 4}, MotionStamp3D{2.0, 8}};

TEST(MotionInterval3D, OwnsOrderedStampsFromAdjacentCompletedFrames) {
  auto stamps = kStamps;
  const auto interval = MotionInterval3D::Create(kFirstFrame, kLastFrame, stamps);
  ASSERT_TRUE(interval.ok()) << interval.status();
  stamps[1].time_seconds = 1.25;
  EXPECT_EQ(interval->stamps()[1].time_seconds, 1.5);
  EXPECT_EQ(interval->stamps()[1].accepted_epoch, 4);

  auto invalid_frame = kLastFrame;
  invalid_frame.frame_ordinal += 1;
  EXPECT_EQ(
      MotionInterval3D::Create(kFirstFrame, invalid_frame, kStamps).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  invalid_frame = kLastFrame;
  invalid_frame.substep_ordinal = 1;
  EXPECT_FALSE(MotionInterval3D::Create(kFirstFrame, invalid_frame, kStamps).ok());
  EXPECT_FALSE(MotionInterval3D::Create(kFirstFrame, kLastFrame, std::span(kStamps).first(1)).ok());
  stamps[1] = stamps[0];
  EXPECT_FALSE(MotionInterval3D::Create(kFirstFrame, kLastFrame, stamps).ok());
  stamps = kStamps;
  stamps[1].accepted_epoch = stamps[2].accepted_epoch;
  EXPECT_FALSE(MotionInterval3D::Create(kFirstFrame, kLastFrame, stamps).ok());
  stamps = kStamps;
  stamps[1].time_seconds = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(MotionInterval3D::Create(kFirstFrame, kLastFrame, stamps).ok());
  stamps = kStamps;
  stamps[0].time_seconds = 1.25;
  EXPECT_FALSE(MotionInterval3D::Create(kFirstFrame, kLastFrame, stamps).ok());
  stamps = kStamps;
  stamps[2].accepted_epoch -= 1;
  EXPECT_FALSE(MotionInterval3D::Create(kFirstFrame, kLastFrame, stamps).ok());
}

TEST(MotionInterval3D, InterpolatesAtOneWorldPositionAndUsesOnlyExactStampAtKnot) {
  const auto interval = MotionInterval3D::Create(kFirstFrame, kLastFrame, kStamps);
  ASSERT_TRUE(interval.ok()) << interval.status();
  const Vec3 world_position{3, -2, 1};
  int calls = 0;
  const auto spatial = [&](std::size_t sample_index, Vec3 position) -> absl::StatusOr<Vec3> {
    ++calls;
    EXPECT_DOUBLE_EQ(position.x, world_position.x);
    EXPECT_DOUBLE_EQ(position.y, world_position.y);
    EXPECT_DOUBLE_EQ(position.z, world_position.z);
    if (sample_index == 2) {
      return absl::NotFoundError("missing donor");
    }
    return Vec3{2 * static_cast<double>(sample_index), 4, -1};
  };
  const auto interpolated = SampleMotionCarrier3D(*interval, world_position, 1.25, spatial);
  ASSERT_TRUE(interpolated.ok()) << interpolated.status();
  EXPECT_DOUBLE_EQ(interpolated->x, 1);
  EXPECT_DOUBLE_EQ(interpolated->y, 4);
  EXPECT_DOUBLE_EQ(interpolated->z, -1);
  EXPECT_EQ(calls, 2);
  const auto exact = SampleMotionCarrier3D(*interval, world_position, 1.5, spatial);
  ASSERT_TRUE(exact.ok()) << exact.status();
  EXPECT_DOUBLE_EQ(exact->x, 2);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(
      SampleMotionCarrier3D(*interval, world_position, 1.75, spatial).status().code(),
      absl::StatusCode::kNotFound
  );
  EXPECT_EQ(calls, 5);
  EXPECT_EQ(
      SampleMotionCarrier3D(*interval, world_position, 2.25, spatial).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(calls, 5);
}

TEST(MotionInterval3D, RejectsNonfiniteSpatialVelocity) {
  const auto interval = MotionInterval3D::Create(kFirstFrame, kLastFrame, kStamps);
  ASSERT_TRUE(interval.ok()) << interval.status();
  const auto spatial = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return Vec3{0, std::numeric_limits<double>::infinity(), 0};
  };
  EXPECT_EQ(SampleMotionCarrier3D(*interval, {}, 1.0, spatial).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(MotionInterval3D, SplitsMidpointTransportAtTimestampKnots) {
  const auto interval = MotionInterval3D::Create(kFirstFrame, kLastFrame, kStamps);
  ASSERT_TRUE(interval.ok()) << interval.status();
  static constexpr std::array kSpeeds{0.0, 2.0, 1.0};
  const auto spatial = [](std::size_t sample_index, Vec3) -> absl::StatusOr<Vec3> {
    return Vec3{kSpeeds[sample_index], 0, 0};
  };
  const auto transported = AdvectMotionPoint3D(*interval, {3, -2, 1}, 1.25, 1.75, 4, 2, spatial);
  ASSERT_TRUE(transported.ok()) << transported.status();
  EXPECT_DOUBLE_EQ(transported->world_position.x, 3.8125);
  EXPECT_DOUBLE_EQ(transported->world_position.y, -2);
  EXPECT_DOUBLE_EQ(transported->world_position.z, 1);
  EXPECT_EQ(transported->accepted_substeps, 2);
  EXPECT_DOUBLE_EQ(transported->maximum_displacement_m, 0.4375);
}

TEST(MotionInterval3D, ReducesStepForRapidlyIncreasingMidpointSpeed) {
  const auto interval = MotionInterval3D::Create(kFirstFrame, kLastFrame, kStamps);
  ASSERT_TRUE(interval.ok()) << interval.status();
  const auto spatial = [](std::size_t sample_index, Vec3) -> absl::StatusOr<Vec3> {
    return Vec3{4 * static_cast<double>(sample_index), 0, 0};
  };
  const auto transported = AdvectMotionPoint3D(*interval, {}, 1.0, 2.0, 1.0, 64, spatial);
  ASSERT_TRUE(transported.ok()) << transported.status();
  EXPECT_NEAR(transported->world_position.x, 4, 1e-14);
  EXPECT_GE(transported->accepted_substeps, 8);
  EXPECT_LE(transported->maximum_displacement_m, 0.5);
}

TEST(MotionInterval3D, FailsWithoutTransportWhenCarrierOrBudgetIsMissing) {
  const auto interval = MotionInterval3D::Create(kFirstFrame, kLastFrame, kStamps);
  ASSERT_TRUE(interval.ok()) << interval.status();
  int calls = 0;
  const auto spatial = [&](std::size_t, Vec3 position) -> absl::StatusOr<Vec3> {
    ++calls;
    if (position.x > 0) {
      return absl::NotFoundError("missing midpoint");
    }
    return Vec3{1, 0, 0};
  };
  const auto no_time = AdvectMotionPoint3D(*interval, {3, 4, 5}, 1.25, 1.25, 1.0, 0, spatial);
  ASSERT_TRUE(no_time.ok()) << no_time.status();
  EXPECT_DOUBLE_EQ(no_time->world_position.x, 3);
  EXPECT_EQ(no_time->accepted_substeps, 0);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(
      AdvectMotionPoint3D(*interval, {}, 1.0, 2.0, 1.0, 0, spatial).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(AdvectMotionPoint3D(*interval, {}, 1.0, 2.0, 1.0, 8, spatial).status().code(), absl::StatusCode::kNotFound);
  const auto constant = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return Vec3{1, 0, 0};
  };
  EXPECT_EQ(
      AdvectMotionPoint3D(*interval, {}, 1.0, 2.0, 1.0, 1, constant).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}

}  // namespace
}  // namespace fluidity
