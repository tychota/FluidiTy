#include <array>

#include "fluidity/host/motion_capture_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

LiquidState3D State(double time, std::uint64_t epoch, bool completed = false) {
  LiquidState3D state;
  state.clock = {
      .time = time,
      .frame_ordinal = completed ? 1u : 0u,
      .substep_ordinal = completed ? 0u : epoch,
      .accepted_step_ordinal = epoch,
      .epoch = epoch,
  };
  return state;
}

TEST(MotionCapture3D, BoundsSamplesAndReplaysFromFrameStart) {
  auto start = State(0, 0);
  auto capture = MotionCapture3D::Create(start, 1.0 / 24, 2);
  ASSERT_TRUE(capture.ok()) << capture.status();
  for (std::uint64_t step = 1; step <= 4; ++step) {
    ASSERT_TRUE(capture->Observe(State(static_cast<double>(step) / 96, step, step == 4), false).ok());
  }
  EXPECT_EQ(capture->sample_count(), 3);
  EXPECT_FALSE(capture->Observe(State(1.0 / 24, 4), false).ok());
  ASSERT_TRUE(capture->Observe(start, true).ok());
  EXPECT_EQ(capture->sample_count(), 1);
  ASSERT_TRUE(capture->Observe(State(1.0 / 48, 1), false).ok());
  ASSERT_TRUE(capture->Observe(State(1.0 / 24, 2, true), false).ok());
  auto encoded = capture->Finish(State(1.0 / 24, 2, true));
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto decoded = DecodeMotionFrame3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->samples.size(), 3);
  EXPECT_DOUBLE_EQ(decoded->samples[1].clock.time, 1.0 / 48);
  EXPECT_EQ(decoded->samples.back().clock.frame_ordinal, 1);
  EXPECT_EQ(decoded->samples.back().clock.substep_ordinal, 0);
  EXPECT_TRUE(decoded->samples.back().particles.values().empty());
}

TEST(MotionCapture3D, ValidatesEndpointsAndFailsWithoutACompleteInterval) {
  auto start = State(0, 0);
  EXPECT_FALSE(MotionCapture3D::Create(start, 1.0 / 24, 0).ok());
  EXPECT_FALSE(MotionCapture3D::Create(start, 0, 16).ok());
  auto capture = MotionCapture3D::Create(start, 1.0 / 24, 16);
  ASSERT_TRUE(capture.ok()) << capture.status();
  EXPECT_FALSE(capture->Finish(State(1.0 / 48, 1, true)).ok());
  EXPECT_FALSE(capture->Observe(State(1.0 / 20, 1), false).ok());
  EXPECT_EQ(capture->sample_count(), 1);
  const auto seed = EncodeMotionSeed3D(start);
  ASSERT_TRUE(seed.ok()) << seed.status();
  auto decoded = DecodeMotionFrame3D(seed->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->samples.size(), 1);
  EXPECT_EQ(decoded->samples[0], start);
}

TEST(MotionCapture3D, RejectsGeometryChangesAndSubstitutedCompletedState) {
  auto start = State(0, 0);
  auto capture = MotionCapture3D::Create(start, 1.0 / 24, 1);
  ASSERT_TRUE(capture.ok());
  auto moved = State(1.0 / 96, 1);
  moved.geometry.world_origin.x = 1;
  EXPECT_EQ(capture->Observe(moved, false).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(capture->sample_count(), 1);
  auto completed = State(1.0 / 24, 2, true);
  ASSERT_TRUE(capture->Observe(completed, false).ok());
  completed.geometry.world_origin.y = 1;
  EXPECT_EQ(capture->Finish(completed).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(MotionCapture3D, RetainsOpposingAcceptedBirthsBetweenCarrierSamplesAndClearsRollback) {
  const auto start = State(0, 0);
  auto capture = MotionCapture3D::Create(start, 1.0 / 24, 1);
  ASSERT_TRUE(capture.ok());
  const auto early = State(1.0 / 96, 1);
  const auto later = State(1.0 / 48, 2);
  const auto completed = State(1.0 / 24, 3, true);
  const std::array first_birth{WhitewaterCellBirth3D{
      .source_cell = {-2, 3, 1},
      .event =
          {
              .time_seconds = 1.0 / 192,
              .position = {-1, 2, 3},
              .velocity = {3, 0, 0},
              .normal = {0, 1, 0},
              .volume_time_m3_seconds = 0.125,
          },
  }};
  auto second_birth = first_birth;
  second_birth[0].event.time_seconds = 1.0 / 64;
  second_birth[0].event.velocity.x = -3;
  ASSERT_TRUE(capture->Observe(early, false, first_birth).ok());
  EXPECT_EQ(capture->sample_count(), 1);
  EXPECT_EQ(capture->birth_count(), 1);
  ASSERT_TRUE(capture->Observe(start, true).ok());
  EXPECT_EQ(capture->birth_count(), 0);
  ASSERT_TRUE(capture->Observe(early, false, first_birth).ok());
  ASSERT_TRUE(capture->Observe(later, false, second_birth).ok());
  ASSERT_TRUE(capture->Observe(completed, false).ok());
  auto bytes = capture->Finish(completed);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto decoded = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->samples.size(), 2);
  ASSERT_EQ(decoded->whitewater_births.size(), 2);
  EXPECT_EQ(decoded->whitewater_births[0].accepted_step_ordinal, 1);
  EXPECT_EQ(decoded->whitewater_births[1].accepted_step_ordinal, 2);
  const auto cell = decoded->whitewater_births[0].birth.source_cell;
  EXPECT_EQ(cell.x, -2);
  EXPECT_EQ(cell.y, 3);
  EXPECT_EQ(cell.z, 1);
  EXPECT_DOUBLE_EQ(decoded->whitewater_births[0].birth.event.velocity.x, 3);
  EXPECT_DOUBLE_EQ(decoded->whitewater_births[1].birth.event.velocity.x, -3);
  EXPECT_DOUBLE_EQ(decoded->whitewater_births[1].birth.event.time_seconds, 1.0 / 64);
  EXPECT_DOUBLE_EQ(decoded->whitewater_births[1].birth.event.volume_time_m3_seconds, 0.125);
}

TEST(MotionCapture3D, RejectsUnorderedOutOfStepAndOverBudgetBirthsWithoutAdvancing) {
  const auto start = State(0, 0);
  auto capture = MotionCapture3D::Create(start, 1.0 / 24, 1, {.maximum_bytes = 8192});
  ASSERT_TRUE(capture.ok());
  const auto accepted = State(1.0 / 96, 1);
  const WhitewaterCellBirth3D valid{
      .source_cell = {0, 0, 0},
      .event =
          {
              .time_seconds = 1.0 / 192,
              .position = {},
              .velocity = {1, 0, 0},
              .normal = {0, 1, 0},
              .volume_time_m3_seconds = 1,
          },
  };
  auto future = valid;
  future.event.time_seconds = 1.0 / 48;
  EXPECT_EQ(capture->Observe(accepted, false, std::span(&future, 1)).code(), absl::StatusCode::kInvalidArgument);
  const std::array duplicate{valid, valid};
  EXPECT_EQ(capture->Observe(accepted, false, duplicate).code(), absl::StatusCode::kInvalidArgument);
  std::vector<WhitewaterCellBirth3D> oversized(100, valid);
  for (std::size_t index = 0; index < oversized.size(); ++index) {
    oversized[index].source_cell.x = static_cast<std::int32_t>(index);
  }
  EXPECT_EQ(capture->Observe(accepted, false, oversized).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(capture->birth_count(), 0);
  EXPECT_EQ(capture->sample_count(), 1);
  auto changed_seed = State(1.0 / 96, 1);
  changed_seed.random_seed = 42;
  EXPECT_EQ(capture->Observe(changed_seed, false).code(), absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(capture->Observe(accepted, false, std::span(&valid, 1)).ok());
  auto repeated_step = State(1.0 / 48, 2);
  repeated_step.clock.accepted_step_ordinal = 1;
  repeated_step.clock.substep_ordinal = 1;
  EXPECT_EQ(capture->Observe(repeated_step, false).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(capture->birth_count(), 1);
}

}  // namespace
}  // namespace fluidity
