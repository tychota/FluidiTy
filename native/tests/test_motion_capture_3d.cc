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

}  // namespace
}  // namespace fluidity
