#include <array>
#include <cmath>

#include "fluidity/host/whitewater_transport_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
absl::StatusOr<MotionInterval3D> TransportInterval() {
  constexpr std::array stamps{MotionStamp3D{0, 0}, MotionStamp3D{0.5, 1}, MotionStamp3D{1, 2}};
  return MotionInterval3D::Create({}, {.time = 1, .frame_ordinal = 1, .accepted_step_ordinal = 2, .epoch = 2}, stamps);
}

TEST(WhitewaterTransport3D, SprayIntegratesFromBirthAndDoesNotReadLiquidCarrier) {
  auto interval = TransportInterval();
  ASSERT_TRUE(interval.ok());
  std::size_t carrier_queries = 0;
  auto carrier = [&](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    ++carrier_queries;
    return absl::FailedPreconditionError("no liquid carrier");
  };
  const auto air = [](Vec3, double) -> absl::StatusOr<double> {
    return 1;
  };
  const WhitewaterTransportPoint3D spray{
      .kinematics = {.position = {}, .velocity = {2, 0, 0}},
      .kind = WhitewaterKind3D::kSpray,
      .birth_time_seconds = 0.75,
  };
  const WhitewaterTransportOptions3D options{.cell_width_m = 1, .gravity = {0, -4, 0}};
  auto result = AdvanceWhitewaterPoint3D(spray, *interval, options, carrier, air);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(carrier_queries, 0);
  EXPECT_EQ(result->point.kind, WhitewaterKind3D::kSpray);
  EXPECT_DOUBLE_EQ(result->point.kinematics.position.x, 0.5);
  EXPECT_DOUBLE_EQ(result->point.kinematics.position.y, -0.125);
  EXPECT_DOUBLE_EQ(result->point.kinematics.velocity.y, -1);
  EXPECT_LE(result->maximum_displacement_m, 0.5);
}

TEST(WhitewaterTransport3D, FoamUsesValidCarrierAndRejectsMissingSupport) {
  auto interval = TransportInterval();
  ASSERT_TRUE(interval.ok());
  const auto surface = [](Vec3, double) -> absl::StatusOr<double> {
    return 0;
  };
  const auto uniform = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return Vec3{1, 0, 0};
  };
  const auto missing = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return absl::FailedPreconditionError("missing halo");
  };
  const WhitewaterTransportPoint3D foam{.kind = WhitewaterKind3D::kFoam};
  const WhitewaterTransportOptions3D options{.cell_width_m = 0.25};
  auto result = AdvanceWhitewaterPoint3D(foam, *interval, options, uniform, surface);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->point.kinematics.position.x, 1);
  EXPECT_DOUBLE_EQ(result->point.kinematics.velocity.x, 1);
  EXPECT_LE(result->maximum_displacement_m, 0.125);
  EXPECT_EQ(
      AdvanceWhitewaterPoint3D(foam, *interval, options, missing, surface).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(WhitewaterTransport3D, BubbleWithMissingCarrierUsesZeroVelocityAndRetainsClass) {
  auto interval = TransportInterval();
  ASSERT_TRUE(interval.ok());
  const auto outside = [](Vec3, double) -> absl::StatusOr<double> {
    return 1;
  };
  const auto missing = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return absl::FailedPreconditionError("missing carrier");
  };
  const WhitewaterTransportPoint3D bubble{
      .kinematics = {.position = {}, .velocity = {2, 0, 0}},
      .kind = WhitewaterKind3D::kBubble,
  };
  const WhitewaterTransportOptions3D options{.cell_width_m = 1, .bubble_drag_per_second = 2};
  auto result = AdvanceWhitewaterPoint3D(bubble, *interval, options, missing, outside);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->point.kind, WhitewaterKind3D::kBubble);
  EXPECT_GT(result->bubble_fallback_substeps, 0);
  EXPECT_NEAR(result->point.kinematics.position.x, 1 - std::exp(-2.0), 1e-14);
  EXPECT_NEAR(result->point.kinematics.velocity.x, 2 * std::exp(-2.0), 1e-14);
}

TEST(WhitewaterTransport3D, FoamExpiresBeforeReadingLaterMissingCarrier) {
  auto interval = TransportInterval();
  ASSERT_TRUE(interval.ok());
  const auto surface = [](Vec3, double) -> absl::StatusOr<double> {
    return 0;
  };
  const auto carrier = [](std::size_t stamp, Vec3) -> absl::StatusOr<Vec3> {
    if (stamp >= 2) {
      return absl::FailedPreconditionError("missing after expiry");
    }
    return Vec3{1, 0, 0};
  };
  const WhitewaterTransportPoint3D foam{.kind = WhitewaterKind3D::kFoam};
  const WhitewaterTransportOptions3D options{.foam_lifetime_seconds = 0.25};
  auto result = AdvanceWhitewaterPoint3D(foam, *interval, options, carrier, surface);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->expired);
  EXPECT_DOUBLE_EQ(result->point.kinematics.position.x, 0.25);
}
}  // namespace
}  // namespace fluidity
