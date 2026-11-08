#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>

#include "absl/status/status.h"
#include "fixture_data.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kMax = std::numeric_limits<double>::max();
constexpr double kTolerance = 2e-14;

Vec3 Scale(Vec3 vector, double scalar) {
  return {vector.x * scalar, vector.y * scalar, vector.z * scalar};
}

void ExpectVectorNear(Vec3 actual, Vec3 expected) {
  EXPECT_NEAR(actual.x, expected.x, kTolerance);
  EXPECT_NEAR(actual.y, expected.y, kTolerance);
  EXPECT_NEAR(actual.z, expected.z, kTolerance);
}

TEST(LiquidWallContact3D, ThinSlabSweepMatchesFixtureOnAllAxesAndReversedNormals) {
  const auto all = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(all, "uniform_mac_liquid_boundary_3d", &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, all.end());
  constexpr std::array<std::string_view, 6> kNames{
      "sweep_start_coordinates",
      "sweep_end_coordinates",
      "sweep_wall_start_centers",
      "sweep_wall_end_centers",
      "sweep_half_widths",
      "expected_first_contact_fractions",
  };
  std::array<std::array<double, 3>, kNames.size()> values{};
  for (std::size_t array_index = 0; array_index < kNames.size(); ++array_index) {
    const auto array = fixtures::FindArray(*fixture, kNames[array_index]);
    ASSERT_TRUE(array.ok()) << array.status();
    ASSERT_EQ((*array)->f64_data.size(), values[array_index].size());
    std::ranges::copy((*array)->f64_data, values[array_index].begin());
  }
  const auto& [starts, ends, wall_starts, wall_ends, widths, fractions] = values;
  constexpr std::array<Vec3, 3> kAxes{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  constexpr std::array<double, 3> kIncidentSides{-1, 1, 0};
  constexpr std::array<double, 3> kExpectedFinalCoordinates{-1.0 / 32.0, 33.0 / 32.0, 1.0};
  for (std::size_t row = 0; row < starts.size(); ++row) {
    SCOPED_TRACE(row);
    for (const Vec3 axis : kAxes) {
      for (const double normal_sign : {-1.0, 1.0}) {
        const Vec3 advecting_velocity = Scale(axis, ends[row] - starts[row]);
        const Vec3 wall_velocity = Scale(axis, wall_ends[row] - wall_starts[row]);
        const auto result = ResolveLiquidWallContact3D({
            .start_world_position = Scale(axis, starts[row]),
            .stored_world_velocity = advecting_velocity,
            .advecting_world_velocity = advecting_velocity,
            .obstacle =
                TranslatingSlab3D{Scale(axis, normal_sign), normal_sign * wall_starts[row], widths[row], wall_velocity},
            .step_duration = 1,
        });
        ASSERT_TRUE(result.ok()) << result.status();
        ExpectVectorNear(result->world_position, Scale(axis, kExpectedFinalCoordinates[row]));
        if (fractions[row] < 0) {
          EXPECT_FALSE(result->contact_fraction.has_value());
          EXPECT_EQ(result->diagnostic, LiquidWallContactDiagnostic3D::kNone);
          ExpectVectorNear(result->stored_world_velocity, advecting_velocity);
          ExpectVectorNear(result->advecting_world_velocity, advecting_velocity);
        } else {
          ASSERT_TRUE(result->contact_fraction.has_value());
          EXPECT_DOUBLE_EQ(*result->contact_fraction, fractions[row]);
          EXPECT_EQ(result->diagnostic, LiquidWallContactDiagnostic3D::kSweptImpact);
          ExpectVectorNear(result->outward_normal, Scale(axis, kIncidentSides[row]));
          ExpectVectorNear(result->stored_world_velocity, wall_velocity);
          ExpectVectorNear(result->advecting_world_velocity, wall_velocity);
          EXPECT_DOUBLE_EQ(result->incoming_stored_relative_normal_speed, -2);
          EXPECT_DOUBLE_EQ(result->incoming_advecting_relative_normal_speed, -2);
        }
      }
    }
  }
}

TEST(LiquidWallContact3D, ObliqueReversedSlabPreservesIncidentSide) {
  constexpr Vec3 kNormal{0.6, 0.8, 0};
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {3, 4, 2},
      .stored_world_velocity = {-6, -8, 3},
      .advecting_world_velocity = {-6, -8, 4},
      .obstacle = TranslatingSlab3D{Scale(kNormal, -1), 0, 1, {}},
      .step_duration = 1,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact_fraction.has_value());
  EXPECT_NEAR(*result->contact_fraction, 0.4, kTolerance);
  ExpectVectorNear(result->outward_normal, kNormal);
  ExpectVectorNear(result->world_position, {0.6, 0.8, 6});
  ExpectVectorNear(result->stored_world_velocity, {0, 0, 3});
  ExpectVectorNear(result->advecting_world_velocity, {0, 0, 4});
  EXPECT_NEAR(result->incoming_advecting_relative_normal_speed, -10, kTolerance);
}

TEST(LiquidWallContact3D, OneSidedPlaneOnlyBlocksMotionIntoSolid) {
  const TranslatingHalfspace3D wall{{-1, 0, 0}, -2, {}};
  LiquidWallContactInputs3D inputs{
      .start_world_position = {0, 1, 0},
      .stored_world_velocity = {-3, 2, 0},
      .advecting_world_velocity = {4, 2, 0},
      .obstacle = wall,
      .step_duration = 1,
  };
  const auto hit = ResolveLiquidWallContact3D(inputs);
  ASSERT_TRUE(hit.ok()) << hit.status();
  ASSERT_TRUE(hit->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*hit->contact_fraction, 0.5);
  ExpectVectorNear(hit->outward_normal, {-1, 0, 0});
  ExpectVectorNear(hit->world_position, {2, 3, 0});
  ExpectVectorNear(hit->stored_world_velocity, {-3, 2, 0});
  ExpectVectorNear(hit->advecting_world_velocity, {0, 2, 0});
  EXPECT_DOUBLE_EQ(hit->incoming_stored_relative_normal_speed, 3);

  inputs.start_world_position = {2, 1, 0};
  inputs.advecting_world_velocity = {-4, 2, 0};
  const auto leaving = ResolveLiquidWallContact3D(inputs);
  ASSERT_TRUE(leaving.ok()) << leaving.status();
  EXPECT_FALSE(leaving->contact_fraction.has_value());
  ExpectVectorNear(leaving->world_position, {-2, 3, 0});
}

TEST(LiquidWallContact3D, TangentialMotionDoesNotTriggerFriction) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {2, 1, 0},
      .stored_world_velocity = {1, 5, 0},
      .advecting_world_velocity = {1, 3, 0},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 2, {1, 1, 0}},
      .step_duration = 2,
      .friction_coefficient = 10,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->contact_fraction.has_value());
  ExpectVectorNear(result->world_position, {4, 7, 0});
  ExpectVectorNear(result->stored_world_velocity, {1, 5, 0});
  ExpectVectorNear(result->advecting_world_velocity, {1, 3, 0});
}

TEST(LiquidWallContact3D, TouchingWallConstrainsStoredInwardVelocityDuringTangentAdvection) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {},
      .stored_world_velocity = {-2, 3, 0},
      .advecting_world_velocity = {0, 4, 0},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {}},
      .step_duration = 1,
      .friction_coefficient = 1,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*result->contact_fraction, 0);
  EXPECT_EQ(result->diagnostic, LiquidWallContactDiagnostic3D::kSweptImpact);
  ExpectVectorNear(result->stored_world_velocity, {0, 1, 0});
  ExpectVectorNear(result->advecting_world_velocity, {0, 4, 0});
  ExpectVectorNear(result->world_position, {0, 4, 0});
}

TEST(LiquidWallContact3D, FrictionUsesEachIncomingRelativeNormalSpeed) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {1, 2, 3},
      .stored_world_velocity = {-1, 1, 3},
      .advecting_world_velocity = {-3, 5, 3},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {1, 2, 3}},
      .step_duration = 1,
      .friction_coefficient = 0.5,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*result->contact_fraction, 0.25);
  EXPECT_DOUBLE_EQ(result->incoming_stored_relative_normal_speed, -2);
  EXPECT_DOUBLE_EQ(result->incoming_advecting_relative_normal_speed, -4);
  ExpectVectorNear(result->stored_world_velocity, {1, 2, 3});
  ExpectVectorNear(result->advecting_world_velocity, {1, 3, 3});
  ExpectVectorNear(result->world_position, {1, 5.5, 6});
}

TEST(LiquidWallContact3D, RecoversTolerablePenetrationOnNearestSide) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {-0.875, 2, 0},
      .stored_world_velocity = {1, 0, 0},
      .advecting_world_velocity = {-1, 3, 0},
      .obstacle = TranslatingSlab3D{{1, 0, 0}, 0, 1, {}},
      .step_duration = 1,
      .penetration_tolerance = 0.125,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*result->contact_fraction, 0);
  EXPECT_EQ(result->diagnostic, LiquidWallContactDiagnostic3D::kInitialPenetration);
  ExpectVectorNear(result->outward_normal, {-1, 0, 0});
  ExpectVectorNear(result->world_position, {-2, 5, 0});
  ExpectVectorNear(result->stored_world_velocity, {0, 0, 0});
  ExpectVectorNear(result->advecting_world_velocity, {-1, 3, 0});
}

TEST(LiquidWallContact3D, RejectsDeepAndAmbiguousPenetration) {
  LiquidWallContactInputs3D inputs{
      .start_world_position = {0.75, 0, 0},
      .obstacle = TranslatingSlab3D{{1, 0, 0}, 0, 1, {}},
      .step_duration = 1,
      .penetration_tolerance = 0.125,
  };
  EXPECT_EQ(ResolveLiquidWallContact3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.start_world_position = {};
  inputs.penetration_tolerance = 2;
  EXPECT_EQ(ResolveLiquidWallContact3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 4, {}};
  EXPECT_EQ(ResolveLiquidWallContact3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(LiquidWallContact3D, DetectsEndpointContactAndFreeAdvection) {
  LiquidWallContactInputs3D inputs{
      .start_world_position = {1, 2, 3},
      .stored_world_velocity = {3, 2, 1},
      .advecting_world_velocity = {-2, 0, 2},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {}},
      .step_duration = 0.5,
  };
  const auto hit = ResolveLiquidWallContact3D(inputs);
  ASSERT_TRUE(hit.ok()) << hit.status();
  ASSERT_TRUE(hit->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*hit->contact_fraction, 1);
  ExpectVectorNear(hit->world_position, {0, 2, 4});
  ExpectVectorNear(hit->advecting_world_velocity, {0, 0, 2});

  inputs.obstacle = std::monostate{};
  const auto free = ResolveLiquidWallContact3D(inputs);
  ASSERT_TRUE(free.ok()) << free.status();
  EXPECT_EQ(free->diagnostic, LiquidWallContactDiagnostic3D::kNone);
  EXPECT_FALSE(free->contact_fraction.has_value());
  ExpectVectorNear(free->world_position, {0, 2, 4});
  ExpectVectorNear(free->advecting_world_velocity, {-2, 0, 2});
  ExpectVectorNear(free->stored_world_velocity, {3, 2, 1});
}

TEST(LiquidWallContact3D, RejectsInvalidContactInputs) {
  const LiquidWallContactInputs3D valid{
      .start_world_position = {1, 0, 0},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {}},
      .step_duration = 1,
  };
  auto invalid = valid;
  invalid.start_world_position.x = kInfinity;
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid = valid;
  invalid.stored_world_velocity.y = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid = valid;
  invalid.step_duration = 0;
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid = valid;
  invalid.friction_coefficient = -1;
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid = valid;
  invalid.penetration_tolerance = kInfinity;
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid = valid;
  invalid.obstacle = TranslatingHalfspace3D{{2, 0, 0}, 0, {}};
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid.obstacle = TranslatingSlab3D{{1, 0, 0}, kInfinity, 1, {}};
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid.obstacle = TranslatingSlab3D{{1, 0, 0}, 0, 0, {}};
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
  invalid.obstacle = TranslatingSlab3D{{1, 0, 0}, 0, 1, {0, kInfinity, 0}};
  EXPECT_EQ(ResolveLiquidWallContact3D(invalid).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidWallContact3D, RejectsUnrepresentableTrajectoryAndWallPose) {
  LiquidWallContactInputs3D inputs{
      .start_world_position = {kMax, 0, 0},
      .advecting_world_velocity = {kMax, 0, 0},
      .step_duration = 2,
  };
  EXPECT_EQ(ResolveLiquidWallContact3D(inputs).status().code(), absl::StatusCode::kOutOfRange);
  inputs.advecting_world_velocity = {};
  inputs.obstacle = TranslatingHalfspace3D{{-1, 0, 0}, -kMax, {kMax, 0, 0}};
  EXPECT_EQ(ResolveLiquidWallContact3D(inputs).status().code(), absl::StatusCode::kOutOfRange);
  inputs.obstacle = TranslatingSlab3D{{1, 0, 0}, 1e20, 1, {}};
  EXPECT_EQ(ResolveLiquidWallContact3D(inputs).status().code(), absl::StatusCode::kOutOfRange);
  inputs.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {-kMax, 0, 0}};
  inputs.advecting_world_velocity = {kMax, 0, 0};
  inputs.step_duration = 0.5;
  EXPECT_EQ(ResolveLiquidWallContact3D(inputs).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(LiquidWallContact3D, CoMovingRoundedEndpointMayLandExactlyOnWall) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {0.125, 0, 0},
      .stored_world_velocity = {1e16, 0, 0},
      .advecting_world_velocity = {1e16, 0, 0},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {1e16, 0, 0}},
      .step_duration = 1,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(result->world_position.x, 1e16);
}

TEST(LiquidWallContact3D, CoMovingRoundedEndpointInsideExactWallRejects) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {1, 0, 0},
      .stored_world_velocity = {1e16, 0, 0},
      .advecting_world_velocity = {1e16, 0, 0},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0.5, {1e16, 0, 0}},
      .step_duration = 1,
  });
  EXPECT_EQ(result.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(LiquidWallContact3D, ExactObliqueTangencyDoesNotBecomeAnImpact) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {0, 0, 0},
      .stored_world_velocity = {0.8, -0.6, 3},
      .advecting_world_velocity = {0.8, -0.6, 2},
      .obstacle = TranslatingHalfspace3D{{0.6, 0.8, 0}, 0, {}},
      .step_duration = 1,
      .friction_coefficient = 2,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->contact_fraction.has_value());
  ExpectVectorNear(result->world_position, {0.8, -0.6, 2});
}

TEST(LiquidWallContact3D, FrictionlessContactPreservesSmallWorldTangentAtFastWall) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {1, 2, 0},
      .stored_world_velocity = {-3, 5, 0},
      .advecting_world_velocity = {-2, 1, 0},
      .obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {0, 1e20, 0}},
      .step_duration = 1,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*result->contact_fraction, 0.5);
  ExpectVectorNear(result->stored_world_velocity, {0, 5, 0});
  ExpectVectorNear(result->advecting_world_velocity, {0, 1, 0});
  ExpectVectorNear(result->world_position, {0, 3, 0});
}

TEST(LiquidWallContact3D, ObliqueMovingPlaneCarriesTheConstrainedRemainder) {
  const auto result = ResolveLiquidWallContact3D({
      .start_world_position = {3, 4, 0},
      .stored_world_velocity = {0, 0, -2},
      .advecting_world_velocity = {0, 0, 2},
      .obstacle = TranslatingHalfspace3D{{0.6, 0.8, 0}, 0, {3, 4, 1}},
      .step_duration = 1.5,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact_fraction.has_value());
  EXPECT_NEAR(*result->contact_fraction, 2.0 / 3.0, kTolerance);
  ExpectVectorNear(result->stored_world_velocity, {3, 4, -2});
  ExpectVectorNear(result->advecting_world_velocity, {3, 4, 2});
  ExpectVectorNear(result->world_position, {4.5, 6, 3});
}

}  // namespace
}  // namespace fluidity
