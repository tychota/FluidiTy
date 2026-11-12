#include <cmath>
#include <numbers>

#include "liquid_wall_contact_3d_cases.h"
namespace fluidity::boundary_test {
TEST(LiquidWallContact3D, RotatingBoxTransfersOneOffCenterImpulse) {
  const RigidBody3D body{
      .body_id = 17,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .orientation = {.w = std::cos(std::numbers::pi / 8.0), .vector = {0.0, 0.0, -std::sin(std::numbers::pi / 8.0)}},
      .angular_velocity = {0.0, 0.0, std::numbers::pi / 2.0},
  };
  const auto step = AdvanceRigidBody3D(body, 1.0);
  ASSERT_TRUE(step.ok()) << step.status();
  constexpr Vec3 kHalfExtent{0.9, 0.1, 0.1};
  LiquidWallContactInputs3D input{
      .start_world_position = {0.75, 0.0, 0.0},
      .step_duration = 1.0,
  };
  const auto result = ResolveRotatingBoxContact3D(input, *step, kHalfExtent, 2.0);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact.contact_fraction.has_value());
  const double normal_x = 2.0 / 15.0;
  const double normal_y = std::sqrt(1.0 - normal_x * normal_x);
  const double normal_wall_speed = 0.75 * (std::numbers::pi / 2.0) * normal_y;
  EXPECT_NEAR(result->contact.outward_normal.x, normal_x, 1e-9);
  EXPECT_NEAR(result->contact.outward_normal.y, normal_y, 1e-9);
  EXPECT_NEAR(result->body_impulse.x, -2.0 * normal_wall_speed * normal_x, 1e-9);
  EXPECT_NEAR(result->body_impulse.y, -2.0 * normal_wall_speed * normal_y, 1e-9);
  EXPECT_NEAR(result->body_angular_impulse.z, 0.75 * result->body_impulse.y, 1e-9);
  EXPECT_DOUBLE_EQ(result->body_impulse.y, -2.0 * result->contact.stored_world_velocity.y);
  const Vec3 endpoint = result->contact.world_position;
  EXPECT_NEAR((endpoint.y - endpoint.x) / std::numbers::sqrt2, kHalfExtent.y, 1e-12);

  const auto next_step = AdvanceRigidBody3D(step->end, 0.125);
  ASSERT_TRUE(next_step.ok()) << next_step.status();
  const LiquidWallContactInputs3D repeated_input{
      .start_world_position = endpoint,
      .stored_world_velocity = result->contact.stored_world_velocity,
      .step_duration = 0.125,
  };
  const auto repeated = ResolveRotatingBoxContact3D(repeated_input, *next_step, kHalfExtent, 2.0);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  ASSERT_TRUE(repeated->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*repeated->contact.contact_fraction, 0.0);
  EXPECT_LT(repeated->contact.incoming_advecting_relative_normal_speed, 0.0);
  input.start_world_position = {1.0, 0.0, 0.0};
  const auto miss = ResolveRotatingBoxContact3D(input, *step, kHalfExtent, 2.0);
  ASSERT_TRUE(miss.ok()) << miss.status();
  EXPECT_FALSE(miss->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(miss->contact.world_position.x, 1.0);
  EXPECT_DOUBLE_EQ(miss->body_angular_impulse.z, 0.0);
}
TEST(LiquidWallContact3D, FiniteBoxDetectsCrossingAndReturnsOneBodyReaction) {
  const TranslatingContactBox3D box{{1.0, 0.0, 0.0}, {1.25, 1.0, 1.0}, {0.25, 0.0, 0.0}};
  const LiquidWallContactInputs3D input{
      .start_world_position = {0.0, 0.75, 0.5},
      .stored_world_velocity = {3.25, 0.0, 0.0},
      .advecting_world_velocity = {4.25, 0.0, 0.0},
      .step_duration = 1.0,
  };
  const auto result = ResolveTranslatingBoxContact3D(input, box, 2.0);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*result->contact.contact_fraction, 0.25);
  EXPECT_DOUBLE_EQ(result->contact.world_position.x, 1.25);
  EXPECT_DOUBLE_EQ(result->contact.stored_world_velocity.x, 0.25);
  EXPECT_DOUBLE_EQ(result->contact.advecting_world_velocity.x, 0.25);
  EXPECT_DOUBLE_EQ(result->body_impulse.x, 6.0);
  EXPECT_DOUBLE_EQ(result->body_angular_impulse.z, -1.5);

  auto grazing = input;
  grazing.start_world_position.y = 1.0;
  const auto miss = ResolveTranslatingBoxContact3D(grazing, box, 2.0);
  ASSERT_TRUE(miss.ok()) << miss.status();
  EXPECT_FALSE(miss->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(miss->contact.world_position.x, 4.25);
  EXPECT_DOUBLE_EQ(miss->body_impulse.x, 0.0);

  auto interior = input;
  interior.start_world_position.x = 1.125;
  EXPECT_EQ(ResolveTranslatingBoxContact3D(interior, box, 2.0).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(LiquidWallContact3D, FiniteBoxResolvesCornerTiesAndEndpointImpact) {
  const TranslatingContactBox3D box{{1.0, 1.0, 0.0}, {2.0, 2.0, 1.0}, {}};
  LiquidWallContactInputs3D input{
      .start_world_position = {0.0, 0.0, 0.5},
      .stored_world_velocity = {1.0, 1.0, 0.0},
      .advecting_world_velocity = {1.0, 1.0, 0.0},
      .step_duration = 1.0,
  };
  const auto tied = ResolveTranslatingBoxContact3D(input, box, 1.0);
  ASSERT_TRUE(tied.ok()) << tied.status();
  ASSERT_TRUE(tied->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*tied->contact.contact_fraction, 1.0);
  EXPECT_DOUBLE_EQ(tied->contact.outward_normal.x, -1.0);
  EXPECT_DOUBLE_EQ(tied->body_impulse.x, 1.0);
  EXPECT_DOUBLE_EQ(tied->body_impulse.y, 0.0);

  input.start_world_position = {0.0, 1.0, 0.5};
  input.advecting_world_velocity = {1.0, 1.0, 0.0};
  const auto corner_graze = ResolveTranslatingBoxContact3D(input, box, 1.0);
  ASSERT_TRUE(corner_graze.ok()) << corner_graze.status();
  EXPECT_FALSE(corner_graze->contact.contact_fraction.has_value());
}
TEST(LiquidWallContact3D, FiniteBoxConstrainsTouchingStoredMotionAndRejectsCollapsedBounds) {
  const TranslatingContactBox3D box{{1.0, 0.0, 0.0}, {1.25, 1.0, 1.0}, {0.25, 0.0, 0.0}};
  const LiquidWallContactInputs3D input{
      .start_world_position = {1.0, 0.25, 0.5},
      .stored_world_velocity = {1.25, 0.0, 0.0},
      .advecting_world_velocity = {0.25, 0.5, 0.0},
      .step_duration = 1.0,
  };
  const auto touching = ResolveTranslatingBoxContact3D(input, box, 3.0);
  ASSERT_TRUE(touching.ok()) << touching.status();
  ASSERT_TRUE(touching->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*touching->contact.contact_fraction, 0.0);
  EXPECT_DOUBLE_EQ(touching->contact.world_position.x, 1.25);
  EXPECT_DOUBLE_EQ(touching->contact.world_position.y, 0.75);
  EXPECT_DOUBLE_EQ(touching->body_impulse.x, 3.0);

  auto unresolved = box;
  unresolved.wall_velocity.x = 0x1p55;
  EXPECT_EQ(ResolveTranslatingBoxContact3D(input, unresolved, 3.0).status().code(), absl::StatusCode::kOutOfRange);
}
TEST(LiquidWallContact3D, OpenTankSweepsAndConstrainsBothVelocities) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.OpenTankSweepsAndConstrainsBothVelocities();
}
TEST(LiquidWallContact3D, OpenTankRejectsInvalidFrictionExtentsAndEndpointWidth) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.OpenTankRejectsInvalidFrictionExtentsAndEndpointWidth();
}
TEST(LiquidWallContact3D, ThinSlabSweepMatchesFixtureOnAllAxesAndReversedNormals) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.ThinSlabSweepMatchesFixtureOnAllAxesAndReversedNormals();
}
TEST(LiquidWallContact3D, ObliqueReversedSlabPreservesIncidentSide) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.ObliqueReversedSlabPreservesIncidentSide();
}
TEST(LiquidWallContact3D, OneSidedPlaneOnlyBlocksMotionIntoSolid) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.OneSidedPlaneOnlyBlocksMotionIntoSolid();
}
TEST(LiquidWallContact3D, TangentialMotionDoesNotTriggerFriction) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.TangentialMotionDoesNotTriggerFriction();
}
TEST(LiquidWallContact3D, TouchingWallConstrainsStoredInwardVelocityDuringTangentAdvection) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.TouchingWallConstrainsStoredInwardVelocityDuringTangentAdvection();
}
TEST(LiquidWallContact3D, FrictionUsesEachIncomingRelativeNormalSpeed) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.FrictionUsesEachIncomingRelativeNormalSpeed();
}
TEST(LiquidWallContact3D, RecoversTolerablePenetrationOnNearestSide) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.RecoversTolerablePenetrationOnNearestSide();
}
TEST(LiquidWallContact3D, RejectsDeepAndAmbiguousPenetration) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.RejectsDeepAndAmbiguousPenetration();
}
TEST(LiquidWallContact3D, DetectsEndpointContactAndFreeAdvection) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.DetectsEndpointContactAndFreeAdvection();
}
TEST(LiquidWallContact3D, RejectsInvalidContactInputs) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.RejectsInvalidContactInputs();
}
TEST(LiquidWallContact3D, RejectsUnrepresentableTrajectoryAndWallPose) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.RejectsUnrepresentableTrajectoryAndWallPose();
}
TEST(LiquidWallContact3D, CoMovingRoundedEndpointMayLandExactlyOnWall) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.CoMovingRoundedEndpointMayLandExactlyOnWall();
}
TEST(LiquidWallContact3D, CoMovingRoundedEndpointInsideExactWallRejects) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.CoMovingRoundedEndpointInsideExactWallRejects();
}
TEST(LiquidWallContact3D, ExactObliqueTangencyDoesNotBecomeAnImpact) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.ExactObliqueTangencyDoesNotBecomeAnImpact();
}
TEST(LiquidWallContact3D, FrictionlessContactPreservesSmallWorldTangentAtFastWall) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.FrictionlessContactPreservesSmallWorldTangentAtFastWall();
}
TEST(LiquidWallContact3D, ObliqueMovingPlaneCarriesTheConstrainedRemainder) {
  ContactCases cases(ResolveLiquidWallContact3D);
  cases.ObliqueMovingPlaneCarriesTheConstrainedRemainder();
}
TEST(ElasticBladeContact3D, MovingThinBladeAccountsForOnePhysicalReaction) {
  const TranslatingSlab3D blade{
      .unit_normal = {1.0, 0.0, 0.0},
      .start_center_coordinate = -1.0,
      .half_width = 1.0 / 32.0,
      .wall_velocity = {2.0, 0.0, 0.0},
  };
  const LiquidWallContactInputs3D inputs{
      .start_world_position = {0.0, 1.0, 0.0},
      .stored_world_velocity = {-1.0, 2.0, 0.0},
      .advecting_world_velocity = {},
      .obstacle = blade,
      .step_duration = 1.0,
      .friction_coefficient = 0.25,
  };
  const auto first = ResolveElasticBladeContact3D(inputs, 2.0, 17, {});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*first->contact.contact_fraction, 31.0 / 64.0);
  EXPECT_DOUBLE_EQ(first->contact.incoming_stored_relative_normal_speed, -3.0);
  EXPECT_DOUBLE_EQ(first->contact.incoming_advecting_relative_normal_speed, -2.0);
  EXPECT_DOUBLE_EQ(first->contact.stored_world_velocity.x, 2.0);
  EXPECT_DOUBLE_EQ(first->contact.stored_world_velocity.y, 1.25);
  EXPECT_DOUBLE_EQ(first->contact.advecting_world_velocity.x, 2.0);
  EXPECT_DOUBLE_EQ(first->particle_impulse.x, 6.0);
  EXPECT_DOUBLE_EQ(first->particle_impulse.y, -1.5);
  EXPECT_DOUBLE_EQ(first->particle_angular_impulse_about_world_origin.z, -6.0);
  EXPECT_DOUBLE_EQ(first->blade_work, 12.0);
  EXPECT_EQ(first->side.blade_id, 17);
  EXPECT_EQ(first->side.sign, 1);

  auto retreating = inputs;
  retreating.obstacle = TranslatingSlab3D{
      .unit_normal = {1.0, 0.0, 0.0},
      .start_center_coordinate = -1.0,
      .half_width = 1.0 / 32.0,
      .wall_velocity = {-2.0, 0.0, 0.0},
  };
  const auto second = ResolveElasticBladeContact3D(retreating, 2.0, 17, first->side);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_FALSE(second->contact.contact_fraction.has_value());
  EXPECT_EQ(second->side.blade_id, first->side.blade_id);
  EXPECT_EQ(second->side.sign, first->side.sign);
  EXPECT_DOUBLE_EQ(second->blade_work, 0.0);

  const ElasticBladeSide3D opposite_side{.blade_id = 17, .sign = -1};
  const auto wrong_side_without_contact = ResolveElasticBladeContact3D(retreating, 2.0, 17, opposite_side);
  EXPECT_EQ(wrong_side_without_contact.status().code(), absl::StatusCode::kFailedPrecondition);
  const auto rejected = ResolveElasticBladeContact3D(inputs, 2.0, 17, opposite_side);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(ElasticBladeContact3D, ReactionUsesCertifiedFacePosition) {
  const LiquidWallContactInputs3D inputs{
      .start_world_position = {100.0, 0.0, 0.0},
      .stored_world_velocity = {-3.0, 2.0, 0.0},
      .advecting_world_velocity = {-3.0, 0.0, 0.0},
      .obstacle =
          TranslatingSlab3D{
              .unit_normal = {1.0, 0.0, 0.0},
              .start_center_coordinate = 0.0,
              .half_width = 0.1,
              .wall_velocity = {},
          },
      .step_duration = 50.0,
      .friction_coefficient = 0.25,
  };
  const auto contact = ResolveElasticBladeContact3D(inputs, 1.0, 17, {});
  ASSERT_TRUE(contact.ok()) << contact.status();
  EXPECT_DOUBLE_EQ(contact->particle_angular_impulse_about_world_origin.z, -0.075);
}

}  // namespace fluidity::boundary_test
