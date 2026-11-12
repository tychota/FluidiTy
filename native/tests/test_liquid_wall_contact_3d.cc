#include "liquid_wall_contact_3d_cases.h"
namespace fluidity::boundary_test {
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
