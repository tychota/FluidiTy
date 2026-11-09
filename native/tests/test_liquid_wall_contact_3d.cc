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
}  // namespace fluidity::boundary_test
