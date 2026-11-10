#include "liquid_boundary_geometry_3d_cases.h"
namespace fluidity::boundary_test {
TEST(LiquidBoundaryGeometry3D, OpenTankClipsVolumeAndUsesPistonOnlyFlux) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.OpenTankClipsVolumeAndUsesPistonOnlyFlux();
}
TEST(LiquidBoundaryGeometry3D, OpenTankPreservesNarrowAndLiquidClippedMeasures) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.OpenTankPreservesNarrowAndLiquidClippedMeasures();
}
TEST(LiquidBoundaryGeometry3D, AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening();
}
TEST(LiquidBoundaryGeometry3D, ObliqueAndSimultaneousLiquidSolidClips) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.ObliqueAndSimultaneousLiquidSolidClips();
}
TEST(LiquidBoundaryGeometry3D, ThinSlabInvisibleToCornersAndSplitCellRefusal) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.ThinSlabInvisibleToCornersAndSplitCellRefusal();
}
TEST(LiquidBoundaryGeometry3D, SharedFaceHasOneOwnerAndMatchingTriangulation) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.SharedFaceHasOneOwnerAndMatchingTriangulation();
}
TEST(LiquidBoundaryGeometry3D, ThetaFixtureAndSignedPressureMapping) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.ThetaFixtureAndSignedPressureMapping();
}
TEST(LiquidBoundaryGeometry3D, PlaneAndSphereCenteredCurvature) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.PlaneAndSphereCenteredCurvature();
}
TEST(LiquidBoundaryGeometry3D, CapillaryLiftOnceAndCorrectionMismatchControl) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.CapillaryLiftOnceAndCorrectionMismatchControl();
}
TEST(LiquidBoundaryGeometry3D, RejectsMissingSamplesOwnersOrderingAndInvalidObstacle) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.RejectsMissingSamplesOwnersOrderingAndInvalidObstacle();
}
TEST(LiquidBoundaryGeometry3D, CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks();
}
TEST(LiquidBoundaryGeometry3D, FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory();
}
TEST(LiquidBoundaryGeometry3D, InternalLinkUsesFullFrozenAreaAndZeroJump) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.InternalLinkUsesFullFrozenAreaAndZeroJump();
}
TEST(LiquidBoundaryGeometry3D, RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor();
}
TEST(LiquidBoundaryGeometry3D, RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs();
}
TEST(LiquidBoundaryGeometry3D, InteriorPositiveSigmaDoesNotRequireUnusedCurvature) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.InteriorPositiveSigmaDoesNotRequireUnusedCurvature();
}
TEST(LiquidBoundaryGeometry3D, SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample();
}
TEST(LiquidBoundaryGeometry3D, ResolvesRepresentableTinyLiquidTetrahedron) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.ResolvesRepresentableTinyLiquidTetrahedron();
}
TEST(LiquidBoundaryGeometry3D, RejectsCapillaryLiftThatUnderflowsToZero) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.RejectsCapillaryLiftThatUnderflowsToZero();
}
TEST(LiquidBoundaryGeometry3D, ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume();
}
TEST(LiquidBoundaryGeometry3D, MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra();
}
TEST(LiquidBoundaryGeometry3D, PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure();
}
TEST(LiquidBoundaryGeometry3D, PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure();
}
}  // namespace fluidity::boundary_test
