#include "liquid_boundary_transfer_3d_cases.h"
namespace fluidity {
namespace {
TEST(LiquidBoundaryTransfer3D, MaskedDepositRecentersMomentumAndQuotientGradient) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.MaskedDepositRecentersMomentumAndQuotientGradient();
}
TEST(LiquidBoundaryTransfer3D, RejectsEmptyZeroSupportAndMissingExcludedOwner) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.RejectsEmptyZeroSupportAndMissingExcludedOwner();
}
TEST(LiquidBoundaryTransfer3D, RoundsActualMaskedContributionsAndHonorsAffineOff) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.RoundsActualMaskedContributionsAndHonorsAffineOff();
}
TEST(LiquidBoundaryTransfer3D, NormalizationSeparatesLowMassFromMissingAndRejectsMalformedQuanta) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.NormalizationSeparatesLowMassFromMissingAndRejectsMalformedQuanta();
}
TEST(LiquidBoundaryTransfer3D, SameSideAffineGhostsExcludePoisonedDonorsAndSeparateHistories) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.SameSideAffineGhostsExcludePoisonedDonorsAndSeparateHistories();
}
TEST(LiquidBoundaryTransfer3D, IdenticalHistoriesKeepParticleRetentionDistinctFromGridIncrement) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.IdenticalHistoriesKeepParticleRetentionDistinctFromGridIncrement();
}
TEST(LiquidBoundaryTransfer3D, WallHistoriesUseSeparateTimesAndConstrainObliqueNormal) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.WallHistoriesUseSeparateTimesAndConstrainObliqueNormal();
}
TEST(LiquidBoundaryTransfer3D, SingularDonorsUseNearestContinuationAndSuppressRetention) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.SingularDonorsUseNearestContinuationAndSuppressRetention();
}
TEST(LiquidBoundaryTransfer3D, LowMassDisablesAffineAndRetentionEvenWithFullRankGhost) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.LowMassDisablesAffineAndRetentionEvenWithFullRankGhost();
}
TEST(LiquidBoundaryTransfer3D, NonaffineCompleteFieldReportsFitSeparatelyFromTrueGradient) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.NonaffineCompleteFieldReportsFitSeparatelyFromTrueGradient();
}
TEST(LiquidBoundaryTransfer3D, RankRevealingQrRejectsNearDependentNormalizedDirection) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.RankRevealingQrRejectsNearDependentNormalizedDirection();
}
TEST(LiquidBoundaryTransfer3D, NearestFallbackBreaksDistanceTiesByOwnedSlotOnlyInNewStage) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.NearestFallbackBreaksDistanceTiesByOwnedSlotOnlyInNewStage();
}
TEST(LiquidBoundaryTransfer3D, QuantizesEachParticleBeforeSummingAndIncludesGuardEndpoint) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.QuantizesEachParticleBeforeSummingAndIncludesGuardEndpoint();
}
TEST(LiquidBoundaryTransfer3D, CompleteBoundaryDepositPreservesUnmaskedApiAtSignedTranslatedScale) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.CompleteBoundaryDepositPreservesUnmaskedApiAtSignedTranslatedScale();
}
TEST(LiquidBoundaryTransfer3D, RejectsMalformedHistoryCapacityAndArithmeticWithoutMutatingInputs) {
  boundary_transfer_test::TransferCases cases{
      DepositCompatibleMacQuanta3D,
      NormalizeLiquidMacFaces3D,
      GatherLiquidBoundaryParticles3D,
      internal::FitLiquidGhostAffine3D
  };
  cases.RejectsMalformedHistoryCapacityAndArithmeticWithoutMutatingInputs();
}
}  // namespace
}  // namespace fluidity
