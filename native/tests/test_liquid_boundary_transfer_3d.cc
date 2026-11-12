#include "liquid_boundary_transfer_3d_cases.h"
namespace fluidity {
namespace {
TEST(LiquidBoundaryTransfer3D, PressureImpulseUsesCompatibleMassAndRejectsUnsupportedFaces) {
  constexpr std::array<BlockIndex3D, 1> kBlocks{{{0, 0, 0}}};
  const auto grid = SparseMacGridView3D::Create({{}, 1.0}, kBlocks);
  ASSERT_TRUE(grid.ok()) << grid.status();
  const auto x_slot = grid->FaceSlot({FaceNormalAxis::kX, {1, 1, 1}});
  const auto y_slot = grid->FaceSlot({FaceNormalAxis::kY, {1, 1, 1}});
  ASSERT_TRUE(x_slot.ok() && y_slot.ok());
  std::array<LiquidBoundaryParticle3D, 3> particles{};
  particles[0].transfer.particle_mass = 2.0;
  particles[1].transfer.particle_mass = 3.0;
  particles[2].transfer.particle_mass = 8.0;
  std::array<std::array<CompatibleMacFaceStencil3D, 3>, 3> stencils{};
  stencils[0][0][0] = {.face_slot = *x_slot, .compatible = true, .normalized_weight = 0.5};
  stencils[1][0][0] = {.face_slot = *x_slot, .compatible = true, .normalized_weight = 1.0};
  stencils[2][0][0] = {.face_slot = *x_slot};
  std::vector<double> face_impulses(grid->face_count());
  std::vector<double> face_masses(grid->face_count());
  std::array<Vec3, 3> particle_impulses{};
  face_impulses[*x_slot] = -8.0;
  const auto distributed =
      DistributeCompatiblePressureImpulse3D(*grid, particles, stencils, face_impulses, face_masses, particle_impulses);
  ASSERT_TRUE(distributed.ok()) << distributed;
  EXPECT_DOUBLE_EQ(particle_impulses[0].x, -2.0);
  EXPECT_DOUBLE_EQ(particle_impulses[1].x, -6.0);
  EXPECT_DOUBLE_EQ(particle_impulses[2].x, 0.0);
  face_impulses[*y_slot] = 1.0;
  const auto unsupported =
      DistributeCompatiblePressureImpulse3D(*grid, particles, stencils, face_impulses, face_masses, particle_impulses);
  EXPECT_EQ(unsupported.code(), absl::StatusCode::kFailedPrecondition);
  const auto aliased = DistributeCompatiblePressureImpulse3D(
      *grid,
      particles,
      stencils,
      face_impulses,
      face_impulses,
      particle_impulses
  );
  EXPECT_EQ(aliased.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_DOUBLE_EQ(face_impulses[*x_slot], -8.0);
}

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
