#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/liquid_step_workspace_3d.h"
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "gtest/gtest.h"
#include "liquid_boundary_transfer_3d_cases.h"
namespace fluidity {
namespace {
constexpr LiquidStateCapacities3D kCap{4, 8, 8, 1536, 0};
constexpr MacDepositQuantization3D kQuant{0x1p-30, 0x1p-30, 4};
struct Fixture {
  SparseMacGrid3D grid = *SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  std::array<CellIndex3D, 1> cells{CellIndex3D{1, 1, 1}};
  std::vector<LiquidCenterDistance3D> centers;
  std::vector<LiquidCornerDistance3D> corners;
  std::array<LiquidBoundaryParticle3D, 1> particles{};
  std::vector<LiquidBoundaryFaceTag3D> tags;
  std::vector<double> old_velocities;
  std::vector<double> new_velocities;
  std::vector<LiquidFaceValidity3D> validity;
  std::vector<LiquidPrescribedWallSample3D> walls;
  Fixture()
      : tags(grid.face_count(), {.side_tag = 0, .structurally_eligible = true}),
        old_velocities(grid.face_count(), 2),
        new_velocities(grid.face_count(), 3),
        validity(grid.face_count(), LiquidFaceValidity3D::kValid),
        walls(grid.face_count()) {
    for (int z = 0; z <= 2; ++z) {
      for (int y = 0; y <= 2; ++y) {
        for (int x = 0; x <= 2; ++x) {
          centers.push_back({{x, y, z}, x == 1 && y == 1 && z == 1 ? -0.5 : 0.5});
          corners.push_back({{x, y, z}, -1});
        }
      }
    }
    auto& p = particles[0];
    p.transfer.sorted_world_position = {2, 2, 2};
    p.transfer.trial_world_position = {2, 2, 2};
    p.transfer.particle_mass = 2;
    p.transfer.particle_velocity = {4, 5, 6};
    p.affine_enabled = true;
  }
  LiquidBoundaryGeometryViewInputs3D geometry() const {
    return {grid.view(), cells, centers, corners, {}, 0.01, 1000, 0};
  }
  LiquidBoundaryDepositViewInputs3D deposit() const {
    return {grid.view(), particles, tags, kQuant};
  }
  LiquidBoundaryVelocityHistory3D old_history() const {
    return {0, 0, old_velocities, validity, tags, walls};
  }
  LiquidBoundaryVelocityHistory3D new_history() const {
    return {0.01, 0.01, new_velocities, validity, tags, walls};
  }
  LiquidBoundaryGatherViewInputs3D gather() const {
    return {grid.view(), particles, 0, 0.01, old_history(), new_history()};
  }
};
TEST(LiquidBoundaryWorkspace3D, FrozenGeometryAndTypedLinksMatchOwningAndUnitCube) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  const auto frozen = FreezeLiquidBoundaryGeometry3D(f.geometry(), *workspace);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  const auto owned = FreezeLiquidBoundaryGeometry3D({f.grid, f.cells, f.centers, f.corners, {}, 0.01, 1000, 0});
  ASSERT_TRUE(owned.ok());
  ASSERT_EQ(frozen->cells.size(), 1);
  EXPECT_DOUBLE_EQ(frozen->cells[0].fluid_volume, 1);
  ASSERT_EQ(frozen->faces.size(), 6);
  EXPECT_DOUBLE_EQ(frozen->cells[0].fluid_volume, owned->cells()[0].fluid_volume);
  const std::array fields{LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 7}};
  const std::array<double, 6> velocities{};
  const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 9}, *workspace);
  ASSERT_TRUE(links.ok()) << links.status();
  const auto owned_links = BuildLiquidPressureLinks3D(*owned, {fields, velocities, 9});
  ASSERT_TRUE(owned_links.ok());
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_DOUBLE_EQ(frozen->faces[i].open_area, 1);
    EXPECT_EQ(frozen->faces[i].face_slot, owned->faces()[i].face_slot);
    EXPECT_DOUBLE_EQ(links->faces[i].pressure_gradient_distance, 0.5);
    EXPECT_DOUBLE_EQ(links->faces[i].pressure_gradient_distance, owned_links->faces[i].pressure_gradient_distance);
    EXPECT_DOUBLE_EQ(std::get<DirichletMacPressureLink3D>(links->faces[i].pressure_link).exterior_pressure, 9);
  }
}
TEST(LiquidBoundaryWorkspace3D, MaskedIntegerDepositsAndNormalizationMatchOwningExactly) {
  Fixture f;
  f.tags[0].structurally_eligible = false;
  f.tags[*f.grid.FaceSlot({FaceNormalAxis::kX, {2, 2, 2}})].side_tag = 1;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  const auto result = DepositCompatibleMacQuanta3D(f.deposit(), *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto owned = DepositCompatibleMacQuanta3D({f.grid, f.particles, f.tags, kQuant});
  ASSERT_TRUE(owned.ok());
  ASSERT_EQ(result->face_quanta.size(), owned->face_quanta.size());
  for (std::size_t i = 0; i < result->face_quanta.size(); ++i) {
    EXPECT_EQ(result->face_quanta[i].mass_quanta_sum, owned->face_quanta[i].mass_quanta_sum);
    EXPECT_EQ(result->face_quanta[i].pure_momentum_quanta_sum, owned->face_quanta[i].pure_momentum_quanta_sum);
    EXPECT_EQ(result->face_quanta[i].stress_force_quanta_sum, 0);
  }
  const auto normalized = NormalizeLiquidMacFaces3D(result->face_quanta, kQuant, 0, *workspace);
  ASSERT_TRUE(normalized.ok());
  const auto owned_normalized = NormalizeLiquidMacFaces3D(owned->face_quanta, kQuant, 0);
  ASSERT_TRUE(owned_normalized.ok());
  for (std::size_t i = 0; i < normalized->face_masses.size(); ++i) {
    EXPECT_EQ(normalized->validity[i], owned_normalized->validity[i]);
    EXPECT_DOUBLE_EQ(normalized->face_masses[i], owned_normalized->face_masses[i]);
    EXPECT_DOUBLE_EQ(normalized->face_velocities[i], owned_normalized->face_velocities[i]);
  }
}
TEST(LiquidBoundaryWorkspace3D, IndependentHistoriesSurviveNormalizationAndGatherMatchesConstantControl) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  const auto old = workspace->SaveHistory(f.old_history(), LiquidVelocityTimeLevel3D::kOld);
  ASSERT_TRUE(old.ok());
  const auto next = workspace->SaveHistory(f.new_history(), LiquidVelocityTimeLevel3D::kNew);
  ASSERT_TRUE(next.ok());
  ASSERT_NE(old->face_velocities.data(), next->face_velocities.data());
  const auto deposits = DepositCompatibleMacQuanta3D(f.deposit(), *workspace);
  ASSERT_TRUE(deposits.ok());
  ASSERT_TRUE(NormalizeLiquidMacFaces3D(deposits->face_quanta, kQuant, 0, *workspace).ok());
  auto inputs = f.gather();
  inputs.old_history = *old;
  inputs.new_history = *next;
  const auto gathered = GatherLiquidBoundaryParticles3D(inputs, *workspace);
  ASSERT_TRUE(gathered.ok()) << gathered.status();
  const auto owned = GatherLiquidBoundaryParticles3D({f.grid, f.particles, 0, 0.01, f.old_history(), f.new_history()});
  ASSERT_TRUE(owned.ok());
  EXPECT_DOUBLE_EQ(old->face_velocities[0], 2);
  EXPECT_DOUBLE_EQ(next->face_velocities[0], 3);
  const auto& p = gathered->particles[0];
  EXPECT_DOUBLE_EQ(p.new_pic_velocity.x, 3);
  EXPECT_DOUBLE_EQ(p.grid_velocity_increment.x, 1);
  EXPECT_DOUBLE_EQ(p.particle_minus_old_pic_velocity.x, 2);
  EXPECT_DOUBLE_EQ(p.fitted_velocity_slopes.x_row.x, 0);
  EXPECT_DOUBLE_EQ(p.velocity_gradient.x_row.x, 0);
  EXPECT_DOUBLE_EQ(p.new_pic_velocity.x, owned->particles[0].new_pic_velocity.x);
}
TEST(LiquidBoundaryWorkspace3D, RejectionsPreservePublishedRegionsIncludingMissingZeroWeightOwner) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  const auto frozen = FreezeLiquidBoundaryGeometry3D(f.geometry(), *workspace);
  ASSERT_TRUE(frozen.ok());
  auto bad_geometry = f.geometry();
  bad_geometry.corner_distances = {};
  EXPECT_EQ(
      FreezeLiquidBoundaryGeometry3D(bad_geometry, *workspace).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_DOUBLE_EQ(frozen->cells[0].fluid_volume, 1);
  const auto deposits = DepositCompatibleMacQuanta3D(f.deposit(), *workspace);
  ASSERT_TRUE(deposits.ok());
  const auto saved = deposits->face_quanta[*f.grid.FaceSlot({FaceNormalAxis::kX, {2, 2, 2}})].mass_quanta_sum;
  f.particles[0].transfer.sorted_world_position.x = 2.5;
  f.particles[0].transfer.trial_world_position.x = 2.5;
  EXPECT_EQ(
      DepositCompatibleMacQuanta3D(f.deposit(), *workspace).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(deposits->face_quanta[*f.grid.FaceSlot({FaceNormalAxis::kX, {2, 2, 2}})].mass_quanta_sum, saved);
  const auto normalized = NormalizeLiquidMacFaces3D(deposits->face_quanta, kQuant, 0, *workspace);
  ASSERT_TRUE(normalized.ok());
  EXPECT_EQ(
      NormalizeLiquidMacFaces3D(deposits->face_quanta, kQuant, -1, *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(normalized->face_masses[*f.grid.FaceSlot({FaceNormalAxis::kX, {2, 2, 2}})], 0.375);
}
TEST(LiquidBoundaryWorkspace3D, SmallRegionAndOverflowFailWithoutPublishing) {
  Fixture f;
  auto capacity = kCap;
  capacity.faces = 5;
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacity);
  ASSERT_TRUE(workspace.ok());
  EXPECT_EQ(
      DepositCompatibleMacQuanta3D(f.deposit(), *workspace).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(
      GatherLiquidBoundaryParticles3D(f.gather(), *workspace).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(
      workspace->SaveHistory(f.old_history(), LiquidVelocityTimeLevel3D::kOld).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  capacity.cells = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(QueryLiquidBoundaryWorkspaceBytes3D(capacity).status().code(), absl::StatusCode::kResourceExhausted);
}
TEST(LiquidBoundaryWorkspace3D, ContactValuePathPreservesOptionalSemanticsAndStatusCategories) {
  LiquidWallContactInputs3D inputs;
  inputs.start_world_position = {1, 0, 0};
  inputs.advecting_world_velocity = {1, 0, 0};
  inputs.step_duration = 0.5;
  auto result = ResolveLiquidWallContactBounded3D(inputs);
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result->contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(result->world_position.x, 1.5);
  inputs.step_duration = -1;
  EXPECT_EQ(ResolveLiquidWallContactBounded3D(inputs).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(LiquidBoundaryWorkspace3D, ByteQueryCoversEveryReservedArrayAndEmptyReuse) {
  const auto bytes = QueryLiquidBoundaryWorkspaceBytes3D(kCap);
  ASSERT_TRUE(bytes.ok());
  EXPECT_EQ(
      bytes->geometry_bytes,
      2 * kCap.cells
          * (27 * sizeof(LiquidCenterDistance3D) + sizeof(FrozenLiquidCell3D) + 6 * sizeof(FrozenLiquidFace3D)
             + sizeof(bool))
  );
  EXPECT_EQ(
      bytes->pressure_link_bytes,
      2 * (kCap.cells * sizeof(MacPressureCell3D) + kCap.faces * sizeof(MacPressureFace3D))
  );
  EXPECT_EQ(
      bytes->deposit_bytes,
      2 * (kCap.faces * sizeof(MacFaceDepositQuanta3D) + kCap.particles * sizeof(LiquidCompatibleStencils3D))
  );
  EXPECT_EQ(bytes->normalization_bytes, 2 * kCap.faces * (2 * sizeof(double) + sizeof(LiquidFaceValidity3D)));
  EXPECT_EQ(
      bytes->history_bytes,
      4 * kCap.faces
          * (sizeof(double) + sizeof(LiquidFaceValidity3D) + sizeof(LiquidBoundaryFaceTag3D)
             + sizeof(LiquidPrescribedWallSample3D))
  );
  EXPECT_EQ(
      bytes->total_bytes,
      bytes->geometry_bytes + bytes->pressure_link_bytes + bytes->deposit_bytes + bytes->normalization_bytes
          + bytes->history_bytes + bytes->gather_bytes
  );
  auto empty = LiquidBoundaryWorkspace3D::Create({});
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty->bytes().total_bytes, 0);
  EXPECT_EQ(empty->used_bytes(), 0);
  const auto grid = SparseMacGridView3D::Create({{0, 0, 0}, 1}, {});
  ASSERT_TRUE(grid.ok());
  EXPECT_TRUE(FreezeLiquidBoundaryGeometry3D({*grid, {}, {}, {}, {}, 0.01, 1000, 0}, *empty).ok());
  EXPECT_TRUE(DepositCompatibleMacQuanta3D({*grid, {}, {}, kQuant}, *empty).ok());
  EXPECT_TRUE(NormalizeLiquidMacFaces3D({}, kQuant, 0, *empty).ok());
  EXPECT_EQ(empty->high_water_bytes(), 0);
}
TEST(LiquidBoundaryWorkspace3D, StepWorkspaceAccountsAndOwnsAllNumericReservations) {
  Fixture f;
  auto step = LiquidStepWorkspace3D::Create(kCap, f.grid.geometry());
  ASSERT_TRUE(step.ok());
  const auto bytes = QueryLiquidWorkspaceBytes3D(kCap);
  ASSERT_TRUE(bytes.ok());
  const auto numeric = QueryLiquidBoundaryWorkspaceBytes3D(kCap);
  ASSERT_TRUE(numeric.ok());
  EXPECT_EQ(bytes->boundary_bytes, numeric->total_bytes);
  EXPECT_EQ(step->boundary().bytes().total_bytes, numeric->total_bytes);
  const auto result = DepositCompatibleMacQuanta3D(f.deposit(), step->boundary());
  ASSERT_TRUE(result.ok());
  EXPECT_GT(step->usage().used_bytes, 0);
  EXPECT_LE(step->usage().used_bytes, step->usage().reserved_bytes);
  EXPECT_EQ(step->boundary().high_water_bytes(), step->boundary().used_bytes());
}
TEST(LiquidBoundaryWorkspace3D, MalformedFrozenViewFailsBeforeLatticeAccessAndPreservesLinks) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  const auto frozen = FreezeLiquidBoundaryGeometry3D(f.geometry(), *workspace);
  ASSERT_TRUE(frozen.ok());
  const std::array fields{LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 0}};
  const std::array<double, 6> velocities{};
  const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 0}, *workspace);
  ASSERT_TRUE(links.ok());
  auto malformed = *frozen;
  malformed.liquid_density = -1;
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(malformed, {fields, velocities, 0}, *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  std::vector<FrozenLiquidFace3D> corrupt_faces(frozen->faces.begin(), frozen->faces.end());
  corrupt_faces[0].face_index.axis = static_cast<FaceNormalAxis>(99);
  malformed = *frozen;
  malformed.faces = corrupt_faces;
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(malformed, {fields, velocities, 0}, *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(links->faces[0].inverse_density, 0.001);
}
boundary_transfer_test::TransferCases BoundedTransferCases(LiquidBoundaryWorkspace3D& workspace) {
  return {
      [&workspace](const LiquidBoundaryDepositInputs3D& inputs) -> absl::StatusOr<LiquidBoundaryFaceDeposits3D> {
        const auto result = DepositCompatibleMacQuanta3D(
            {inputs.mac_grid.view(), inputs.particles, inputs.face_tags, inputs.quantization},
            workspace
        );
        if (!result.ok()) {
          return result.status();
        }
        return LiquidBoundaryFaceDeposits3D{
            {result->face_quanta.begin(), result->face_quanta.end()},
            {result->compatible_stencils.begin(), result->compatible_stencils.end()}
        };
      },
      [&workspace](
          std::span<const MacFaceDepositQuanta3D> quanta,
          MacDepositQuantization3D quantization,
          double minimum_mass
      ) -> absl::StatusOr<NormalizedLiquidMacFaces3D> {
        const auto result = NormalizeLiquidMacFaces3D(quanta, quantization, minimum_mass, workspace);
        if (!result.ok()) {
          return result.status();
        }
        return NormalizedLiquidMacFaces3D{
            {result->face_masses.begin(), result->face_masses.end()},
            {result->face_velocities.begin(), result->face_velocities.end()},
            {result->validity.begin(), result->validity.end()}
        };
      },
      [&workspace](const LiquidBoundaryGatherInputs3D& inputs) -> absl::StatusOr<LiquidBoundaryGatherResults3D> {
        const auto result = GatherLiquidBoundaryParticles3D(
            {inputs.mac_grid.view(),
             inputs.particles,
             inputs.old_time,
             inputs.new_time,
             inputs.old_history,
             inputs.new_history},
            workspace
        );
        if (!result.ok()) {
          return result.status();
        }
        return LiquidBoundaryGatherResults3D{
            {result->particles.begin(), result->particles.end()},
            result->old_pic_fallback_particle_count,
            result->new_pic_fallback_particle_count,
            result->low_mass_fallback_particle_count,
        };
      },
      internal::FitLiquidGhostAffine3D,
  };
}
TEST(LiquidBoundaryWorkspace3D, MaskedDepositRecentersMomentumAndQuotientGradient) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.MaskedDepositRecentersMomentumAndQuotientGradient();
}
TEST(LiquidBoundaryWorkspace3D, RejectsEmptyZeroSupportAndMissingExcludedOwner) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.RejectsEmptyZeroSupportAndMissingExcludedOwner();
}
TEST(LiquidBoundaryWorkspace3D, RoundsActualMaskedContributionsAndHonorsAffineOff) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.RoundsActualMaskedContributionsAndHonorsAffineOff();
}
TEST(LiquidBoundaryWorkspace3D, NormalizationSeparatesLowMassFromMissingAndRejectsMalformedQuanta) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.NormalizationSeparatesLowMassFromMissingAndRejectsMalformedQuanta();
}
TEST(LiquidBoundaryWorkspace3D, SameSideAffineGhostsExcludePoisonedDonorsAndSeparateHistories) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.SameSideAffineGhostsExcludePoisonedDonorsAndSeparateHistories();
}
TEST(LiquidBoundaryWorkspace3D, IdenticalHistoriesKeepParticleRetentionDistinctFromGridIncrement) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.IdenticalHistoriesKeepParticleRetentionDistinctFromGridIncrement();
}
TEST(LiquidBoundaryWorkspace3D, WallHistoriesUseSeparateTimesAndConstrainObliqueNormal) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.WallHistoriesUseSeparateTimesAndConstrainObliqueNormal();
}
TEST(LiquidBoundaryWorkspace3D, SingularDonorsUseNearestContinuationAndSuppressRetention) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.SingularDonorsUseNearestContinuationAndSuppressRetention();
}
TEST(LiquidBoundaryWorkspace3D, LowMassDisablesAffineAndRetentionEvenWithFullRankGhost) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.LowMassDisablesAffineAndRetentionEvenWithFullRankGhost();
}
TEST(LiquidBoundaryWorkspace3D, NonaffineCompleteFieldReportsFitSeparatelyFromTrueGradient) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.NonaffineCompleteFieldReportsFitSeparatelyFromTrueGradient();
}
TEST(LiquidBoundaryWorkspace3D, NearestFallbackBreaksDistanceTiesByOwnedSlotOnlyInNewStage) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.NearestFallbackBreaksDistanceTiesByOwnedSlotOnlyInNewStage();
}
TEST(LiquidBoundaryWorkspace3D, QuantizesEachParticleBeforeSummingAndIncludesGuardEndpoint) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.QuantizesEachParticleBeforeSummingAndIncludesGuardEndpoint();
}
TEST(LiquidBoundaryWorkspace3D, CompleteBoundaryDepositPreservesUnmaskedApiAtSignedTranslatedScale) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.CompleteBoundaryDepositPreservesUnmaskedApiAtSignedTranslatedScale();
}
TEST(LiquidBoundaryWorkspace3D, RejectsMalformedHistoryCapacityAndArithmeticWithoutMutatingInputs) {
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto cases = BoundedTransferCases(*workspace);
  cases.RejectsMalformedHistoryCapacityAndArithmeticWithoutMutatingInputs();
}
TEST(LiquidBoundaryWorkspace3D, MovingWallAndThinSlabGeometryParityAndRejectedLinksKeepPriorView) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  auto inputs = f.geometry();
  inputs.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 1.25, {2, 3, 4}};
  const auto frozen = FreezeLiquidBoundaryGeometry3D(inputs, *workspace);
  ASSERT_TRUE(frozen.ok());
  EXPECT_DOUBLE_EQ(frozen->cells[0].fluid_volume, 0.75);
  const auto owned =
      FreezeLiquidBoundaryGeometry3D({f.grid, f.cells, f.centers, f.corners, inputs.obstacle, 0.01, 1000, 0});
  ASSERT_TRUE(owned.ok());
  for (std::size_t i = 0; i < frozen->faces.size(); ++i) {
    EXPECT_DOUBLE_EQ(frozen->faces[i].open_area, owned->faces()[i].open_area);
    EXPECT_DOUBLE_EQ(frozen->faces[i].wall_normal_velocity, 2 + static_cast<int>(frozen->faces[i].face_index.axis));
  }
  const std::array fields{LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 0}};
  std::array<double, 6> velocities{};
  const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 5}, *workspace);
  ASSERT_TRUE(links.ok());
  velocities.back() = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 5}, *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(links->faces.back().predicted_normal_velocity, 0);
  inputs.obstacle = TranslatingSlab3D{{1, 0, 0}, 1.5, 0.01, {}};
  EXPECT_EQ(FreezeLiquidBoundaryGeometry3D(inputs, *workspace).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_DOUBLE_EQ(frozen->cells[0].fluid_volume, 0.75);
}
TEST(LiquidBoundaryWorkspace3D, PositiveSigmaCurvatureAndCapillaryBoundMatchOwning) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  // Sphere centered one unit left of the liquid center: |gradient|=1, curvature=2/R.
  for (auto& sample : f.centers) {
    const double x = sample.cell_index.x + 0.5;
    const double y = sample.cell_index.y + 0.5;
    const double z = sample.cell_index.z + 0.5;
    sample.signed_distance = std::sqrt((x - 0.5) * (x - 0.5) + (y - 1.5) * (y - 1.5) + (z - 1.5) * (z - 1.5)) - 1.1;
  }
  auto inputs = f.geometry();
  inputs.surface_tension = 0.5;
  const auto frozen = FreezeLiquidBoundaryGeometry3D(inputs, *workspace);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  const auto owned = FreezeLiquidBoundaryGeometry3D({f.grid, f.cells, f.centers, f.corners, {}, 0.01, 1000, 0.5});
  ASSERT_TRUE(owned.ok());
  EXPECT_DOUBLE_EQ(frozen->cells[0].curvature, owned->cells()[0].curvature);
  EXPECT_DOUBLE_EQ(frozen->capillary_step_bound, owned->capillary_step_bound());
  inputs.step_duration = frozen->capillary_step_bound * 2;
  EXPECT_EQ(FreezeLiquidBoundaryGeometry3D(inputs, *workspace).status().code(), absl::StatusCode::kFailedPrecondition);
}
TEST(LiquidBoundaryWorkspace3D, LateGatherAndHistoryFailuresPreservePublishedResultsAcrossReuse) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  const auto gathered = GatherLiquidBoundaryParticles3D(f.gather(), *workspace);
  ASSERT_TRUE(gathered.ok());
  const auto old = workspace->SaveHistory(f.old_history(), LiquidVelocityTimeLevel3D::kOld);
  ASSERT_TRUE(old.ok());
  f.new_velocities.back() = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      GatherLiquidBoundaryParticles3D(f.gather(), *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  f.old_velocities.back() = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      workspace->SaveHistory(f.old_history(), LiquidVelocityTimeLevel3D::kOld).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(old->face_velocities.back(), 2);
  EXPECT_DOUBLE_EQ(gathered->particles[0].new_pic_velocity.x, 3);
  f.old_velocities.back() = 2;
  f.new_velocities.back() = 3;
  auto empty = f.gather();
  empty.particles = {};
  ASSERT_TRUE(GatherLiquidBoundaryParticles3D(empty, *workspace).ok());
  const auto refilled = GatherLiquidBoundaryParticles3D(f.gather(), *workspace);
  ASSERT_TRUE(refilled.ok());
  EXPECT_DOUBLE_EQ(refilled->particles[0].grid_velocity_increment.x, 1);
  EXPECT_FALSE(refilled->particles[0].new_pic_fallback);
  EXPECT_GE(workspace->high_water_bytes(), workspace->used_bytes());
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t allocations = 0;
std::size_t frees = 0;
std::size_t fail_request = 0;
_CRT_ALLOC_HOOK previous_hook = nullptr;
int AllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int block_type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++allocations;
    if (allocations == fail_request) {
      return 0;
    }
  }
  if (operation == _HOOK_FREE) {
    ++frees;
  }
  return previous_hook == nullptr ? 1 : previous_hook(operation, data, size, block_type, request, filename, line);
}
TEST(LiquidBoundaryWorkspace3D, SuccessfulAndRejectedOperationsAndStatusDestructionHaveNoHeapActivity) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create({1, 1, 1, 192, 0});
  ASSERT_TRUE(workspace.ok());
  const std::array fields{LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 0}};
  const std::array<double, 6> velocities{};
  bool success = true;
  bool rejected = true;
  allocations = 0;
  frees = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    const auto frozen = FreezeLiquidBoundaryGeometry3D(f.geometry(), *workspace);
    success = success && frozen.ok();
    if (frozen.ok()) {
      const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 0}, *workspace);
      success = success && links.ok();
    }
    const auto deposits = DepositCompatibleMacQuanta3D(f.deposit(), *workspace);
    success = success && deposits.ok();
    if (deposits.ok()) {
      const auto normalized = NormalizeLiquidMacFaces3D(deposits->face_quanta, kQuant, 0, *workspace);
      success = success && normalized.ok();
    }
    const auto old = workspace->SaveHistory(f.old_history(), LiquidVelocityTimeLevel3D::kOld);
    success = success && old.ok();
    const auto next = workspace->SaveHistory(f.new_history(), LiquidVelocityTimeLevel3D::kNew);
    success = success && next.ok();
    const auto gathered = GatherLiquidBoundaryParticles3D(f.gather(), *workspace);
    success = success && gathered.ok();
    auto bad = f.geometry();
    bad.step_duration = -1;
    const auto geometry_error = FreezeLiquidBoundaryGeometry3D(bad, *workspace);
    rejected = rejected && !geometry_error.ok();
    auto bad_gather = f.gather();
    bad_gather.new_time = 0;
    const auto gather_error = GatherLiquidBoundaryParticles3D(bad_gather, *workspace);
    rejected = rejected && !gather_error.ok();
    f.particles[0].transfer.particle_mass = -1;
    const auto deposit_error = DepositCompatibleMacQuanta3D(f.deposit(), *workspace);
    rejected = rejected && !deposit_error.ok();
    const auto contact_error = ResolveLiquidWallContactBounded3D({});
    rejected = rejected && !contact_error.ok();
  }
  _CrtSetAllocHook(previous_hook);
  EXPECT_TRUE(success);
  EXPECT_TRUE(rejected);
  EXPECT_EQ(allocations, 0);
  EXPECT_EQ(frees, 0);
}
TEST(LiquidBoundaryWorkspace3D, GhostAndLateRejectedAttemptsDoNotAllocateOrDestroyHeapStorage) {
  Fixture f;
  auto workspace = LiquidBoundaryWorkspace3D::Create(kCap);
  ASSERT_TRUE(workspace.ok());
  const auto missing_slot = *f.grid.FaceSlot({FaceNormalAxis::kX, {2, 2, 2}});
  f.validity[missing_slot] = LiquidFaceValidity3D::kMissing;
  const auto frozen = FreezeLiquidBoundaryGeometry3D(f.geometry(), *workspace);
  ASSERT_TRUE(frozen.ok());
  const auto deposited = DepositCompatibleMacQuanta3D(f.deposit(), *workspace);
  ASSERT_TRUE(deposited.ok());
  const auto normalized = NormalizeLiquidMacFaces3D(deposited->face_quanta, kQuant, 0, *workspace);
  ASSERT_TRUE(normalized.ok());
  const auto gathered = GatherLiquidBoundaryParticles3D(f.gather(), *workspace);
  ASSERT_TRUE(gathered.ok());
  std::array particles{f.particles[0], f.particles[0]};
  particles[1].transfer.particle_velocity.x = std::numeric_limits<double>::max();
  auto invalid_deposit = f.deposit();
  invalid_deposit.particles = particles;
  auto invalid_gather = f.gather();
  invalid_gather.particles = particles;
  std::vector<MacFaceDepositQuanta3D> corrupt_quanta(deposited->face_quanta.begin(), deposited->face_quanta.end());
  corrupt_quanta.back().stress_force_quanta_sum = 1;
  std::array<LiquidPressureCellFields3D, 1> fields{
      LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 0}
  };
  std::array<double, 6> velocities{};
  velocities.back() = std::numeric_limits<double>::infinity();
  bool successful_ghost = false;
  std::array<absl::StatusCode, 5> codes{};
  allocations = 0;
  frees = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    const auto ghost = GatherLiquidBoundaryParticles3D(f.gather(), *workspace);
    successful_ghost = ghost.ok();
    const auto deposit_error = DepositCompatibleMacQuanta3D(invalid_deposit, *workspace);
    codes[0] = deposit_error.status().code();
    const auto normalize_error = NormalizeLiquidMacFaces3D(corrupt_quanta, kQuant, 0, *workspace);
    codes[1] = normalize_error.status().code();
    particles[1].transfer.trial_world_position.x = 2.5;
    particles[1].transfer.sorted_world_position.x = 2.5;
    const auto gather_error = GatherLiquidBoundaryParticles3D(invalid_gather, *workspace);
    codes[2] = gather_error.status().code();
    const auto link_error = BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 0}, *workspace);
    codes[3] = link_error.status().code();
    auto invalid_geometry = f.geometry();
    invalid_geometry.corner_distances = {};
    const auto geometry_error = FreezeLiquidBoundaryGeometry3D(invalid_geometry, *workspace);
    codes[4] = geometry_error.status().code();
  }
  _CrtSetAllocHook(previous_hook);
  EXPECT_TRUE(successful_ghost);
  EXPECT_EQ(allocations, 0);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(codes[0], absl::StatusCode::kOutOfRange);
  EXPECT_EQ(codes[1], absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(codes[2], absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(codes[3], absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(codes[4], absl::StatusCode::kFailedPrecondition);
  EXPECT_DOUBLE_EQ(frozen->cells[0].fluid_volume, 1);
  EXPECT_EQ(deposited->face_quanta[missing_slot].mass_quanta_sum, std::int64_t{3} << 27);
  EXPECT_DOUBLE_EQ(normalized->face_masses[missing_slot], 0.375);
}
TEST(LiquidBoundaryWorkspace3D, ActualPartialReservationFailureHasNoSecondaryErrorAllocation) {
  allocations = 0;
  frees = 0;
  fail_request = 4;
  absl::StatusCode code{};
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    const auto result = LiquidBoundaryWorkspace3D::Create(kCap);
    code = result.status().code();
  }
  _CrtSetAllocHook(previous_hook);
  fail_request = 0;
  EXPECT_EQ(code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(allocations, 4);
  EXPECT_EQ(frees, 3);
}
#endif
}  // namespace
}  // namespace fluidity
