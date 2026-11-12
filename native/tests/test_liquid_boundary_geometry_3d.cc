#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/rigid_body_3d.h"
#include "liquid_boundary_geometry_3d_cases.h"
namespace fluidity::boundary_test {
namespace {
constexpr auto kOwningFreeze =
    static_cast<absl::StatusOr<FrozenLiquidBoundaryGeometry3D> (*)(const LiquidBoundaryGeometryInputs3D&)>(
        FreezeLiquidBoundaryGeometry3D
    );
constexpr auto kOwningLinks = static_cast<
    absl::StatusOr<LiquidPressureProblem3D> (*)(const FrozenLiquidBoundaryGeometry3D&, const LiquidPressureFields3D&)>(
    BuildLiquidPressureLinks3D
);
}  // namespace

TEST(LiquidBoundaryGeometry3D, OrientedBoxSubtractsWetVolumeAndBlocksFace) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  const auto grid = cases.Grid();
  const auto samples = cases.Sample([](double, double y, double) {
    return y - 0.75;
  });
  const LiquidBoundaryGeometryInputs3D inputs{
      grid,
      samples.candidates,
      samples.centers,
      samples.corners,
      {},
      0.01,
      1.0,
      0.0,
  };
  const OrientedFloatingBoxGeometry3D box{
      .pose =
          {
              .center = {0.0, 0.75, 0.5},
              .orientation = {.w = std::sqrt(0.9), .vector = {0.0, std::sqrt(0.1), 0.0}},
          },
      .body_half_extent = {0.25, 0.25, 0.25},
      .liquid_height = 0.75,
  };
  const auto frozen = FreezeOrientedFloatingBoxLiquidBoundaryGeometry3D(inputs, box);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  // The x=0 face and y=0.75 surface each bisect the yawed cube.
  EXPECT_NEAR(frozen->cells().front().fluid_volume, 0.71875, 1e-13);
  const auto low_x = std::ranges::find_if(frozen->faces(), [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == -2;
  });
  ASSERT_NE(low_x, frozen->faces().end());
  EXPECT_NEAR(low_x->blocked_area, 0.3125, 1e-13);
  EXPECT_NEAR(low_x->open_area, 0.6875, 1e-13);

  const LiquidBoundaryGeometryViewInputs3D tank_inputs{
      grid.view(),
      samples.candidates,
      samples.centers,
      samples.corners,
      AxisAlignedLiquidTank3D{-0.75, 0.875, 0.125, 0.125, 0.875, 0.0},
      0.01,
      1.0,
      0.0,
  };
  auto workspace = LiquidBoundaryWorkspace3D::Create({.blocks = 8, .cells = 8, .faces = 48});
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto in_tank = workspace->FreezeOrientedFloatingBox(tank_inputs, box);
  ASSERT_TRUE(in_tank.ok()) << in_tank.status();
  ASSERT_EQ(in_tank->cells.size(), 1);
  EXPECT_NEAR(in_tank->cells.front().fluid_volume, 0.37890625, 1e-13);
  const auto tank_low_x = std::ranges::find_if(in_tank->faces, [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == -2;
  });
  ASSERT_NE(tank_low_x, in_tank->faces.end());
  EXPECT_NEAR(tank_low_x->open_area, 0.34375, 1e-13);

  auto overlaps_tank = box;
  overlaps_tank.pose.center.z = 0.25;
  const auto rejected = workspace->FreezeOrientedFloatingBox(tank_inputs, overlaps_tank);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(rejected.status().message().empty());
  EXPECT_NEAR(in_tank->cells.front().fluid_volume, 0.37890625, 1e-13);
}
TEST(LiquidBoundaryGeometry3D, FiniteBoxClipsPlanarLiquidVolumeAndApertures) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  const auto grid = cases.Grid();
  const auto samples = cases.Sample([](double, double y, double) {
    return y - 0.75;
  });
  const LiquidBoundaryGeometryInputs3D inputs{
      grid,
      samples.candidates,
      samples.centers,
      samples.corners,
      {},
      0.01,
      1000.0,
      0.0,
  };
  const AxisAlignedFloatingBoxGeometry3D box{
      .world_low = {0.0, 0.25, 0.25},
      .world_high = {0.5, 1.0, 0.75},
      .liquid_height = 0.75,
  };
  const auto frozen = FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, box);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  EXPECT_DOUBLE_EQ(frozen->cells().front().fluid_volume, 0.625);
  ASSERT_EQ(frozen->faces().size(), 6);
  const auto low_x = std::ranges::find_if(frozen->faces(), [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == -2;
  });
  ASSERT_NE(low_x, frozen->faces().end());
  EXPECT_DOUBLE_EQ(low_x->open_area, 0.625);
  EXPECT_DOUBLE_EQ(low_x->blocked_area, 0.375);
  EXPECT_DOUBLE_EQ(low_x->wall_normal_velocity, 0.0);

  auto interior = box;
  interior.world_low = {0.25, 0.25, 0.0};
  interior.world_high = {0.75, 0.5, 1.0};
  const auto connected = FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, interior);
  ASSERT_TRUE(connected.ok()) << connected.status();
  EXPECT_DOUBLE_EQ(connected->cells().front().fluid_volume, 0.625);

  auto split = box;
  split.world_low = {0.25, 0.0, 0.0};
  split.world_high = {0.5, 1.0, 1.0};
  EXPECT_EQ(
      FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, split).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto workspace = LiquidBoundaryWorkspace3D::Create({.blocks = 8, .cells = 8, .faces = 48});
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const LiquidBoundaryGeometryViewInputs3D view{
      grid.view(),
      samples.candidates,
      samples.centers,
      samples.corners,
      {},
      0.01,
      1000.0,
      0.0,
  };
  const auto accepted = workspace->FreezeFloatingBox(view, box);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  const auto rejected = workspace->FreezeFloatingBox(view, split);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(rejected.status().message().empty());
  EXPECT_DOUBLE_EQ(accepted->cells.front().fluid_volume, 0.625);
  auto overlap = inputs;
  overlap.obstacle = TranslatingHalfspace3D{{0.0, 1.0, 0.0}, 0.125, {}};
  EXPECT_EQ(
      FreezeFloatingBoxLiquidBoundaryGeometry3D(overlap, box).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
TEST(LiquidBoundaryGeometry3D, DisjointBoxAndTankDriveReciprocalSolvedPressure) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  const auto grid = cases.Grid();
  const auto samples = cases.Sample([](double, double y, double) {
    return y - 0.75;
  });
  const AxisAlignedLiquidTank3D tank{0.125, 0.875, 0.125, 0.125, 0.875, 0.0};
  const LiquidBoundaryGeometryInputs3D inputs{
      grid,
      samples.candidates,
      samples.centers,
      samples.corners,
      tank,
      0.125,
      1.0,
      0.0,
  };
  const AxisAlignedFloatingBoxGeometry3D box{
      .world_low = {0.25, 0.25, 0.25},
      .world_high = {0.5, 0.875, 0.5},
      .liquid_height = 0.75,
  };
  const auto frozen = FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, box);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  EXPECT_DOUBLE_EQ(frozen->cells().front().fluid_volume, 0.3203125);
  auto problem = cases.Links(*frozen);
  ASSERT_EQ(problem.cells.size(), 1);
  const auto open_count = std::ranges::count_if(problem.faces, [](const auto& face) {
    return face.open_area > 0.0;
  });
  EXPECT_EQ(open_count, 1);
  for (const auto& face : problem.faces) {
    if (face.open_area > 0.0) {
      EXPECT_EQ(face.face_index.axis, FaceNormalAxis::kY);
      EXPECT_DOUBLE_EQ(face.open_area, 0.5625);
    }
  }

  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.375, 0.5625, 0.375},
      .linear_velocity = {0.0, 0.5, 0.0},
  };
  const auto patches = BuildFloatingBoxPressurePatches3D(
      body.body_id,
      box.world_low,
      box.world_high,
      box.liquid_height,
      grid.geometry(),
      samples.candidates
  );
  ASSERT_TRUE(patches.ok()) << patches.status();
  const auto flux = ComputeRigidBodyInwardWallFlux3D(body, problem.cells.size(), *patches);
  ASSERT_TRUE(flux.ok()) << flux.status();
  EXPECT_DOUBLE_EQ(flux->front(), -0.03125);
  problem.cells.front().integrated_inward_wall_flux = flux->front();
  const auto solved = ProjectMacPressure3D({grid, inputs.step_duration, problem.cells, problem.faces});
  ASSERT_TRUE(solved.ok()) << solved.status();
  EXPECT_DOUBLE_EQ(solved->integrated_rhs.front(), -0.03125);
  EXPECT_NEAR(solved->cell_pressures.front(), -1.0 / 9.0, 1e-7);
  const auto loaded = AccumulateRigidBodyPressure3D(body, solved->cell_pressures, 0.0, *patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_NEAR(loaded->force.y, -1.0 / 144.0, 1e-8);
  EXPECT_DOUBLE_EQ(loaded->force.y * body.linear_velocity.y + solved->cell_pressures.front() * flux->front(), 0.0);

  auto touching = box;
  touching.world_low.x = tank.start_low_x;
  EXPECT_EQ(
      FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, touching).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}
TEST(LiquidBoundaryGeometry3D, OpenTankClipsVolumeAndUsesPistonOnlyFlux) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.OpenTankClipsVolumeAndUsesPistonOnlyFlux();
}
TEST(LiquidBoundaryGeometry3D, OpenTankPreservesNarrowAndLiquidClippedMeasures) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.OpenTankPreservesNarrowAndLiquidClippedMeasures();
}
TEST(LiquidBoundaryGeometry3D, AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening();
}
TEST(LiquidBoundaryGeometry3D, ObliqueAndSimultaneousLiquidSolidClips) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ObliqueAndSimultaneousLiquidSolidClips();
}
TEST(LiquidBoundaryGeometry3D, ThinSlabInvisibleToCornersAndSplitCellRefusal) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ThinSlabInvisibleToCornersAndSplitCellRefusal();
}
TEST(LiquidBoundaryGeometry3D, SharedFaceHasOneOwnerAndMatchingTriangulation) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.SharedFaceHasOneOwnerAndMatchingTriangulation();
}
TEST(LiquidBoundaryGeometry3D, ThetaFixtureAndSignedPressureMapping) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ThetaFixtureAndSignedPressureMapping();
}
TEST(LiquidBoundaryGeometry3D, PlaneAndSphereCenteredCurvature) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.PlaneAndSphereCenteredCurvature();
}
TEST(LiquidBoundaryGeometry3D, CapillaryLiftOnceAndCorrectionMismatchControl) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.CapillaryLiftOnceAndCorrectionMismatchControl();
}
TEST(LiquidBoundaryGeometry3D, RejectsMissingSamplesOwnersOrderingAndInvalidObstacle) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsMissingSamplesOwnersOrderingAndInvalidObstacle();
}
TEST(LiquidBoundaryGeometry3D, CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks();
}
TEST(LiquidBoundaryGeometry3D, FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory();
}
TEST(LiquidBoundaryGeometry3D, InternalLinkUsesFullFrozenAreaAndZeroJump) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.InternalLinkUsesFullFrozenAreaAndZeroJump();
}
TEST(LiquidBoundaryGeometry3D, RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor();
}
TEST(LiquidBoundaryGeometry3D, RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs();
}
TEST(LiquidBoundaryGeometry3D, InteriorPositiveSigmaDoesNotRequireUnusedCurvature) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.InteriorPositiveSigmaDoesNotRequireUnusedCurvature();
}
TEST(LiquidBoundaryGeometry3D, SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample();
}
TEST(LiquidBoundaryGeometry3D, ResolvesRepresentableTinyLiquidTetrahedron) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ResolvesRepresentableTinyLiquidTetrahedron();
}
TEST(LiquidBoundaryGeometry3D, RejectsCapillaryLiftThatUnderflowsToZero) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsCapillaryLiftThatUnderflowsToZero();
}
TEST(LiquidBoundaryGeometry3D, ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume();
}
TEST(LiquidBoundaryGeometry3D, MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra();
}
TEST(LiquidBoundaryGeometry3D, PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure();
}
TEST(LiquidBoundaryGeometry3D, PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure();
}
}  // namespace fluidity::boundary_test
