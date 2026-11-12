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
TEST(LiquidBoundaryGeometry3D, PredictorDensitySetsSharedAndFreeSurfaceMobility) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  const auto grid = cases.Grid();
  auto samples = cases.IsolatedLiquid(-1.0);
  samples.candidates.push_back({-1, -2, -2});
  for (auto& center : samples.centers) {
    const auto cell = center.cell_index;
    if (cell.x == -1 && cell.y == -2 && cell.z == -2) {
      center.signed_distance = -1.0;
    }
  }
  constexpr double kStepDuration = 0.01;
  const auto frozen = cases.Freeze(grid, samples, {}, 0.0, kStepDuration);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 2U);
  std::array fields{
      LiquidPressureCellFields3D{0.0, 100.0, 3.0, 2.0},
      LiquidPressureCellFields3D{0.0, 100.0, 1.0, 6.0},
  };
  const std::vector<double> velocity(frozen->faces().size());
  const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, velocity, 0.0});
  ASSERT_TRUE(links.ok()) << links.status();
  const auto projection = ProjectMacPressure3D({grid, kStepDuration, links->cells, links->faces});
  ASSERT_TRUE(projection.ok()) << projection.status();
  std::size_t shared_faces = 0;
  std::size_t surface_faces = 0;
  for (std::size_t face = 0; face < links->faces.size(); ++face) {
    const auto& link = links->faces[face];
    if (std::holds_alternative<InternalMacPressureLink3D>(link.pressure_link)) {
      ++shared_faces;
      EXPECT_EQ(link.inverse_density, 0.25);
      const double pressure_difference = projection->cell_pressures[0] - projection->cell_pressures[1];
      EXPECT_GT(pressure_difference, 1.0);
      EXPECT_NEAR(projection->corrected_face_velocities[face], kStepDuration * 0.25 * pressure_difference, 1e-12);
    } else if (std::holds_alternative<DirichletMacPressureLink3D>(link.pressure_link)) {
      ++surface_faces;
      const auto lattice = link.face_index.lattice_index;
      const int liquid_x = link.face_index.axis == FaceNormalAxis::kX && lattice.x == 0 ? -1 : lattice.x;
      EXPECT_EQ(link.inverse_density, liquid_x == -2 ? 0.5 : 1.0 / 6.0);
    }
  }
  EXPECT_EQ(shared_faces, 1U);
  EXPECT_EQ(surface_faces, 10U);
  fields[1].density = 0.0;
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*frozen, {fields, velocity, 0.0}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidBoundaryGeometry3D, ClipsGasVolumeAndSharedFaceAreas) {
  LiquidAirCellMeasureInputs3D inputs{
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .cell_index = {0, 0, 0},
      .obstacle = {},
      .liquid_corner_distances = {-0.25, 0.75, -0.25, 0.75, -0.25, 0.75, -0.25, 0.75},
  };
  const auto gas = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(gas.ok()) << gas.status();
  EXPECT_NEAR(gas->gas_volume, 0.75, 1e-15);
  EXPECT_EQ(gas->gas_face_areas, (std::array{0.0, 1.0, 0.75, 0.75, 0.75, 0.75}));
  inputs.cell_index.y = 1;
  const auto neighbor = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(neighbor.ok()) << neighbor.status();
  EXPECT_EQ(gas->gas_face_areas[3], neighbor->gas_face_areas[2]);

  inputs.cell_index.y = 0;
  inputs.obstacle = TranslatingHalfspace3D{{0.0, 1.0, 0.0}, 0.5, {}};
  const auto halfspace = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(halfspace.ok()) << halfspace.status();
  EXPECT_NEAR(halfspace->gas_volume, 0.375, 1e-15);
  EXPECT_EQ(halfspace->gas_face_areas, (std::array{0.0, 0.5, 0.0, 0.75, 0.375, 0.375}));
  inputs.obstacle = AxisAlignedLiquidTank3D{0.5, 1.0, 0.0, 0.0, 1.0, 0.0};
  const auto tank = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(tank.ok()) << tank.status();
  EXPECT_NEAR(tank->gas_volume, 0.5, 1e-15);
  EXPECT_EQ(tank->gas_face_areas, (std::array{0.0, 0.0, 0.0, 0.5, 0.0, 0.0}));
}

TEST(LiquidBoundaryGeometry3D, GasConnectivityRequiresSharedTriangleOpening) {
  LiquidAirCellMeasureInputs3D inputs{
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .cell_index = {0, 0, 0},
      .obstacle = {},
      .liquid_corner_distances = {-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},
  };
  EXPECT_EQ(MeasureLiquidAirCell3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.liquid_corner_distances[0] = 0.0;
  inputs.liquid_corner_distances[7] = 0.0;
  EXPECT_EQ(MeasureLiquidAirCell3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.liquid_corner_distances[0] = 1.0;
  const auto connected = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(connected.ok()) << connected.status();
  EXPECT_GT(connected->gas_volume, 0.0);
  EXPECT_GT(connected->gas_face_areas[0], 0.0);
  EXPECT_GT(connected->gas_face_areas[1], 0.0);
}

TEST(LiquidBoundaryGeometry3D, GasMeasureRejectsSlabSplitAndIgnoresZeroPlateaus) {
  LiquidAirCellMeasureInputs3D inputs{
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .cell_index = {0, 0, 0},
      .obstacle = TranslatingSlab3D{{1.0, 0.0, 0.0}, 0.5, 0.1, {}},
      .liquid_corner_distances = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
  };
  EXPECT_EQ(MeasureLiquidAirCell3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.obstacle = {};
  inputs.liquid_corner_distances.fill(0.0);
  const auto plateau = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(plateau.ok()) << plateau.status();
  EXPECT_EQ(plateau->gas_volume, 0.0);
  EXPECT_EQ(plateau->gas_face_areas, (std::array<double, 6>{}));
}

TEST(LiquidBoundaryGeometry3D, CavityGasPressureAddsSignedCapillaryJumpOnce) {
  GeometryCases cases(FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D);
  cases.CavityGasPressureAddsSignedCapillaryJumpOnce();
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
