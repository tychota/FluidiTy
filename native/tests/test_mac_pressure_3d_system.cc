#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "../src/mac_pressure_3d_system.h"
#include "absl/status/status.h"
#include "adaptive_pressure_checkpoint_fixture.h"
#include "fluidity/host/mac_pressure_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"
#include "mac_pressure_explicit_cases.h"

namespace fluidity {
namespace {

using pressure_internal::EvaluateMacPressure3D;
using pressure_internal::PrepareMacPressureSystem3D;
using pressure_test::kExplicitPressureCases;
using pressure_test::MakeExplicitPressureFixture;

constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
constexpr std::array kUnitBlock{BlockIndex3D{0, 0, 0}};
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr auto kMinIndex = std::numeric_limits<std::int32_t>::min();
constexpr auto kMaxIndex = std::numeric_limits<std::int32_t>::max();

std::array<MacPressureFace3D, 6> SealedFacesAtX(std::int32_t x) {
  return {{
      {.face_index = {FaceNormalAxis::kX, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kX, {x + 1, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kY, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kY, {x, 1, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kZ, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kZ, {x, 0, 1}}, .pressure_link = SealedMacPressureLink3D{}},
  }};
}

MacPressureCell3D IncompressibleCellAtX(std::int32_t x) {
  return {.cell_index = {x, 0, 0}, .fluid_volume = 1.0, .bulk_modulus = kInfinity};
}

TEST(MacPressureExplicitGraph3D, BindsPhysicalInputsToCanonicalAdaptiveSubfaces) {
  auto fixture = pressure_test::MakeAdaptiveCutPressureFixture();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto& topology = fixture->topology;
  const auto& cells = fixture->cells;
  auto& subfaces = fixture->subfaces;
  const auto internal_slot = fixture->internal_slot;
  const auto graph = pressure_internal::PrepareAdaptiveMacPressureSystem3D(topology, 1.0, cells, subfaces);
  ASSERT_TRUE(graph.ok()) << graph.status();
  EXPECT_EQ(graph->links().size(), subfaces.size());
  EXPECT_DOUBLE_EQ(graph->links()[internal_slot].blocked_area, 0.125);
  EXPECT_DOUBLE_EQ(graph->links()[internal_slot].prescribed_wall_flux, 0.0625);
  EXPECT_DOUBLE_EQ(graph->cells()[0].integrated_target_flux, 0.09375);
  const std::array pressure{1.28125, 1.09375};
  const auto result = EvaluateMacPressure3D(*graph, pressure);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NEAR(result->diagnostics.max_cell_constraint_defect, 0, 1e-12);

  auto workspace = MacPressureWorkspace3D::Create(cells.size(), subfaces.size());
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto projected = workspace->ProjectPrepared(*graph);
  ASSERT_TRUE(projected.ok()) << projected.status();
  EXPECT_NEAR(projected->cell_pressures[0], pressure[0], 1e-6);
  EXPECT_NEAR(projected->cell_pressures[1], pressure[1], 1e-6);
  EXPECT_NEAR(projected->corrected_face_velocities[internal_slot], 0.25, 1e-6);
  EXPECT_NEAR(projected->corrected_face_velocities[fixture->outlet_slot], 0.375, 1e-6);

  auto sealed_cells = cells;
  sealed_cells[0].target_divergence = 0.0625;
  subfaces[internal_slot].open_area_m2 = 0;
  const auto sealed = pressure_internal::PrepareAdaptiveMacPressureSystem3D(topology, 1, sealed_cells, subfaces);
  ASSERT_TRUE(sealed.ok()) << sealed.status();
  ASSERT_EQ(sealed->gauge_cell_ordinals().size(), 1U);
  EXPECT_EQ(sealed->gauge_cell_ordinals().front(), 0U);
  const std::array sealed_pressures{0.0, 1.0625};
  const auto sealed_result = EvaluateMacPressure3D(*sealed, sealed_pressures);
  ASSERT_TRUE(sealed_result.ok()) << sealed_result.status();
  EXPECT_NEAR(sealed_result->diagnostics.max_cell_constraint_defect, 0, 1e-12);
  subfaces[internal_slot].open_area_m2 = 0.5;
  EXPECT_EQ(
      pressure_internal::PrepareAdaptiveMacPressureSystem3D(topology, 1.0, cells, subfaces).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
TEST(MacPressureExplicitGraph3D, CheckpointReplaysAcceptedAdaptivePressure) {
  auto workspace = MacPressureWorkspace3D::Create(2, 14);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  pressure_test::QualifyAdaptivePressureCheckpoint([&](const auto& system) {
    return workspace->ProjectPrepared(system);
  });
}
TEST(MacPressureExplicitGraph3D, ProjectsExteriorPressureAndSignedWallFlux) {
  const std::array cells{
      IncompressibleCellAtX(0),
      MacPressureCell3D{
          .cell_index = {2, 0, 0},
          .fluid_volume = 0.125,
          .target_divergence = 0.0,
          .bulk_modulus = kInfinity,
          .previous_pressure = 0.0,
      },
  };
  const std::array interior{
      pressure_internal::ExplicitMacPressureFace3D{
          .face_slot = 1,
          .tail_cell_ordinal = 0,
          .head_cell_ordinal = 1,
          .open_area = 0.25,
          .inverse_density = 1.0,
          .pressure_gradient_distance = 0.75,
          .predicted_normal_velocity = 0.0,
      },
  };
  std::array exterior{
      pressure_internal::ExplicitMacPressureBoundaryFace3D{
          .face_slot = 0,
          .cell_ordinal = 1,
          .cell_is_tail = false,
          .full_face_area = 0.25,
          .open_area = 0.125,
          .inverse_density = 1.0,
          .pressure_gradient_distance = 0.25,
          .predicted_normal_velocity = 0.0,
          .prescribed_wall_flux = 0.0625,
          .exterior_pressure = 1.0,
      },
      pressure_internal::ExplicitMacPressureBoundaryFace3D{
          .face_slot = 2,
          .cell_ordinal = 0,
          .cell_is_tail = true,
          .full_face_area = 1.0,
          .open_area = 0.0,
          .inverse_density = 0.0,
          .pressure_gradient_distance = 0.0,
          .predicted_normal_velocity = 0.0,
          .prescribed_wall_flux = -0.125,
          .exterior_pressure = std::nullopt,
      },
  };

  const auto prepared = pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, cells, interior, exterior);
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  EXPECT_TRUE(prepared->gauge_cell_ordinals().empty());
  const std::array analytic_pressures{1.75, 1.375};
  const auto analytic = EvaluateMacPressure3D(*prepared, analytic_pressures);
  ASSERT_TRUE(analytic.ok()) << analytic.status();
  EXPECT_NEAR(analytic->diagnostics.max_cell_constraint_defect, 0.0, 1e-12);
  EXPECT_NEAR(analytic->corrected_face_velocities[0], -1.5, 1e-12);
  EXPECT_NEAR(analytic->corrected_face_velocities[1], 0.5, 1e-12);

  auto workspace = MacPressureWorkspace3D::Create(cells.size(), interior.size() + exterior.size());
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto projected = workspace->ProjectPrepared(*prepared);
  ASSERT_TRUE(projected.ok()) << projected.status();
  EXPECT_NEAR(projected->cell_pressures[0], analytic_pressures[0], 1e-6);
  EXPECT_NEAR(projected->cell_pressures[1], analytic_pressures[1], 1e-6);
  EXPECT_NEAR(projected->frozen_measurement.inward_prescribed_wall_flux_m3_per_s, 0.1875, 1e-12);

  for (auto& face : exterior) {
    face.cell_is_tail = !face.cell_is_tail;
    face.prescribed_wall_flux = -face.prescribed_wall_flux;
  }
  const auto reversed = pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, cells, interior, exterior);
  ASSERT_TRUE(reversed.ok()) << reversed.status();
  const auto reversed_projection = EvaluateMacPressure3D(*reversed, analytic_pressures);
  ASSERT_TRUE(reversed_projection.ok()) << reversed_projection.status();
  EXPECT_NEAR(reversed_projection->diagnostics.max_cell_constraint_defect, 0.0, 1e-12);
  EXPECT_NEAR(reversed_projection->corrected_face_velocities[0], 1.5, 1e-12);
  EXPECT_NEAR(reversed_projection->corrected_face_velocities[1], 0.5, 1e-12);

  exterior[0].face_slot = 1;
  EXPECT_EQ(
      pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, cells, interior, exterior).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  exterior[0].face_slot = 0;
  exterior[0].open_area = exterior[0].full_face_area;
  EXPECT_EQ(
      pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, cells, interior, exterior).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_NEAR(projected->cell_pressures[0], analytic_pressures[0], 1e-6);
}

TEST(MacPressureExplicitGraph3D, BoundedSolverMatchesEdgeAndCornerReferencePressure) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  for (const auto& pressure_case : kExplicitPressureCases) {
    SCOPED_TRACE(pressure_case.plane_count);
    SCOPED_TRACE(pressure_case.reversed);
    const auto fixture = MakeExplicitPressureFixture(pressure_case);
    const auto system = pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, fixture.cells, fixture.faces);
    ASSERT_TRUE(system.ok()) << system.status();
    auto workspace = MacPressureWorkspace3D::Create(fixture.cells.size(), fixture.faces.size());
    ASSERT_TRUE(workspace.ok()) << workspace.status();
    const auto projection = workspace->ProjectPrepared(*system);
    ASSERT_TRUE(projection.ok()) << projection.status();
    EXPECT_GT(projection->diagnostics.iteration_count, 0);
    EXPECT_LE(projection->diagnostics.max_cell_constraint_defect, 2e-5);
    EXPECT_LE(projection->diagnostics.max_volume_scaled_residual, 2e-5);
    ASSERT_EQ(projection->cell_pressures.size(), fixture.cells.size());
    for (std::size_t cell = 0; cell < fixture.cells.size(); ++cell) {
      EXPECT_NEAR(projection->cell_pressures[cell], pressure_case.reference_pressure[cell], 2e-5);
    }
    const auto oracle_pressure = std::span(pressure_case.reference_pressure).first(fixture.cells.size());
    const auto oracle = EvaluateMacPressure3D(*system, oracle_pressure);
    ASSERT_TRUE(oracle.ok()) << oracle.status();
    ASSERT_EQ(projection->corrected_face_velocities.size(), fixture.faces.size());
    double max_face_flux_error = 0.0;
    for (std::size_t face = 0; face < fixture.faces.size(); ++face) {
      const double velocity_error =
          projection->corrected_face_velocities[face] - oracle->corrected_face_velocities[face];
      max_face_flux_error = std::max(max_face_flux_error, std::abs(fixture.faces[face].open_area * velocity_error));
    }
    EXPECT_LE(max_face_flux_error, 5e-6);
    std::cout << "explicit_graph_cpu planes=" << pressure_case.plane_count << " reversed=" << pressure_case.reversed
              << " iterations=" << projection->diagnostics.iteration_count
              << " max_divergence=" << projection->diagnostics.max_cell_constraint_defect
              << " max_face_flux_error=" << max_face_flux_error << '\n';
    const std::vector<double> accepted_pressures(projection->cell_pressures.begin(), projection->cell_pressures.end());
    const std::vector<double> accepted_velocities(
        projection->corrected_face_velocities.begin(),
        projection->corrected_face_velocities.end()
    );
    const auto rejected = workspace->ProjectPrepared(*system, {.max_iterations = 1});
    EXPECT_EQ(rejected.status().code(), absl::StatusCode::kAborted);
    EXPECT_TRUE(std::ranges::equal(projection->cell_pressures, accepted_pressures));
    EXPECT_TRUE(std::ranges::equal(projection->corrected_face_velocities, accepted_velocities));
    const auto replayed = workspace->ProjectPrepared(*system);
    ASSERT_TRUE(replayed.ok()) << replayed.status();
    EXPECT_TRUE(std::ranges::equal(replayed->cell_pressures, accepted_pressures));
    EXPECT_TRUE(std::ranges::equal(replayed->corrected_face_velocities, accepted_velocities));
  }
}

TEST(MacPressureExplicitGraph3D, MatchesEdgeAndCornerReferencePressure) {
  for (const auto& pressure_case : kExplicitPressureCases) {
    SCOPED_TRACE(pressure_case.plane_count);
    SCOPED_TRACE(pressure_case.reversed);
    const auto fixture = MakeExplicitPressureFixture(pressure_case);
    const auto system = pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, fixture.cells, fixture.faces);
    ASSERT_TRUE(system.ok()) << system.status();
    ASSERT_EQ(system->gauge_cell_ordinals().size(), 1);
    EXPECT_EQ(system->gauge_cell_ordinals().front(), 0);

    const auto constant = EvaluateMacPressure3D(*system, std::vector<double>(fixture.cells.size(), 1.3));
    ASSERT_TRUE(constant.ok()) << constant.status();
    for (std::size_t cell = 0; cell < fixture.cells.size(); ++cell) {
      EXPECT_DOUBLE_EQ(constant->integrated_rhs[cell], constant->true_residuals[cell]);
    }
    for (std::size_t coarse = 0; coarse < pressure_case.plane_count; ++coarse) {
      EXPECT_DOUBLE_EQ(constant->cell_constraint_defects[coarse], 0.25);
    }
    double fine_outward_flux = 0.0;
    for (std::size_t fine = pressure_case.plane_count; fine < fixture.cells.size(); ++fine) {
      fine_outward_flux += fixture.cells[fine].fluid_volume * constant->cell_constraint_defects[fine];
    }
    EXPECT_DOUBLE_EQ(fine_outward_flux, -0.25 * static_cast<double>(pressure_case.plane_count));

    const auto pressure = std::span(pressure_case.reference_pressure).first(fixture.cells.size());
    const auto projected = EvaluateMacPressure3D(*system, pressure);
    ASSERT_TRUE(projected.ok()) << projected.status();
    EXPECT_LE(projected->diagnostics.max_cell_constraint_defect, 1e-12);
    EXPECT_LE(projected->diagnostics.max_volume_scaled_residual, 1e-12);
    for (std::size_t cell = 0; cell < fixture.cells.size(); ++cell) {
      EXPECT_NEAR(
          projected->true_residuals[cell],
          -fixture.cells[cell].fluid_volume * projected->cell_constraint_defects[cell],
          1e-15
      );
    }

    auto corrupt_faces = fixture.faces;
    std::swap(corrupt_faces.front().tail_cell_ordinal, corrupt_faces.front().head_cell_ordinal);
    const auto corrupt_system =
        pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, fixture.cells, corrupt_faces);
    ASSERT_TRUE(corrupt_system.ok()) << corrupt_system.status();
    const auto corrupt_projection = EvaluateMacPressure3D(*corrupt_system, pressure);
    ASSERT_TRUE(corrupt_projection.ok()) << corrupt_projection.status();
    EXPECT_GE(corrupt_projection->diagnostics.max_cell_constraint_defect, 0.75);
  }
}

TEST(MacPressureExplicitGraph3D, RejectsInvalidOwnershipAndCoefficients) {
  const auto fixture = MakeExplicitPressureFixture(kExplicitPressureCases.front());
  auto faces = fixture.faces;
  faces[1].face_slot = faces[0].face_slot;
  EXPECT_EQ(
      pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, fixture.cells, faces).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  faces = fixture.faces;
  faces[0].tail_cell_ordinal = fixture.cells.size();
  EXPECT_EQ(
      pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, fixture.cells, faces).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  faces = fixture.faces;
  faces[0].pressure_gradient_distance = 0.0;
  EXPECT_EQ(
      pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, fixture.cells, faces).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  const std::vector<MacPressureCell3D> oversized_cells(4097, fixture.cells.front());
  EXPECT_EQ(
      pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, oversized_cells, fixture.faces).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}

TEST(MacPressureExplicitGraph3D, RejectsInexactSubnormalMobilityProduct) {
  const auto fixture = MakeExplicitPressureFixture(kExplicitPressureCases.front());
  auto face = fixture.faces.front();
  face.head_cell_ordinal = 1;
  face.open_area = 1.0;
  face.inverse_density = 1.5;
  face.pressure_gradient_distance = std::numeric_limits<double>::denorm_min();
  const auto cells = std::span(fixture.cells).first(2);
  const auto graph = pressure_internal::PrepareExplicitMacPressureSystem3D(
      face.pressure_gradient_distance,
      cells,
      std::span(&face, 1)
  );
  EXPECT_EQ(graph.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(MacPressureExplicitGraph3D, BoundedExplicitEntryRejectsUniformPreparedSystem) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{IncompressibleCellAtX(0)};
  const auto faces = SealedFacesAtX(0);
  const auto uniform = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(uniform.ok()) << uniform.status();
  auto workspace = MacPressureWorkspace3D::Create(cells.size(), faces.size());
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto projected = workspace->ProjectPrepared(*uniform);
  EXPECT_EQ(projected.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(MacPressureSystem3D, EvaluatesAllIndependentFixtureCases) {
  std::vector<pressure_test::MacPressureFixtureCase3D> cases;
  ASSERT_NO_FATAL_FAILURE(pressure_test::LoadMacPressureFixtureCases3D(cases));
  ASSERT_EQ(cases.size(), 8);
  for (const auto& fixture_case : cases) {
    SCOPED_TRACE(fixture_case.name);
    const auto system = PrepareMacPressureSystem3D(
        {fixture_case.grid, fixture_case.step_duration, fixture_case.cells, fixture_case.faces}
    );
    ASSERT_TRUE(system.ok()) << system.status();
    const auto projection = EvaluateMacPressure3D(*system, fixture_case.expected_projection.cell_pressures);
    ASSERT_TRUE(projection.ok()) << projection.status();
    pressure_test::ExpectMacPressureFixtureProjection3D(
        *projection,
        fixture_case.expected_projection,
        fixture_case.absolute_tolerance,
        fixture_case.relative_tolerance
    );
  }
}

TEST(MacPressureSystem3D, AcceptsLowBoundaryAtMinimumSignedIndex) {
  constexpr std::array blocks{BlockIndex3D{kMinIndex / 4, 0, 0}};
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, blocks, blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{IncompressibleCellAtX(kMinIndex)};
  auto faces = SealedFacesAtX(kMinIndex);
  faces.front().open_area = 1.0;
  faces.front().inverse_density = 1.0;
  faces.front().pressure_gradient_distance = 1.0;
  faces.front().pressure_link = DirichletMacPressureLink3D{3.0};
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  EXPECT_FALSE(system->links().front().tail_cell_ordinal.has_value());
  EXPECT_EQ(system->links().front().head_cell_ordinal, 0);
  const auto projection = EvaluateMacPressure3D(*system, std::array{3.0});
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_TRUE(projection->gauge_cell_ordinals.empty());
  EXPECT_EQ(projection->integrated_rhs, std::vector{3.0});
  EXPECT_EQ(projection->true_residuals, std::vector{0.0});
}

TEST(MacPressureSystem3D, RejectsHighFaceOverflowAndMissingOwner) {
  constexpr std::array blocks{BlockIndex3D{kMaxIndex / 4, 0, 0}};
  const auto extreme_grid = SparseMacGrid3D::Create(kUnitGeometry, blocks, blocks.size());
  ASSERT_TRUE(extreme_grid.ok()) << extreme_grid.status();
  const auto faces = SealedFacesAtX(kMaxIndex - 1);
  const std::array overflowing_cells{IncompressibleCellAtX(kMaxIndex)};
  EXPECT_EQ(
      PrepareMacPressureSystem3D({*extreme_grid, 1.0, overflowing_cells, faces}).status().code(),
      absl::StatusCode::kOutOfRange
  );

  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array missing_owner_cells{IncompressibleCellAtX(3)};
  EXPECT_EQ(
      PrepareMacPressureSystem3D({*grid, 1.0, missing_owner_cells, SealedFacesAtX(3)}).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(MacPressureSystem3D, RejectsUnsortedRepeatedOmittedAndExtraneousEntries) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{IncompressibleCellAtX(0)};
  const auto faces = SealedFacesAtX(0);
  const auto expect_invalid = [&](std::span<const MacPressureCell3D> pressure_cells,
                                  std::span<const MacPressureFace3D> pressure_faces) {
    EXPECT_EQ(
        PrepareMacPressureSystem3D({*grid, 1.0, pressure_cells, pressure_faces}).status().code(),
        absl::StatusCode::kInvalidArgument
    );
  };
  auto wrong_order = faces;
  std::swap(wrong_order[0], wrong_order[1]);
  expect_invalid(cells, wrong_order);
  auto repeated = faces;
  repeated[1] = repeated[0];
  expect_invalid(cells, repeated);
  expect_invalid(cells, std::span(faces).first(5));
  auto extraneous = faces;
  extraneous.back().face_index.lattice_index.z = 2;
  expect_invalid(cells, extraneous);
  auto invalid_axis = faces;
  invalid_axis[0].face_index.axis = static_cast<FaceNormalAxis>(3);
  expect_invalid(cells, invalid_axis);
  expect_invalid(std::array{cells[0], cells[0]}, faces);
  expect_invalid(std::array{IncompressibleCellAtX(1), cells[0]}, faces);
  expect_invalid({}, faces);
  expect_invalid(cells, {});
}

TEST(MacPressureSystem3D, RejectsCapacityBeforeReadingEntries) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::vector<MacPressureCell3D> too_many_cells(4097);
  const std::vector<MacPressureFace3D> too_many_faces(24577);
  const std::array cells{IncompressibleCellAtX(0)};
  const auto faces = SealedFacesAtX(0);
  EXPECT_EQ(
      PrepareMacPressureSystem3D({*grid, 1.0, too_many_cells, faces}).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(
      PrepareMacPressureSystem3D({*grid, 1.0, cells, too_many_faces}).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}

TEST(MacPressureSystem3D, ValidatesTypedLinksAndTheirPhysicalInputs) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  std::array cells{IncompressibleCellAtX(0)};
  auto faces = SealedFacesAtX(0);
  const auto expect_status = [&](absl::StatusCode code, double duration = 1.0) {
    EXPECT_EQ(PrepareMacPressureSystem3D({*grid, duration, cells, faces}).status().code(), code);
  };
  auto& face = faces.front();
  face.inverse_density = -1.0;
  face.pressure_gradient_distance = -1.0;
  face.predicted_normal_velocity = 6.0;
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto sealed = EvaluateMacPressure3D(*system, std::array{0.0});
  ASSERT_TRUE(sealed.ok()) << sealed.status();
  EXPECT_EQ(sealed->corrected_face_velocities.front(), 6.0);

  face.inverse_density = kInfinity;
  expect_status(absl::StatusCode::kInvalidArgument);
  face.inverse_density = 1.0;
  face.pressure_gradient_distance = 1.0;
  face.pressure_link = InternalMacPressureLink3D{0.0};
  expect_status(absl::StatusCode::kInvalidArgument);
  face.pressure_link = DirichletMacPressureLink3D{0.0};
  expect_status(absl::StatusCode::kInvalidArgument);
  face.open_area = 1.0;
  face.pressure_link = SealedMacPressureLink3D{};
  expect_status(absl::StatusCode::kInvalidArgument);
  face.pressure_link = DirichletMacPressureLink3D{kInfinity};
  expect_status(absl::StatusCode::kInvalidArgument);
  face.pressure_link = DirichletMacPressureLink3D{0.0};
  face.open_area = 1.01;
  expect_status(absl::StatusCode::kInvalidArgument);
  face.open_area = 1.0;
  face.inverse_density = 0.0;
  expect_status(absl::StatusCode::kInvalidArgument);
  face.inverse_density = 1.0;
  face.pressure_gradient_distance = 0.0;
  expect_status(absl::StatusCode::kInvalidArgument);
  face.pressure_gradient_distance = 1.0;
  expect_status(absl::StatusCode::kInvalidArgument, kInfinity);
  expect_status(absl::StatusCode::kInvalidArgument, 0.0);
  cells.front().bulk_modulus = -kInfinity;
  expect_status(absl::StatusCode::kInvalidArgument);
  cells.front().bulk_modulus = kInfinity;
  cells.front().fluid_volume = 2.0;
  expect_status(absl::StatusCode::kInvalidArgument);
  cells.front().fluid_volume = 1.0;
  cells.front().target_divergence = std::numeric_limits<double>::quiet_NaN();
  expect_status(absl::StatusCode::kInvalidArgument);
}

TEST(MacPressureSystem3D, RejectsUnrepresentableActiveCoefficients) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  std::array cells{IncompressibleCellAtX(0)};
  auto faces = SealedFacesAtX(0);
  auto& face = faces.front();
  face.open_area = 1.0;
  face.inverse_density = std::numeric_limits<double>::max();
  face.pressure_gradient_distance = 1.0;
  face.pressure_link = DirichletMacPressureLink3D{0.0};
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 2.0, cells, faces}).status().code(), absl::StatusCode::kOutOfRange);

  face.inverse_density = std::numeric_limits<double>::min();
  face.open_area = std::numeric_limits<double>::min();
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(), absl::StatusCode::kOutOfRange);

  faces = SealedFacesAtX(0);
  cells.front().bulk_modulus = std::numeric_limits<double>::max();
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 2.0, cells, faces}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(MacPressureSystem3D, RejectsLossySubnormalTargetFluxes) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  constexpr double kFluxQuantum = std::numeric_limits<double>::denorm_min();
  std::array cells{IncompressibleCellAtX(0)};
  auto faces = SealedFacesAtX(0);
  cells.front().fluid_volume = kFluxQuantum;
  cells.front().target_divergence = 0.25;
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(), absl::StatusCode::kOutOfRange);

  // The exact source is 1.25 quanta. Rounding it to one quantum would
  // incorrectly balance the prescribed wall and conceal a −0.25 s⁻¹ defect.
  cells.front().target_divergence = 1.25;
  faces[1].wall_normal_velocity = kFluxQuantum;
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(MacPressureSystem3D, RejectsLossySubnormalFluidAndWallFluxes) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  constexpr double kFluxQuantum = std::numeric_limits<double>::denorm_min();
  auto cell = IncompressibleCellAtX(0);
  cell.fluid_volume = kFluxQuantum;
  const std::array cells{cell};
  auto faces = SealedFacesAtX(0);
  auto& boundary = faces[1];
  boundary.open_area = 0.25;
  boundary.inverse_density = 1.0;
  boundary.pressure_gradient_distance = 1.0;
  boundary.pressure_link = DirichletMacPressureLink3D{0.0};

  // Three quarters of a quantum round to a nonzero but wrong wall flux.
  boundary.wall_normal_velocity = kFluxQuantum;
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(), absl::StatusCode::kOutOfRange);

  // One quarter of a quantum rounds away the predicted fluid flux.
  boundary.wall_normal_velocity = 0.0;
  boundary.predicted_normal_velocity = kFluxQuantum;
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(), absl::StatusCode::kOutOfRange);

  boundary.predicted_normal_velocity = 0.0;
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  EXPECT_EQ(EvaluateMacPressure3D(*system, std::array{kFluxQuantum}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(MacPressureSystem3D, RejectsLossySubnormalComplianceFlux) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto cell = IncompressibleCellAtX(0);
  cell.fluid_volume = std::numeric_limits<double>::denorm_min();
  cell.bulk_modulus = 1.0;
  const std::array cells{cell};
  const auto faces = SealedFacesAtX(0);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  EXPECT_EQ(EvaluateMacPressure3D(*system, std::array{0.25}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(MacPressureSystem3D, PreservesExactlyRepresentableSubnormalFlux) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  constexpr double kFluxQuantum = std::numeric_limits<double>::denorm_min();
  auto cell = IncompressibleCellAtX(0);
  cell.fluid_volume = 8.0 * kFluxQuantum;
  cell.target_divergence = 0.25;
  const std::array cells{cell};
  auto faces = SealedFacesAtX(0);
  faces[1].wall_normal_velocity = 2.0 * kFluxQuantum;
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto projection = EvaluateMacPressure3D(*system, std::array{0.0});
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_EQ(projection->integrated_rhs, std::vector{0.0});
  EXPECT_EQ(projection->cell_constraint_defects, std::vector{0.0});
}

TEST(MacPressureSystem3D, ChecksPhysicalCompatibilityPerComponentWithoutAbsoluteFloor) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  std::array cells{IncompressibleCellAtX(0), IncompressibleCellAtX(1)};
  const auto first = SealedFacesAtX(0);
  const auto second = SealedFacesAtX(1);
  std::vector faces{
      first[0],
      first[1],
      second[1],
      first[2],
      second[2],
      first[3],
      second[3],
      first[4],
      second[4],
      first[5],
      second[5]
  };
  faces[1].pressure_link = InternalMacPressureLink3D{12.0};
  faces[1].wall_normal_velocity = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(
      PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(),
      absl::StatusCode::kFailedPrecondition
  );

  cells[0].target_divergence = faces[1].wall_normal_velocity;
  cells[1].target_divergence = -faces[1].wall_normal_velocity;
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto projection = EvaluateMacPressure3D(*system, std::array{0.0, 0.0});
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_EQ(projection->gauge_cell_ordinals, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(projection->true_residuals, (std::vector{0.0, 0.0}));

  // The two-endpoint separator cannot be prescribed as a Dirichlet face.
  auto& separator = faces[1];
  separator.open_area = 1.0;
  separator.inverse_density = 1.0;
  separator.pressure_gradient_distance = 1.0;
  separator.wall_normal_velocity = 0.0;
  separator.pressure_link = DirichletMacPressureLink3D{0.0};
  EXPECT_EQ(PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(), absl::StatusCode::kInvalidArgument);

  // A large internal lift can round the unit source out of the assembled RHS.
  // It cancels physically inside the component and cannot mask incompatibility.
  separator.pressure_link = InternalMacPressureLink3D{1e16};
  cells[0].target_divergence = 1.0;
  cells[1].target_divergence = 0.0;
  EXPECT_EQ(
      PrepareMacPressureSystem3D({*grid, 1.0, cells, faces}).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(MacPressureSystem3D, EvaluatesUnconvergedCandidateOnEveryOriginalRow) {
  constexpr std::array blocks{BlockIndex3D{-1, 0, 0}, BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, blocks, blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto tiny_cell = IncompressibleCellAtX(-1);
  tiny_cell.fluid_volume = 1e-6;
  std::array cells{tiny_cell, IncompressibleCellAtX(0)};
  const auto first = SealedFacesAtX(-1);
  const auto second = SealedFacesAtX(0);
  std::vector faces{
      first[0],
      first[2],
      first[3],
      first[4],
      first[5],
      first[1],
      second[1],
      second[2],
      second[3],
      second[4],
      second[5]
  };
  auto& interface = faces[5];
  interface.open_area = 1.0;
  interface.inverse_density = 1.0;
  interface.pressure_gradient_distance = 1.0;
  interface.pressure_link = InternalMacPressureLink3D{1e-7};
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  EXPECT_EQ(system->component_ordinals().size(), 2);
  EXPECT_EQ(system->component_ordinals()[0], 0);
  EXPECT_EQ(system->component_ordinals()[1], 0);

  // Inputs can change once preparation succeeds. Frozen ordering and coefficients
  // are the only inputs to candidate evaluation.
  faces.clear();
  cells[0].fluid_volume = 1.0;
  const std::array candidate{0.0, 0.0};
  const auto projection = EvaluateMacPressure3D(*system, candidate);
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_EQ(projection->gauge_cell_ordinals, std::vector<std::size_t>{0});
  EXPECT_EQ(projection->integrated_rhs, (std::vector{-1e-7, 1e-7}));
  EXPECT_EQ(projection->true_residuals, (std::vector{-1e-7, 1e-7}));
  EXPECT_DOUBLE_EQ(projection->corrected_face_velocities[5], 1e-7);
  EXPECT_DOUBLE_EQ(projection->cell_constraint_defects[0], 0.1);
  EXPECT_DOUBLE_EQ(projection->cell_constraint_defects[1], -1e-7);
  EXPECT_DOUBLE_EQ(projection->diagnostics.true_residual_l2, std::sqrt(2.0) * 1e-7);
  EXPECT_DOUBLE_EQ(projection->diagnostics.max_volume_scaled_residual, 0.1);
  EXPECT_DOUBLE_EQ(projection->diagnostics.max_cell_constraint_defect, 0.1);
  EXPECT_EQ(projection->diagnostics.iteration_count, 0);
  EXPECT_EQ(projection->diagnostics.multigrid_level_count, 0);
  EXPECT_EQ(EvaluateMacPressure3D(*system, std::array{0.0}).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      EvaluateMacPressure3D(*system, std::array{0.0, kInfinity}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(MacPressureSystem3D, PreservesIsolatedComplianceAndRejectsNonfiniteEvaluation) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1.0,
      .target_divergence = 0.5,
      .bulk_modulus = 2.0,
      .previous_pressure = 4.0,
  }};
  const auto faces = SealedFacesAtX(0);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto projection = EvaluateMacPressure3D(*system, std::array{4.0});
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_TRUE(projection->gauge_cell_ordinals.empty());
  EXPECT_EQ(projection->integrated_rhs, std::vector{2.5});
  EXPECT_EQ(projection->true_residuals, std::vector{0.5});
  EXPECT_EQ(projection->cell_constraint_defects, std::vector{-0.5});

  auto overflow_faces = faces;
  overflow_faces[1] = {
      .face_index = faces[1].face_index,
      .open_area = 1.0,
      .inverse_density = 2.0,
      .pressure_gradient_distance = 1.0,
      .pressure_link = DirichletMacPressureLink3D{0.0},
  };
  const auto overflow_system = PrepareMacPressureSystem3D({*grid, 1.0, cells, overflow_faces});
  ASSERT_TRUE(overflow_system.ok()) << overflow_system.status();
  const std::array large_candidate{std::numeric_limits<double>::max()};
  EXPECT_EQ(EvaluateMacPressure3D(*overflow_system, large_candidate).status().code(), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace fluidity
