#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <variant>
#include <vector>

#include "../src/mac_pressure_3d_system.h"
#include "absl/status/status.h"
#include "fluidity/host/mac_pressure_3d.h"
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"

namespace fluidity {
namespace {

using pressure_internal::EvaluateMacPressure3D;
using pressure_internal::PrepareMacPressureSystem3D;

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
