#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "../src/mac_pressure_3d_system.h"
#include "absl/status/status.h"
#include "fixture_data.h"
#include "fluidity/host/mac_pressure_3d.h"
#include "gtest/gtest.h"

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

template <typename Scalar>
std::span<const Scalar> ReadFixtureArray(
    const fixtures::FixtureRecord& fixture,
    std::string_view name,
    std::size_t row_count,
    std::size_t column_count,
    std::string_view units
) {
  const auto array = fixtures::FindArray(fixture, name);
  if (!array.ok()) {
    ADD_FAILURE() << array.status();
    return {};
  }
  SCOPED_TRACE(name);
  EXPECT_EQ((*array)->units, units);
  if (column_count == 1) {
    EXPECT_EQ((*array)->shape.size(), 1);
  } else {
    EXPECT_EQ((*array)->shape.size(), 2);
    if ((*array)->shape.size() == 2) {
      EXPECT_EQ((*array)->shape[1], column_count);
    }
  }
  EXPECT_FALSE((*array)->shape.empty());
  if (!(*array)->shape.empty()) {
    EXPECT_EQ((*array)->shape.front(), row_count);
  }
  std::span<const Scalar> values;
  if constexpr (std::is_same_v<Scalar, double>) {
    EXPECT_EQ((*array)->dtype, fixtures::DType::kFloat64);
    values = (*array)->f64_data;
    EXPECT_TRUE(std::ranges::all_of(values, [](double value) {
      return std::isfinite(value);
    }));
  } else {
    static_assert(std::is_same_v<Scalar, std::int64_t>);
    EXPECT_EQ((*array)->dtype, fixtures::DType::kInt64);
    values = (*array)->i64_data;
  }
  if (values.size() != row_count * column_count) {
    ADD_FAILURE() << "Wrong scalar count";
    return {};
  }
  return values;
}

bool OffsetsCoverRows(std::span<const std::int64_t> offsets) {
  return !offsets.empty() && offsets.front() == 0 && std::ranges::is_sorted(offsets) && offsets.back() > 0;
}

std::int32_t FixtureOwnerAxis(std::int32_t index) {
  return index / 4 - (index % 4 < 0 ? 1 : 0);
}

void ExpectFixtureValues(
    std::span<const double> actual,
    std::span<const double> expected,
    const fixtures::FixtureRecord& fixture
) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t ordinal = 0; ordinal < actual.size(); ++ordinal) {
    EXPECT_NEAR(actual[ordinal], expected[ordinal], fixture.atol + fixture.rtol * std::abs(expected[ordinal]))
        << "ordinal " << ordinal;
  }
}

TEST(MacPressureSystem3D, EvaluatesAllIndependentFixtureCases) {
  constexpr std::array<std::string_view, 8> kCaseNames{
      "jump_across_x_block",
      "jump_across_y_block",
      "jump_across_z_block",
      "low_dirichlet",
      "high_dirichlet",
      "disconnected_neumann",
      "isolated_compliance",
      "mixed",
  };
  const auto fixtures = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(fixtures, "uniform_mac_pressure_3d", &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, fixtures.end());

  const std::size_t offset_count = kCaseNames.size() + 1;
  const auto cell_offsets = ReadFixtureArray<std::int64_t>(*fixture, "case_cell_offsets", offset_count, 1, "1");
  const auto face_offsets = ReadFixtureArray<std::int64_t>(*fixture, "case_face_offsets", offset_count, 1, "1");
  const auto gauge_offsets = ReadFixtureArray<std::int64_t>(*fixture, "case_gauge_offsets", offset_count, 1, "1");
  ASSERT_TRUE(OffsetsCoverRows(cell_offsets));
  ASSERT_TRUE(OffsetsCoverRows(face_offsets));
  ASSERT_TRUE(OffsetsCoverRows(gauge_offsets));
  const auto cell_count = static_cast<std::size_t>(cell_offsets.back());
  const auto face_count = static_cast<std::size_t>(face_offsets.back());
  const auto gauge_count = static_cast<std::size_t>(gauge_offsets.back());
  const auto step_durations = ReadFixtureArray<double>(*fixture, "step_durations", kCaseNames.size(), 1, "s");
  const auto cell_indices = ReadFixtureArray<std::int64_t>(*fixture, "cell_indices", cell_count, 3, "1");
  const auto face_indices = ReadFixtureArray<std::int64_t>(*fixture, "face_indices", face_count, 4, "1");
  const auto face_link_kinds = ReadFixtureArray<std::int64_t>(*fixture, "face_link_kinds", face_count, 1, "1");
  const auto expected_gauges =
      ReadFixtureArray<std::int64_t>(*fixture, "expected_gauge_cell_ordinals", gauge_count, 1, "1");
  ASSERT_FALSE(step_durations.empty());
  ASSERT_FALSE(cell_indices.empty());
  ASSERT_FALSE(face_indices.empty());
  ASSERT_FALSE(face_link_kinds.empty());
  ASSERT_FALSE(expected_gauges.empty());

  const auto cell_values = [&](std::string_view name, std::string_view units) {
    return ReadFixtureArray<double>(*fixture, name, cell_count, 1, units);
  };
  const auto face_values = [&](std::string_view name, std::string_view units) {
    return ReadFixtureArray<double>(*fixture, name, face_count, 1, units);
  };
  const auto fluid_volumes = cell_values("cell_fluid_volumes", "m^3");
  const auto target_divergences = cell_values("target_cell_divergences", "s^-1");
  const auto bulk_moduli = cell_values("cell_bulk_moduli", "Pa");
  const auto previous_pressures = cell_values("previous_cell_pressures", "Pa");
  const auto open_areas = face_values("face_open_areas", "m^2");
  const auto inverse_densities = face_values("face_inverse_densities", "m^3/kg");
  const auto distances = face_values("face_pressure_gradient_distances", "m");
  const auto predicted_velocities = face_values("predicted_face_velocities", "m/s");
  const auto wall_velocities = face_values("wall_face_velocities", "m/s");
  const auto pressure_lifts = face_values("face_pressure_lifts", "Pa");
  const auto expected_pressures = cell_values("expected_cell_pressures", "Pa");
  const auto expected_rhs = cell_values("expected_integrated_rhs", "m^3/s");
  const auto expected_residuals = cell_values("expected_true_residuals", "m^3/s");
  const auto expected_defects = cell_values("expected_cell_constraint_defects", "s^-1");
  const auto expected_velocities = face_values("expected_corrected_face_velocities", "m/s");
  for (const auto values :
       {fluid_volumes,
        target_divergences,
        bulk_moduli,
        previous_pressures,
        open_areas,
        inverse_densities,
        distances,
        predicted_velocities,
        wall_velocities,
        pressure_lifts,
        expected_pressures,
        expected_rhs,
        expected_residuals,
        expected_defects,
        expected_velocities}) {
    ASSERT_FALSE(values.empty());
  }

  for (std::size_t case_ordinal = 0; case_ordinal < kCaseNames.size(); ++case_ordinal) {
    SCOPED_TRACE(kCaseNames[case_ordinal]);
    const auto first_cell = static_cast<std::size_t>(cell_offsets[case_ordinal]);
    const auto end_cell = static_cast<std::size_t>(cell_offsets[case_ordinal + 1]);
    const auto first_face = static_cast<std::size_t>(face_offsets[case_ordinal]);
    const auto end_face = static_cast<std::size_t>(face_offsets[case_ordinal + 1]);
    const auto first_gauge = static_cast<std::size_t>(gauge_offsets[case_ordinal]);
    const auto end_gauge = static_cast<std::size_t>(gauge_offsets[case_ordinal + 1]);
    ASSERT_LT(first_cell, end_cell);
    ASSERT_LT(first_face, end_face);

    std::vector<MacPressureCell3D> cells;
    for (std::size_t cell = first_cell; cell < end_cell; ++cell) {
      const auto coordinates = cell_indices.subspan(3 * cell, 3);
      for (const auto coordinate : coordinates) {
        ASSERT_GE(coordinate, kMinIndex);
        ASSERT_LE(coordinate, kMaxIndex);
      }
      cells.push_back({
          .cell_index =
              {static_cast<std::int32_t>(coordinates[0]),
               static_cast<std::int32_t>(coordinates[1]),
               static_cast<std::int32_t>(coordinates[2])},
          .fluid_volume = fluid_volumes[cell],
          .target_divergence = target_divergences[cell],
          .bulk_modulus = bulk_moduli[cell] == 0.0 ? kInfinity : bulk_moduli[cell],
          .previous_pressure = previous_pressures[cell],
      });
    }

    std::vector<MacPressureFace3D> faces;
    std::set<std::tuple<std::int32_t, std::int32_t, std::int32_t>> owner_block_coordinates;
    for (std::size_t face = first_face; face < end_face; ++face) {
      const auto index = face_indices.subspan(4 * face, 4);
      ASSERT_GE(index[0], 0);
      ASSERT_LE(index[0], 2);
      for (const auto coordinate : index.subspan(1)) {
        ASSERT_GE(coordinate, kMinIndex);
        ASSERT_LE(coordinate, kMaxIndex);
      }
      const FaceLatticeIndex3D lattice_index{
          static_cast<std::int32_t>(index[1]),
          static_cast<std::int32_t>(index[2]),
          static_cast<std::int32_t>(index[3])
      };
      owner_block_coordinates.emplace(
          FixtureOwnerAxis(lattice_index.x),
          FixtureOwnerAxis(lattice_index.y),
          FixtureOwnerAxis(lattice_index.z)
      );
      MacPressureFace3D pressure_face{
          .face_index = {static_cast<FaceNormalAxis>(index[0]), lattice_index},
          .open_area = open_areas[face],
          .inverse_density = inverse_densities[face],
          .pressure_gradient_distance = distances[face],
          .predicted_normal_velocity = predicted_velocities[face],
          .wall_normal_velocity = wall_velocities[face],
          .pressure_link = SealedMacPressureLink3D{},
      };
      switch (face_link_kinds[face]) {
        case 0:
          EXPECT_EQ(pressure_lifts[face], 0.0);
          break;
        case 1:
          pressure_face.pressure_link = InternalMacPressureLink3D{pressure_lifts[face]};
          break;
        case 2:
          pressure_face.pressure_link = DirichletMacPressureLink3D{pressure_lifts[face]};
          break;
        default:
          FAIL() << "Invalid fixture pressure link kind";
      }
      faces.push_back(pressure_face);
    }
    std::vector<BlockIndex3D> owner_blocks;
    for (const auto& [x, y, z] : owner_block_coordinates) {
      owner_blocks.push_back({x, y, z});
    }
    const auto grid = SparseMacGrid3D::Create(kUnitGeometry, owner_blocks, owner_blocks.size());
    ASSERT_TRUE(grid.ok()) << grid.status();
    const auto system = PrepareMacPressureSystem3D({*grid, step_durations[case_ordinal], cells, faces});
    ASSERT_TRUE(system.ok()) << system.status();
    const auto pressures = expected_pressures.subspan(first_cell, cells.size());
    const auto projection = EvaluateMacPressure3D(*system, pressures);
    ASSERT_TRUE(projection.ok()) << projection.status();
    ExpectFixtureValues(projection->cell_pressures, pressures, *fixture);
    ExpectFixtureValues(projection->integrated_rhs, expected_rhs.subspan(first_cell, cells.size()), *fixture);
    ExpectFixtureValues(projection->true_residuals, expected_residuals.subspan(first_cell, cells.size()), *fixture);
    ExpectFixtureValues(
        projection->cell_constraint_defects,
        expected_defects.subspan(first_cell, cells.size()),
        *fixture
    );
    ExpectFixtureValues(
        projection->corrected_face_velocities,
        expected_velocities.subspan(first_face, faces.size()),
        *fixture
    );
    std::vector<std::size_t> gauges;
    for (const auto ordinal : expected_gauges.subspan(first_gauge, end_gauge - first_gauge)) {
      ASSERT_GE(ordinal, 0);
      ASSERT_LT(ordinal, static_cast<std::int64_t>(cells.size()));
      gauges.push_back(static_cast<std::size_t>(ordinal));
    }
    EXPECT_EQ(projection->gauge_cell_ordinals, gauges);
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
