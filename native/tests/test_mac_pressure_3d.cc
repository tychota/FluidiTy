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

#include "../src/mac_pressure_3d_multigrid.h"
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

TEST(MacPressure3D, SolvesAllIndependentFixtureCases) {
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
              {
                  static_cast<std::int32_t>(coordinates[0]),
                  static_cast<std::int32_t>(coordinates[1]),
                  static_cast<std::int32_t>(coordinates[2]),
              },
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
          static_cast<std::int32_t>(index[3]),
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
    const auto projection = ProjectMacPressure3D({*grid, step_durations[case_ordinal], cells, faces});
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

// Build an independent affine solution: each predicted face velocity is the
// analytic pressure slope, so correction leaves zero flux on every open face.
struct AffineProblem {
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  std::vector<BlockIndex3D> blocks;
};

double AffinePressure(CellIndex3D index) {
  return index.x + 2.0 * index.y + 4.0 * index.z;
}

AffineProblem MakeAffineProblem(std::int32_t minimum, std::int32_t extent, bool anchored) {
  AffineProblem problem;
  std::set<std::tuple<int, int, int, int>> face_keys;
  for (int z = minimum; z < minimum + extent; ++z) {
    for (int y = minimum; y < minimum + extent; ++y) {
      for (int x = minimum; x < minimum + extent; ++x) {
        problem.cells.push_back({
            .cell_index = {x, y, z},
            .fluid_volume = 1.0,
            .bulk_modulus = kInfinity,
        });
        for (int axis = 0; axis < 3; ++axis) {
          for (int side = 0; side < 2; ++side) {
            face_keys.emplace(axis, x + (axis == 0 ? side : 0), y + (axis == 1 ? side : 0), z + (axis == 2 ? side : 0));
          }
        }
      }
    }
  }
  std::set<std::tuple<int, int, int>> block_keys;
  for (const auto& [axis, x, y, z] : face_keys) {
    const std::array coordinates{x, y, z};
    const bool low = coordinates[axis] == minimum;
    const bool high = coordinates[axis] == minimum + extent;
    const bool boundary = low || high;
    MacPressureFace3D face{
        .face_index = {static_cast<FaceNormalAxis>(axis), {x, y, z}},
        .open_area = boundary && !anchored ? 0.0 : 1.0,
        .inverse_density = 1.0,
        .pressure_gradient_distance = 1.0,
        .predicted_normal_velocity = boundary && !anchored ? 0.0 : static_cast<double>(1 << axis),
        .pressure_link = InternalMacPressureLink3D{0.0},
    };
    if (boundary && anchored) {
      const CellIndex3D exterior{
          x - (axis == 0 && low ? 1 : 0),
          y - (axis == 1 && low ? 1 : 0),
          z - (axis == 2 && low ? 1 : 0),
      };
      face.pressure_link = DirichletMacPressureLink3D{AffinePressure(exterior)};
    } else if (boundary) {
      face.pressure_link = SealedMacPressureLink3D{};
    }
    problem.faces.push_back(face);
    block_keys.emplace(FixtureOwnerAxis(x), FixtureOwnerAxis(y), FixtureOwnerAxis(z));
  }
  for (const auto& [x, y, z] : block_keys) {
    problem.blocks.push_back({x, y, z});
  }
  return problem;
}

void SortFaces(AffineProblem& problem, const SparseMacGrid3D& grid) {
  std::ranges::sort(problem.faces, [&](const auto& left, const auto& right) {
    return *grid.FaceSlot(left.face_index) < *grid.FaceSlot(right.face_index);
  });
}

TEST(MacPressure3D, SolvesNegativeCoordinateMultilevelAffineFieldWithAllSixBoundaries) {
  for (const bool anchored : {false, true}) {
    SCOPED_TRACE(anchored);
    auto problem = MakeAffineProblem(-3, 4, anchored);
    const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
    ASSERT_TRUE(grid.ok()) << grid.status();
    SortFaces(problem, *grid);
    const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_GE(result->diagnostics.multigrid_level_count, 2);
    EXPECT_GT(result->diagnostics.iteration_count, 0);
    EXPECT_LE(result->diagnostics.max_volume_scaled_residual, 2e-5);
    EXPECT_LE(result->diagnostics.max_cell_constraint_defect, 2e-5);
    double rhs_norm = 0.0;
    for (double rhs : result->integrated_rhs) {
      rhs_norm = std::hypot(rhs_norm, rhs);
    }
    EXPECT_LE(result->diagnostics.true_residual_l2, 2e-6 + 2e-6 * rhs_norm);
    for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
      const double expected = AffinePressure(problem.cells[cell].cell_index) + (anchored ? 0.0 : 21.0);
      EXPECT_NEAR(result->cell_pressures[cell], expected, 5e-5);
      EXPECT_NEAR(result->true_residuals[cell], -result->cell_constraint_defects[cell], 1e-12);
    }
    for (double velocity : result->corrected_face_velocities) {
      EXPECT_NEAR(velocity, 0.0, 5e-5);
    }
    RecordProperty(
        anchored ? "anchored_true_residual_l2" : "gauged_true_residual_l2",
        std::to_string(result->diagnostics.true_residual_l2)
    );
    RecordProperty(
        anchored ? "anchored_max_defect" : "gauged_max_defect",
        std::to_string(result->diagnostics.max_cell_constraint_defect)
    );
  }
}

TEST(MacPressure3D, ShiftsPreviousPressureByEachNeumannPinBeforeIterating) {
  auto problem = MakeAffineProblem(-3, 4, false);
  for (auto& cell : problem.cells) {
    cell.previous_pressure = AffinePressure(cell.cell_index) + 100.0;
  }
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces}, {.max_iterations = 1});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->diagnostics.iteration_count, 0);
  EXPECT_DOUBLE_EQ(result->cell_pressures.front(), 0.0);
  EXPECT_DOUBLE_EQ(result->cell_pressures.back(), 21.0);
}

TEST(MacPressure3D, RetainsAnchoredPreviousPressureMean) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1.0,
      .target_divergence = 0.5,
      .bulk_modulus = 2.0,
      .previous_pressure = 4.0,
  }};
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, SealedFacesAtX(0)});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->gauge_cell_ordinals.empty());
  EXPECT_DOUBLE_EQ(result->cell_pressures.front(), 5.0);
}

TEST(MacPressure3D, ExhaustsOneIterationWithoutPublishingOrMutatingInputs) {
  auto problem = MakeAffineProblem(-3, 4, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces}, {.max_iterations = 1});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kAborted);
  EXPECT_EQ(problem.cells.front().previous_pressure, 0.0);
  EXPECT_EQ(problem.cells.back().previous_pressure, 0.0);
  const auto converged = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(converged.ok()) << converged.status();
  EXPECT_GT(converged->diagnostics.iteration_count, 1);
}

TEST(MacPressure3D, TinyCellMustPassLocalGateEvenWhenInitialGlobalGatePasses) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1e-6,
      .target_divergence = 0.1,
      .bulk_modulus = kInfinity,
  }};
  auto faces = SealedFacesAtX(0);
  faces[1].open_area = 1.0;
  faces[1].inverse_density = 1.0;
  faces[1].pressure_gradient_distance = 1.0;
  faces[1].pressure_link = DirichletMacPressureLink3D{0.0};
  const auto prepared = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  const auto initial = EvaluateMacPressure3D(*prepared, std::array{0.0});
  ASSERT_TRUE(initial.ok()) << initial.status();
  EXPECT_LT(initial->diagnostics.true_residual_l2, 2e-6);
  EXPECT_NEAR(initial->diagnostics.max_volume_scaled_residual, 0.1, 1e-15);
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_GT(result->diagnostics.iteration_count, 0);
  EXPECT_NEAR(result->cell_pressures.front(), 1e-7, 2e-11);
  EXPECT_LE(result->diagnostics.max_volume_scaled_residual, 2e-5);
  EXPECT_LE(result->diagnostics.max_cell_constraint_defect, 2e-5);
}

TEST(MacPressure3D, RejectsIterationBudgetsOutsideFixedBound) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{IncompressibleCellAtX(0)};
  const auto faces = SealedFacesAtX(0);
  for (const auto iterations : {std::size_t{0}, std::size_t{257}}) {
    const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces}, {.max_iterations = iterations});
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

double GraphEnergy(const pressure_internal::MacPressureLevel3D& level, std::span<const double> pressure) {
  double energy = 0.0;
  for (std::size_t unknown = 0; unknown < pressure.size(); ++unknown) {
    energy += level.anchor_diagonal[unknown] * pressure[unknown] * pressure[unknown];
  }
  for (const auto& edge : level.edges) {
    const double difference = pressure[edge.lower_unknown_ordinal] - pressure[edge.higher_unknown_ordinal];
    energy += edge.conductance * difference * difference;
  }
  return energy;
}

TEST(MacPressure3D, SharedHierarchyPreservesGalerkinEnergyAndBoundsTerminalBlocks) {
  auto problem = MakeAffineProblem(-4, 8, true);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  ASSERT_TRUE(hierarchy.ok()) << hierarchy.status();
  ASSERT_GE(hierarchy->levels.size(), 3);
  for (std::size_t ordinal = 0; ordinal + 1 < hierarchy->levels.size(); ++ordinal) {
    const auto& fine = hierarchy->levels[ordinal];
    const auto& coarse = hierarchy->levels[ordinal + 1];
    std::vector<double> coarse_pressure(coarse.coordinates.size());
    for (std::size_t unknown = 0; unknown < coarse_pressure.size(); ++unknown) {
      coarse_pressure[unknown] = static_cast<double>(unknown % 7) - 3.0;
    }
    std::vector<double> fine_pressure(fine.coordinates.size());
    for (std::size_t unknown = 0; unknown < fine_pressure.size(); ++unknown) {
      fine_pressure[unknown] = coarse_pressure[fine.aggregate_ordinals[unknown]];
    }
    EXPECT_DOUBLE_EQ(GraphEnergy(fine, fine_pressure), GraphEnergy(coarse, coarse_pressure));
    std::vector<int> mirror_counts(fine.edges.size(), 0);
    for (std::size_t unknown = 0; unknown < fine.coordinates.size(); ++unknown) {
      for (std::size_t offset = fine.row_offsets[unknown]; offset < fine.row_offsets[unknown + 1]; ++offset) {
        const auto neighbor = fine.neighbors[offset];
        const auto& edge = fine.edges[neighbor.edge_ordinal];
        EXPECT_TRUE(
            (edge.lower_unknown_ordinal == unknown && edge.higher_unknown_ordinal == neighbor.unknown_ordinal)
            || (edge.higher_unknown_ordinal == unknown && edge.lower_unknown_ordinal == neighbor.unknown_ordinal)
        );
        ++mirror_counts[neighbor.edge_ordinal];
      }
    }
    for (const auto count : mirror_counts) {
      EXPECT_EQ(count, 2);
    }
  }
  for (const auto& block : hierarchy->levels.back().terminal_blocks) {
    EXPECT_LE(block.unknown_ordinals.size(), 32);
  }
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_GE(result->diagnostics.multigrid_level_count, 3);
  EXPECT_GT(result->diagnostics.iteration_count, 0);
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    EXPECT_NEAR(result->cell_pressures[cell], AffinePressure(problem.cells[cell].cell_index), 5e-5);
  }
}

TEST(MacPressure3D, SignedCoordinatePlateauUsesLinkedPairCoarsening) {
  auto problem = MakeAffineProblem(-1, 2, true);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  ASSERT_TRUE(hierarchy.ok()) << hierarchy.status();
  ASSERT_EQ(hierarchy->levels.size(), 2);
  const auto& fine = hierarchy->levels.front();
  EXPECT_LT(hierarchy->levels.back().coordinates.size(), fine.coordinates.size());
  for (std::size_t unknown = 0; unknown < fine.coordinates.size(); ++unknown) {
    bool has_linked_partner = false;
    for (std::size_t offset = fine.row_offsets[unknown]; offset < fine.row_offsets[unknown + 1]; ++offset) {
      has_linked_partner |=
          fine.aggregate_ordinals[unknown] == fine.aggregate_ordinals[fine.neighbors[offset].unknown_ordinal];
    }
    EXPECT_TRUE(has_linked_partner);
  }
}

TEST(MacPressure3D, EliminatedArticulationPinRetainsConductanceInSeparateSingletons) {
  auto problem = MakeAffineProblem(0, 2, false);
  problem.cells.resize(3);  // (0,0,0), (1,0,0), (0,1,0): the lowest cell joins two arms.
  const auto contains = [&](CellIndex3D index) {
    return std::ranges::any_of(problem.cells, [&](const auto& cell) {
      return cell.cell_index.x == index.x && cell.cell_index.y == index.y && cell.cell_index.z == index.z;
    });
  };
  std::erase_if(problem.faces, [&](auto& face) {
    const auto index = face.face_index.lattice_index;
    const auto axis = face.face_index.axis;
    const CellIndex3D head{index.x, index.y, index.z};
    const CellIndex3D tail{
        index.x - (axis == FaceNormalAxis::kX ? 1 : 0),
        index.y - (axis == FaceNormalAxis::kY ? 1 : 0),
        index.z - (axis == FaceNormalAxis::kZ ? 1 : 0),
    };
    const bool has_tail = contains(tail);
    const bool has_head = contains(head);
    if (!has_tail && !has_head) {
      return true;
    }
    if (!has_tail || !has_head) {
      face.open_area = 0.0;
      face.predicted_normal_velocity = 0.0;
      face.pressure_link = SealedMacPressureLink3D{};
    }
    return false;
  });
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  ASSERT_TRUE(hierarchy.ok()) << hierarchy.status();
  EXPECT_EQ(hierarchy->unknown_to_cell, (std::vector<std::size_t>{1, 2}));
  EXPECT_EQ(hierarchy->cell_to_unknown.front(), pressure_internal::kNoPressureUnknown3D);
  ASSERT_EQ(hierarchy->levels.size(), 1);
  EXPECT_TRUE(hierarchy->levels.front().edges.empty());
  EXPECT_EQ(hierarchy->levels.front().anchor_diagonal, (std::vector{1.0, 1.0}));
  EXPECT_EQ(hierarchy->levels.front().terminal_blocks.size(), 2);
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->cell_pressures, (std::vector{0.0, 1.0, 2.0}));
}

TEST(MacPressure3D, ReplacesFalseConvergedFp32ResidualAndRestartsWithinSameBudget) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1e-6,
      .target_divergence = 1e6,
      .bulk_modulus = 1e-6 / 1.1,
  }};
  const auto faces = SealedFacesAtX(0);
  const auto exhausted = ProjectMacPressure3D({*grid, 1.0, cells, faces}, {.max_iterations = 1});
  EXPECT_EQ(exhausted.status().code(), absl::StatusCode::kAborted);
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_GE(result->diagnostics.iteration_count, 2);
  EXPECT_NEAR(result->cell_pressures.front(), 1.0 / 1.1, 2e-11);
  EXPECT_LE(result->diagnostics.max_volume_scaled_residual, 2e-5);
  EXPECT_LE(result->diagnostics.max_cell_constraint_defect, 2e-5);
}

TEST(MacPressure3D, RejectsCoefficientsOutsideFineRangeAndPreservesPreparedRangeChecks) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto cells = std::array{IncompressibleCellAtX(0)};
  auto faces = SealedFacesAtX(0);
  faces[1].open_area = 1.0;
  faces[1].pressure_gradient_distance = 1.0;
  faces[1].predicted_normal_velocity = 1.0;
  faces[1].pressure_link = DirichletMacPressureLink3D{0.0};
  for (const double inverse_density : {1e-100, 1e39}) {
    faces[1].inverse_density = inverse_density;
    const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces});
    EXPECT_EQ(result.status().code(), absl::StatusCode::kOutOfRange);
  }
  cells[0].fluid_volume = std::numeric_limits<double>::denorm_min();
  cells[0].target_divergence = 0.25;
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, SealedFacesAtX(0)});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kOutOfRange);
}
}  // namespace
}  // namespace fluidity
