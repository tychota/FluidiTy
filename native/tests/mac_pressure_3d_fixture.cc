#include "mac_pressure_3d_fixture.h"

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
#include <utility>
#include <vector>

#include "fixture_data.h"
#include "gtest/gtest.h"

namespace fluidity::pressure_test {
namespace {

constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr auto kMinIndex = std::numeric_limits<std::int32_t>::min();
constexpr auto kMaxIndex = std::numeric_limits<std::int32_t>::max();

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
    double absolute_tolerance,
    double relative_tolerance
) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t ordinal = 0; ordinal < actual.size(); ++ordinal) {
    EXPECT_NEAR(
        actual[ordinal],
        expected[ordinal],
        absolute_tolerance + relative_tolerance * std::abs(expected[ordinal])
    ) << "ordinal "
      << ordinal;
  }
}

}  // namespace

void LoadMacPressureFixtureCases3D(std::vector<MacPressureFixtureCase3D>& cases) {
  cases.clear();
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
    auto grid = SparseMacGrid3D::Create(kUnitGeometry, owner_blocks, owner_blocks.size());
    ASSERT_TRUE(grid.ok()) << grid.status();
    std::vector<std::size_t> gauges;
    for (const auto ordinal : expected_gauges.subspan(first_gauge, end_gauge - first_gauge)) {
      ASSERT_GE(ordinal, 0);
      ASSERT_LT(ordinal, static_cast<std::int64_t>(cells.size()));
      gauges.push_back(static_cast<std::size_t>(ordinal));
    }
    MacPressureProjection3D expected;
    expected.cell_pressures.assign(expected_pressures.begin() + first_cell, expected_pressures.begin() + end_cell);
    expected.integrated_rhs.assign(expected_rhs.begin() + first_cell, expected_rhs.begin() + end_cell);
    expected.true_residuals.assign(expected_residuals.begin() + first_cell, expected_residuals.begin() + end_cell);
    expected.cell_constraint_defects.assign(expected_defects.begin() + first_cell, expected_defects.begin() + end_cell);
    expected.corrected_face_velocities.assign(
        expected_velocities.begin() + first_face,
        expected_velocities.begin() + end_face
    );
    expected.gauge_cell_ordinals = std::move(gauges);
    cases.push_back({
        .name = kCaseNames[case_ordinal],
        .grid = std::move(*grid),
        .step_duration = step_durations[case_ordinal],
        .cells = std::move(cells),
        .faces = std::move(faces),
        .expected_projection = std::move(expected),
        .absolute_tolerance = fixture->atol,
        .relative_tolerance = fixture->rtol,
    });
  }
}

void ExpectMacPressureFixtureProjection3D(
    const MacPressureProjection3D& actual,
    const MacPressureProjection3D& expected,
    double absolute_tolerance,
    double relative_tolerance
) {
  ExpectFixtureValues(actual.cell_pressures, expected.cell_pressures, absolute_tolerance, relative_tolerance);
  ExpectFixtureValues(actual.integrated_rhs, expected.integrated_rhs, absolute_tolerance, relative_tolerance);
  ExpectFixtureValues(actual.true_residuals, expected.true_residuals, absolute_tolerance, relative_tolerance);
  ExpectFixtureValues(
      actual.cell_constraint_defects,
      expected.cell_constraint_defects,
      absolute_tolerance,
      relative_tolerance
  );
  ExpectFixtureValues(
      actual.corrected_face_velocities,
      expected.corrected_face_velocities,
      absolute_tolerance,
      relative_tolerance
  );
  EXPECT_EQ(actual.gauge_cell_ordinals, expected.gauge_cell_ordinals);
}

}  // namespace fluidity::pressure_test
