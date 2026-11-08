#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string_view>

#include "absl/status/status.h"
#include "fixture_data.h"
#include "gtest/gtest.h"

namespace fluidity::fixtures {
namespace {

TEST(FixtureBridge, CanonicalFixturesPresent) {
  constexpr std::array<std::string_view, 7> kFixtureNames{
      "affine_apic",
      "compliant_pressure",
      "stiff_viscosity",
      "two_cell_pressure",
      "uniform_mac_liquid_boundary_3d",
      "uniform_mac_pressure_3d",
      "uniform_mac_transfer_3d",
  };
  const auto fixtures = AllFixtures();
  ASSERT_EQ(fixtures.size(), kFixtureNames.size());

  for (std::size_t fixture_index = 0; fixture_index < fixtures.size(); ++fixture_index) {
    const auto& fixture = fixtures[fixture_index];
    SCOPED_TRACE(fixture.name);
    EXPECT_EQ(fixture.name, kFixtureNames[fixture_index]);
    EXPECT_FALSE(fixture.arrays.empty());
    EXPECT_TRUE(std::isfinite(fixture.rtol));
    EXPECT_TRUE(std::isfinite(fixture.atol));
    EXPECT_GE(fixture.rtol, 0.0);
    EXPECT_GE(fixture.atol, 0.0);
    EXPECT_FALSE(fixture.provenance.empty());
    EXPECT_FALSE(fixture.default_policy.empty());

    const bool uses_si_units = fixture.name == "uniform_mac_liquid_boundary_3d"
        || fixture.name == "uniform_mac_pressure_3d" || fixture.name == "uniform_mac_transfer_3d";
    for (const auto& array : fixture.arrays) {
      SCOPED_TRACE(array.name);
      const auto found = FindArray(fixture, array.name);
      ASSERT_TRUE(found.ok()) << found.status();
      EXPECT_EQ(*found, &array);
      EXPECT_FALSE(array.units.empty());
      if (!uses_si_units) {
        EXPECT_EQ(array.units, "1");
      }
      const auto count = std::accumulate(
          array.shape.begin(),
          array.shape.end(),
          std::size_t{1},
          [](std::size_t product, std::size_t extent) {
            return product * extent;
          }
      );
      ASSERT_GT(count, 0);
      switch (array.dtype) {
        case DType::kFloat64:
          EXPECT_EQ(array.f64_data.size(), count);
          EXPECT_TRUE(array.i64_data.empty());
          for (double value : array.f64_data) {
            EXPECT_TRUE(std::isfinite(value));
          }
          break;
        case DType::kInt64:
          EXPECT_EQ(array.i64_data.size(), count);
          EXPECT_TRUE(array.f64_data.empty());
          break;
        default:
          ADD_FAILURE() << "Invalid dtype";
      }
    }

    if (fixture.name.ends_with("_pressure")) {
      const auto grid = FindArray(fixture, "grid");
      const auto spacing = FindArray(fixture, "spacing");
      const auto origin = FindArray(fixture, "origin");
      ASSERT_TRUE(grid.ok()) << grid.status();
      ASSERT_TRUE(spacing.ok()) << spacing.status();
      ASSERT_TRUE(origin.ok()) << origin.status();
      ASSERT_EQ((*grid)->i64_data.size(), 2);
      ASSERT_EQ((*spacing)->f64_data.size(), 1);
      ASSERT_EQ((*origin)->f64_data.size(), 2);
      EXPECT_GT((*spacing)->f64_data.front(), 0.0);
    }

    const auto missing = FindArray(fixture, "missing_array");
    ASSERT_FALSE(missing.ok());
    EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
  }
}

}  // namespace
}  // namespace fluidity::fixtures
