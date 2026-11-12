#include <array>
#include <cmath>
#include <limits>
#include <numbers>

#include "fluidity/host/liquid_air_components.h"
#include "fluidity/host/liquid_cavity.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(LiquidCavity3D, SampledGasDomainSeparatesEnclosureFromUnknownAndVentedFrontiers) {
  constexpr UniformMacGeometry3D kGeometry{{0.0, 0.0, 0.0}, 1.0};
  std::array<CellIndex3D, 8> sampled_cells;
  std::array<LiquidCornerDistance3D, 27> corners;
  std::size_t cell_count = 0;
  std::size_t corner_count = 0;
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        corners[corner_count++] = {{x, y, z}, x == 0 && y == 0 && z == 0 ? 1.0 : -1.0};
        if (x <= 0 && y <= 0 && z <= 0) {
          sampled_cells[cell_count++] = {x, y, z};
        }
      }
    }
  }
  std::array<LiquidAirVolumeCell3D, 8> gas_cells;
  std::array<std::size_t, 8> labels;
  std::array<std::size_t, 8> queue;
  std::array<LiquidAirComponent3D, 1> components;
  const auto classify = [&](std::span<const CellIndex3D> cells, std::span<const FaceIndex3D> outlets) {
    const auto built = BuildLiquidAirDomain3D(kGeometry, {}, cells, corners, outlets, gas_cells);
    ASSERT_TRUE(built.ok()) << built.status();
    EXPECT_EQ(*built, cells.size());
    const auto count = LabelLiquidAirComponents3D(
        std::span(gas_cells).first(*built),
        std::span(labels).first(*built),
        queue,
        components
    );
    ASSERT_TRUE(count.ok()) << count.status();
    EXPECT_EQ(*count, 1U);
  };
  ASSERT_NO_FATAL_FAILURE(classify(sampled_cells, {}));
  EXPECT_EQ(components[0].connectivity, LiquidAirConnectivity3D::kEnclosed);
  EXPECT_NEAR(components[0].gas_volume, 0.5, 1e-15);
  const auto cropped_cells = std::span(sampled_cells).first(7);
  ASSERT_NO_FATAL_FAILURE(classify(cropped_cells, {}));
  EXPECT_EQ(components[0].connectivity, LiquidAirConnectivity3D::kUnclassified);
  EXPECT_NEAR(components[0].gas_volume, 0.375, 1e-15);
  const std::array outlets{FaceIndex3D{FaceNormalAxis::kX, {0, 0, 0}}};
  ASSERT_NO_FATAL_FAILURE(classify(cropped_cells, outlets));
  EXPECT_EQ(components[0].connectivity, LiquidAirConnectivity3D::kVented);
  EXPECT_EQ(
      BuildLiquidAirDomain3D(kGeometry, {}, sampled_cells, corners, outlets, gas_cells).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      BuildLiquidAirDomain3D(kGeometry, {}, sampled_cells, std::span(corners).first(26), {}, gas_cells).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(LiquidCavity3D, LabelsEnclosedVentedAndUnclassifiedAirComponents) {
  using Passage = LiquidAirPassage3D;
  using Connectivity = LiquidAirConnectivity3D;
  std::array cells{
      LiquidAirVolumeCell3D{.cell_index = {-3, 0, 0}, .gas_volume = 0.25},
      LiquidAirVolumeCell3D{.cell_index = {-2, 0, 0}, .gas_volume = 0.75},
      LiquidAirVolumeCell3D{.cell_index = {0, 0, 0}, .gas_volume = 1.0},
      LiquidAirVolumeCell3D{.cell_index = {1, 0, 0}, .gas_volume = 0.5},
      LiquidAirVolumeCell3D{.cell_index = {3, 0, 0}, .gas_volume = 0.125},
  };
  cells[0].passages[1] = cells[1].passages[0] = Passage::kOpen;
  cells[2].passages[1] = cells[3].passages[0] = Passage::kOpen;
  cells[2].passages[2] = Passage::kOpen;  // Unclassified frontier.
  cells[3].passages[3] = Passage::kAtmosphere;
  cells[4].passages[5] = Passage::kOpen;
  std::array<std::size_t, 5> labels;
  std::array<std::size_t, 5> queue;
  std::array<LiquidAirComponent3D, 3> components;
  const auto result = LabelLiquidAirComponents3D(cells, labels, queue, components);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 3U);
  EXPECT_EQ(labels, (std::array<std::size_t, 5>{0, 0, 1, 1, 2}));
  EXPECT_EQ(components[0].first_cell.x, -3);
  EXPECT_EQ(components[0].gas_volume, 1.0);
  EXPECT_EQ(components[0].connectivity, Connectivity::kEnclosed);
  EXPECT_EQ(components[1].gas_volume, 1.5);
  EXPECT_EQ(components[1].connectivity, Connectivity::kVented);
  EXPECT_EQ(components[2].gas_volume, 0.125);
  EXPECT_EQ(components[2].connectivity, Connectivity::kUnclassified);
  EXPECT_EQ(
      LabelLiquidAirComponents3D(cells, labels, queue, std::span(components).first(2)).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(
      LabelLiquidAirComponents3D(cells, labels, labels, components).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  cells[0].passages[1] = Passage::kClosed;
  EXPECT_EQ(
      LabelLiquidAirComponents3D(cells, labels, queue, components).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidCavity3D, AirConnectivityDoesNotWrapSignedCoordinates) {
  using Passage = LiquidAirPassage3D;
  constexpr auto kMin = std::numeric_limits<std::int32_t>::min();
  constexpr auto kMax = std::numeric_limits<std::int32_t>::max();
  std::array cells{
      LiquidAirVolumeCell3D{.cell_index = {kMin, 0, 0}, .gas_volume = 1.0},
      LiquidAirVolumeCell3D{.cell_index = {kMax, 0, 0}, .gas_volume = 1.0},
  };
  cells[0].passages[0] = Passage::kOpen;
  cells[1].passages[1] = Passage::kAtmosphere;
  std::array<std::size_t, 2> labels;
  std::array<std::size_t, 2> queue;
  std::array<LiquidAirComponent3D, 2> components;
  const auto result = LabelLiquidAirComponents3D(cells, labels, queue, components);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 2U);
  EXPECT_EQ(components[0].connectivity, LiquidAirConnectivity3D::kUnclassified);
  EXPECT_EQ(components[1].connectivity, LiquidAirConnectivity3D::kVented);
}

TEST(LiquidCavity3D, PreparedBoundariesUseOnlyResolvedEnclosedInventories) {
  using Connectivity = LiquidAirConnectivity3D;
  const std::array gas_cells{
      LiquidAirVolumeCell3D{.cell_index = {-2, 0, 0}, .gas_volume = 0.5},
      LiquidAirVolumeCell3D{.cell_index = {-1, 0, 0}, .gas_volume = 0.5},
      LiquidAirVolumeCell3D{.cell_index = {0, 0, 0}, .gas_volume = 0.01},
      LiquidAirVolumeCell3D{.cell_index = {1, 0, 0}, .gas_volume = 1.0},
      LiquidAirVolumeCell3D{.cell_index = {2, 0, 0}, .gas_volume = 1.0},
      LiquidAirVolumeCell3D{.cell_index = {3, 0, 0}, .gas_volume = 1.0},
  };
  std::array<std::size_t, 6> component_labels{0, 0, 1, 2, 3, 4};
  const std::array components{
      LiquidAirComponent3D{{-2, 0, 0}, 1.0, Connectivity::kEnclosed},
      LiquidAirComponent3D{{0, 0, 0}, 0.01, Connectivity::kEnclosed},
      LiquidAirComponent3D{{1, 0, 0}, 1.0, Connectivity::kVented},
      LiquidAirComponent3D{{2, 0, 0}, 1.0, Connectivity::kUnclassified},
      LiquidAirComponent3D{{3, 0, 0}, 1.0, Connectivity::kEnclosed},
  };
  const std::array<std::optional<LiquidCavityInventory3D>, 5> inventories{
      LiquidCavityInventory3D{2.0, 100000.0},
      LiquidCavityInventory3D{0.01, 100000.0},
      LiquidCavityInventory3D{1.0, 110000.0},
      LiquidCavityInventory3D{1.0, 110000.0},
      std::nullopt,
  };
  const std::array centers{
      LiquidCenterDistance3D{{-2, 0, 0}, -0.1},
      LiquidCenterDistance3D{{-1, 0, 0}, 0.1},
      LiquidCenterDistance3D{{0, 0, 0}, 0.1},
      LiquidCenterDistance3D{{1, 0, 0}, 0.1},
      LiquidCenterDistance3D{{2, 0, 0}, 0.1},
      LiquidCenterDistance3D{{3, 0, 0}, 0.1},
  };
  const LiquidCavityBoundaryInputs3D inputs{
      .gas_cells = gas_cells,
      .labels = component_labels,
      .components = components,
      .inventories = inventories,
      .center_distances = centers,
      .atmospheric_absolute_pressure = 100000.0,
      .liquid_density = 1000.0,
      .cell_width = 0.25,
  };
  std::array<LiquidCavityBoundaryComponent3D, 5> scratch;
  std::array<LiquidAirCellPressure3D, 1> air_pressures;
  const auto result = PrepareLiquidCavityBoundaries3D(inputs, scratch, air_pressures);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->active_cavity_count, 1U);
  EXPECT_EQ(result->subresolution_inventory_count, 1U);
  ASSERT_EQ(result->air_pressure_count, 1U);
  EXPECT_EQ(air_pressures[0].cell_index.x, -1);
  const double expected_absolute = 100000.0 * std::pow(2.0, 1.4);
  EXPECT_NEAR(air_pressures[0].gas_gauge_pressure, expected_absolute - 100000.0, 1e-8);
  const double expected_radius = std::cbrt(3.0 / (4.0 * std::numbers::pi));
  EXPECT_NEAR(
      result->maximum_step_duration,
      0.5 * expected_radius / std::sqrt(4.2 * expected_absolute / 1000.0),
      1e-15
  );
  EXPECT_EQ(PrepareLiquidCavityBoundaries3D(inputs, scratch, {}).status().code(), absl::StatusCode::kResourceExhausted);
  component_labels.back() = 0;
  EXPECT_EQ(
      PrepareLiquidCavityBoundaries3D(inputs, scratch, air_pressures).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidCavity3D, SubmergedSphereBalancesHydrostaticPressureAndSignedCurvature) {
  constexpr double kRadius = 0.25;
  constexpr double kVolume = 4.0 * std::numbers::pi * kRadius * kRadius * kRadius / 3.0;
  constexpr double kAtmosphere = 101325.0;
  constexpr double kHydrostatic = 9810.0;
  constexpr double kSurfaceTension = 0.072;
  const auto inventory = InitializeSubmergedLiquidCavity3D(kVolume, kAtmosphere, kHydrostatic, kSurfaceTension);
  ASSERT_TRUE(inventory.ok()) << inventory.status();
  const double expected_pressure = kAtmosphere + kHydrostatic + 2.0 * kSurfaceTension / kRadius;
  EXPECT_NEAR(inventory->initial_absolute_pressure, expected_pressure, 1e-10);
  const auto state =
      EvaluateLiquidCavity3D(*inventory, kVolume, kAtmosphere, 1000.0, kSurfaceTension, -2.0 / kRadius, 0.12);
  ASSERT_TRUE(state.ok()) << state.status();
  EXPECT_NEAR(state->boundary_gauge_pressure, kHydrostatic, 1e-10);
  EXPECT_NEAR(state->equivalent_radius, kRadius, 1e-15);
  EXPECT_TRUE(state->resolved);
  EXPECT_NEAR(state->maximum_step_duration, 0.5 * kRadius / std::sqrt(4.2 * expected_pressure / 1000.0), 1e-15);
}

TEST(LiquidCavity3D, FixedInventoryCompressesAndFallsBelowResolution) {
  const LiquidCavityInventory3D inventory{1.0, 100000.0};
  const auto compressed = EvaluateLiquidCavity3D(inventory, 0.5, 100000.0, 1000.0, 0.0, 0.0, 0.3);
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  const double expected_absolute = 100000.0 * std::pow(2.0, 1.4);
  EXPECT_NEAR(compressed->absolute_pressure, expected_absolute, 1e-8);
  EXPECT_NEAR(compressed->boundary_gauge_pressure, expected_absolute - 100000.0, 1e-8);
  EXPECT_FALSE(compressed->resolved);
  const auto expanded = EvaluateLiquidCavity3D(inventory, 2.0, 100000.0, 1000.0, 0.0, 0.0, 0.3);
  ASSERT_TRUE(expanded.ok()) << expanded.status();
  EXPECT_LT(expanded->boundary_gauge_pressure, 0.0);
  EXPECT_GT(expanded->maximum_step_duration, compressed->maximum_step_duration);
  EXPECT_TRUE(expanded->resolved);
  const auto threshold =
      EvaluateLiquidCavity3D(inventory, 2.0, 100000.0, 1000.0, 0.0, 0.0, expanded->equivalent_radius / 2.0);
  ASSERT_TRUE(threshold.ok()) << threshold.status();
  EXPECT_TRUE(threshold->resolved);
}

TEST(LiquidCavity3D, PreservesSmallGaugeIncrementAndChecksRange) {
  const double volume = std::nextafter(1.0, 2.0);
  const auto expanded = EvaluateLiquidCavity3D({1.0, 101325.0}, volume, 101325.0, 1000.0, 0.0, 0.0, 0.25);
  ASSERT_TRUE(expanded.ok()) << expanded.status();
  const double linearized_gauge = -1.4 * 101325.0 * (volume - 1.0);
  EXPECT_NEAR(expanded->boundary_gauge_pressure, linearized_gauge, 1e-24);
  EXPECT_EQ(InitializeSubmergedLiquidCavity3D(0.0, 1.0, 0.0, 0.0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      EvaluateLiquidCavity3D({1.0, 1.0}, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      EvaluateLiquidCavity3D({1.0, 1e300}, 1e-30, 1.0, 1.0, 0.0, 0.0, 1.0).status().code(),
      absl::StatusCode::kOutOfRange
  );
  const auto scaled = EvaluateLiquidCavity3D({1e-300, 1e-320}, 1e-300, 1e-320, 1e308, 0.0, 0.0, 1.0);
  ASSERT_TRUE(scaled.ok()) << scaled.status();
  EXPECT_TRUE(std::isfinite(scaled->maximum_step_duration));
  EXPECT_GT(scaled->maximum_step_duration, 1e210);
}

}  // namespace
}  // namespace fluidity
