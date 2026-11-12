#include <array>
#include <cmath>
#include <limits>
#include <numbers>

#include "fluidity/host/liquid_air_components.h"
#include "fluidity/host/liquid_cavity.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(LiquidCavity3D, PreparesSampledCavityThenRetainsItsInventoryUnderCompression) {
  constexpr int kHalfExtent = 4;
  constexpr std::size_t kCellCount = 512;
  constexpr std::size_t kCornerCount = 729;
  std::array<CellIndex3D, kCellCount> sampled_cells;
  std::array<LiquidCornerDistance3D, kCornerCount> corners;
  std::array<LiquidCenterDistance3D, kCellCount> centers;
  auto sample = [&](double radius) {
    std::size_t cell = 0;
    std::size_t corner = 0;
    for (int z = -kHalfExtent; z <= kHalfExtent; ++z) {
      for (int y = -kHalfExtent; y <= kHalfExtent; ++y) {
        for (int x = -kHalfExtent; x <= kHalfExtent; ++x) {
          const CellIndex3D index{x, y, z};
          corners[corner++] = {
              {x, y, z},
              radius - std::hypot(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z))
          };
          if (x < kHalfExtent && y < kHalfExtent && z < kHalfExtent) {
            sampled_cells[cell] = index;
            centers[cell++] = {index, radius - std::hypot(x + 0.5, y + 0.5, z + 0.5)};
          }
        }
      }
    }
  };
  struct Scratch {
    std::array<LiquidAirVolumeCell3D, kCellCount> gas;
    std::array<std::size_t, kCellCount> labels;
    std::array<std::size_t, kCellCount> queue;
    std::array<LiquidAirComponent3D, 1> components;
    std::array<LiquidAirComponentOverlap3D, kCellCount> overlaps;
    std::array<std::optional<LiquidCavityInventory3D>, 1> inventories;
    std::array<LiquidCavityLineageCandidate3D, 1> lineage;
    std::array<LiquidCavityBoundaryComponent3D, 1> boundary;
    std::array<LiquidAirCellPressure3D, kCellCount> pressures;
    LiquidCavityStepScratch3D View() {
      return {gas, labels, queue, components, overlaps, inventories, lineage, boundary, pressures};
    }
  } first, second;
  const LiquidObstacle3D obstacle;
  int initializations = 0;
  LiquidCavityStepInputs3D inputs{
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .obstacle = obstacle,
      .sampled_cells = sampled_cells,
      .corners = corners,
      .centers = centers,
      .atmosphere_outlets = {},
      .previous_snapshot = {},
      .previous_lineage = {},
      .next_id = 1,
      .atmospheric_absolute_pressure = 100000.0,
      .liquid_density = 1000.0,
      .initialization_context = &initializations,
      .initialize = [](void* context,
                       const LiquidAirComponent3D& component) -> absl::StatusOr<LiquidCavityInventory3D> {
        ++*static_cast<int*>(context);
        return InitializeSubmergedLiquidCavity3D(component.gas_volume, 100000.0, 4000.0, 0.0);
      },
  };
  sample(3.2);
  const auto start = PrepareLiquidCavityStep3D(inputs, first.View());
  ASSERT_TRUE(start.ok()) << start.status();
  ASSERT_EQ(start->component_count, 1);
  ASSERT_GT(start->boundaries.air_pressure_count, 0);
  EXPECT_DOUBLE_EQ(first.pressures.front().gas_gauge_pressure, 4000.0);
  EXPECT_EQ(first.lineage.front().state.id, 1);

  const std::array previous_lineage{first.lineage.front().state};
  inputs.previous_snapshot = {
      std::span(first.gas).first(start->gas_cell_count),
      std::span(first.labels).first(start->gas_cell_count),
      1
  };
  inputs.previous_lineage = previous_lineage;
  inputs.next_id = start->lineage.next_id;
  sample(3.0);
  const auto compressed = PrepareLiquidCavityStep3D(inputs, second.View());
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  EXPECT_EQ(initializations, 1);
  EXPECT_EQ(second.lineage.front().state.id, 1);
  EXPECT_EQ(compressed->lineage.next_id, 2);
  EXPECT_LT(compressed->boundaries.maximum_step_duration, start->boundaries.maximum_step_duration);
  const double pressure =
      104000.0 * std::pow(first.components.front().gas_volume / second.components.front().gas_volume, 1.4);
  ASSERT_GT(compressed->boundaries.air_pressure_count, 0);
  EXPECT_NEAR(second.pressures.front().gas_gauge_pressure, pressure - 100000.0, 1.0e-8);
}

TEST(LiquidCavity3D, LineageAssignsSplitMergeBirthAndLossDeterministically) {
  using Connectivity = LiquidAirConnectivity3D;
  const std::array<LiquidCavityLineageState3D, 3> previous{{
      {10, LiquidCavityInventory3D{10.0, 100000.0}},
      {11, LiquidCavityInventory3D{4.0, 100000.0}},
      {12, LiquidCavityInventory3D{2.0, 100000.0}},
  }};
  std::array<LiquidAirComponent3D, 5> current{{
      {{0, 0, 0}, 6.0, Connectivity::kEnclosed},
      {{1, 0, 0}, 6.0, Connectivity::kEnclosed},
      {{2, 0, 0}, 2.0, Connectivity::kVented},
      {{3, 0, 0}, 0.001, Connectivity::kEnclosed},
      {{4, 0, 0}, 4.0, Connectivity::kEnclosed},
  }};
  constexpr std::array<LiquidAirComponentOverlap3D, 5> kOverlaps{{
      {0, 0, 0.004},
      {0, 1, 0.003},
      {0, 2, 0.002},
      {0, 3, 0.001},
      {1, 0, 0.003},
  }};
  std::array<std::optional<LiquidCavityInventory3D>, 5> initial{};
  initial.back() = LiquidCavityInventory3D{4.0, 110000.0};
  const LiquidCavityLineageInputs3D inputs{
      .previous = previous,
      .current = current,
      .overlaps = kOverlaps,
      .initial_inventories = initial,
      .next_id = 13,
      .cell_width = 0.1,
  };
  std::array<LiquidCavityLineageCandidate3D, 5> candidates;
  const auto result = PrepareLiquidCavityLineage3D(inputs, candidates);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->next_id, 15);
  ASSERT_TRUE(candidates[0].state.inventory);
  EXPECT_EQ(candidates[0].state.id, 10);
  EXPECT_DOUBLE_EQ(candidates[0].state.inventory->initial_volume, 8.0);
  ASSERT_TRUE(candidates[1].state.inventory);
  EXPECT_EQ(candidates[1].state.id, 13);
  EXPECT_DOUBLE_EQ(candidates[1].state.inventory->initial_volume, 3.0);
  EXPECT_FALSE(candidates[2].state.inventory);
  EXPECT_FALSE(candidates[3].state.inventory);
  ASSERT_TRUE(candidates[4].state.inventory);
  EXPECT_EQ(candidates[4].state.id, 14);
  EXPECT_DOUBLE_EQ(candidates[4].state.inventory->initial_absolute_pressure, 110000.0);
  const double pressure_root = std::pow(100000.0, 1.0 / 1.4);
  EXPECT_DOUBLE_EQ(result->vented_inventory, 2.0 * pressure_root);
  EXPECT_DOUBLE_EQ(result->subresolution_inventory, pressure_root);
  EXPECT_DOUBLE_EQ(result->unmatched_inventory, 2.0 * pressure_root);

  current[2].connectivity = Connectivity::kUnclassified;
  EXPECT_EQ(PrepareLiquidCavityLineage3D(inputs, candidates).status().code(), absl::StatusCode::kFailedPrecondition);
  current[2].connectivity = Connectivity::kVented;
  initial.back().reset();
  EXPECT_EQ(PrepareLiquidCavityLineage3D(inputs, candidates).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(LiquidCavity3D, MeasuresCanonicalOverlapEdgesForSplitAndMerge) {
  constexpr std::array<LiquidAirVolumeCell3D, 5> kPrevious{{
      {{-2, 0, 0}, 0.25},
      {{-1, 0, 0}, 0.5},
      {{0, 0, 0}, 0.5},
      {{1, 0, 0}, 0.75},
      {{3, 0, 0}, 1.0},
  }};
  constexpr std::array<LiquidAirVolumeCell3D, 5> kCurrent{{
      {{-1, 0, 0}, 0.75},
      {{0, 0, 0}, 0.25},
      {{1, 0, 0}, 0.5},
      {{2, 0, 0}, 1.0},
      {{3, 0, 0}, 0.25},
  }};
  constexpr std::array<std::size_t, 5> kPreviousLabels{0, 0, 0, 0, 1};
  constexpr std::array<std::size_t, 5> kCurrentLabels{0, 0, 1, 1, 1};
  const LiquidAirComponentSnapshot3D previous{kPrevious, kPreviousLabels, 2};
  const LiquidAirComponentSnapshot3D current{kCurrent, kCurrentLabels, 2};
  std::array<LiquidAirComponentOverlap3D, 4> overlaps;
  const auto count = MeasureLiquidAirComponentOverlaps3D(previous, current, overlaps);
  ASSERT_TRUE(count.ok()) << count.status();
  ASSERT_EQ(*count, 3);
  EXPECT_EQ(overlaps[0].previous_component, 0);
  EXPECT_EQ(overlaps[0].current_component, 0);
  EXPECT_DOUBLE_EQ(overlaps[0].shared_cell_volume, 0.75);
  EXPECT_EQ(overlaps[1].previous_component, 0);
  EXPECT_EQ(overlaps[1].current_component, 1);
  EXPECT_DOUBLE_EQ(overlaps[1].shared_cell_volume, 0.5);
  EXPECT_EQ(overlaps[2].previous_component, 1);
  EXPECT_EQ(overlaps[2].current_component, 1);
  EXPECT_DOUBLE_EQ(overlaps[2].shared_cell_volume, 0.25);
  EXPECT_EQ(
      MeasureLiquidAirComponentOverlaps3D(previous, current, std::span(overlaps).first(3)).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  const LiquidAirComponentSnapshot3D orphaned{kPrevious, kPreviousLabels, 3};
  EXPECT_EQ(
      MeasureLiquidAirComponentOverlaps3D(orphaned, current, overlaps).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidCavity3D, SplitAndMergePreserveAdditivePolytropicInventory) {
  constexpr LiquidCavityInventory3D kParent{8.0, 120000.0};
  constexpr std::array<LiquidCavityInventoryShare3D, 1> kLeftShare{{{kParent, 0.25}}};
  constexpr std::array<LiquidCavityInventoryShare3D, 1> kRightShare{{{kParent, 0.75}}};
  const auto left = CombineLiquidCavityInventoryShares3D(kLeftShare);
  const auto right = CombineLiquidCavityInventoryShares3D(kRightShare);
  ASSERT_TRUE(left.ok()) << left.status();
  ASSERT_TRUE(right.ok()) << right.status();
  EXPECT_DOUBLE_EQ(left->initial_volume, 2.0);
  EXPECT_DOUBLE_EQ(right->initial_volume, 6.0);
  EXPECT_DOUBLE_EQ(left->initial_absolute_pressure, kParent.initial_absolute_pressure);
  const std::array<LiquidCavityInventoryShare3D, 2> reunion{{{*left, 1.0}, {*right, 1.0}}};
  const auto reunited = CombineLiquidCavityInventoryShares3D(reunion);
  ASSERT_TRUE(reunited.ok()) << reunited.status();
  EXPECT_DOUBLE_EQ(reunited->initial_volume, kParent.initial_volume);
  EXPECT_DOUBLE_EQ(reunited->initial_absolute_pressure, kParent.initial_absolute_pressure);

  constexpr LiquidCavityInventory3D kOther{3.0, 280000.0};
  const std::array<LiquidCavityInventoryShare3D, 2> different_pressures{{{*left, 1.0}, {kOther, 1.0}}};
  const auto combined = CombineLiquidCavityInventoryShares3D(different_pressures);
  ASSERT_TRUE(combined.ok()) << combined.status();
  constexpr double kVolume = 4.0;
  const double left_pressure_root = 2.0 * std::pow(120000.0, 1.0 / 1.4);
  const double other_pressure_root = 3.0 * std::pow(280000.0, 1.0 / 1.4);
  const double expected_pressure = std::pow((left_pressure_root + other_pressure_root) / kVolume, 1.4);
  const auto pressure = EvaluateLiquidCavity3D(*combined, kVolume, 101325.0, 1000.0, 0.0, 0.0, 0.1);
  ASSERT_TRUE(pressure.ok()) << pressure.status();
  EXPECT_NEAR(pressure->absolute_pressure, expected_pressure, 1.0e-8);

  constexpr std::array<LiquidCavityInventoryShare3D, 1> kInvalidShare{{{kParent, 1.01}}};
  EXPECT_EQ(CombineLiquidCavityInventoryShares3D(kInvalidShare).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(CombineLiquidCavityInventoryShares3D({}).status().code(), absl::StatusCode::kInvalidArgument);
}

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
