#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "fluidity/host/adaptive_liquid_core_transport_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr std::array<AdaptiveMacCellKey3D, 3> kTransportLeaves{{
    {{0, 0, 0}, 1},
    {{2, 0, 0}, 0},
    {{2, 1, 0}, 0},
}};

std::vector<LiquidAdaptiveCellState3D> CoreCells(const AdaptiveMacTopology3D& topology) {
  const std::array<LiquidRemapInventory3D, 3> inventories{{
      {8, 6, 4, {12, -4, 20}, -16},
      {2, 0.125, 0.25, {-1, 3, 2}, 4},
      {3, 1, 0.5, {-2, 1, 4}, -4},
  }};
  std::vector<LiquidAdaptiveCellState3D> cells;
  for (std::size_t ordinal = 0; ordinal < topology.cells().size(); ++ordinal) {
    cells.push_back({
        .key = topology.cells()[ordinal].key,
        .material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian,
        .inventory = inventories[ordinal],
        .previous_pressure = static_cast<double>(ordinal),
    });
  }
  return cells;
}

auto Lanes(const LiquidRemapInventory3D& inventory) {
  return std::array{
      inventory.mass_kg,
      inventory.reference_volume_m3,
      inventory.current_volume_m3,
      inventory.momentum_kg_m_per_s.x,
      inventory.momentum_kg_m_per_s.y,
      inventory.momentum_kg_m_per_s.z,
      inventory.enthalpy_j,
  };
}

TEST(AdaptiveLiquidCoreTransport3D, AdvectsBothOrientationsAcrossCoarseFineSubfaces) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  ASSERT_EQ(topology->interior_faces().size(), 3);
  EXPECT_EQ(topology->interior_faces()[0].tail_cell_ordinal, 0);
  EXPECT_EQ(topology->interior_faces()[0].head_cell_ordinal, 1);
  EXPECT_EQ(topology->interior_faces()[1].tail_cell_ordinal, 0);
  EXPECT_EQ(topology->interior_faces()[1].head_cell_ordinal, 2);
  const auto cells = CoreCells(*topology);
  const std::array fluxes{1.0, -0.5, 0.0};
  const auto result = PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, fluxes, 0.5);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), cells.size());

  EXPECT_EQ(Lanes((*result)[0].inventory), (std::array{8.5, 5.75, 3.75, 9.5, -3.0, 19.5, -16.0}));
  EXPECT_EQ(Lanes((*result)[1].inventory), (std::array{3.0, 0.875, 0.75, 0.5, 2.5, 4.5, 2.0}));
  EXPECT_EQ(Lanes((*result)[2].inventory), (std::array{1.5, 0.5, 0.25, -1.0, 0.5, 2.0, -2.0}));

  auto committed_total = Lanes(cells.front().inventory);
  auto candidate_total = Lanes(result->front().inventory);
  for (std::size_t ordinal = 0; ordinal < cells.size(); ++ordinal) {
    auto candidate_cell = (*result)[ordinal];
    candidate_cell.inventory = cells[ordinal].inventory;
    EXPECT_EQ(candidate_cell, cells[ordinal]);
    if (ordinal == 0) {
      continue;
    }
    const auto committed_lanes = Lanes(cells[ordinal].inventory);
    const auto candidate_lanes = Lanes((*result)[ordinal].inventory);
    for (std::size_t lane = 0; lane < committed_total.size(); ++lane) {
      committed_total[lane] += committed_lanes[lane];
      candidate_total[lane] += candidate_lanes[lane];
    }
  }
  EXPECT_EQ(candidate_total, committed_total);
}

TEST(AdaptiveLiquidCoreTransport3D, WorkspaceReusesRecordsAndPreservesLastSuccessOnRefusal) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto cells = CoreCells(*topology);
  const std::array fluxes{1.0, -0.5, 0.0};
  const auto expected = PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, fluxes, 0.5);
  ASSERT_TRUE(expected.ok()) << expected.status();
  auto workspace = AdaptiveLiquidCoreTransportWorkspace3D::Create(cells.size());
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto reserved_bytes = workspace->reserved_bytes();
  EXPECT_GT(reserved_bytes, cells.size() * sizeof(LiquidAdaptiveCellState3D));

  const auto first = workspace->Prepare(*topology, cells, fluxes, 0.5);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(std::ranges::equal(*first, *expected));
  const auto* output_records = first->data();
  const auto overfill = workspace->Prepare(*topology, cells, std::array{2.0, 0.0, 0.0}, 0.5);
  EXPECT_EQ(overfill.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(std::ranges::equal(*first, *expected));

  const std::array zero_fluxes{0.0, 0.0, 0.0};
  const auto second = workspace->Prepare(*topology, cells, zero_fluxes, 0.5);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->data(), output_records);
  EXPECT_TRUE(std::ranges::equal(*second, cells));
  EXPECT_EQ(workspace->reserved_bytes(), reserved_bytes);
  EXPECT_EQ(
      workspace->Prepare(*topology, *second, zero_fluxes, 0.5).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_TRUE(std::ranges::equal(*second, cells));

  auto undersized = AdaptiveLiquidCoreTransportWorkspace3D::Create(cells.size() - 1);
  ASSERT_TRUE(undersized.ok()) << undersized.status();
  EXPECT_EQ(undersized->Prepare(*topology, cells, fluxes, 0.5).status().code(), absl::StatusCode::kResourceExhausted);
}

TEST(AdaptiveLiquidCoreTransport3D, RejectsGrossOverdrawEvenWithIncomingVolume) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  const auto committed = cells;
  // The third face draws 0.5 m³ from the first fine leaf, which starts with
  // 0.25 m³ and receives 0.5 m³ through the first face during this step.
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, std::array{1.0, 0.0, 1.0}, 0.5).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(cells, committed);
}

TEST(AdaptiveLiquidCoreTransport3D, RejectsCandidateCapacityOverflow) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  const auto committed = cells;
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, std::array{2.0, 0.0, 0.0}, 0.5).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(cells, committed);
}

TEST(AdaptiveLiquidCoreTransport3D, RejectsNonzeroExchangeWithParticleAndAirLeaves) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  cells[1].material_owner = AdaptiveLiquidMaterialOwner3D::kParticles;
  cells[2].occupancy = LiquidCellOccupancy3D::kAir;
  cells[2].inventory = {};
  const auto committed = cells;
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, std::array{1.0, 0.0, 0.0}, 0.5).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, std::array{0.0, -1.0, 0.0}, 0.5).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(cells, committed);
  const auto outside_core = PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, std::array{0.0, 0.0, 1.0}, 0.5);
  ASSERT_TRUE(outside_core.ok()) << outside_core.status();
  EXPECT_EQ(*outside_core, committed);
}

TEST(AdaptiveLiquidCoreTransport3D, RejectsUnrepresentableSweep) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  const auto committed = cells;
  const std::array fluxes{std::numeric_limits<double>::denorm_min(), 0.0, 0.0};
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, fluxes, 0.5).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(cells, committed);
}

TEST(AdaptiveLiquidCoreTransport3D, RejectsInexactSubnormalDonorFraction) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  cells[0].inventory.current_volume_m3 = 0.75;
  cells[0].inventory.mass_kg = std::numeric_limits<double>::max();
  const std::array fluxes{std::numeric_limits<double>::denorm_min(), 0.0, 0.0};
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, fluxes, 1.0).status().code(),
      absl::StatusCode::kOutOfRange
  );
}

TEST(AdaptiveLiquidCoreTransport3D, ExactSplitDrainLeavesNoStrandedInventory) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  cells[0].inventory = {1.0, 2.0, 1.5, {1.0, -2.0, 3.0}, -1.0};
  cells[1].inventory = {};
  cells[2].inventory = {};
  const auto result = PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, std::array{0.5, 1.0, 0.0}, 1.0);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(Lanes((*result)[0].inventory), (std::array<double, 7>{}));
  const auto first = Lanes((*result)[1].inventory);
  const auto second = Lanes((*result)[2].inventory);
  const auto initial = Lanes(cells[0].inventory);
  for (std::size_t lane = 0; lane < initial.size(); ++lane) {
    EXPECT_DOUBLE_EQ(first[lane] + second[lane], initial[lane]);
  }
}

}  // namespace
}  // namespace fluidity
