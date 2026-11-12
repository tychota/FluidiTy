#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "fluidity/host/adaptive_liquid_core_transport_3d.h"
#include "fluidity/host/adaptive_liquid_owner_exchange_3d.h"
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

AdaptiveLiquidOwnerExchangeFace3D ExchangeFace(double velocity, double area = 1) {
  return {
      .projected_carrier = {.carrier_velocity = velocity, .validity = LiquidFaceValidity3D::kValid},
      .open_area_m2 = area,
  };
}

TEST(AdaptiveLiquidOwnerExchange3D, StagesSignedCrossOwnerSubfacesExactlyOnce) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  cells[1].material_owner = AdaptiveLiquidMaterialOwner3D::kParticles;
  cells[2].material_owner = AdaptiveLiquidMaterialOwner3D::kParticles;
  const auto committed = cells;
  const std::array faces{ExchangeFace(1), ExchangeFace(-1, 0.5), ExchangeFace(7)};
  const auto result = PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, faces, 0.5);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(cells, committed);
  EXPECT_EQ(result->particle_reconciliation_leaf_ordinals, (std::vector<std::size_t>{1, 2}));
  ASSERT_EQ(result->transfers.size(), 2);
  const auto& to_particles = result->transfers[0];
  EXPECT_EQ(to_particles.interior_face_ordinal, 0);
  EXPECT_EQ(to_particles.donor_leaf_ordinal, 0);
  EXPECT_EQ(to_particles.receiver_leaf_ordinal, 1);
  EXPECT_EQ(to_particles.signed_flux_m3_per_s, 1);
  EXPECT_EQ(Lanes(to_particles.transferred_inventory), (std::array{1.0, 0.75, 0.5, 1.5, -0.5, 2.5, -2.0}));
  const auto& to_core = result->transfers[1];
  EXPECT_EQ(to_core.interior_face_ordinal, 1);
  EXPECT_EQ(to_core.donor_leaf_ordinal, 2);
  EXPECT_EQ(to_core.receiver_leaf_ordinal, 0);
  EXPECT_EQ(to_core.signed_flux_m3_per_s, -0.5);
  EXPECT_EQ(Lanes(to_core.transferred_inventory), (std::array{1.5, 0.5, 0.25, -1.0, 0.5, 2.0, -2.0}));
  ASSERT_EQ(result->candidate_cells.size(), 3);
  EXPECT_EQ(Lanes(result->candidate_cells[0].inventory), (std::array{8.5, 5.75, 3.75, 9.5, -3.0, 19.5, -16.0}));
  EXPECT_EQ(Lanes(result->candidate_cells[1].inventory), (std::array{3.0, 0.875, 0.75, 0.5, 2.5, 4.5, 2.0}));
  EXPECT_EQ(Lanes(result->candidate_cells[2].inventory), (std::array{1.5, 0.5, 0.25, -1.0, 0.5, 2.0, -2.0}));
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    auto metadata = result->candidate_cells[cell];
    metadata.inventory = cells[cell].inventory;
    EXPECT_EQ(metadata, cells[cell]);
  }
  const std::array legacy_fluxes{1.0, -0.5, 7.0};
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreTransport3D(*topology, cells, legacy_fluxes, 0.5).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(AdaptiveLiquidOwnerExchange3D, RefusesOverdrawCapacityAndInvalidCarrierWithoutMutatingCandidates) {
  const auto topology = AdaptiveMacTopology3D::Create(kTransportLeaves, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto cells = CoreCells(*topology);
  cells[0].inventory.current_volume_m3 = 1;
  cells[2].inventory.current_volume_m3 = 0.25;
  cells[2].material_owner = AdaptiveLiquidMaterialOwner3D::kParticles;
  const auto committed = cells;
  const std::array valid_faces{ExchangeFace(0.25), ExchangeFace(0.25), ExchangeFace(0)};
  const auto first = PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, valid_faces, 1);
  ASSERT_TRUE(first.ok()) << first.status();
  const auto prepared_cells = first->candidate_cells;
  ASSERT_EQ(first->transfers.size(), 2);
  EXPECT_EQ(first->particle_reconciliation_leaf_ordinals, (std::vector<std::size_t>{2}));

  auto refused = valid_faces;
  refused[0] = ExchangeFace(0.75);
  refused[1] = ExchangeFace(0.75);
  EXPECT_EQ(
      PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, refused, 1).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  refused = valid_faces;
  refused[0] = ExchangeFace(0);
  refused[1] = ExchangeFace(1);
  EXPECT_EQ(
      PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, refused, 1).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  refused = valid_faces;
  refused[1].projected_carrier.validity = LiquidFaceValidity3D::kMissing;
  EXPECT_EQ(
      PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, refused, 1).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  refused = valid_faces;
  refused[1].projected_carrier.side_tag = 1;
  EXPECT_EQ(
      PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, refused, 1).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  refused = valid_faces;
  refused[1].open_area_m2 = 2;
  EXPECT_EQ(
      PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, refused, 1).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  refused = {ExchangeFace(0), ExchangeFace(0x1p-60), ExchangeFace(0)};
  EXPECT_EQ(
      PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, refused, 1).status().code(),
      absl::StatusCode::kOutOfRange
  );
  auto solid = cells;
  solid[2].occupancy = LiquidCellOccupancy3D::kSolid;
  EXPECT_EQ(
      PrepareAdaptiveLiquidOwnerExchange3D(*topology, solid, valid_faces, 1).status().code(),
      absl::StatusCode::kFailedPrecondition
  );

  const auto replay = PrepareAdaptiveLiquidOwnerExchange3D(*topology, cells, valid_faces, 1);
  ASSERT_TRUE(replay.ok()) << replay.status();
  EXPECT_EQ(cells, committed);
  EXPECT_EQ(first->candidate_cells, prepared_cells);
  EXPECT_EQ(replay->candidate_cells, prepared_cells);
  EXPECT_EQ(replay->particle_reconciliation_leaf_ordinals, first->particle_reconciliation_leaf_ordinals);
  ASSERT_EQ(replay->transfers.size(), first->transfers.size());
  for (std::size_t face = 0; face < first->transfers.size(); ++face) {
    const auto& left = first->transfers[face];
    const auto& right = replay->transfers[face];
    EXPECT_EQ(left.interior_face_ordinal, right.interior_face_ordinal);
    EXPECT_EQ(left.donor_leaf_ordinal, right.donor_leaf_ordinal);
    EXPECT_EQ(left.receiver_leaf_ordinal, right.receiver_leaf_ordinal);
    EXPECT_EQ(left.signed_flux_m3_per_s, right.signed_flux_m3_per_s);
    EXPECT_EQ(Lanes(left.transferred_inventory), Lanes(right.transferred_inventory));
  }
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
