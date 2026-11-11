#include <array>
#include <cstdint>
#include <limits>
#include <optional>

#include "fluidity/host/liquid_band_handoff_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr LiquidHandoffCell3D kCell{
    .world_lower_corner = {-2.0, 4.0, -1.0},
    .cell_width = 2.0,
    .occupancy = LiquidHandoffOccupancy3D::kFullLiquid,
};

constexpr LiquidHandoffCarrier3D kCarrier{
    .current_density = 1.0,
    .volume_ratio = 0.5,
    .cell_center_velocity = {0.0, 1.25, -2.0},
    .velocity_slopes = {{2.0, -1.0, 3.0}, {0.0, 2.0, 0.0}, {-1.0, 0.0, 1.0}},
    .tag = 19,
    .side_tag = -9,
};

constexpr std::array<LiquidParticleState3D, 2> kBand{
    LiquidParticleState3D{
        .id = 3,
        .tag = 19,
        .world_position = {-1.5, 4.5, -0.5},
        .world_velocity = {3.0, -1.0, 1.0},
        .mass = 2.0,
        .reference_volume = 4.0,
        .volume_ratio = 0.5,
        .side_tag = -9,
    },
    LiquidParticleState3D{
        .id = 1,
        .tag = 19,
        .world_position = {-0.5, 5.5, 0.5},
        .world_velocity = {-1.0, 2.0, -3.0},
        .mass = 6.0,
        .reference_volume = 12.0,
        .volume_ratio = 0.5,
        .side_tag = -9,
    },
};

constexpr LiquidRemapInventory3D kCore{
    .mass_kg = 8.0,
    .reference_volume_m3 = 16.0,
    .current_volume_m3 = 8.0,
    .momentum_kg_m_per_s = {0.0, 10.0, -16.0},
    .enthalpy_j = -7.5,
};

void ExpectInventory(const LiquidRemapInventory3D& actual, const LiquidRemapInventory3D& expected) {
  EXPECT_DOUBLE_EQ(actual.mass_kg, expected.mass_kg);
  EXPECT_DOUBLE_EQ(actual.reference_volume_m3, expected.reference_volume_m3);
  EXPECT_DOUBLE_EQ(actual.current_volume_m3, expected.current_volume_m3);
  EXPECT_DOUBLE_EQ(actual.momentum_kg_m_per_s.x, expected.momentum_kg_m_per_s.x);
  EXPECT_DOUBLE_EQ(actual.momentum_kg_m_per_s.y, expected.momentum_kg_m_per_s.y);
  EXPECT_DOUBLE_EQ(actual.momentum_kg_m_per_s.z, expected.momentum_kg_m_per_s.z);
  EXPECT_DOUBLE_EQ(actual.enthalpy_j, expected.enthalpy_j);
}

TEST(LiquidBandHandoff3D, ConservesSignedInventoryAcrossExclusiveOwnershipRoundTrip) {
  const auto core = RetireLiquidBandCell3D(kCell, kBand, kCore.enthalpy_j, kCarrier);
  ASSERT_TRUE(core.ok()) << core.status();
  ExpectInventory(*core, kCore);

  const auto band = ReseedLiquidCoreCell3D(kCell, *core, kCarrier, 8, 8, 30);
  ASSERT_TRUE(band.ok()) << band.status();
  ASSERT_EQ(band->particles().size(), 8);
  EXPECT_EQ(band->next_particle_id(), 38);
  EXPECT_DOUBLE_EQ(band->enthalpy_j(), kCore.enthalpy_j);
  double mass = 0.0;
  double reference_volume = 0.0;
  double current_volume = 0.0;
  Vec3 momentum{};
  for (std::size_t ordinal = 0; ordinal < band->particles().size(); ++ordinal) {
    const auto& particle = band->particles()[ordinal];
    EXPECT_EQ(particle.id, 30 + ordinal);
    EXPECT_EQ(particle.tag, kCarrier.tag);
    EXPECT_EQ(particle.side_tag, kCarrier.side_tag);
    EXPECT_TRUE(particle.affine_enabled);
    EXPECT_FALSE(particle.pic_initialized);
    EXPECT_DOUBLE_EQ(particle.mass, 1.0);
    EXPECT_DOUBLE_EQ(particle.reference_volume, 2.0);
    EXPECT_DOUBLE_EQ(particle.volume_ratio, 0.5);
    EXPECT_DOUBLE_EQ(particle.velocity_slopes.x_row.z, 3.0);
    const double dx = particle.world_position.x + 1.0;
    const double dy = particle.world_position.y - 5.0;
    const double dz = particle.world_position.z;
    EXPECT_DOUBLE_EQ(particle.world_velocity.x, 2.0 * dx - dy + 3.0 * dz);
    EXPECT_DOUBLE_EQ(particle.world_velocity.y, 1.25 + 2.0 * dy);
    EXPECT_DOUBLE_EQ(particle.world_velocity.z, -2.0 - dx + dz);
    mass += particle.mass;
    reference_volume += particle.reference_volume;
    current_volume += particle.volume_ratio * particle.reference_volume;
    momentum.x += particle.mass * particle.world_velocity.x;
    momentum.y += particle.mass * particle.world_velocity.y;
    momentum.z += particle.mass * particle.world_velocity.z;
  }
  ExpectInventory({mass, reference_volume, current_volume, momentum, band->enthalpy_j()}, kCore);
  const auto round_trip = RetireLiquidBandCell3D(kCell, band->particles(), band->enthalpy_j(), kCarrier);
  ASSERT_TRUE(round_trip.ok()) << round_trip.status();
  ExpectInventory(*round_trip, kCore);
}

TEST(LiquidBandHandoff3D, ReplaysOddSeedCountsAndPreservesIdOnCapacityRefusal) {
  constexpr auto max_id = std::numeric_limits<std::uint64_t>::max();
  const auto denied = ReseedLiquidCoreCell3D(kCell, kCore, kCarrier, 3, 2, max_id - 3);
  EXPECT_EQ(denied.status().code(), absl::StatusCode::kResourceExhausted);
  const auto overflow = ReseedLiquidCoreCell3D(kCell, kCore, kCarrier, 3, 3, max_id - 2);
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
  const auto first = ReseedLiquidCoreCell3D(kCell, kCore, kCarrier, 3, 3, max_id - 3);
  const auto replay = ReseedLiquidCoreCell3D(kCell, kCore, kCarrier, 3, 7, max_id - 3);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(replay.ok()) << replay.status();
  EXPECT_EQ(first->next_particle_id(), max_id);
  EXPECT_EQ(first->particles().front().id, max_id - 3);
  EXPECT_DOUBLE_EQ(first->particles().front().world_position.x, -1.0);
  EXPECT_DOUBLE_EQ(first->particles()[1].world_position.x + first->particles()[2].world_position.x, -2.0);
  for (std::size_t i = 0; i < first->particles().size(); ++i) {
    EXPECT_EQ(first->particles()[i], replay->particles()[i]);
  }
  const auto core = RetireLiquidBandCell3D(kCell, first->particles(), first->enthalpy_j(), kCarrier);
  ASSERT_TRUE(core.ok()) << core.status();
  ExpectInventory(*core, kCore);
}

TEST(LiquidBandHandoff3D, RejectsPartialOrMissingOccupancyAndInvalidCarrier) {
  for (const auto occupancy : {LiquidHandoffOccupancy3D::kPartial, LiquidHandoffOccupancy3D::kMissing}) {
    auto cell = kCell;
    cell.occupancy = occupancy;
    EXPECT_EQ(
        RetireLiquidBandCell3D(cell, kBand, kCore.enthalpy_j, kCarrier).status().code(),
        absl::StatusCode::kFailedPrecondition
    );
    EXPECT_EQ(
        ReseedLiquidCoreCell3D(cell, kCore, kCarrier, 1, 1, 30).status().code(),
        absl::StatusCode::kFailedPrecondition
    );
  }
  EXPECT_EQ(
      RetireLiquidBandCell3D(kCell, kBand, kCore.enthalpy_j, std::nullopt).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(
      ReseedLiquidCoreCell3D(kCell, kCore, std::nullopt, 1, 1, 30).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto carrier = kCarrier;
  carrier.velocity_slopes.y_row.z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      ReseedLiquidCoreCell3D(kCell, kCore, carrier, 1, 1, 30).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  carrier = kCarrier;
  carrier.volume_ratio = 0.0;
  EXPECT_EQ(
      RetireLiquidBandCell3D(kCell, kBand, kCore.enthalpy_j, carrier).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidBandHandoff3D, RejectsInconsistentLedgerAndDuplicateOrOutOfCellParticles) {
  auto carrier = kCarrier;
  carrier.current_density = 2.0;
  EXPECT_EQ(
      ReseedLiquidCoreCell3D(kCell, kCore, carrier, 4, 4, 30).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  carrier = kCarrier;
  carrier.volume_ratio = 1.0;
  EXPECT_EQ(
      ReseedLiquidCoreCell3D(kCell, kCore, carrier, 4, 4, 30).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  carrier = kCarrier;
  carrier.cell_center_velocity.x = 0.25;
  EXPECT_EQ(
      ReseedLiquidCoreCell3D(kCell, kCore, carrier, 4, 4, 30).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(
      RetireLiquidBandCell3D(kCell, kBand, kCore.enthalpy_j, carrier).status().code(),
      absl::StatusCode::kFailedPrecondition
  );

  auto particles = kBand;
  particles[1].id = particles[0].id;
  EXPECT_EQ(
      RetireLiquidBandCell3D(kCell, particles, kCore.enthalpy_j, kCarrier).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  particles = kBand;
  particles[1].world_position.x = 0.0;
  EXPECT_EQ(
      RetireLiquidBandCell3D(kCell, particles, kCore.enthalpy_j, kCarrier).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  particles = kBand;
  particles[1].side_tag++;
  EXPECT_EQ(
      RetireLiquidBandCell3D(kCell, particles, kCore.enthalpy_j, kCarrier).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

}  // namespace
}  // namespace fluidity
