#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "fluidity/host/liquid_band_handoff_3d.h"
#include "fluidity/host/liquid_cache_3d.h"
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

TEST(LiquidBandHandoff3D, ConservesSamePhaseConstituentsThroughGridAndParticleOwners) {
  auto carrier = kCarrier;
  carrier.phase_id = 4;
  auto particles = kBand;
  particles[0].phase_id = 4;
  particles[0].constituent_count = 1;
  particles[0].constituents[0] = {14, 1};
  particles[1].phase_id = 4;
  particles[1].constituent_count = 1;
  particles[1].constituents[0] = {8, 1};
  const auto core = RetireLiquidBandCell3D(kCell, particles, kCore.enthalpy_j, carrier);
  ASSERT_TRUE(core.ok()) << core.status();
  ExpectInventory(*core, kCore);
  ASSERT_EQ(core->constituent_count, 2);
  EXPECT_EQ(core->phase_id, 4);
  EXPECT_EQ(core->constituents[0].substance_id, 8);
  EXPECT_DOUBLE_EQ(core->constituents[0].volume_m3, 6);
  EXPECT_EQ(core->constituents[1].substance_id, 14);
  EXPECT_DOUBLE_EQ(core->constituents[1].volume_m3, 2);

  LiquidRemapStencil3D neighbors{};
  neighbors.fill({*core, *core});
  auto& oil = neighbors[0][1];
  oil.phase_id = 5;
  oil.constituent_count = 1;
  oil.constituents[0] = {25, oil.current_volume_m3};
  oil.mass_kg *= 2;
  const auto children = ProlongLiquidInventory3D(*core, neighbors);
  ASSERT_TRUE(children.ok()) << children.status();
  for (const auto& child : *children) {
    ASSERT_EQ(child.constituent_count, 2);
    EXPECT_EQ(child.phase_id, 4);
    EXPECT_DOUBLE_EQ(child.mass_kg, 1);
    EXPECT_DOUBLE_EQ(child.constituents[0].volume_m3, 0.75);
    EXPECT_DOUBLE_EQ(child.constituents[1].volume_m3, 0.25);
  }
  const auto restricted = RestrictLiquidInventory3D(*children);
  ASSERT_TRUE(restricted.ok()) << restricted.status();
  const auto band = ReseedLiquidCoreCell3D(kCell, *restricted, carrier, 8, 8, 30);
  ASSERT_TRUE(band.ok()) << band.status();
  for (const auto& particle : band->particles()) {
    ASSERT_EQ(particle.constituent_count, 2);
    EXPECT_EQ(particle.phase_id, 4);
    EXPECT_EQ(particle.constituents[0], (LiquidConstituentFraction3D{8, 0.75}));
    EXPECT_EQ(particle.constituents[1], (LiquidConstituentFraction3D{14, 0.25}));
  }
  const auto retired_again = RetireLiquidBandCell3D(kCell, band->particles(), band->enthalpy_j(), carrier);
  ASSERT_TRUE(retired_again.ok()) << retired_again.status();
  ExpectInventory(*retired_again, kCore);
  EXPECT_DOUBLE_EQ(retired_again->constituents[0].volume_m3, 6);
  EXPECT_DOUBLE_EQ(retired_again->constituents[1].volume_m3, 2);

  particles[0].phase_id = 5;
  particles[0].constituents[0] = {25, 1};
  EXPECT_EQ(
      RetireLiquidBandCell3D(kCell, particles, kCore.enthalpy_j, carrier).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto mixed_children = *children;
  mixed_children[0] = oil;
  EXPECT_EQ(RestrictLiquidInventory3D(mixed_children).status().code(), absl::StatusCode::kFailedPrecondition);
  carrier.phase_id = 5;
  EXPECT_EQ(
      ReseedLiquidCoreCell3D(kCell, *restricted, carrier, 8, 8, 30).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
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

constexpr AdaptiveMacCellKey3D kAdaptiveBandLeaf{.lower_corner = {-2, 0, 0}, .width_exponent = 1};
constexpr LiquidStateCapacities3D kAdaptiveBandCapacities{9, 0, 2, 11, 0};
constexpr auto kFullOccupancy = LiquidHandoffOccupancy3D::kFullLiquid;

absl::StatusOr<LiquidState3D> AdaptiveBandState() {
  LiquidState3D state;
  state.geometry = {.world_origin = {0, 4, -1}, .cell_width = 1};
  state.clock = {.time = 1, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1};
  state.random_seed = 71;
  state.next_particle_id = 10;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(3);
  if (!particles.ok()) {
    return particles.status();
  }
  state.particles = std::move(*particles);
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(2);
  if (!cells.ok()) {
    return cells.status();
  }
  state.adaptive_cells = std::move(*cells);
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(11);
  if (!faces.ok()) {
    return faces.status();
  }
  state.adaptive_faces = std::move(*faces);
  (void)state.particles.Resize(3);
  (void)state.adaptive_cells.Resize(2);
  (void)state.adaptive_faces.Resize(11);
  state.particles[0] = kBand[0];
  state.particles[1] = {.id = 5, .tag = 33, .world_position = {0.5, 4.5, -0.5}};
  state.particles[2] = kBand[1];
  state.adaptive_cells[0] = {
      .key = kAdaptiveBandLeaf,
      .inventory = kCore,
      .previous_pressure = 4,
  };
  state.adaptive_cells[1] = {
      .key = {.lower_corner = {0, 0, 0}, .width_exponent = 1},
      .inventory = {.mass_kg = 1, .reference_volume_m3 = 1, .current_volume_m3 = 1},
      .previous_pressure = 7,
  };
  state.adaptive_faces[0] = {.carrier_velocity = -0.25, .validity = LiquidFaceValidity3D::kValid};
  return state;
}

TEST(AdaptiveLiquidBandHandoff3D, ConservesInventoryAndPreservesUnrelatedRecordsAcrossFreshCheckpoint) {
  const auto initial = AdaptiveBandState();
  ASSERT_TRUE(initial.ok()) << initial.status();
  const auto before = EncodeLiquidState3D(*initial);
  ASSERT_TRUE(before.ok()) << before.status();
  const auto core = PrepareAdaptiveLiquidBandRetirement3D(
      *initial,
      kAdaptiveBandCapacities,
      kAdaptiveBandLeaf,
      kFullOccupancy,
      kCarrier
  );
  ASSERT_TRUE(core.ok()) << core.status();
  EXPECT_EQ(core->clock, initial->clock);
  EXPECT_EQ(core->random_seed, initial->random_seed);
  EXPECT_EQ(core->next_particle_id, initial->next_particle_id);
  ASSERT_EQ(core->particles.size(), 1);
  EXPECT_EQ(core->particles[0], initial->particles[1]);
  EXPECT_EQ(core->adaptive_cells[0].material_owner, AdaptiveLiquidMaterialOwner3D::kEulerian);
  EXPECT_EQ(core->adaptive_cells[0].previous_pressure, initial->adaptive_cells[0].previous_pressure);
  ExpectInventory(core->adaptive_cells[0].inventory, kCore);
  EXPECT_EQ(core->adaptive_cells[1], initial->adaptive_cells[1]);
  EXPECT_EQ(core->adaptive_faces, initial->adaptive_faces);

  LiquidResumeMetadata3D resume{};
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  resume.capacities = kAdaptiveBandCapacities;
  const auto checkpoint = EncodeLiquidCheckpoint3D(*core, resume);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeLiquidCheckpoint3D(checkpoint->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(restored->state, *core);
  EXPECT_EQ(restored->resume, resume);
  const auto repeated_checkpoint = EncodeLiquidCheckpoint3D(restored->state, restored->resume);
  ASSERT_TRUE(repeated_checkpoint.ok()) << repeated_checkpoint.status();
  EXPECT_EQ(*repeated_checkpoint, *checkpoint);

  const auto fresh_band =
      PrepareAdaptiveLiquidCoreReseed3D(*core, kAdaptiveBandCapacities, kAdaptiveBandLeaf, kFullOccupancy, kCarrier, 8);
  ASSERT_TRUE(fresh_band.ok()) << fresh_band.status();
  const auto resumed_band = PrepareAdaptiveLiquidCoreReseed3D(
      restored->state,
      kAdaptiveBandCapacities,
      kAdaptiveBandLeaf,
      kFullOccupancy,
      kCarrier,
      8
  );
  ASSERT_TRUE(resumed_band.ok()) << resumed_band.status();
  EXPECT_EQ(*fresh_band, *resumed_band);
  EXPECT_EQ(fresh_band->clock, initial->clock);
  EXPECT_EQ(fresh_band->next_particle_id, 18);
  ASSERT_EQ(fresh_band->particles.size(), 9);
  EXPECT_EQ(fresh_band->particles[0], initial->particles[1]);
  for (std::size_t ordinal = 1; ordinal < fresh_band->particles.size(); ++ordinal) {
    EXPECT_EQ(fresh_band->particles[ordinal].id, 9 + ordinal);
    EXPECT_FALSE(fresh_band->particles[ordinal].pic_initialized);
  }
  EXPECT_EQ(fresh_band->adaptive_cells, initial->adaptive_cells);
  EXPECT_EQ(fresh_band->adaptive_faces, initial->adaptive_faces);

  const auto fresh_bytes = EncodeLiquidState3D(*fresh_band);
  ASSERT_TRUE(fresh_bytes.ok()) << fresh_bytes.status();
  const auto resumed_bytes = EncodeLiquidState3D(*resumed_band);
  ASSERT_TRUE(resumed_bytes.ok()) << resumed_bytes.status();
  EXPECT_EQ(*fresh_bytes, *resumed_bytes);
  const auto unchanged = EncodeLiquidState3D(*initial);
  ASSERT_TRUE(unchanged.ok()) << unchanged.status();
  EXPECT_EQ(*before, *unchanged);
}

TEST(AdaptiveLiquidBandHandoff3D, RejectsUncertifiedOrIncompatibleCandidatesWithoutChangingInput) {
  auto initial = AdaptiveBandState();
  ASSERT_TRUE(initial.ok()) << initial.status();
  const auto before = EncodeLiquidState3D(*initial);
  ASSERT_TRUE(before.ok()) << before.status();
  EXPECT_EQ(
      PrepareAdaptiveLiquidBandRetirement3D(
          *initial,
          kAdaptiveBandCapacities,
          kAdaptiveBandLeaf,
          kFullOccupancy,
          std::nullopt
      )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(
      PrepareAdaptiveLiquidBandRetirement3D(
          *initial,
          kAdaptiveBandCapacities,
          kAdaptiveBandLeaf,
          LiquidHandoffOccupancy3D::kPartial,
          kCarrier
      )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto absent_leaf = kAdaptiveBandLeaf;
  absent_leaf.width_exponent = 0;
  EXPECT_EQ(
      PrepareAdaptiveLiquidBandRetirement3D(*initial, kAdaptiveBandCapacities, absent_leaf, kFullOccupancy, kCarrier)
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto insufficient = kAdaptiveBandCapacities;
  insufficient.faces = 10;
  EXPECT_EQ(
      PrepareAdaptiveLiquidBandRetirement3D(*initial, insufficient, kAdaptiveBandLeaf, kFullOccupancy, kCarrier)
          .status()
          .code(),
      absl::StatusCode::kResourceExhausted
  );

  auto incompatible = CloneLiquidState3D(*initial, kAdaptiveBandCapacities);
  ASSERT_TRUE(incompatible.ok()) << incompatible.status();
  incompatible->adaptive_cells[0].inventory.mass_kg = std::nextafter(kCore.mass_kg, 0.0);
  EXPECT_EQ(
      PrepareAdaptiveLiquidBandRetirement3D(
          *incompatible,
          kAdaptiveBandCapacities,
          kAdaptiveBandLeaf,
          kFullOccupancy,
          kCarrier
      )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  incompatible->adaptive_cells[0].inventory = kCore;
  incompatible->adaptive_cells[0].inventory.current_volume_m3 = 7;
  EXPECT_EQ(
      PrepareAdaptiveLiquidBandRetirement3D(
          *incompatible,
          kAdaptiveBandCapacities,
          kAdaptiveBandLeaf,
          kFullOccupancy,
          kCarrier
      )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  incompatible->adaptive_cells[0].inventory = kCore;
  incompatible->adaptive_cells[1].material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
  EXPECT_EQ(
      PrepareAdaptiveLiquidBandRetirement3D(
          *incompatible,
          kAdaptiveBandCapacities,
          kAdaptiveBandLeaf,
          kFullOccupancy,
          kCarrier
      )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  const auto unchanged = EncodeLiquidState3D(*initial);
  ASSERT_TRUE(unchanged.ok()) << unchanged.status();
  EXPECT_EQ(*before, *unchanged);

  const auto core = PrepareAdaptiveLiquidBandRetirement3D(
      *initial,
      kAdaptiveBandCapacities,
      kAdaptiveBandLeaf,
      kFullOccupancy,
      kCarrier
  );
  ASSERT_TRUE(core.ok()) << core.status();
  const auto core_bytes = EncodeLiquidState3D(*core);
  ASSERT_TRUE(core_bytes.ok()) << core_bytes.status();
  insufficient = kAdaptiveBandCapacities;
  insufficient.particles = 8;
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreReseed3D(*core, insufficient, kAdaptiveBandLeaf, kFullOccupancy, kCarrier, 8)
          .status()
          .code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoreReseed3D(
          *core,
          kAdaptiveBandCapacities,
          kAdaptiveBandLeaf,
          kFullOccupancy,
          std::nullopt,
          8
      )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  const auto unchanged_core = EncodeLiquidState3D(*core);
  ASSERT_TRUE(unchanged_core.ok()) << unchanged_core.status();
  EXPECT_EQ(*core_bytes, *unchanged_core);
}

}  // namespace
}  // namespace fluidity
