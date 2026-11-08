#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "fluidity/host/liquid_step_workspace_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
constexpr UniformMacGeometry3D kGeometry{{0.0, 0.0, 0.0}, 1.0};
constexpr LiquidStateCapacities3D kCapacity{
    .particles = 16,
    .blocks = 32,
    .cells = 16,
    .faces = 32 * 192,
    .sources = 0,
};
LiquidParticleState3D Particle(Vec3 position, std::uint64_t id = 1) {
  LiquidParticleState3D particle;
  particle.world_position = position;
  particle.id = id;
  return particle;
}
template <typename Index>
auto Key(Index index) {
  return std::tuple(index.z, index.y, index.x);
}
auto FaceKey(FaceIndex3D face) {
  return std::tuple(face.axis, Key(face.lattice_index));
}
LiquidState3D EmptyState(LiquidStateCapacities3D capacity = kCapacity) {
  LiquidState3D empty;
  auto state = CloneLiquidState3D(empty, capacity);
  EXPECT_TRUE(state.ok());
  return std::move(*state);
}
void ExpectBlockKeys(std::span<const BlockIndex3D> actual, std::span<const BlockIndex3D> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_EQ(Key(actual[index]), Key(expected[index]));
  }
}

TEST(LiquidStepWorkspace3D, NegativeBinsAndEqualBinsPreserveIncomingOrder) {
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const std::array particles{
      Particle({2.0, 2.0, 2.0}, 8),
      Particle({-2.0, 2.0, 2.0}, 7),
      Particle({2.1, 2.0, 2.0}, 6),
      Particle({2.0, -2.0, 2.0}, 5),
      Particle({2.0, 2.0, -2.0}, 4),
      Particle({-2.1, 2.0, 2.0}, 3),
  };
  ASSERT_TRUE(workspace->PrepareTrialTopology(particles, {}).ok());
  auto view = workspace->prepared();
  ASSERT_TRUE(view.ok());
  constexpr std::array<std::size_t, 6> order{4, 3, 1, 5, 0, 2};
  for (std::size_t index = 0; index < order.size(); ++index) {
    EXPECT_EQ(view->particle_order[index].original_ordinal, order[index]);
    EXPECT_EQ(view->particles[index], particles[order[index]]);
  }
  EXPECT_EQ(Key(view->particle_order[0].bin_index), std::make_tuple(-1, 0, 0));
  EXPECT_EQ(Key(view->particle_order[2].bin_index), std::make_tuple(0, 0, -1));
}

TEST(LiquidStepWorkspace3D, ActivatesZeroWeightOriginalNodesRegardlessOfTag) {
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  auto particle = Particle({3.0, 3.0, 3.0});
  particle.tag = std::numeric_limits<std::uint32_t>::max();
  const std::array particles{particle};
  ASSERT_TRUE(workspace->PrepareTrialTopology(particles, {}).ok());
  auto view = workspace->prepared();
  ASSERT_TRUE(view.ok());
  // x support is [2,4]x[2,4]x[2,4]. Its transverse index 4 has zero
  // weight at X=3, but all eight owners must still be activated.
  constexpr std::array blocks{
      BlockIndex3D{0, 0, 0},
      BlockIndex3D{1, 0, 0},
      BlockIndex3D{0, 1, 0},
      BlockIndex3D{1, 1, 0},
      BlockIndex3D{0, 0, 1},
      BlockIndex3D{1, 0, 1},
      BlockIndex3D{0, 1, 1},
      BlockIndex3D{1, 1, 1},
  };
  ExpectBlockKeys(view->grid.active_block_indices(), blocks);
  EXPECT_EQ(view->faces.size(), 8 * 192);
  EXPECT_EQ(workspace->usage().activation_keys, 81);
  for (FaceNormalAxis axis : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
    for (int z = 2; z <= 4; ++z) {
      for (int y = 2; y <= 4; ++y) {
        for (int x = 2; x <= 4; ++x) {
          EXPECT_TRUE(view->grid.FaceSlot({axis, {x, y, z}}).ok());
        }
      }
    }
  }
}

TEST(LiquidStepWorkspace3D, ActivatesSixPressureFaceOwnersAndCanonicalUniqueCells) {
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  constexpr std::array cells{CellIndex3D{3, 3, 3}, CellIndex3D{-1, -1, -1}, CellIndex3D{3, 3, 3}};
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, cells).ok());
  auto view = workspace->prepared();
  ASSERT_TRUE(view.ok());
  constexpr std::array blocks{
      BlockIndex3D{-1, -1, -1},
      BlockIndex3D{0, -1, -1},
      BlockIndex3D{-1, 0, -1},
      BlockIndex3D{-1, -1, 0},
      BlockIndex3D{0, 0, 0},
      BlockIndex3D{1, 0, 0},
      BlockIndex3D{0, 1, 0},
      BlockIndex3D{0, 0, 1},
  };
  ExpectBlockKeys(view->grid.active_block_indices(), blocks);
  ASSERT_EQ(view->cells.size(), 2);
  EXPECT_EQ(Key(view->cells[0].cell_index), std::make_tuple(-1, -1, -1));
  EXPECT_EQ(Key(view->cells[1].cell_index), std::make_tuple(3, 3, 3));
  constexpr std::array faces{
      FaceIndex3D{FaceNormalAxis::kX, {-1, -1, -1}},
      FaceIndex3D{FaceNormalAxis::kX, {0, -1, -1}},
      FaceIndex3D{FaceNormalAxis::kY, {-1, -1, -1}},
      FaceIndex3D{FaceNormalAxis::kY, {-1, 0, -1}},
      FaceIndex3D{FaceNormalAxis::kZ, {-1, -1, -1}},
      FaceIndex3D{FaceNormalAxis::kZ, {-1, -1, 0}},
  };
  for (auto face : faces) {
    EXPECT_TRUE(view->grid.FaceSlot(face).ok());
  }
}

TEST(LiquidStepWorkspace3D, RemapsHistoryByIdentityAcrossShiftedBlockOrdinals) {
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  constexpr std::array old_cells{CellIndex3D{1, 1, 1}, CellIndex3D{9, 1, 1}};
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, old_cells).ok());
  auto previous = EmptyState();
  ASSERT_TRUE(workspace->ApplyTrialTopology(previous).ok());
  previous.clock.epoch = 11;
  previous.blocks[0].generation = 4;
  previous.blocks[0].activation_age = 7;
  previous.blocks[0].inactive_age = 2;
  previous.cells[0].previous_pressure = 123.5;
  previous.cells[0].occupancy = LiquidCellOccupancy3D::kSolid;
  for (auto& face : previous.faces.values()) {
    face.carrier_velocity = 10.0 + static_cast<double>(face.face_slot);
    face.validity = LiquidFaceValidity3D::kLowMass;
    face.side_tag = -17;
    face.structurally_eligible = false;
  }
  auto snapshot = CloneLiquidState3D(previous, kCapacity);
  ASSERT_TRUE(snapshot.ok());
  constexpr std::array new_cells{CellIndex3D{-3, 1, 1}, CellIndex3D{1, 1, 1}};
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, new_cells, &previous).ok());
  auto view = workspace->prepared();
  ASSERT_TRUE(view.ok());
  ASSERT_EQ(view->blocks.size(), 2);
  EXPECT_EQ(Key(view->blocks[0].block_index), std::make_tuple(0, 0, -1));
  EXPECT_EQ(view->blocks[0].generation, 12);
  EXPECT_EQ(view->blocks[0].activation_age, 0);
  EXPECT_EQ(view->blocks[1].generation, 4);
  EXPECT_EQ(view->blocks[1].activation_age, 7);
  EXPECT_EQ(view->blocks[1].inactive_age, 0);
  EXPECT_EQ(view->cells[0].previous_pressure, 0);
  EXPECT_EQ(view->cells[0].occupancy, LiquidCellOccupancy3D::kLiquid);
  EXPECT_EQ(view->cells[1].previous_pressure, 123.5);
  EXPECT_EQ(view->cells[1].occupancy, LiquidCellOccupancy3D::kSolid);
  for (std::size_t index = 0; index < 192; ++index) {
    EXPECT_EQ(view->faces[index].carrier_velocity, 0);
    EXPECT_EQ(view->faces[index].validity, LiquidFaceValidity3D::kMissing);
    auto expected = previous.faces[index];
    expected.face_slot += 192;
    EXPECT_EQ(view->faces[index + 192], expected);
    auto identity = view->grid.FaceIndex(index + 192);
    ASSERT_TRUE(identity.ok());
    EXPECT_EQ(FaceKey(*identity), FaceKey(expected.face_index));
  }
  EXPECT_EQ(previous, *snapshot);
  auto trial = EmptyState();
  ASSERT_TRUE(workspace->ApplyTrialTopology(trial).ok());
  EXPECT_TRUE(ValidateLiquidState3D(trial).ok());
  EXPECT_EQ(trial.clock.epoch, 0);  // Applying topology cannot publish a clock.
}

TEST(LiquidStepWorkspace3D, CapacityFailuresPreservePublishedValuesAndAddresses) {
  auto capacity = kCapacity;
  capacity.particles = 1;
  capacity.blocks = 1;
  capacity.cells = 1;
  capacity.faces = 192;
  auto workspace = LiquidStepWorkspace3D::Create(capacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  constexpr std::array initial_cells{CellIndex3D{1, 1, 1}};
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, initial_cells).ok());
  const auto before = *workspace->prepared();
  const std::array corner{Particle({3, 3, 3})};
  EXPECT_EQ(workspace->PrepareTrialTopology(corner, {}).code(), absl::StatusCode::kResourceExhausted);
  const std::array too_many{Particle({2, 2, 2}), Particle({2, 2, 2}, 2)};
  EXPECT_EQ(workspace->PrepareTrialTopology(too_many, {}).code(), absl::StatusCode::kResourceExhausted);
  constexpr std::array too_many_cells{CellIndex3D{1, 1, 1}, CellIndex3D{2, 1, 1}};
  EXPECT_EQ(workspace->PrepareTrialTopology({}, too_many_cells).code(), absl::StatusCode::kResourceExhausted);
  auto after = *workspace->prepared();
  EXPECT_EQ(after.blocks.data(), before.blocks.data());
  EXPECT_EQ(after.faces.data(), before.faces.data());
  EXPECT_EQ(after.cells.data(), before.cells.data());
  EXPECT_EQ(after.cells[0].previous_pressure, 0);
  EXPECT_EQ(Key(after.cells[0].cell_index), std::make_tuple(1, 1, 1));
  auto undersized = EmptyState({});
  auto saved = CloneLiquidState3D(undersized, {});
  ASSERT_TRUE(saved.ok());
  EXPECT_EQ(workspace->ApplyTrialTopology(undersized).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(undersized, *saved);
}

TEST(LiquidStepWorkspace3D, FaceCapacityIsCheckedBeforePublish) {
  auto capacity = kCapacity;
  capacity.faces = 191;
  auto workspace = LiquidStepWorkspace3D::Create(capacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  constexpr std::array cells{CellIndex3D{1, 1, 1}};
  EXPECT_EQ(workspace->PrepareTrialTopology({}, cells).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(workspace->prepared().status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(LiquidStepWorkspace3D, RejectsLatticeAndGenerationOverflowAndInvalidInputTransactionally) {
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, {}).ok());
  const auto before = *workspace->prepared();
  constexpr std::array cells{CellIndex3D{std::numeric_limits<std::int32_t>::max(), 0, 0}};
  EXPECT_EQ(workspace->PrepareTrialTopology({}, cells).code(), absl::StatusCode::kOutOfRange);
  const std::array outside{Particle({2147483647.0, 0, 0})};
  EXPECT_EQ(workspace->PrepareTrialTopology(outside, {}).code(), absl::StatusCode::kOutOfRange);
  const std::array invalid{Particle({std::numeric_limits<double>::quiet_NaN(), 0, 0})};
  EXPECT_EQ(workspace->PrepareTrialTopology(invalid, {}).code(), absl::StatusCode::kInvalidArgument);
  auto history = EmptyState();
  history.clock.epoch = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(workspace->PrepareTrialTopology({}, {}, &history).code(), absl::StatusCode::kOutOfRange);
  history.clock.epoch = 0;
  history.geometry.cell_width = 2;
  EXPECT_EQ(workspace->PrepareTrialTopology({}, {}, &history).code(), absl::StatusCode::kInvalidArgument);
  const auto after = *workspace->prepared();
  EXPECT_EQ(after.blocks.data(), before.blocks.data());
  EXPECT_TRUE(after.faces.empty());
}

TEST(LiquidStepWorkspace3D, AccountsEveryReservedRegionAndRejectsByteOverflow) {
  const auto bytes = QueryLiquidWorkspaceBytes3D(kCapacity);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ(bytes->activation_key_capacity, 81 * 16 + 7 * 16);
  EXPECT_EQ(bytes->particle_order_bytes, 2 * 16 * sizeof(LiquidParticleSortEntry3D));
  EXPECT_EQ(bytes->particle_state_bytes, 2 * 16 * sizeof(LiquidParticleState3D));
  EXPECT_EQ(bytes->block_key_bytes, 2 * 32 * sizeof(BlockIndex3D));
  EXPECT_EQ(bytes->block_state_bytes, 2 * 32 * sizeof(LiquidBlockState3D));
  EXPECT_EQ(bytes->cell_state_bytes, 2 * 16 * sizeof(LiquidCellState3D));
  EXPECT_EQ(bytes->face_state_bytes, 2 * 32 * 192 * sizeof(LiquidFaceState3D));
  EXPECT_EQ(bytes->activation_key_bytes, (81 * 16 + 7 * 16) * sizeof(BlockIndex3D));
  EXPECT_EQ(
      bytes->total_bytes,
      bytes->particle_order_bytes + bytes->particle_state_bytes + bytes->block_key_bytes + bytes->block_state_bytes
          + bytes->cell_state_bytes + bytes->face_state_bytes + bytes->activation_key_bytes + bytes->boundary_bytes
  );
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok());
  EXPECT_EQ(workspace->usage().used_bytes, 0);
  EXPECT_EQ(workspace->usage().reserved_bytes, bytes->total_bytes);
  constexpr std::array cells{CellIndex3D{1, 1, 1}};
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, cells).ok());
  const auto usage = workspace->usage();
  EXPECT_EQ(usage.active.blocks, 1);
  EXPECT_EQ(usage.active.cells, 1);
  EXPECT_EQ(usage.active.faces, 192);
  EXPECT_EQ(usage.activation_keys, 7);
  EXPECT_EQ(
      usage.used_bytes,
      8 * sizeof(BlockIndex3D) + sizeof(LiquidBlockState3D) + sizeof(LiquidCellState3D)
          + 192 * sizeof(LiquidFaceState3D)
  );
  EXPECT_LE(usage.used_bytes, usage.reserved_bytes);
  for (int region = 0; region < 4; ++region) {
    auto huge = kCapacity;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (region == 0) {
      huge.particles = maximum;
    }
    if (region == 1) {
      huge.blocks = maximum;
    }
    if (region == 2) {
      huge.cells = maximum;
    }
    if (region == 3) {
      huge.faces = maximum;
    }
    EXPECT_EQ(QueryLiquidWorkspaceBytes3D(huge).status().code(), absl::StatusCode::kResourceExhausted);
    EXPECT_EQ(LiquidStepWorkspace3D::Create(huge, kGeometry).status().code(), absl::StatusCode::kResourceExhausted);
  }
}

TEST(LiquidStepWorkspace3D, EmptyPreparationClearsExtentsAndReactivationInitializesHistory) {
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  constexpr std::array cells{CellIndex3D{1, 1, 1}};
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, cells).ok());
  auto previous = EmptyState();
  ASSERT_TRUE(workspace->ApplyTrialTopology(previous).ok());
  previous.cells[0].previous_pressure = 99;
  previous.faces[0].carrier_velocity = 23;
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, {}, &previous).ok());
  auto empty = workspace->prepared();
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->blocks.empty());
  EXPECT_TRUE(empty->faces.empty());
  EXPECT_EQ(workspace->usage().activation_keys, 0);
  auto empty_state = EmptyState();
  empty_state.clock.epoch = 9;
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, cells, &empty_state).ok());
  const auto restored = *workspace->prepared();
  EXPECT_EQ(restored.blocks[0].generation, 10);
  EXPECT_EQ(restored.cells[0].previous_pressure, 0);
  EXPECT_EQ(restored.faces[0].carrier_velocity, 0);
  EXPECT_EQ(restored.faces[0].validity, LiquidFaceValidity3D::kMissing);
}

TEST(LiquidStepWorkspace3D, CompensatedSortBoundaryRetainsNegativeSideOfTranslatedTie) {
  // The translated subtraction rounds these neighboring coordinates onto X=1.5.
  // The compensated geometry must still distinguish the exact sides.
  const UniformMacGeometry3D geometry{{-1.5, -1.5, -1.5}, 1.0};
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, geometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const std::array particles{Particle({0, 0, 0}, 2), Particle({-0x1p-54, 0, 0}, 1)};
  ASSERT_TRUE(workspace->PrepareTrialTopology(particles, {}).ok());
  const auto view = *workspace->prepared();
  EXPECT_EQ(view.particle_order[0].original_ordinal, 1);
  EXPECT_EQ(view.particle_order[0].bin_index.x, -1);
  EXPECT_EQ(view.particle_order[1].bin_index.x, 0);
}

TEST(LiquidStepWorkspace3D, BorrowedGridChecksCanonicalKeysAndSignedSlotInverse) {
  constexpr std::array unordered{BlockIndex3D{0, 0, 0}, BlockIndex3D{-1, 0, 0}};
  EXPECT_EQ(SparseMacGridView3D::Create(kGeometry, unordered).status().code(), absl::StatusCode::kInvalidArgument);
  constexpr std::array duplicates{BlockIndex3D{0, 0, 0}, BlockIndex3D{0, 0, 0}};
  EXPECT_EQ(SparseMacGridView3D::Create(kGeometry, duplicates).status().code(), absl::StatusCode::kInvalidArgument);
  constexpr std::array outside{BlockIndex3D{536870912, 0, 0}};
  EXPECT_EQ(SparseMacGridView3D::Create(kGeometry, outside).status().code(), absl::StatusCode::kOutOfRange);
  constexpr std::array blocks{BlockIndex3D{-1, -1, -1}, BlockIndex3D{0, -1, -1}};
  const auto grid = SparseMacGridView3D::Create(kGeometry, blocks);
  ASSERT_TRUE(grid.ok());
  EXPECT_EQ(grid->active_block_indices().data(), blocks.data());
  EXPECT_EQ(grid->FaceIndex(384).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(
      grid->FaceSlot({static_cast<FaceNormalAxis>(3), {0, 0, 0}}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(grid->FaceSlot({FaceNormalAxis::kX, {0, 0, 0}}).status().code(), absl::StatusCode::kFailedPrecondition);
  const auto face = grid->FaceIndex(191);
  ASSERT_TRUE(face.ok());
  EXPECT_EQ(FaceKey(*face), FaceKey({FaceNormalAxis::kZ, {-1, -1, -1}}));
  const auto next_face = grid->FaceIndex(192);
  ASSERT_TRUE(next_face.ok());
  EXPECT_EQ(FaceKey(*next_face), FaceKey({FaceNormalAxis::kX, {0, -4, -4}}));
  const auto high_face = grid->FaceIndex(383);
  ASSERT_TRUE(high_face.ok());
  EXPECT_EQ(FaceKey(*high_face), FaceKey({FaceNormalAxis::kZ, {3, -1, -1}}));
  for (std::size_t slot = 0; slot < 384; ++slot) {
    const auto identity = grid->FaceIndex(slot);
    ASSERT_TRUE(identity.ok());
    EXPECT_EQ(*grid->FaceSlot(*identity), slot);
  }
}

TEST(LiquidStepWorkspace3D, UnownedGeometryIncludesExactZeroEdgeAndSharedOwnedNumerics) {
  const auto geometry = BuildMacFaceGeometryStencil3D(kGeometry, FaceNormalAxis::kX, {3, 3, 3}, {3, 3, 3});
  ASSERT_TRUE(geometry.ok());
  EXPECT_EQ(FaceKey((*geometry)[0].face_index), FaceKey({FaceNormalAxis::kX, {2, 2, 2}}));
  EXPECT_EQ(FaceKey((*geometry)[26].face_index), FaceKey({FaceNormalAxis::kX, {4, 4, 4}}));
  EXPECT_EQ((*geometry)[26].weight, 0);
  EXPECT_EQ((*geometry)[13].weight, 0.1875);
  EXPECT_EQ((*geometry)[13].weight_gradient.x, 0);
  EXPECT_EQ((*geometry)[13].weight_gradient.y, 0.375);
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok());
  const std::array particles{Particle({3, 3, 3})};
  ASSERT_TRUE(workspace->PrepareTrialTopology(particles, {}).ok());
  const auto view = *workspace->prepared();
  const auto borrowed = BuildOwnedMacFaceStencil3D(view.grid, FaceNormalAxis::kX, {3, 3, 3}, {3, 3, 3});
  ASSERT_TRUE(borrowed.ok());
  EXPECT_EQ((*borrowed)[26].weight, 0);
  EXPECT_EQ((*borrowed)[13].weight, 0.1875);
  EXPECT_EQ((*borrowed)[13].weight_gradient.y, 0.375);
  EXPECT_EQ((*borrowed)[13].face_slot, 63);
}

TEST(LiquidStepWorkspace3D, EmptyCapacityAndBorrowedParticleReprepareAndMove) {
  auto empty = LiquidStepWorkspace3D::Create({}, kGeometry);
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty->prepared().status().code(), absl::StatusCode::kFailedPrecondition);
  auto state = EmptyState({});
  EXPECT_EQ(empty->ApplyTrialTopology(state).code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(empty->PrepareTrialTopology({}, {}).ok());
  EXPECT_TRUE(empty->prepared()->faces.empty());
  EXPECT_EQ(empty->usage().used_bytes, 0);
  EXPECT_EQ(empty->usage().reserved_bytes, 0);
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok());
  const std::array particles{Particle({2, 2, 2}, 2), Particle({-2, 2, 2}, 1)};
  ASSERT_TRUE(workspace->PrepareTrialTopology(particles, {}).ok());
  const auto first = *workspace->prepared();
  ASSERT_TRUE(workspace->PrepareTrialTopology(first.particles, {}).ok());
  auto moved = std::move(*workspace);
  const auto second = *moved.prepared();
  EXPECT_EQ(second.particles[0].id, 1);
  EXPECT_EQ(second.particles[1].id, 2);
  EXPECT_EQ(second.particle_order[0].original_ordinal, 0);
  EXPECT_EQ(second.particle_order[1].original_ordinal, 1);
}

TEST(LiquidStepWorkspace3D, InvalidGeometryAndHistoryLeaveTrialAndPublishedTopologyUnchanged) {
  auto invalid_geometry = kGeometry;
  invalid_geometry.cell_width = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(
      LiquidStepWorkspace3D::Create(kCapacity, invalid_geometry).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok());
  constexpr std::array cells{CellIndex3D{1, 1, 1}};
  ASSERT_TRUE(workspace->PrepareTrialTopology({}, cells).ok());
  const auto first = *workspace->prepared();
  auto history = EmptyState();
  ASSERT_TRUE(workspace->ApplyTrialTopology(history).ok());
  history.faces[0].carrier_velocity = std::numeric_limits<double>::infinity();
  EXPECT_EQ(workspace->PrepareTrialTopology({}, cells, &history).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(workspace->prepared()->blocks.data(), first.blocks.data());
  auto trial = EmptyState();
  trial.geometry.cell_width = 2;
  auto saved = CloneLiquidState3D(trial, kCapacity);
  ASSERT_TRUE(saved.ok());
  EXPECT_EQ(workspace->ApplyTrialTopology(trial).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(trial, *saved);
}

#if defined(_WIN32) && defined(_DEBUG)
std::size_t allocation_count = 0;
std::size_t deallocation_count = 0;
std::size_t fail_allocation = 0;
_CRT_ALLOC_HOOK previous_hook = nullptr;
int AllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int block_type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++allocation_count;
    if (allocation_count == fail_allocation) {
      return 0;
    }
  }
  if (operation == _HOOK_FREE) {
    ++deallocation_count;
  }
  return previous_hook == nullptr ? 1 : previous_hook(operation, data, size, block_type, request, filename, line);
}
TEST(LiquidStepWorkspace3D, SuccessfulPreparationAndApplyHaveNoHeapActivityIncludingDestruction) {
  auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  auto state = EmptyState();
  state.next_particle_id = 3;
  const std::array particles{Particle({3, 3, 3}, 2), Particle({-1, -1, -1}, 1)};
  constexpr std::array cells{CellIndex3D{3, 3, 3}};
  bool prepared_ok = false;
  bool applied_ok = false;
  allocation_count = 0;
  deallocation_count = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    const auto status = workspace->PrepareTrialTopology(particles, cells, &state);
    prepared_ok = status.ok();
    const auto applied = workspace->ApplyTrialTopology(state);
    applied_ok = applied.ok();
    const auto view = workspace->prepared();
    prepared_ok = prepared_ok && view.ok();
  }
  _CrtSetAllocHook(previous_hook);
  EXPECT_TRUE(prepared_ok);
  EXPECT_TRUE(applied_ok);
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(deallocation_count, 0);
  EXPECT_TRUE(ValidateLiquidState3D(state).ok());
}
TEST(LiquidStepWorkspace3D, RejectedPreparationsAndResultDestructionHaveNoHeapActivity) {
  auto capacity = kCapacity;
  capacity.blocks = 1;
  auto workspace = LiquidStepWorkspace3D::Create(capacity, kGeometry);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const std::array corner{Particle({3, 3, 3})};
  const std::array invalid{Particle({std::numeric_limits<double>::infinity(), 0, 0})};
  constexpr std::array overflow{CellIndex3D{0, std::numeric_limits<std::int32_t>::max(), 0}};
  std::array<absl::StatusCode, 3> codes{};
  allocation_count = 0;
  deallocation_count = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    auto status = workspace->PrepareTrialTopology(corner, {});
    codes[0] = status.code();
    status = workspace->PrepareTrialTopology(invalid, {});
    codes[1] = status.code();
    status = workspace->PrepareTrialTopology({}, overflow);
    codes[2] = status.code();
  }
  _CrtSetAllocHook(previous_hook);
  EXPECT_EQ(codes[0], absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(codes[1], absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(codes[2], absl::StatusCode::kOutOfRange);
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(deallocation_count, 0);
}
TEST(LiquidStepWorkspace3D, ActualPartialAllocationFailureMakesNoSecondaryErrorAllocation) {
  absl::StatusCode code = absl::StatusCode::kOk;
  allocation_count = 0;
  deallocation_count = 0;
  fail_allocation = 4;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    auto workspace = LiquidStepWorkspace3D::Create(kCapacity, kGeometry);
    code = workspace.status().code();
  }
  _CrtSetAllocHook(previous_hook);
  fail_allocation = 0;
  EXPECT_EQ(code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(allocation_count, 4);
  EXPECT_EQ(deallocation_count, 3);
}
#endif
}  // namespace
}  // namespace fluidity
