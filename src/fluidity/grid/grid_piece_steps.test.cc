#include "fluidity/grid/grid_piece_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// The worked example of the header: the whole grid of blocks 0..9, the range {4, 3} and the border {1, 2, 8}.
constexpr std::array<CellBlockIndex, 3> kBorderBlocks = {{{.value = 1}, {.value = 2}, {.value = 8}}};
constexpr CellBlockRange kRange = {.first_block = 4, .block_count = 3};

TEST(ToPiecePlace, RangeBetweenBorderEntries_GivesAscendingMap) {
  const std::array<std::uint32_t, 6> expected_places = {0, 1, 5, 2, 3, 4};
  std::array<std::uint32_t, 6> map = {};
  for (std::uint32_t loaded_item = 0; loaded_item < 6; ++loaded_item) {
    const CellBlockIndex place = ToPiecePlace(kBorderBlocks.data(), 3, kRange, loaded_item);
    EXPECT_EQ(place.value, expected_places.at(loaded_item)) << "loaded item " << loaded_item;
    const std::uint32_t whole_block = loaded_item < 3 ? kBorderBlocks.at(loaded_item).value : 4 + (loaded_item - 3);
    map.at(place.value) = whole_block;
  }
  EXPECT_EQ(map, (std::array<std::uint32_t, 6>{1, 2, 4, 5, 6, 8}));
}

TEST(ToPiecePlace, NoBorder_GivesTheRangeFromPlaceZero) {
  EXPECT_EQ(ToPiecePlace(nullptr, 0, kRange, 0).value, 0U);
  EXPECT_EQ(ToPiecePlace(nullptr, 0, kRange, 2).value, 2U);
}

TEST(ToPieceNeighbour, BlockNotLoaded_IsNoIndex) {
  const std::array<CellBlockIndex, 6> map = {{{1}, {2}, {4}, {5}, {6}, {8}}};
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 7), kNoIndex);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 0), kNoIndex);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 9), kNoIndex);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, kNoIndex), kNoIndex);
}

TEST(ToPieceNeighbour, BlockLoaded_IsItsPlaceInTheMap) {
  const std::array<CellBlockIndex, 6> map = {{{1}, {2}, {4}, {5}, {6}, {8}}};
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 1), 0U);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 5), 3U);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 8), 5U);
}

// The tables of the grid of BuildPlaneFourCells (3 blocks), which the steps below read.
struct PlaneFourGrid {
  std::expected<TestGrid, GridFailure> test_grid;
  GridTables tables;
};

PlaneFourGrid BuildPlaneFourGrid() {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  EXPECT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  return {.test_grid = std::move(test_grid), .tables = tables};
}

TEST(ComputePieceBorderFailureKey, EntryOutsideRangeAndAscending_IsNoFailure) {
  const PieceLoad load = {.range = kRange, .border_blocks = kBorderBlocks.data(), .border_block_count = 3};
  for (std::uint32_t entry = 0; entry < 3; ++entry) {
    EXPECT_EQ(ComputePieceBorderFailureKey(load, 10, entry), kNoFailureKey) << entry;
  }
}

TEST(ComputePieceBorderFailureKey, EntryInRangeNotAscendingOrBeyondGrid_IsTheKeyOfStepOne) {
  const std::array<CellBlockIndex, 3> in_range = {{{.value = 1}, {.value = 2}, {.value = 5}}};
  const PieceLoad range_load = {.range = kRange, .border_blocks = in_range.data(), .border_block_count = 3};
  EXPECT_EQ(ComputePieceBorderFailureKey(range_load, 10, 2), ToFailureKey(kPieceBorderStep, 2, 0));
  const std::array<CellBlockIndex, 3> unsorted = {{{.value = 1}, {.value = 8}, {.value = 2}}};
  const PieceLoad unsorted_load = {.range = kRange, .border_blocks = unsorted.data(), .border_block_count = 3};
  EXPECT_EQ(ComputePieceBorderFailureKey(unsorted_load, 10, 2), ToFailureKey(kPieceBorderStep, 2, 0));
  EXPECT_EQ(ComputePieceBorderFailureKey(unsorted_load, 10, 1), kNoFailureKey);
  const std::array<CellBlockIndex, 2> beyond = {{{.value = 1}, {.value = 10}}};
  const PieceLoad beyond_load = {.range = kRange, .border_blocks = beyond.data(), .border_block_count = 2};
  EXPECT_EQ(ComputePieceBorderFailureKey(beyond_load, 10, 1), ToFailureKey(kPieceBorderStep, 1, 0));
}

TEST(ComputeIsBorderEntryBelowRange, EntriesAroundTheRange_CountThoseBelow) {
  const PieceLoad load = {.range = kRange, .border_blocks = kBorderBlocks.data(), .border_block_count = 3};
  EXPECT_EQ(ComputeIsBorderEntryBelowRange(load, 0), 1U);
  EXPECT_EQ(ComputeIsBorderEntryBelowRange(load, 1), 1U);
  EXPECT_EQ(ComputeIsBorderEntryBelowRange(load, 2), 0U);
}

TEST(ApplyPieceMapWriteAt, EveryLoadedItem_WritesTheAscendingMap) {
  const PieceLoad load = {.range = kRange, .border_blocks = kBorderBlocks.data(), .border_block_count = 3};
  std::array<CellBlockIndex, 6> map = {};
  for (std::uint32_t loaded_item = 5; loaded_item < 6; --loaded_item) {
    ApplyPieceMapWriteAt(load, loaded_item, map.data());
  }
  const std::array<std::uint32_t, 6> expected_map = {1, 2, 4, 5, 6, 8};
  for (std::size_t place = 0; place < map.size(); ++place) {
    EXPECT_EQ(map.at(place).value, expected_map.at(place)) << place;
  }
}

// The piece map that holds every block but `missing_block` of the grid of `block_count` blocks.
std::vector<CellBlockIndex> BuildMapWithout(std::uint32_t block_count, CellBlockIndex missing_block) {
  std::vector<CellBlockIndex> map;
  for (std::uint32_t block = 0; block < block_count; ++block) {
    if (block != missing_block.value) {
      map.push_back({.value = block});
    }
  }
  return map;
}

// The arrays one piece block's steps write, for `block_count` blocks.
struct PieceArraysOwner {
  explicit PieceArraysOwner(std::size_t block_count)
      : block_ids(block_count),
        same_size_level_blocks(block_count * 27),
        bigger_block(block_count),
        smaller_blocks(block_count * 8),
        cell_states(block_count * kCellPlacesPerBlock),
        face_piece_kinds(block_count * kFacePlacesPerBlock) {
  }

  [[nodiscard]] PieceTableArrays ToArrays() {
    return {
        .block_ids = block_ids.data(),
        .same_size_level_blocks = same_size_level_blocks.data(),
        .bigger_block = bigger_block.data(),
        .smaller_blocks = smaller_blocks.data(),
        .cell_states = cell_states.data(),
        .face_piece_kinds = face_piece_kinds.data(),
    };
  }

  std::vector<CellBlockId> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks;
  std::vector<std::uint32_t> bigger_block;
  std::vector<std::uint32_t> smaller_blocks;
  std::vector<CellState> cell_states;
  std::vector<FacePieceKind> face_piece_kinds;
};

// Checks that the ids written for the three blocks are the whole grid's.
void CheckBlockIdsCopied(const std::vector<CellBlockId>& block_ids, const GridTables& whole) {
  for (std::size_t block = 0; block < 3; ++block) {
    EXPECT_EQ(block_ids.at(block).value, whole.block_ids[block].value);
  }
}

TEST(ApplyPieceBlockCopyAt, EveryBlockLoaded_RenumbersEveryEntry) {
  const PlaneFourGrid grid = BuildPlaneFourGrid();
  const std::vector<CellBlockIndex> map = BuildMapWithout(3, {.value = kNoIndex});
  const PieceBlocks blocks = {.range = {.first_block = 0, .block_count = 3}, .map = map.data(), .count = 3};
  PieceArraysOwner owner(3);
  for (std::uint32_t piece_block = 0; piece_block < 3; ++piece_block) {
    EXPECT_EQ(ApplyPieceBlockCopyAt(grid.tables, blocks, owner.ToArrays(), piece_block), kNoFailureKey);
  }
  CheckBlockIdsCopied(owner.block_ids, grid.tables);
  EXPECT_EQ(owner.same_size_level_blocks.at(0), grid.tables.same_size_level_blocks[0]);
  EXPECT_EQ(owner.bigger_block.at(0), grid.tables.bigger_block[0]);
  EXPECT_EQ(owner.smaller_blocks.at(0), grid.tables.smaller_blocks[0]);
}

// Finds a whole block and an entry of its 27 same-size-level entries that names another block.
struct NeighbourEntry {
  std::uint32_t block;
  std::uint32_t entry;
  std::uint32_t neighbour;
};

NeighbourEntry FindEntryNamingAnotherBlock(const GridTables& tables) {
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    for (std::uint32_t entry = 0; entry < 27; ++entry) {
      const std::uint32_t neighbour = tables.same_size_level_blocks[(std::size_t{block} * 27) + entry];
      if (neighbour != kNoIndex && neighbour != block) {
        return {.block = block, .entry = entry, .neighbour = neighbour};
      }
    }
  }
  return {.block = 0, .entry = 0, .neighbour = kNoIndex};
}

TEST(ApplyPieceBlockCopyAt, EntryNamesBlockOfRangeMissingFromMap_IsTheKeyOfAFault) {
  const PlaneFourGrid grid = BuildPlaneFourGrid();
  const NeighbourEntry named = FindEntryNamingAnotherBlock(grid.tables);
  ASSERT_NE(named.neighbour, kNoIndex);
  const std::vector<CellBlockIndex> map = BuildMapWithout(3, {.value = named.neighbour});
  const PieceBlocks blocks = {
      .range = {.first_block = named.neighbour, .block_count = 1},
      .map = map.data(),
      .count = static_cast<std::uint32_t>(map.size()),
  };
  PieceArraysOwner owner(map.size());
  const auto piece_block =
      static_cast<std::uint32_t>(std::ranges::find(map, named.block, &CellBlockIndex::value) - map.begin());
  const std::uint64_t failure_key = ApplyPieceBlockCopyAt(grid.tables, blocks, owner.ToArrays(), piece_block);
  EXPECT_EQ(failure_key, ToFailureKey(kPieceFaultStep, piece_block, static_cast<std::uint8_t>(named.entry)));
  EXPECT_EQ(owner.same_size_level_blocks.at((std::size_t{piece_block} * 27) + named.entry), kNoIndex);
}

TEST(ApplyPieceBlockCopyAt, EntryNamesBlockOfTheBorderNotLoaded_IsNoIndexWithoutAFault) {
  const PlaneFourGrid grid = BuildPlaneFourGrid();
  const NeighbourEntry named = FindEntryNamingAnotherBlock(grid.tables);
  ASSERT_NE(named.neighbour, kNoIndex);
  const std::vector<CellBlockIndex> map = BuildMapWithout(3, {.value = named.neighbour});
  const std::uint32_t range_block = named.block;
  const PieceBlocks blocks = {
      .range = {.first_block = range_block, .block_count = 1},
      .map = map.data(),
      .count = static_cast<std::uint32_t>(map.size()),
  };
  PieceArraysOwner owner(map.size());
  const auto piece_block =
      static_cast<std::uint32_t>(std::ranges::find(map, named.block, &CellBlockIndex::value) - map.begin());
  EXPECT_EQ(ApplyPieceBlockCopyAt(grid.tables, blocks, owner.ToArrays(), piece_block), kNoFailureKey);
  EXPECT_EQ(owner.same_size_level_blocks.at((std::size_t{piece_block} * 27) + named.entry), kNoIndex);
}

TEST(ApplyPieceCellStateCopyAt, EveryPlaceOfEveryBlock_CopiesTheStateAndGivesTheSizeLevelOfActiveCells) {
  const PlaneFourGrid grid = BuildPlaneFourGrid();
  const std::vector<CellBlockIndex> map = BuildMapWithout(3, {.value = kNoIndex});
  const PieceBlocks blocks = {.range = {.first_block = 0, .block_count = 3}, .map = map.data(), .count = 3};
  PieceArraysOwner owner(3);
  int active_count = 0;
  for (std::uint32_t place = 0; place < 3 * kCellPlacesPerBlock; ++place) {
    const int size_level = ApplyPieceCellStateCopyAt(grid.tables, blocks, place, owner.cell_states.data());
    EXPECT_EQ(owner.cell_states.at(place), grid.tables.cell_states[place]) << place;
    const bool is_active = grid.tables.cell_states[place] == CellState::kActive;
    EXPECT_EQ(size_level >= 0, is_active) << place;
    active_count += is_active ? 1 : 0;
  }
  EXPECT_GT(active_count, 0);
}

TEST(ApplyPieceFaceKindWriteAt, WholeGridLoaded_KeepsEveryKind) {
  const PlaneFourGrid grid = BuildPlaneFourGrid();
  const std::vector<CellBlockIndex> map = BuildMapWithout(3, {.value = kNoIndex});
  const PieceBlocks blocks = {.range = {.first_block = 0, .block_count = 3}, .map = map.data(), .count = 3};
  PieceArraysOwner owner(3);
  for (std::uint32_t place = 0; place < 3 * kFacePlacesPerBlock; ++place) {
    const std::uint32_t is_face_piece =
        ApplyPieceFaceKindWriteAt(grid.tables, blocks, place, owner.face_piece_kinds.data());
    EXPECT_EQ(owner.face_piece_kinds.at(place), grid.tables.face_piece_kinds[place]) << place;
    EXPECT_EQ(is_face_piece, grid.tables.face_piece_kinds[place] != FacePieceKind::kNotAPiece ? 1U : 0U) << place;
  }
}

TEST(ApplyPieceFaceKindWriteAt, EachBlockAlone_SomeFacePieceIsOutsideThePiece) {
  const PlaneFourGrid grid = BuildPlaneFourGrid();
  bool is_outside_piece_found = false;
  for (std::uint32_t block = 0; block < 3; ++block) {
    const std::vector<CellBlockIndex> map = {{.value = block}};
    const PieceBlocks blocks = {.range = {.first_block = block, .block_count = 1}, .map = map.data(), .count = 1};
    PieceArraysOwner owner(1);
    for (std::uint32_t place = 0; place < kFacePlacesPerBlock; ++place) {
      ApplyPieceFaceKindWriteAt(grid.tables, blocks, place, owner.face_piece_kinds.data());
    }
    is_outside_piece_found = is_outside_piece_found
        || std::ranges::find(owner.face_piece_kinds, FacePieceKind::kOutsidePiece) != owner.face_piece_kinds.end();
  }
  EXPECT_TRUE(is_outside_piece_found);
}

}  // namespace
}  // namespace fluidity::grid
