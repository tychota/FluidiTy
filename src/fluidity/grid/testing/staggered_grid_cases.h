// Test-only: the helpers that the unit tests and the functional tests of staggered_grid share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToUnitFrame, ToCell, ToInputs, BuildCellsInBlocksOfTheirOwn, ToSmallCells, CheckFailure,
//          CheckOneSplitCellBlockTables, CheckOneSplitCellStates, BuildMoreCellsThanEightBlocksHold
// Uses:    the headers it includes
// Used by: src/fluidity/grid/staggered_grid.test.cc and tests/grid/build_grid.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// A frame with its origin at 0 and biggest cells 1 m wide.
inline GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

inline CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The inputs of a CPU list of version 1.
inline GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// Size-level-0 cells {4 i, 0, 0} for i = 0..count - 1: each alone in a block of its own.
inline std::vector<CellId> BuildCellsInBlocksOfTheirOwn(std::int32_t count) {
  std::vector<CellId> cells;
  cells.reserve(static_cast<std::size_t>(count));
  for (std::int32_t index = 0; index < count; ++index) {
    cells.push_back(ToCell(0, 4 * index, 0, 0));
  }
  return cells;
}

// The 8 small cells of a big cell, by the test's own arithmetic: 2 c and 2 c + 1 on each axis, in eighth order.
inline std::vector<CellId> ToSmallCells(CellId big_cell) {
  const CellCoord coord = ToCellCoord(big_cell);
  const int small_size_level = static_cast<int>(ToSizeLevel(big_cell)) + 1;
  std::vector<CellId> small_cells;
  for (std::int32_t dz = 0; dz < 2; ++dz) {
    for (std::int32_t dy = 0; dy < 2; ++dy) {
      for (std::int32_t dx = 0; dx < 2; ++dx) {
        small_cells.push_back(ToCell(small_size_level, (2 * coord.x) + dx, (2 * coord.y) + dy, (2 * coord.z) + dz));
      }
    }
  }
  return small_cells;
}

inline void CheckFailure(
    const std::expected<StaggeredGrid, GridFailure>& result,
    GridError error,
    CellId cell = CellId{0},
    CellId other_cell = CellId{0}
) {
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, error);
  EXPECT_EQ(result.error().cell.value, cell.value);
  EXPECT_EQ(result.error().other_cell.value, other_cell.value);
  EXPECT_EQ(result.error().magnitude, 0U);
}

// The block tables of the fixture of OneSplitCell_GivesTheWorkedTables, in a room of 8 blocks.
inline void CheckOneSplitCellBlockTables(const GridTables& tables) {
  const std::vector<std::uint64_t> block_ids = {tables.block_ids[0].value, tables.block_ids[1].value};
  EXPECT_EQ(block_ids, (std::vector<std::uint64_t>{ToCell(0, 0, 0, 0).value, ToCell(1, 0, 0, 0).value}));
  EXPECT_EQ(
      std::vector<std::uint32_t>(tables.bigger_block, tables.bigger_block + 2),
      (std::vector<std::uint32_t>{kNoIndex, 0})
  );
  EXPECT_EQ(tables.smaller_blocks, tables.bigger_block + 8);
  std::vector<std::uint32_t> expected_smaller_blocks(16, kNoIndex);
  expected_smaller_blocks.at(0) = 1;
  EXPECT_EQ(std::vector<std::uint32_t>(tables.smaller_blocks, tables.smaller_blocks + 16), expected_smaller_blocks);
}

// The cell states of the fixture of OneSplitCell_GivesTheWorkedTables.
inline void CheckOneSplitCellStates(const GridTables& tables) {
  std::map<std::pair<int, CellState>, int> state_counts;
  for (int place = 0; place < 2 * kCellPlacesPerBlock; ++place) {
    ++state_counts[{place / kCellPlacesPerBlock, tables.cell_states[place]}];
  }
  const std::map<std::pair<int, CellState>, int> expected_state_counts = {
      {{0, CellState::kAbsent}, 62},
      {{0, CellState::kActive}, 1},
      {{0, CellState::kSplit}, 1},
      {{1, CellState::kAbsent}, 48},
      {{1, CellState::kInsideBiggerCell}, 8},
      {{1, CellState::kActive}, 8},
  };
  EXPECT_EQ(state_counts, expected_state_counts);
  // Block 0: place 0 the active cell, place 1 the split cell. Block 1: size level 1 place (1, 1, 1), at 21, lies inside
  // the big cell; (2, 1, 1), at 22, is a small cell.
  const std::vector<CellState> chosen_states = {
      tables.cell_states[0],
      tables.cell_states[1],
      tables.cell_states[64 + 21],
      tables.cell_states[64 + 22],
  };
  const std::vector<CellState> expected_chosen_states = {
      CellState::kActive,
      CellState::kSplit,
      CellState::kInsideBiggerCell,
      CellState::kActive,
  };
  EXPECT_EQ(chosen_states, expected_chosen_states);
}

// 64 x 8 + 1 = 513 size-level-0 cells {i, 0, 0} into a room of 8 blocks: the length check, before any id is read.
inline std::vector<CellId> BuildMoreCellsThanEightBlocksHold() {
  constexpr std::int32_t kCellCount = (64 * 8) + 1;
  std::vector<CellId> cells;
  cells.reserve(kCellCount);
  for (std::int32_t index = 0; index < kCellCount; ++index) {
    cells.push_back(ToCell(0, index, 0, 0));
  }
  return cells;
}

}  // namespace fluidity::grid
