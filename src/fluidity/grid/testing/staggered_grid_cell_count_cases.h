// Test-only: the helpers that the unit tests and the functional tests of staggered_grid_cell_count share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToUnitFrame, ToCell, ToSizeLevelNumber, ToInputs, ComputeCountsOfList, ToFloorOfHalf, ToBigCellByDivision,
//          ComputeCheckerBlockCount, ComputeActiveCellsPerSizeLevel, ToCountFields, CheckCheckerCounts
// Uses:    the headers it includes
// Used by: src/fluidity/grid/staggered_grid.cell_count.test.cc and tests/grid/grid_counts_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
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

inline int ToSizeLevelNumber(CellId cell) {
  return static_cast<int>(ToSizeLevel(cell));
}

// The inputs of a CPU list of version 1.
inline GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

inline std::expected<GridCounts, GridFailure>
ComputeCountsOfList(GridFrame frame, std::span<const CellId> active_cells) {
  return ComputeGridCounts(ToInputs(frame, active_cells));
}

// floor(coord / 2) by integer division and a correction for a negative remainder: the test's own arithmetic, apart
// from the shift the grid uses.
inline std::int32_t ToFloorOfHalf(std::int32_t coord) {
  const std::int32_t quotient = coord / 2;
  return (coord % 2 != 0 && coord < 0) ? quotient - 1 : quotient;
}

// The big cell, one size level up, that holds `cell`, by the test's own floor.
inline CellId ToBigCellByDivision(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  return ToCell(ToSizeLevelNumber(cell) - 1, ToFloorOfHalf(coord.x), ToFloorOfHalf(coord.y), ToFloorOfHalf(coord.z));
}

// The cell blocks of a valid list by the rule of <sec-cell-blocks>, apart from the grid's split cells and sort: the
// distinct ToCellBlockId of every active cell and of every bigger cell that holds one.
inline std::size_t ComputeCheckerBlockCount(std::span<const CellId> active_cells) {
  std::set<std::uint64_t> block_ids;
  for (const CellId active_cell : active_cells) {
    CellId cell = active_cell;
    block_ids.insert(ToCellBlockId(cell).value);
    while (ToSizeLevelNumber(cell) > 0) {
      cell = ToBigCellByDivision(cell);
      block_ids.insert(ToCellBlockId(cell).value);
    }
  }
  return block_ids.size();
}

// The active cells of each size level of a list, counted from the list itself.
inline std::array<std::uint64_t, kMaxSizeLevels> ComputeActiveCellsPerSizeLevel(std::span<const CellId> active_cells) {
  std::array<std::uint64_t, kMaxSizeLevels> per_size_level{};
  for (const CellId cell : active_cells) {
    ++per_size_level.at(static_cast<std::size_t>(ToSizeLevelNumber(cell)));
  }
  return per_size_level;
}

// Every field of counts, as one comparable value.
inline std::tuple<std::uint32_t, std::array<std::uint64_t, kMaxSizeLevels>, std::uint64_t, std::uint64_t> ToCountFields(
    const GridCounts& counts
) {
  std::array<std::uint64_t, kMaxSizeLevels> per_size_level{};
  for (std::size_t size_level = 0; size_level < per_size_level.size(); ++size_level) {
    per_size_level.at(size_level) = counts.active_cells_per_size_level[size_level];
  }
  return {counts.cell_blocks, per_size_level, counts.active_cells, counts.face_pieces};
}

// Expects the counts the checker gives for a valid list: its face pieces, the blocks of every active cell and the
// bigger cells holding it, and the active cells counted from the list.
inline void CheckCheckerCounts(
    const std::expected<GridCounts, GridFailure>& counts,
    GridFrame frame,
    const std::vector<CellId>& active_cells
) {
  ASSERT_TRUE(counts.has_value());
  const GeometricGrid checker(frame, active_cells);
  EXPECT_EQ(counts->face_pieces, checker.FacePieces().size());
  EXPECT_EQ(counts->cell_blocks, ComputeCheckerBlockCount(active_cells));
  EXPECT_EQ(counts->active_cells, active_cells.size());
  EXPECT_EQ(std::get<1>(ToCountFields(*counts)), ComputeActiveCellsPerSizeLevel(active_cells));
}

}  // namespace fluidity::grid
