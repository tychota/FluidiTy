// Test-only: the helpers that the unit tests and the functional tests of find_cell_at share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToUnitFrame, ToCell, ToInputs, ToFoundCellId, BuildEveryNearValue, FindCellIdAt,
//          BuildCellsOverSeveralBlocks, ExpectCheckerCellAt
// Uses:    the headers it includes
// Used by: src/fluidity/grid/find_cell_at.test.cc and tests/grid/find_cell_at_near_form_on_positions.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// A frame with its origin at 0 and biggest cells 1 m wide: every cell side in it is a double with few significant
// bits, so a position on a side is exactly on it.
constexpr GridFrame ToUnitFrame(int size_level_count) {
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

// The id of the cell at a place FindCellAt gave, or CellId{0} for kNoIndex. A found place must be kActive.
inline CellId ToFoundCellId(const GridTables& tables, CellPlaceIndex cell) {
  if (cell.value == kNoIndex) {
    return CellId{0};
  }
  EXPECT_EQ(tables.cell_states[cell.value], CellState::kActive);
  return ToCellId(tables, cell);
}

// The near values every hand-built test tries: each block of the grid, kNoIndex, block_count and a large index that is
// not kNoIndex.
inline std::vector<CellBlockIndex> BuildEveryNearValue(const GridTables& tables) {
  std::vector<CellBlockIndex> near_values;
  near_values.reserve(tables.block_count + 3);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    near_values.push_back({.value = block});
  }
  near_values.push_back({.value = kNoIndex});
  near_values.push_back({.value = tables.block_count});
  near_values.push_back({.value = kNoIndex - 1});
  return near_values;
}

// The id of the active cell that holds `position` by the search form, or CellId{0}, after checking that the near
// form gives the same place for every value of BuildEveryNearValue.
inline CellId FindCellIdAt(const GridTables& tables, Vec3 position) {
  const CellPlaceIndex searched = FindCellAt(tables, position);
  for (const CellBlockIndex near : BuildEveryNearValue(tables)) {
    EXPECT_EQ(FindCellAt(tables, position, near).value, searched.value) << "near " << near.value;
  }
  return ToFoundCellId(tables, searched);
}

// Size-level-0 cells over 4 x 2 x 2 size-level-0 blocks, block coordinates (0..3, 0..1, 0..1), without block
// (1, 1, 0), and one cell split, so the near form crosses blocks: one or two moves away, three moves away, and past the
// block that does not exist. 15 size-level-0 blocks and 1 size-level-1 block, in block (2, 0, 0).
inline std::vector<CellId> BuildCellsOverSeveralBlocks() {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 8; ++z) {
    for (std::int32_t y = 0; y < 8; ++y) {
      for (std::int32_t x = 0; x < 16; ++x) {
        const bool is_in_missing_block = x >= 4 && x < 8 && y >= 4 && z < 4;
        if (is_in_missing_block) {
          continue;
        }
        if (x == 9 && y == 2 && z == 1) {
          for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
            cells.push_back(ComputeSmallCell(ToCell(0, x, y, z), cell_eighth));
          }
          continue;
        }
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  return cells;
}

// Expects FindCellIdAt, both forms, to give the checker's cell at `position`, and says whether a cell holds it.
inline bool ExpectCheckerCellAt(const GridTables& tables, const GeometricGrid& checker, Vec3 position) {
  SCOPED_TRACE(testing::Message() << "position " << position.x << ", " << position.y << ", " << position.z);
  const std::optional<CellId> checker_cell = checker.ContainingCell(position);
  const CellId found_cell = FindCellIdAt(tables, position);
  EXPECT_EQ(found_cell.value, checker_cell.has_value() ? checker_cell->value : 0U);
  return found_cell.value != 0;
}

}  // namespace fluidity::grid
