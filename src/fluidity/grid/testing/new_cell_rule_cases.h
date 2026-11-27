// Test-only: the hand-made committed grids and new cells of the new cell rule, with the facts worked out by hand.
//
// Purpose: let the unit tests of the reference, the unit tests of FindNewCellNeighbors and its GPU test use the same
//   cases, written once.
//
// Idea: the frame has 4 size levels, biggest cells 1 m wide and origin 0. The near-small grid holds
//   A    the size-level-0 cell {1,-1,0};
//   P    the size-level-0 cell {0,0,0}, split into its 8 size-level-1 cells; the one at {1,1,0} is split into its 8
//        size-level-2 cells, and the one of those at {3,3,0} into its 8 size-level-3 cells.
//   That is 1 + 7 + 7 + 8 = 23 active cells. The size step rule holds: the size-level-3 cells lie at x 0.75 to 1 m,
//   y 0.75 to 1 m and z 0 to 0.25 m, and touch only size-level-2 cells and cells outside the grid. Each case gives the
//   new cell, the touching committed cells with their size levels, and the two facts.
//
// Owns:    HandCase, ToHandGridFrame, ToHandCell, BuildNearSmallCells, BuildNearSmallCases, BuildRangeEndCells,
//          BuildRangeEndCases
// Uses:    CPU-only guard (base/cpu_only.h), GridFrame (grid/grid_tables.h), NewCellNeighbors (grid/new_cell_rule.h),
//          the id types and constants (grid/grid_ids.h)
// Used by: src/fluidity/grid/new_cell_rule.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <vector>

#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/new_cell_rule.h"

namespace fluidity::grid {

// A new cell and the two facts the committed cells around it give, worked out by hand.
struct HandCase {
  CellId new_cell;
  NewCellNeighbors expected;
};

// The frame of the hand grids: origin 0, biggest cells 1 m wide, 4 size levels.
constexpr GridFrame ToHandGridFrame() {
  return {.origin = {.x = 0, .y = 0, .z = 0}, .biggest_cell_width = 1.0, .size_level_count = 4};
}

inline CellId ToHandCell(int size_level, std::int32_t cell_x, std::int32_t cell_y, std::int32_t cell_z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = cell_x, .y = cell_y, .z = cell_z});
}

// Appends the 8 small cells of `big_cell` to `cells`.
inline void ApplySmallCellsToHandList(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

// Appends the cells of the 2 x 2 x 2 box of `size_level` cells that start at `first_coord`, except the one at
// `split_coord`, which the caller splits.
inline void ApplyEightCellsExceptOneToHandList(
    int size_level,
    CellCoord first_coord,
    CellCoord split_coord,
    std::vector<CellId>& cells
) {
  for (std::int32_t offset_z = 0; offset_z < 2; ++offset_z) {
    for (std::int32_t offset_y = 0; offset_y < 2; ++offset_y) {
      for (std::int32_t offset_x = 0; offset_x < 2; ++offset_x) {
        const CellCoord coord = {
            .x = first_coord.x + offset_x,
            .y = first_coord.y + offset_y,
            .z = first_coord.z + offset_z,
        };
        const bool is_split = coord.x == split_coord.x && coord.y == split_coord.y && coord.z == split_coord.z;
        if (!is_split) {
          cells.push_back(ToHandCell(size_level, coord.x, coord.y, coord.z));
        }
      }
    }
  }
}

// Builds the 23 active cells of the near-small grid described in the file header: A, then the 7 size-level-1 cells of
// P, the 7 size-level-2 cells inside its cell {1,1,0}, and the 8 size-level-3 cells inside its cell {3,3,0}.
inline std::vector<CellId> BuildNearSmallCells() {
  std::vector<CellId> cells = {ToHandCell(0, 1, -1, 0)};
  ApplyEightCellsExceptOneToHandList(1, {.x = 0, .y = 0, .z = 0}, {.x = 1, .y = 1, .z = 0}, cells);
  ApplyEightCellsExceptOneToHandList(2, {.x = 2, .y = 2, .z = 0}, {.x = 3, .y = 3, .z = 0}, cells);
  ApplySmallCellsToHandList(ToHandCell(2, 3, 3, 0), cells);
  return cells;
}

// The new cells of the near-small grid with the facts worked out by hand. The new cell Q is the size-level-0 cell
// {1,0,0}: x 1 to 2 m, y 0 to 1 m, z 0 to 1 m. Each entry lists the committed cells that touch it (size level in
// brackets) and says which of them decide the facts. A cell touches when its closed box meets the new cell's closed
// box without overlapping it; "much smaller" is 2 or more size levels, "bigger" is 1 or more size levels bigger.
inline std::vector<HandCase> BuildNearSmallCases() {
  constexpr NewCellNeighbors kSplitIt = {.touches_much_smaller_cell = true, .touches_bigger_cell = false};
  constexpr NewCellNeighbors kBiggerOnly = {.touches_much_smaller_cell = false, .touches_bigger_cell = true};
  constexpr NewCellNeighbors kNeither = {.touches_much_smaller_cell = false, .touches_bigger_cell = false};
  return {
      // Q: touches A [0] across y = 0, the P cells {1,0,*} and {1,1,1} [1], the size-level-2 cells with x 0.75 to 1 m
      // [2] and the four size-level-3 cells at x 0.875 to 1 m [3]. A size-level-3 cell is 3 >= 0 + 2 levels smaller:
      // much smaller. No cell has a size level below 0: not bigger.
      {.new_cell = ToHandCell(0, 1, 0, 0), .expected = kSplitIt},
      // Q's size-level-1 cell {2,1,0} (x 1 to 1.5, y 0.5 to 1, z 0 to 0.5): touches the size-level-3 cells {7,6,0},
      // {7,6,1}, {7,7,0}, {7,7,1} [3] across x = 1: 3 >= 1 + 2, much smaller. A [0] lies at y below 0: not touching.
      // No cell of size level 0 touches it: not bigger.
      {.new_cell = ToHandCell(1, 2, 1, 0), .expected = kSplitIt},
      // Q's size-level-1 cell {2,0,0} (y 0 to 0.5): touches A [0] across y = 0, so bigger. It touches {1,0,0} [1]
      // across x = 1 and the size-level-2 cells {3,2,0}, {3,2,1} [2] along the edge x = 1, y = 0.5; no size-level-3
      // cell
      // reaches below y = 0.75: not much smaller.
      {.new_cell = ToHandCell(1, 2, 0, 0), .expected = kBiggerOnly},
      // Q's size-level-1 cell {2,1,1} (x 1 to 1.5, y 0.5 to 1, z 0.5 to 1): touches {1,1,1} [1] across x = 1 and the
      // size-level-2 cells {3,2,1}, {3,3,1} [2] at z = 0.5; the size-level-3 cells end at z 0.25: not much smaller.
      // No size-level-0 cell touches it: not bigger.
      {.new_cell = ToHandCell(1, 2, 1, 1), .expected = kNeither},
      // Q's size-level-2 cell {4,2,0} (x 1 to 1.25, y 0.5 to 0.75, z 0 to 0.25): touches {1,0,0} [1] along the edge
      // x = 1, y = 0.5, so bigger. It touches {3,2,0} [2] across x = 1 and the size-level-3 cell {7,6,0} [3] along
      // the edge x = 1, y = 0.75. A cell 2 size levels smaller would be size level 4: it cannot exist: not much
      // smaller.
      {.new_cell = ToHandCell(2, 4, 2, 0), .expected = kBiggerOnly},
      // Q's size-level-2 cell {4,3,0} (x 1 to 1.25, y 0.75 to 1, z 0 to 0.25): touches {3,3,1} [2] along the edge
      // x = 1, z = 0.25, {3,2,0} [2] along x = 1, y = 0.75 and the size-level-3 cells {7,6,0}, {7,7,0}, {7,7,1},
      // {7,6,1} [3]. No cell of size level 1 or below touches it: not bigger.
      {.new_cell = ToHandCell(2, 4, 3, 0), .expected = kNeither},
      // Q's size-level-3 cell {8,7,0} (x 1 to 1.125, y 0.875 to 1, z 0 to 0.125): touches {7,7,0} [3] across x = 1 and
      // {7,7,1} [3] along an edge. {3,3,1} [2] starts at z 0.25: not touching. Not bigger, and not much smaller:
      // the last size level has no cell 2 levels smaller.
      {.new_cell = ToHandCell(3, 8, 7, 0), .expected = kNeither},
      // Q's size-level-3 cell {8,5,0} (y 0.625 to 0.75): touches {3,2,0} [2] across x = 1, so bigger; {7,6,0} [3]
      // touches
      // it along the edge x = 1, y = 0.75.
      {.new_cell = ToHandCell(3, 8, 5, 0), .expected = kBiggerOnly},
      // The size-level-0 cell {5,5,5} is far from every committed cell.
      {.new_cell = ToHandCell(0, 5, 5, 5), .expected = kNeither},
      // The size-level-0 cell {2,0,1} (x 2 to 3, y 0 to 1, z 1 to 2) meets A [0] (x 1 to 2, y -1 to 0, z 0 to 1) at
      // the point (2, 0, 1) only. A has the same size level: neither fact.
      {.new_cell = ToHandCell(0, 2, 0, 1), .expected = kNeither},
      // The size-level-1 cell {4,0,2} (x 2 to 2.5, y 0 to 0.5, z 1 to 1.5) meets A [0] at the point (2, 0, 1) only:
      // A is one size level bigger.
      {.new_cell = ToHandCell(1, 4, 0, 2), .expected = kBiggerOnly},
      // The size-level-0 cell {1,1,-1} (x 1 to 2, y 1 to 2, z -1 to 0) meets the size-level-3 cell {7,7,0} [3] at the
      // point (1, 1, 0) only: much smaller.
      {.new_cell = ToHandCell(0, 1, 1, -1), .expected = kSplitIt},
      // The size-level-0 cell {1,1,0} (x 1 to 2, y 1 to 2, z 0 to 1) meets no committed cell across a face or at a
      // corner only. It meets the size-level-3 cells {7,7,0} and {7,7,1} [3] along the edge x = 1, y = 1 (z 0 to 0.25),
      // so a much smaller cell touches along an edge: much smaller. Along the same edge it meets {1,1,1} [1] (z 0.5
      // to 1) and {3,3,1} [2] (z 0.25 to 0.5), and no cell has a size level below 0: not bigger.
      {.new_cell = ToHandCell(0, 1, 1, 0), .expected = kSplitIt},
  };
}

// Builds the 2 active cells of the range-end grid: the size-level-0 cells {kMaxCoord - 1,0,0} and {1 - kMaxCoord,0,0},
// next to the last cells of the range. Only size-level-0 cells reach the ends of the range: a cell of size level l has
// coordinates in -kMaxCoord..kMaxCoord in its own cells, so it stays within 2^-l of the range.
inline std::vector<CellId> BuildRangeEndCells() {
  return {ToHandCell(0, kMaxCoord - 1, 0, 0), ToHandCell(0, 1 - kMaxCoord, 0, 0)};
}

// The new cells of the range-end grid: the size-level-0 cells {kMaxCoord,0,0} and {-kMaxCoord,0,0}, the last cells of
// the range. Each touches one committed cell of its own size across x: neither fact. The places beyond the range on
// its other side hold no cell, and a look-up there must count as empty.
inline std::vector<HandCase> BuildRangeEndCases() {
  constexpr NewCellNeighbors kNeither = {.touches_much_smaller_cell = false, .touches_bigger_cell = false};
  return {
      {.new_cell = ToHandCell(0, kMaxCoord, 0, 0), .expected = kNeither},
      {.new_cell = ToHandCell(0, -kMaxCoord, 0, 0), .expected = kNeither},
  };
}

}  // namespace fluidity::grid
