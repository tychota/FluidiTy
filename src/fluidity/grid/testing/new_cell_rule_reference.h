// Test-only: the two facts about a new cell, found by looking at every committed cell.
//
// Purpose: give the tests of FindNewCellNeighbors an answer computed another way.
//
// Idea: it walks every active cell of the grid, turns the cell and the new cell into boxes of whole smallest cells,
//   keeps the cells whose closed box meets the new cell's closed box without overlapping it in volume, and compares
//   size levels. It makes no look-up by place and uses none of the 26 places or 56 parts of the method it checks.
//
// Words:
//   smallest cell   a cell of the frame's last size level; every box here is whole smallest cells
// Owns:    ComputeNewCellNeighborsByAllCells; and, for this header's own functions only, SmallestCellBox,
//          ToSmallestCellBox, ComputeBoxesTouch
// Uses:    CPU-only guard (base/cpu_only.h), NewCellNeighbors (grid/new_cell_rule.h), GridTables, ToCellId
//          (grid/grid_tables.h), the id types (grid/grid_ids.h)
// Used by: src/fluidity/grid/new_cell_rule.test.cc and tests/grid/new_cell_neighbors_on_random_grids.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/new_cell_rule.h"

namespace fluidity::grid {

// A cell as a box of whole smallest cells: it covers low[axis] <= u < low[axis] + width on each axis.
struct SmallestCellBox {
  std::array<std::int64_t, 3> low;
  std::int64_t width;
};

// Gives the box of a cell in smallest cells: a cell of size level l at coordinate c is 2^(L - 1 - l) smallest cells
// wide and starts at c times that width, L the number of size levels of the frame.
inline SmallestCellBox ToSmallestCellBox(CellId cell, int size_level_count) {
  const int size_level = static_cast<int>(ToSizeLevel(cell));
  const CellCoord coord = ToCellCoord(cell);
  const auto width =
      static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level_count - 1 - size_level));
  return {.low = {coord.x * width, coord.y * width, coord.z * width}, .width = width};
}

// Says whether two boxes share at least a point and do not overlap in volume. A box that overlaps the other in volume
// is no neighbour: the new cell must lie in space with no cell.
inline bool ComputeBoxesTouch(const SmallestCellBox& first, const SmallestCellBox& second) {
  bool shares_a_point = true;
  bool overlaps_in_volume = true;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const std::int64_t first_high = first.low.at(axis) + first.width;
    const std::int64_t second_high = second.low.at(axis) + second.width;
    shares_a_point = shares_a_point && first.low.at(axis) <= second_high && second.low.at(axis) <= first_high;
    overlaps_in_volume = overlaps_in_volume && first.low.at(axis) < second_high && second.low.at(axis) < first_high;
  }
  return shares_a_point && !overlaps_in_volume;
}

// Finds the two facts about a new cell by looking at every active cell of the committed grid.
//
// Method: for each place of the tables that is kActive, makes the cell's id, then both boxes in smallest cells, and
//   keeps the cell when the closed boxes meet and do not overlap in volume. A kept cell two or more size levels
//   smaller than the new cell sets touches_much_smaller_cell; a kept cell one or more size levels bigger sets
//   touches_bigger_cell.
// Requires: `new_cell` is a valid id, in the frame of `committed`, with no cell of the grid overlapping it.
inline NewCellNeighbors ComputeNewCellNeighborsByAllCells(const GridTables& committed, CellId new_cell) {
  const int size_level_count = committed.frame.size_level_count;
  const int new_cell_size_level = static_cast<int>(ToSizeLevel(new_cell));
  const SmallestCellBox new_cell_box = ToSmallestCellBox(new_cell, size_level_count);
  NewCellNeighbors neighbors = {.touches_much_smaller_cell = false, .touches_bigger_cell = false};
  const std::uint64_t place_count = std::uint64_t{committed.block_count} * kCellPlacesPerBlock;
  for (std::uint64_t place = 0; place < place_count; ++place) {
    if (committed.cell_states[place] != CellState::kActive) {
      continue;
    }
    const CellId committed_cell = ToCellId(committed, CellPlaceIndex{static_cast<std::uint32_t>(place)});
    const SmallestCellBox committed_box = ToSmallestCellBox(committed_cell, size_level_count);
    if (!ComputeBoxesTouch(new_cell_box, committed_box)) {
      continue;
    }
    const int committed_size_level = static_cast<int>(ToSizeLevel(committed_cell));
    neighbors.touches_much_smaller_cell =
        neighbors.touches_much_smaller_cell || committed_size_level >= new_cell_size_level + 2;
    neighbors.touches_bigger_cell = neighbors.touches_bigger_cell || committed_size_level <= new_cell_size_level - 1;
  }
  return neighbors;
}

}  // namespace fluidity::grid
