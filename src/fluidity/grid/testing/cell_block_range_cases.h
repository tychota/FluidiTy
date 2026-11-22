// Test-only: the grids that the unit tests and the functional tests of ranges and borders share.
//
// Purpose: let the tests of cell_block_range.h and border_block_steps.h build the same small grids, written once.
//
// Idea: two grids of one size level, cells 1 m wide, whose borders are known by hand. The plane-four grid has the
//   active cells {3,3,0} (block 0), {4,3,0} (block 1) and {6,6,0} (block 2). The cell {4,3,0} touches {3,3,0}, and its
//   +Y face piece is plane 4 of block 1, which is plane 0 of block 2, so block 2 is in the border of block 0 though no
//   cell of block 2 touches {3,3,0}; and block 1 is in the border of block 2, since that face piece's cell_low is
//   {4,3,0}. The distant-corner grid has {0,0,0} (block 0) and {7,7,7} (block 1), two cells that touch nothing but lie
//   in blocks that are neighbours. The corner-beyond grid, of 2 size levels, has the size-level-0 cell {-1,0,0} and
//   the 8 size-level-1 cells of the size-level-0 place {1,1,1}, which touch nothing. The blocks are the size-level-0
//   blocks (-1,0,0) and (0,0,0), then the size-level-1 block (0,0,0): 0, 1 and 2. The +X side of {-1,0,0} lies at
//   plane 0 of block 1. Its 4 corners at x = 0 m are points of the lattices of blocks 1 and 2, kept in block 2, of
//   the smaller cells, though no cell of block 2 touches {-1,0,0}.
//   The split-corner grid, of 2 size levels, has the size-level-0 cells {0..3}^3, all active but {0..2}^3, which are
//   split into their size-level-1 cells. Block 0 is the size-level-0 block, block 1 the size-level-1 block (0,0,0).
//   Every place a reader returns from a place of block 1 is kept at size level 1, yet FindCellAt walks down from block
//   0, so block 0 is in the border of block 1.
//   The nested-ball grid, of 4 size levels, is larger: the size-level-0 cells {0..11}^3, 27 size-level-0 blocks, and
//   around the point (6, 6, 6) m, the middle of the middle size-level-0 block, balls of smaller cells, each size level
//   in a region of its own, far from the grid's edge.
//
// Owns:    BuildPlaneFourCells, BuildDistantCornerCells, BuildCornerBeyondCells, BuildSplitCornerCells,
//          BuildNestedBallCells, FindDrawnAscendingBlocks, ToRangeBlocks, FindChosenBlockSets, ToBlocks
// Uses:    CPU-only guard (base/cpu_only.h), ToCell (grid/testing/staggered_grid_cases.h), BuildSizeStepClosure,
//          ToUniformIndex (grid/testing/grid_support.h), BuildRanges (grid/cell_block_range.h), StaggeredGrid
//          (grid/staggered_grid.h), GridFailure (grid/grid_error.h)
// Used by: src/fluidity/grid/cell_block_range.test.cc, the functional tests of border blocks (tests/grid/)
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {

// The cells {3,3,0}, {4,3,0} and {6,6,0} of one size level, in blocks 0, 1 and 2 of a frame of 1 size level.
inline std::vector<CellId> BuildPlaneFourCells() {
  return {ToCell(0, 3, 3, 0), ToCell(0, 4, 3, 0), ToCell(0, 6, 6, 0)};
}

// The cells {0,0,0} and {7,7,7} of one size level, in blocks 0 and 1 of a frame of 1 size level.
inline std::vector<CellId> BuildDistantCornerCells() {
  return {ToCell(0, 0, 0, 0), ToCell(0, 7, 7, 7)};
}

// The size-level-0 cell {-1,0,0} and the 8 size-level-1 cells of the place {1,1,1}, in a frame of 2 size levels:
// blocks 0 (size level 0, at (-1,0,0)), 1 (size level 0, at (0,0,0)) and 2 (size level 1, at (0,0,0)).
inline std::vector<CellId> BuildCornerBeyondCells() {
  std::vector<CellId> cells = {ToCell(0, -1, 0, 0)};
  for (std::int32_t small_z = 2; small_z <= 3; ++small_z) {
    for (std::int32_t small_y = 2; small_y <= 3; ++small_y) {
      for (std::int32_t small_x = 2; small_x <= 3; ++small_x) {
        cells.push_back(ToCell(1, small_x, small_y, small_z));
      }
    }
  }
  return cells;
}

// The size-level-0 cells {0..3}^3, all active but {0..2}^3, which give their 8 size-level-1 cells each, in a frame of
// 2 size levels: 37 size-level-0 cells and 216 size-level-1 cells.
inline std::vector<CellId> BuildSplitCornerCells() {
  std::vector<CellId> cells;
  for (std::int32_t big_z = 0; big_z <= 3; ++big_z) {
    for (std::int32_t big_y = 0; big_y <= 3; ++big_y) {
      for (std::int32_t big_x = 0; big_x <= 3; ++big_x) {
        const bool is_split = big_x <= 2 && big_y <= 2 && big_z <= 2;
        if (!is_split) {
          cells.push_back(ToCell(0, big_x, big_y, big_z));
          continue;
        }
        for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
          cells.push_back(ToCell(
              1,
              (2 * big_x) + (cell_eighth % 2),
              (2 * big_y) + ((cell_eighth / 2) % 2),
              (2 * big_z) + (cell_eighth / 4)
          ));
        }
      }
    }
  }
  return cells;
}

// The nested-ball grid of the file header, for a frame of 4 size levels with biggest cells 1 m wide: a cell of size
// level l is split while the distance from (6, 6, 6) m to its box is below the radius of l, 3.5 m, 2 m and 1 m for
// size levels 0, 1 and 2; the size-step closure then splits what the size step rule needs. About 11,000 cells in
// about 200 blocks, with a whole size-level-0 block in the middle and uniform regions of every size level.
inline std::vector<CellId> BuildNestedBallCells() {
  constexpr std::array<double, 3> kSplitRadius = {3.5, 2.0, 1.0};
  constexpr double kBallCenter = 6.0;
  std::vector<CellId> cells;
  std::vector<CellId> to_visit;
  for (std::int32_t big_z = 0; big_z < 12; ++big_z) {
    for (std::int32_t big_y = 0; big_y < 12; ++big_y) {
      for (std::int32_t big_x = 0; big_x < 12; ++big_x) {
        to_visit.push_back(ToCell(0, big_x, big_y, big_z));
      }
    }
  }
  while (!to_visit.empty()) {
    const CellId cell = to_visit.back();
    to_visit.pop_back();
    const auto size_level = static_cast<int>(ToSizeLevel(cell));
    const CellCoord coord = ToCellCoord(cell);
    const double cell_width = 1.0 / static_cast<double>(1U << static_cast<unsigned>(size_level));
    // The distance from the ball's centre to the cell's box, by the nearest point of the box on each axis.
    double squared_distance = 0;
    for (const std::int32_t axis_coord : {coord.x, coord.y, coord.z}) {
      const double low_side = axis_coord * cell_width;
      const double nearest = std::clamp(kBallCenter, low_side, low_side + cell_width);
      squared_distance += (nearest - kBallCenter) * (nearest - kBallCenter);
    }
    const bool is_split = size_level < 3
        && squared_distance < kSplitRadius.at(static_cast<std::size_t>(size_level))
                * kSplitRadius.at(static_cast<std::size_t>(size_level));
    if (!is_split) {
      cells.push_back(cell);
      continue;
    }
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      to_visit.push_back(ToCell(
          size_level + 1,
          (2 * coord.x) + (cell_eighth % 2),
          (2 * coord.y) + ((cell_eighth / 2) % 2),
          (2 * coord.z) + (cell_eighth / 4)
      ));
    }
  }
  return BuildSizeStepClosure(std::move(cells), 4);
}

// Draws an ascending set of blocks of a grid: each block with probability 1 in `one_in`, and at least one block.
inline std::vector<CellBlockIndex>
FindDrawnAscendingBlocks(std::mt19937_64& generator, std::uint32_t block_count, int one_in) {
  std::vector<CellBlockIndex> blocks;
  for (std::uint32_t block = 0; block < block_count; ++block) {
    if (ToUniformIndex(generator(), static_cast<std::uint64_t>(one_in)) == 0) {
      blocks.push_back({.value = block});
    }
  }
  if (blocks.empty()) {
    blocks.push_back({.value = static_cast<std::uint32_t>(ToUniformIndex(generator(), block_count))});
  }
  return blocks;
}

// The blocks of a range, in order.
inline std::vector<CellBlockIndex> ToRangeBlocks(CellBlockRange range) {
  std::vector<CellBlockIndex> blocks;
  blocks.reserve(range.block_count);
  for (std::uint32_t entry = 0; entry < range.block_count; ++entry) {
    blocks.push_back({.value = range.first_block + entry});
  }
  return blocks;
}

// The four block sets of random grid `grid_entry`, from a generator of its own: one block alone, the middle range of
// a cut into ranges of 8 blocks (BuildRanges), a sparse set and a dense set. One block shows the border at its
// smallest; a range is what a stage gets; a sparse set has border blocks between its blocks; a dense set leaves few
// blocks to find.
inline std::vector<std::vector<CellBlockIndex>>
FindChosenBlockSets(std::uint64_t grid_entry, const StaggeredGrid& grid) {
  const std::uint32_t block_count = grid.counts().cell_blocks;
  std::mt19937_64 generator(50000 + grid_entry);
  std::vector<std::vector<CellBlockIndex>> sets;
  sets.push_back({{.value = static_cast<std::uint32_t>(ToUniformIndex(generator(), block_count))}});
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(grid, 8);
  if (ranges.has_value() && !ranges->empty()) {
    sets.push_back(ToRangeBlocks(ranges->at(ranges->size() / 2)));
  }
  sets.push_back(FindDrawnAscendingBlocks(generator, block_count, 6));
  sets.push_back(FindDrawnAscendingBlocks(generator, block_count, 2));
  return sets;
}

// The block indexes of a list, in the order given.
inline std::vector<CellBlockIndex> ToBlocks(std::initializer_list<std::uint32_t> indexes) {
  std::vector<CellBlockIndex> blocks;
  for (const std::uint32_t index : indexes) {
    blocks.push_back({.value = index});
  }
  return blocks;
}

}  // namespace fluidity::grid
