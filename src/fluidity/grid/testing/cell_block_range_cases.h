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
//   The carried-sums grid, of 2 size levels, is one column of cells along x, 1 m wide at size level 0: the cells
//   [-2,-1), [0,1), [1,2) and [3,4) are active, and [-1,0) and [2,3) are split into their 8 smaller cells. A particle
//   of the cell [0,1) near x = 0.95 lists the face piece at x = -0.5, kept in the size-level-1 block of [-2,0), and the
//   face piece at x = 2, kept in the size-level-1 block of [2,4): the second block alone is a range whose near border
//   does not hold the first block.
//   The nested-ball grid, of 4 size levels, is larger: the size-level-0 cells {0..11}^3, 27 size-level-0 blocks, and
//   around the point (6, 6, 6) m, the middle of the middle size-level-0 block, balls of smaller cells, each size level
//   in a region of its own, far from the grid's edge.
//
//   The border search rig holds the room of the buffer form of FindBorderBlocks for one CPU grid: search buffers and
//   two buffers of one entry per block, so a test finds one layer of a stacked border without repeating the set-up.
//
// Owns:    BuildPlaneFourCells, BuildDistantCornerCells, BuildCornerBeyondCells, BuildSplitCornerCells,
//          BuildCarriedSumsCells, BuildNestedBallCells, BuildTouchingCellsInManyBlocks, FindDrawnAscendingBlocks,
//          ToRangeBlocks, FindChosenBlockSets, ToBlocks, BorderSearchRig, BuildBorderSearchRig, FindLayerOrFail,
//          kLayerWidths, kNestedBallLayerWidths, BuildTestedRanges, BuildNestedBallTestedRanges
// Uses:    CPU-only guard (base/cpu_only.h), Buffer, WriteBuffer, ReadBuffer (base/buffer.h), RunMemory
//          (base/run_memory.h), BuildBorderSearchMemoryEstimate (grid/grid_memory_estimate.h), ToCell
//          (grid/testing/staggered_grid_cases.h), BuildSizeStepClosure, ToUniformIndex (grid/testing/grid_support.h),
//          BuildRanges (grid/cell_block_range.h), StaggeredGrid (grid/staggered_grid.h), GridFailure
//          (grid/grid_error.h), ToName (grid/grid_error_message.h)
// Used by: src/fluidity/grid/cell_block_range.test.cc, the functional tests of border blocks (tests/grid/)
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_error_message.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
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

// The carried-sums grid of the file header, for a frame of 2 size levels: 4 size-level-0 cells and 16 size-level-1
// cells.
inline std::vector<CellId> BuildCarriedSumsCells() {
  std::vector<CellId> cells = {ToCell(0, -2, 0, 0), ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0), ToCell(0, 3, 0, 0)};
  for (const std::int32_t split_x : {-1, 2}) {
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      cells.push_back(ToCell(1, (2 * split_x) + (cell_eighth % 2), (cell_eighth / 2) % 2, cell_eighth / 4));
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

// The cells {4 i, 0, 0} and {4 i + 3, 0, 0} of one size level for i = 0 .. block_count - 1: block i of the first size
// level holds two cells, and the last cell of each block touches the first cell of the next, so the border of a run of
// blocks holds the block after it. A grid of this many blocks makes a set larger than one marking launch cheaply.
inline std::vector<CellId> BuildTouchingCellsInManyBlocks(std::int32_t block_count) {
  std::vector<CellId> cells;
  cells.reserve(std::size_t{2} * static_cast<std::size_t>(block_count));
  for (std::int32_t block = 0; block < block_count; ++block) {
    cells.push_back(ToCell(0, 4 * block, 0, 0));
    cells.push_back(ToCell(0, (4 * block) + 3, 0, 0));
  }
  return cells;
}

// The block indexes of a list, in the order given.
inline std::vector<CellBlockIndex> ToBlocks(std::initializer_list<std::uint32_t> indexes) {
  std::vector<CellBlockIndex> blocks;
  for (const std::uint32_t index : indexes) {
    blocks.push_back({.value = index});
  }
  return blocks;
}

// The room of the buffer form of FindBorderBlocks for one CPU grid: the block that the search buffers are pieces of,
// the search buffers, and two buffers of one entry per block of the grid, one for the set found so far and one for the
// border found.
struct BorderSearchRig {
  const StaggeredGrid& grid;  // The grid the room is for.
  RunMemory run_memory;
  BorderSearchBuffers search_buffers;
  Buffer<CellBlockIndex> so_far_buffer;
  Buffer<CellBlockIndex> border_buffer;
};

// Builds the rig of `grid`, a grid of at least one block, on the processor of the grid: search buffers for its block
// count and two buffers of that many entries.
inline std::expected<BorderSearchRig, GridFailure> BuildBorderSearchRig(const StaggeredGrid& grid) {
  const std::uint32_t block_count = grid.counts().cell_blocks;
  const std::vector<MemoryEstimateEntry> estimate = BuildBorderSearchMemoryEstimate(grid.run_on(), block_count);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(
      grid.run_on(),
      ComputeRunMemoryBytes(estimate, ToMemoryCounts(grid.counts(), 0), MemoryWhere::kRunOn)
  );
  if (!run_memory.has_value()) {
    return std::unexpected(GridFailure{.error = GridError::kOutOfMemory});
  }
  std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(grid.run_on(), block_count, *run_memory);
  std::expected<Buffer<CellBlockIndex>, BufferError> so_far_buffer =
      BuildBuffer<CellBlockIndex>(grid.run_on(), block_count);
  std::expected<Buffer<CellBlockIndex>, BufferError> border_buffer =
      BuildBuffer<CellBlockIndex>(grid.run_on(), block_count);
  if (!search_buffers.has_value()) {
    return std::unexpected(search_buffers.error());
  }
  if (!so_far_buffer.has_value() || !border_buffer.has_value()) {
    return std::unexpected(GridFailure{.error = GridError::kOutOfMemory});
  }
  return BorderSearchRig{
      .grid = grid,
      .run_memory = std::move(*run_memory),
      .search_buffers = std::move(*search_buffers),
      .so_far_buffer = std::move(*so_far_buffer),
      .border_buffer = std::move(*border_buffer),
  };
}

// One layer of a stacked border by the buffer form: the blocks outside `range` that are in `so_far` or in the border
// of range plus `so_far` at `width`, ascending. `so_far` is copied into the rig's first buffer. It fails the calling
// test, and gives an empty list, when a call fails.
inline std::vector<CellBlockIndex>
FindLayerOrFail(BorderSearchRig& rig, CellBlockRange range, std::span<const CellBlockIndex> so_far, BorderWidth width) {
  const std::expected<void, BufferError> written = WriteBuffer<CellBlockIndex>(so_far, rig.so_far_buffer, 0);
  EXPECT_TRUE(written.has_value());
  const std::expected<std::uint32_t, GridFailure> count = FindBorderBlocks(
      rig.grid,
      range,
      rig.so_far_buffer,
      static_cast<std::uint32_t>(so_far.size()),
      width,
      rig.search_buffers,
      rig.border_buffer
  );
  EXPECT_TRUE(count.has_value()) << (count.has_value() ? "" : ToName(count.error().error));
  if (!written.has_value() || !count.has_value() || *count > rig.border_buffer.size()) {
    return {};
  }
  std::vector<CellBlockIndex> layer(*count);
  EXPECT_TRUE(ReadBuffer<CellBlockIndex>(rig.border_buffer, 0, layer).has_value());
  return layer;
}

// The widths of the layers of a pass of four stages of widths (touching, touching, four widths, touching), found from
// the last stage backwards: layer 0 serves the last stage, layer 3 the first.
inline constexpr std::array<BorderWidth, 4> kLayerWidths = {
    BorderWidth::kTouchingCells,
    BorderWidth::kFourCellWidths,
    BorderWidth::kTouchingCells,
    BorderWidth::kTouchingCells,
};

// The widths of the two layers of the nested-ball check: touching, then four widths.
inline constexpr std::array<BorderWidth, 2> kNestedBallLayerWidths = {
    BorderWidth::kTouchingCells,
    BorderWidth::kFourCellWidths,
};

// The ranges a grid is tested on: the first range, the middle range and the last range of BuildRanges(grid,
// range_blocks), each once (fewer when the grid has fewer ranges), then the range of the whole grid. The first and the
// last lie at the edges of the grid, where readers meet the end of the tables, and the last is often shorter than the
// others. The middle range has a border on every side. The whole-grid range has an empty border. Testing every range
// would take minutes, and the ranges between these differ from the middle one only in position.
inline std::vector<CellBlockRange> BuildTestedRanges(const StaggeredGrid& grid, std::uint32_t range_blocks) {
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(grid, range_blocks);
  EXPECT_TRUE(ranges.has_value());
  if (!ranges.has_value() || ranges->empty()) {
    return {};
  }
  std::vector<CellBlockRange> tested_ranges;
  for (const std::size_t position : {std::size_t{0}, ranges->size() / 2, ranges->size() - 1}) {
    const CellBlockRange range = ranges->at(position);
    const bool is_listed = std::ranges::any_of(tested_ranges, [range](CellBlockRange listed) {
      return listed.first_block == range.first_block && listed.block_count == range.block_count;
    });
    if (!is_listed) {
      tested_ranges.push_back(range);
    }
  }
  const std::uint32_t block_count = grid.counts().cell_blocks;
  const bool whole_grid_is_listed = std::ranges::any_of(tested_ranges, [block_count](CellBlockRange listed) {
    return listed.first_block == 0 && listed.block_count == block_count;
  });
  if (!whole_grid_is_listed) {
    tested_ranges.push_back({.first_block = 0, .block_count = block_count});
  }
  return tested_ranges;
}

// The ranges of the nested-ball grid: BuildTestedRanges plus every fourth range of BuildRanges, which the dense
// layout of that grid makes worth the time.
inline std::vector<CellBlockRange> BuildNestedBallTestedRanges(const StaggeredGrid& grid, std::uint32_t range_blocks) {
  std::vector<CellBlockRange> tested_ranges = BuildTestedRanges(grid, range_blocks);
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(grid, range_blocks);
  EXPECT_TRUE(ranges.has_value());
  if (!ranges.has_value()) {
    return tested_ranges;
  }
  for (std::size_t position = 0; position < ranges->size(); position += 4) {
    const CellBlockRange range = ranges->at(position);
    const bool is_listed = std::ranges::any_of(tested_ranges, [range](CellBlockRange listed) {
      return listed.first_block == range.first_block && listed.block_count == range.block_count;
    });
    if (!is_listed) {
      tested_ranges.push_back(range);
    }
  }
  return tested_ranges;
}

}  // namespace fluidity::grid
