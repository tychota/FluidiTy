// Test-only: the grids, ranges and borders on which the grid pieces are tested, shared by the CPU and GPU tests.
//
// Purpose: give every test of grid pieces the same cases: random grids and the nested-ball grid, the ranges cut from
//   each, and for each range its near border and its stacked border, found on the CPU.
//
// Idea: a test of a piece builder compares it with a reference on the same cases, so the cases are written once. A grid
//   is cut into ranges (every range, or the first, middle, last and whole-grid ranges), and each range gets the border
//   of one stage at kTouchingCells (the first layer) and the border of two stages stacked, kTouchingCells then
//   kFourCellWidths around the range plus the first layer (both layers, ascending).
//
// Words:
//   stacked border  the layers of the borders of several stages, merged: the border of the last stage, then of the
//                   stage before around range plus that border, and so on
// Owns:    kEqualityGridCount, kReaderGridCount, kRangeBlocks, kNestedBallRangeBlocks, GridUnderTest,
//          FindNestedBallGrid, RangeChoice, BuildGridsUnderTest, StackedBorder, FindStackedBorder
// Uses:    CPU-only guard (base/cpu_only.h), StaggeredGrid (grid/staggered_grid.h), CellBlockIndex, CellBlockRange
//          (grid/grid_ids.h), BorderWidth (grid/border_block_steps.h), BuildRanges, ToRangeBlocks
//          (grid/cell_block_range.h and grid/testing/cell_block_range_cases.h), FindBorderOrFail
//          (grid/testing/border_block_cases.h), TestGrid, BuildTestGrid, FindRandomGridsWithCheckers,
//          FindRandomGridsSpreadBySize (grid/testing/grid_support.h), GridFailure (grid/grid_error.h),
//          BuildNestedBallCells, ToInputs, ToUnitFrame (grid/testing/staggered_grid_cases.h)
// Used by: tests/grid/grid_pieces_on_random_grids.test.cc, tests/grid/grid_piece_steps_in_every_thread_order.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iterator>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/border_block_cases.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {

// The random grids whose pieces are compared with the reference tables, the random grids whose pieces are asked the
// readers, and the blocks of a range of each kind of grid.
inline constexpr std::size_t kEqualityGridCount = 100;
inline constexpr std::size_t kReaderGridCount = 5;
inline constexpr std::uint32_t kRangeBlocks = 4;
inline constexpr std::uint32_t kNestedBallRangeBlocks = 16;

// Which ranges of a grid a test runs.
enum class RangeChoice : std::uint8_t {
  kTestedRanges,  // The first, middle, last and whole-grid ranges.
  kEveryRange,  // Every range of BuildRanges.
};

// A grid of the tests and the ranges it is cut at.
struct GridUnderTest {
  const StaggeredGrid* grid;
  std::vector<CellBlockRange> ranges;
  std::string name;
};

// The nested-ball grid, built once per program.
inline const TestGrid& FindNestedBallGrid() {
  static const TestGrid built = [] {
    const std::vector<CellId> cells = BuildNestedBallCells();
    std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(4), cells));
    EXPECT_TRUE(test_grid.has_value());
    return std::move(*test_grid);
  }();
  return built;
}

// The grids and their ranges: the first `random_count` random grids, then the nested-ball grid. Every range of
// BuildRanges, or the first, middle, last and whole-grid ranges, as `range_choice` says.
inline std::vector<GridUnderTest> BuildGridsUnderTest(std::size_t random_count, RangeChoice range_choice) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  EXPECT_GE(built.grids.size(), random_count);
  std::vector<GridUnderTest> grids;
  const auto find_ranges = [range_choice](const StaggeredGrid& grid, std::uint32_t range_blocks) {
    if (range_choice == RangeChoice::kEveryRange) {
      return BuildRanges(grid, range_blocks).value_or(std::vector<CellBlockRange>{});
    }
    return BuildTestedRanges(grid, range_blocks);
  };
  for (const std::size_t entry : FindRandomGridsSpreadBySize(random_count)) {
    const StaggeredGrid& grid = built.grids.at(entry).grid;
    grids.push_back(
        {.grid = &grid, .ranges = find_ranges(grid, kRangeBlocks), .name = "random grid " + std::to_string(entry)}
    );
  }
  const StaggeredGrid& ball = FindNestedBallGrid().grid;
  grids.push_back({.grid = &ball, .ranges = find_ranges(ball, kNestedBallRangeBlocks), .name = "nested-ball grid"});
  return grids;
}

// A range and the two layers of its stacked border: the layer at kTouchingCells around the range, then the layer at
// kFourCellWidths around the range plus the first layer.
struct StackedBorder {
  CellBlockRange range;
  std::vector<CellBlockIndex> first_layer;
  std::vector<CellBlockIndex> border;  // Both layers, ascending.
};

inline StackedBorder FindStackedBorder(const StaggeredGrid& grid, CellBlockRange range) {
  const std::vector<CellBlockIndex> range_blocks = ToRangeBlocks(range);
  StackedBorder stacked = {.range = range, .first_layer = {}, .border = {}};
  stacked.first_layer = FindBorderOrFail(grid, range_blocks, BorderWidth::kTouchingCells);
  const std::vector<CellBlockIndex> set_with_first_layer = BuildMergedBlocks(range_blocks, stacked.first_layer);
  const std::vector<CellBlockIndex> second_layer =
      FindBorderOrFail(grid, set_with_first_layer, BorderWidth::kFourCellWidths);
  stacked.border = BuildMergedBlocks(stacked.first_layer, second_layer);
  return stacked;
}

}  // namespace fluidity::grid
