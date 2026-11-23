// Test-only: the block sets of the random grids and their borders, which the functional tests of borders share.
//
// Purpose: let the tests that compare borders with the geometry, the tests that compare the forms of the search and the
//   tests that build pieces find the same sets and borders of the random grids, written once.
//
// Idea: each random grid has four chosen block sets (FindChosenBlockSets). Their borders at both widths are found once
//   per test program, when a test first asks for the grid, since every test of a program asks for them.
//
// Owns:    BlockNumbers, ToBlockNumbers, ToBlockList, FindBorderOrFail, GridBorderCase, FindGridBorderCase
// Uses:    CPU-only guard (base/cpu_only.h), FindBorderBlocks, BorderWidth (grid/cell_block_range.h),
//          FindChosenBlockSets (grid/testing/cell_block_range_cases.h), FindRandomGridsWithCheckers
//          (grid/testing/grid_support.h), StaggeredGrid (grid/staggered_grid.h), GridFailure (grid/grid_error.h),
//          CellBlockIndex (grid/grid_ids.h)
// Used by: the functional tests of border blocks (tests/grid/)
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

using BlockNumbers = std::set<std::uint32_t>;

// The block numbers of a list of blocks, each once.
inline BlockNumbers ToBlockNumbers(std::span<const CellBlockIndex> blocks) {
  BlockNumbers numbers;
  for (const CellBlockIndex block : blocks) {
    numbers.insert(block.value);
  }
  return numbers;
}

// The blocks of a set of block numbers, ascending.
inline std::vector<CellBlockIndex> ToBlockList(const BlockNumbers& numbers) {
  std::vector<CellBlockIndex> blocks;
  for (const std::uint32_t number : numbers) {
    blocks.push_back({.value = number});
  }
  return blocks;
}

// The border of a set, which every check asks for. It fails the calling test when the call fails.
inline std::vector<CellBlockIndex>
FindBorderOrFail(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border = FindBorderBlocks(grid, blocks, width);
  EXPECT_TRUE(border.has_value());
  return border.has_value() ? *border : std::vector<CellBlockIndex>{};
}

// The chosen sets of one random grid and their borders at both widths.
struct GridBorderCase {
  std::vector<std::vector<CellBlockIndex>> sets;
  std::vector<std::vector<CellBlockIndex>> touching_borders;  // FindBorderBlocks of each set at kTouchingCells.
  std::vector<std::vector<CellBlockIndex>> wide_borders;  // FindBorderBlocks of each set at kFourCellWidths.
};

// The sets and borders of random grid `grid`, found the first time they are asked for and kept for the program.
inline const GridBorderCase& FindGridBorderCase(std::size_t grid) {
  static std::vector<std::optional<GridBorderCase>> cases(kRandomGridCount);
  std::optional<GridBorderCase>& grid_case = cases.at(grid);
  if (!grid_case.has_value()) {
    const StaggeredGrid& staggered_grid = FindRandomGridsWithCheckers().grids.at(grid).grid;
    grid_case = GridBorderCase{
        .sets = FindChosenBlockSets(grid, staggered_grid),
        .touching_borders = {},
        .wide_borders = {},
    };
    for (const std::vector<CellBlockIndex>& blocks : grid_case->sets) {
      grid_case->touching_borders.push_back(FindBorderOrFail(staggered_grid, blocks, BorderWidth::kTouchingCells));
      grid_case->wide_borders.push_back(FindBorderOrFail(staggered_grid, blocks, BorderWidth::kFourCellWidths));
    }
  }
  return *grid_case;
}

}  // namespace fluidity::grid
