// Test-only: the block sets of the random grids and their borders, which the functional tests of borders share.
//
// Purpose: let the tests that compare borders with the geometry, the tests that compare the forms of the search and the
//   tests that build pieces find the same sets and borders of the random grids, written once.
//
// Idea: each random grid has four chosen block sets (FindChosenBlockSets). Their borders at both widths are found once
//   per test program, when a test first asks for the grid, since every test of a program asks for them.
//
// Owns:    BlockNumbers, ToBlockNumbers, ToBlockList, FindBorderOrFail, GridBorderCase, FindGridBorderCase,
//          BuildMergedBlocks, CheckLayersAnswerAsWholeGrid, CheckStackedLayersAnswerAsTheWholeGrid
// Uses:    CPU-only guard (base/cpu_only.h), the border search steps (grid/border_block_steps.h), FindBorderBlocks,
//          BorderWidth (grid/cell_block_range.h),
//          FindChosenBlockSets (grid/testing/cell_block_range_cases.h), FindRandomGridsWithCheckers
//          (grid/testing/grid_support.h), StaggeredGrid (grid/staggered_grid.h), GridFailure (grid/grid_error.h),
//          CellBlockIndex (grid/grid_ids.h), FindFirstReaderDifference (grid/testing/grid_piece_support.h)
// Used by: the functional tests of border blocks (tests/grid/)
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
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

// The sets and borders of grid `grid` of the set `built`, found the first time they are asked for and kept for the
// program, one list of cases for each set.
inline const GridBorderCase& FindGridBorderCase(const RandomGridsWithCheckers& built, std::size_t grid) {
  static std::map<const RandomGridsWithCheckers*, std::vector<std::optional<GridBorderCase>>> cases_of_sets;
  std::vector<std::optional<GridBorderCase>>& cases = cases_of_sets[&built];
  if (cases.empty()) {
    cases.resize(built.grids.size());
  }
  std::optional<GridBorderCase>& grid_case = cases.at(grid);
  if (!grid_case.has_value()) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
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

// The sets and borders of random grid `grid` of the first 200 random grids.
inline const GridBorderCase& FindGridBorderCase(std::size_t grid) {
  return FindGridBorderCase(FindRandomGridsWithCheckers(), grid);
}

// The blocks of two ascending lists that share no block, in ascending order.
inline std::vector<CellBlockIndex>
BuildMergedBlocks(std::span<const CellBlockIndex> first, std::span<const CellBlockIndex> second) {
  std::vector<CellBlockIndex> merged;
  std::ranges::merge(first, second, std::back_inserter(merged), {}, &CellBlockIndex::value, &CellBlockIndex::value);
  return merged;
}

// Checks the stacked border of `range` at `layer_widths` in two kinds of piece. Layer j is the buffer form's list B(j)
// without B(j - 1): the blocks the stage adds, B(-1) being empty.
//   - Layer by layer: the piece of the range and B(j) answers every reader of layer j's width from every place of the
//     range and B(j - 1), as the whole grid does.
//   - The piece a pass loads: the range and B(last). Every stage's width, from every place of the range and the layers
//     inside it, is asked in that piece.
// The layer-by-layer check implies the check of the full piece, because of this property: loading more blocks never
// changes a reader's answer for a place whose reads were already all loaded. A reader that touched only loaded blocks
// reads the same entries in a bigger piece, and a block that was loaded stays loaded. The property is tested on its
// own, on random grids, by CheckMorePiecesChangeNoAnswer. The full-piece check is made here as well, so the piece the
// pass really loads is never taken on trust.
inline void
CheckLayersAnswerAsWholeGrid(BorderSearchRig& rig, CellBlockRange range, std::span<const BorderWidth> layer_widths) {
  const GridTables whole = rig.grid.tables(RunOn::kCpu);
  std::vector<std::vector<CellBlockIndex>> borders_so_far;
  std::vector<CellBlockIndex> border_so_far;
  for (std::size_t layer = 0; layer < layer_widths.size(); ++layer) {
    SCOPED_TRACE(testing::Message() << "layer " << layer);
    const BorderWidth width = layer_widths[layer];
    const std::vector<CellBlockIndex> outer_border = FindLayerOrFail(rig, range, border_so_far, width);
    EXPECT_TRUE(std::ranges::includes(outer_border, border_so_far, {}, &CellBlockIndex::value, &CellBlockIndex::value));
    std::vector<CellBlockIndex> layer_blocks;
    std::ranges::set_difference(
        outer_border,
        border_so_far,
        std::back_inserter(layer_blocks),
        {},
        &CellBlockIndex::value,
        &CellBlockIndex::value
    );
    const std::vector<CellBlockIndex> written_blocks = BuildMergedBlocks(ToRangeBlocks(range), border_so_far);
    const std::optional<std::string> difference = FindFirstReaderDifference(whole, written_blocks, layer_blocks, width);
    EXPECT_FALSE(difference.has_value()) << difference.value_or("");
    borders_so_far.push_back(outer_border);
    border_so_far = outer_border;
  }
  if (borders_so_far.empty()) {
    return;
  }
  const std::vector<CellBlockIndex>& loaded_border = borders_so_far.back();
  for (std::size_t stage = 0; stage < layer_widths.size(); ++stage) {
    SCOPED_TRACE(testing::Message() << "stage " << stage << " in the full piece");
    const std::vector<CellBlockIndex> inner_border =
        stage == 0 ? std::vector<CellBlockIndex>{} : borders_so_far[stage - 1];
    std::vector<CellBlockIndex> outer_blocks;
    std::ranges::set_difference(
        loaded_border,
        inner_border,
        std::back_inserter(outer_blocks),
        {},
        &CellBlockIndex::value,
        &CellBlockIndex::value
    );
    const std::vector<CellBlockIndex> written_blocks = BuildMergedBlocks(ToRangeBlocks(range), inner_border);
    const std::optional<std::string> difference =
        FindFirstReaderDifference(whole, written_blocks, outer_blocks, layer_widths[stage]);
    EXPECT_FALSE(difference.has_value()) << difference.value_or("");
  }
}

// A pass of four stages, on the tested ranges of 4 blocks of 5 random grids spread by size, from the
// grid with the most cells to the one with the fewest. A layer that missed a block a reader walks through would fail
// here, though every place the readers return lay in the layers.
inline void CheckStackedLayersAnswerAsTheWholeGrid(
    const RandomGridsWithCheckers& built,
    std::span<const std::size_t> grids,
    std::uint32_t range_blocks
) {
  for (const std::size_t grid : grids) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(staggered_grid);
    ASSERT_TRUE(rig.has_value());
    for (const CellBlockRange range : BuildTestedRanges(staggered_grid, range_blocks)) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
      CheckLayersAnswerAsWholeGrid(*rig, range, kLayerWidths);
    }
  }
}

}  // namespace fluidity::grid
