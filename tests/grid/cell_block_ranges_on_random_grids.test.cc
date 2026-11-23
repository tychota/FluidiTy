#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/border_block_cases.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

bool ComputeIsAscendingWithoutRepeat(std::span<const CellBlockIndex> blocks) {
  return std::ranges::adjacent_find(
             blocks,
             [](CellBlockIndex first, CellBlockIndex second) {
               return first.value >= second.value;
             }
         )
      == blocks.end();
}
// The cut of a grid into ranges of at most max_blocks_per_range blocks. It fails the calling test when the call fails.
std::vector<CellBlockRange> BuildRangesOrFail(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range) {
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(grid, max_blocks_per_range);
  EXPECT_TRUE(ranges.has_value());
  return ranges.has_value() ? *ranges : std::vector<CellBlockRange>{};
}
// The random grids on which the two forms are compared: their chosen sets, and every block alone of the first ones.
constexpr std::size_t kTwoFormsGridCount = 60;
constexpr std::size_t kBlockAloneGridCount = 10;

// The border of a set by the form that asks each block from its own side (ComputeIsBorderBlock): the form of
// FindBorderBlocks that takes an output span, which has no working memory. It fails the calling test when the call
// fails or the span is too short.
std::vector<CellBlockIndex>
FindBorderBlockByBlockOrFail(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  std::vector<CellBlockIndex> border(grid.counts().cell_blocks);
  const std::expected<std::uint32_t, GridFailure> count = FindBorderBlocks(grid, blocks, width, border);
  EXPECT_TRUE(count.has_value());
  const std::uint32_t border_count = count.value_or(0);
  EXPECT_LE(border_count, border.size());
  border.resize(std::min<std::size_t>(border_count, border.size()));
  return border;
}

// Checks that the forward form (FindBorderByReadMarks) gives, for each chosen set of random grid `grid`, the borders
// the form that asks each block from its own side finds.
void CheckForwardFormOnChosenSets(std::size_t grid) {
  const GridTables tables = FindRandomGridsWithCheckers().grids.at(grid).grid.tables(RunOn::kCpu);
  const GridBorderCase& grid_case = FindGridBorderCase(grid);
  for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
    const std::vector<CellBlockIndex>& set = grid_case.sets.at(set_number);
    const StaggeredGrid& staggered_grid = FindRandomGridsWithCheckers().grids.at(grid).grid;
    for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
      EXPECT_EQ(
          ToBlockNumbers(FindBorderByReadMarks(tables, set, width)),
          ToBlockNumbers(FindBorderBlockByBlockOrFail(staggered_grid, set, width))
      );
    }
  }
}

// Checks that the forward form and the form that asks each block from its own side give the same border for every
// block alone of random grid `grid`.
void CheckForwardFormOnEveryBlockAlone(std::size_t grid) {
  const StaggeredGrid& staggered_grid = FindRandomGridsWithCheckers().grids.at(grid).grid;
  const GridTables tables = staggered_grid.tables(RunOn::kCpu);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", block " << block);
    const std::vector<CellBlockIndex> set = {{.value = block}};
    for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
      EXPECT_EQ(
          ToBlockNumbers(FindBorderByReadMarks(tables, set, width)),
          ToBlockNumbers(FindBorderBlockByBlockOrFail(staggered_grid, set, width))
      );
    }
  }
}

// The form that asks each block from its own side (FindBorderBlocks with an output span, through
// ComputeIsBorderBlock) and the forward form, in which every place of the set marks what its readers read
// (FindBorderByReadMarks, through ApplyCellPlaceReadMarks and ApplyFacePlaceReadMarks), give the same border: for
// every chosen set of the first 60 random grids, and for every block alone of the first 10, at both widths.
TEST(FindBorderBlocks, BlockByBlockForm_EqualsForwardForm) {
  for (std::size_t grid = 0; grid < kTwoFormsGridCount; ++grid) {
    CheckForwardFormOnChosenSets(grid);
  }
  for (std::size_t grid = 0; grid < kBlockAloneGridCount; ++grid) {
    CheckForwardFormOnEveryBlockAlone(grid);
  }
}

// Checks that a border is ascending, repeats no block and holds no block of `set`.
void CheckBorderIsAscendingOutsideSet(const BlockNumbers& set, const std::vector<CellBlockIndex>& border) {
  EXPECT_TRUE(ComputeIsAscendingWithoutRepeat(border));
  EXPECT_TRUE(std::ranges::none_of(border, [&set](CellBlockIndex block) {
    return set.contains(block.value);
  }));
}

TEST(FindBorderBlocks, RandomSets_GiveAscendingBlocksOutsideTheSet) {
  for (std::size_t grid = 0; grid < kTwoFormsGridCount; ++grid) {
    const GridBorderCase& grid_case = FindGridBorderCase(grid);
    for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
      const BlockNumbers set = ToBlockNumbers(grid_case.sets.at(set_number));
      CheckBorderIsAscendingOutsideSet(set, grid_case.touching_borders.at(set_number));
      CheckBorderIsAscendingOutsideSet(set, grid_case.wide_borders.at(set_number));
    }
  }
}

// The random grids, of the first ones, whose ranges are checked through pieces, and the blocks of one range.
constexpr std::size_t kRangePieceGridCount = 6;
constexpr std::uint32_t kRangePieceBlocks = 3;

// Checks that the piece of one range and its border (BuildRangeWithBorder) answers every reader of kTouchingCells, from
// every place of the range, as the whole grid does.
void CheckPieceOfRangeAnswersAsWholeGrid(const StaggeredGrid& grid, CellBlockRange range) {
  const RangeWithBorder range_with_border = BuildRangeWithBorder(grid, range);
  const std::optional<std::string> difference = FindFirstReaderDifference(
      grid.tables(RunOn::kCpu),
      ToRangeBlocks(range),
      range_with_border.border_blocks,
      BorderWidth::kTouchingCells
  );
  EXPECT_FALSE(difference.has_value()) << difference.value_or("");
}

// In the piece of a range and its border, every reader of kTouchingCells, run from every place kept in the range,
// answers as on the whole grid, including what transfer needs: every range of 3 blocks of the first 6 random grids.
TEST(BuildRangeWithBorder, PiecesOfRandomGridRanges_AnswerAsTheWholeGrid) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kRangePieceGridCount);
  for (std::size_t grid = 0; grid < kRangePieceGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    for (const CellBlockRange range : BuildRangesOrFail(staggered_grid, kRangePieceBlocks)) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
      CheckPieceOfRangeAnswersAsWholeGrid(staggered_grid, range);
    }
  }
}

// The widths of the four stages of a pass, from the first stage run to the last.
constexpr std::array<BorderWidth, 4> kStageWidths = {
    BorderWidth::kTouchingCells,
    BorderWidth::kTouchingCells,
    BorderWidth::kFourCellWidths,
    BorderWidth::kTouchingCells,
};

// The random grids and range size of the stacked check.
constexpr std::size_t kStackedGridCount = 5;
constexpr std::uint32_t kStackedRangeBlocks = 4;

// Finds the layers of stacked borders around `range`, from the last stage backwards: layer 0 is the border of the range
// at the last stage's width, layer 1 the border of the range plus layer 0 at the width of the stage before, and so on.
std::vector<BlockNumbers> FindStackedLayers(const StaggeredGrid& grid, CellBlockRange range) {
  std::vector<BlockNumbers> layers;
  BlockNumbers loaded = ToBlockNumbers(ToRangeBlocks(range));
  for (std::size_t layer = 0; layer < kStageWidths.size(); ++layer) {
    const BorderWidth width = kStageWidths.at(kStageWidths.size() - 1 - layer);
    layers.push_back(ToBlockNumbers(FindBorderOrFail(grid, ToBlockList(loaded), width)));
    loaded.insert(layers.back().begin(), layers.back().end());
  }
  return layers;
}

// Checks that the piece of the range and every layer answers, for each layer k, every reader of the width of layer k
// from every place kept in the range and the layers before k. Layer k holds what the stage that writes those places
// reads at that width: the last stage writes the range, and each stage before it writes what the one after it reads.
void CheckStackedLayersAnswerEveryStage(const StaggeredGrid& grid, CellBlockRange range) {
  const std::vector<BlockNumbers> layers = FindStackedLayers(grid, range);
  BlockNumbers written = ToBlockNumbers(ToRangeBlocks(range));
  for (std::size_t layer = 0; layer < layers.size(); ++layer) {
    BlockNumbers border;
    for (std::size_t later_layer = layer; later_layer < layers.size(); ++later_layer) {
      border.insert(layers.at(later_layer).begin(), layers.at(later_layer).end());
    }
    const BorderWidth width = kStageWidths.at(kStageWidths.size() - 1 - layer);
    SCOPED_TRACE(testing::Message() << "layer " << layer);
    const std::optional<std::string> difference =
        FindFirstReaderDifference(grid.tables(RunOn::kCpu), ToBlockList(written), ToBlockList(border), width);
    EXPECT_FALSE(difference.has_value()) << difference.value_or("");
    written.insert(layers.at(layer).begin(), layers.at(layer).end());
  }
}

// A pass of four stages of widths (touching, touching, four widths, touching), on the middle range of 4 blocks of each
// of the first 5 random grids. The layers are found from the last stage backwards (FindStackedLayers). The piece of the
// range and all four layers must answer every reader of layer k's width from every place of the range and layers 0 to k
// - 1: layer 0 serves the last stage, which writes the range and reads at the last stage's width; layer 1 serves the
// stage before, which writes the range and layer 0, and so on. A layer that missed a block a reader walks through would
// fail here, though every place the readers return lay in the layers.
TEST(FindBorderBlocks, StackedLayersOfRandomGrids_AnswerEveryStage) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kStackedGridCount);
  for (std::size_t grid = 0; grid < kStackedGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    const std::vector<CellBlockRange> ranges = BuildRangesOrFail(staggered_grid, kStackedRangeBlocks);
    ASSERT_FALSE(ranges.empty());
    const CellBlockRange range = ranges.at(ranges.size() / 2);
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
    CheckStackedLayersAnswerEveryStage(staggered_grid, range);
  }
}

}  // namespace
}  // namespace fluidity::grid
