#include <gtest/gtest.h>

#include <algorithm>
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

// The border of a set by the form that asks each block from its own side: ComputeIsBorderBlock for every block of the
// grid, in ascending index. It has no working memory and costs every block of the grid and every block under it.
std::vector<CellBlockIndex>
FindBorderBlockByBlock(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  const GridTables tables = grid.tables(RunOn::kCpu);
  const BlockSet set = {
      .range = {.first_block = 0, .block_count = 0},
      .sorted_blocks = blocks.data(),
      .sorted_count = static_cast<std::uint32_t>(blocks.size()),
  };
  std::vector<CellBlockIndex> border;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    if (ComputeIsBorderBlock(tables, set, CellBlockIndex{block}, width)) {
      border.push_back(CellBlockIndex{block});
    }
  }
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
          ToBlockNumbers(FindBorderBlockByBlock(staggered_grid, set, width))
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
          ToBlockNumbers(FindBorderBlockByBlock(staggered_grid, set, width))
      );
    }
  }
}

// The form that asks each block from its own side (ComputeIsBorderBlock for every block) and the forward form, in which
// every place of the set marks what its readers read (FindBorderByReadMarks, through ApplyCellPlaceReadMarks and
// ApplyFacePlaceReadMarks), give the same border: for every chosen set of the first 60 random grids, and for every
// block alone of the first 10, at both widths.
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
constexpr std::size_t kRangePieceGridCount = 16;
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
// answers as on the whole grid, including what transfer needs: every range of 3 blocks of the first 16 random grids.
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

}  // namespace
}  // namespace fluidity::grid
