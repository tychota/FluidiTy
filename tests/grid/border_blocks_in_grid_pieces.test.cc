#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/border_block_cases.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// The random grids whose pieces are checked: the first 10, each with its 4 chosen sets at both widths.
constexpr std::size_t kPieceGridCount = 10;

// The most blocks of a range of the nested-ball grid.
constexpr std::uint32_t kNestedBallRangeBlocks = 16;

// The nested-ball grid, built once per program.
const TestGrid& FindNestedBallGrid() {
  static const TestGrid built = [] {
    const std::vector<CellId> cells = BuildNestedBallCells();
    std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(4), cells));
    EXPECT_TRUE(test_grid.has_value());
    return std::move(*test_grid);
  }();
  return built;
}

// Checks that every reader of `width`, run from every place of `set`, answers in the piece of the set and its border
// as in the whole grid.
void CheckPieceAnswersAsWholeGrid(const StaggeredGrid& grid, std::span<const CellBlockIndex> set, BorderWidth width) {
  const std::vector<CellBlockIndex> border = FindBorderOrFail(grid, set, width);
  const std::optional<std::string> difference = FindFirstReaderDifference(grid.tables(RunOn::kCpu), set, border, width);
  EXPECT_FALSE(difference.has_value()) << difference.value_or("");
}

// In the piece made of a set and its border, every reader of the width, run from every place kept in the set, gives
// the answer it gives on the whole grid: the first 10 random grids, 4 sets each, both widths.
TEST(FindBorderBlocks, PiecesOfRandomGrids_AnswerAsTheWholeGrid) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kPieceGridCount);
  for (std::size_t grid = 0; grid < kPieceGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    const std::vector<std::vector<CellBlockIndex>> sets = FindChosenBlockSets(grid, staggered_grid);
    for (std::size_t set = 0; set < sets.size(); ++set) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set);
      for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
        CheckPieceAnswersAsWholeGrid(staggered_grid, sets.at(set), width);
      }
    }
  }
}

// The same on the nested-ball grid, for every range of a cut into ranges of 16 blocks, at both widths.
TEST(FindBorderBlocks, PiecesOfNestedBallRanges_AnswerAsTheWholeGrid) {
  const StaggeredGrid& grid = FindNestedBallGrid().grid;
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(grid, kNestedBallRangeBlocks);
  ASSERT_TRUE(ranges.has_value());
  for (const CellBlockRange range : *ranges) {
    SCOPED_TRACE(testing::Message() << "range from block " << range.first_block);
    for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
      CheckPieceAnswersAsWholeGrid(grid, ToRangeBlocks(range), width);
    }
  }
}

// The random grids whose borders are checked block by block for need, 3 sets of one block each.
constexpr std::size_t kMinimalGridCount = 10;

// Checks, for the border of `set` at `width`, that the piece without any one border block that `set` reads directly
// makes some reader differ. A border block that is only a bigger block of another may or may not be needed: the rule
// takes every bigger block (border_block_steps.h, ComputeIsBorderBlock). Returns how many border blocks the readers
// do not need, all of them bigger blocks off every walk.
int CheckEveryDirectBorderBlockIsNeeded(
    const StaggeredGrid& grid,
    const GeometricReadBlocks& read_blocks,
    std::span<const CellBlockIndex> set,
    BorderWidth width
) {
  const std::vector<CellBlockIndex> border = FindBorderOrFail(grid, set, width);
  std::set<std::uint32_t> direct_blocks;
  for (const CellBlockIndex block : FindDirectReadsGeometrically(read_blocks, set, width)) {
    direct_blocks.insert(block.value);
  }
  int unneeded_count = 0;
  for (std::size_t left_out = 0; left_out < border.size(); ++left_out) {
    std::vector<CellBlockIndex> smaller_border = border;
    smaller_border.erase(smaller_border.begin() + static_cast<std::ptrdiff_t>(left_out));
    const bool does_reader_differ =
        FindFirstReaderDifference(grid.tables(RunOn::kCpu), set, smaller_border, width).has_value();
    if (direct_blocks.contains(border.at(left_out).value)) {
      EXPECT_TRUE(does_reader_differ) << "border block " << border.at(left_out).value << " is not needed";
    } else if (!does_reader_differ) {
      ++unneeded_count;
    }
  }
  return unneeded_count;
}

// Every border block that a set reads directly is needed: the piece without it makes some reader differ. The sets are
// the first, middle and last block alone of the first 10 random grids, and the size-level-1 block of the split-corner
// grid, at both widths. Some bigger blocks are not needed, and the count shows that the extra bigger blocks the rule
// takes are few.
TEST(FindBorderBlocks, OneDirectBlockLeftOut_MakesSomeReaderDiffer) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kMinimalGridCount);
  int unneeded_count = 0;
  int border_block_count = 0;
  for (std::size_t grid = 0; grid < kMinimalGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    const GeometricReadBlocks read_blocks =
        BuildGeometricReadBlocks(built.checkers.at(grid), staggered_grid.tables(RunOn::kCpu));
    const std::uint32_t block_count = staggered_grid.counts().cell_blocks;
    for (const std::uint32_t block : {0U, block_count / 2, block_count - 1}) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", block " << block);
      const std::vector<CellBlockIndex> set = {{.value = block}};
      for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
        unneeded_count += CheckEveryDirectBorderBlockIsNeeded(staggered_grid, read_blocks, set, width);
        border_block_count += static_cast<int>(FindBorderOrFail(staggered_grid, set, width).size());
      }
    }
  }
  const std::vector<CellId> cells = BuildSplitCornerCells();
  const std::expected<TestGrid, GridFailure> split_corner = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(split_corner.has_value());
  const GeometricReadBlocks read_blocks =
      BuildGeometricReadBlocks(GeometricGrid(ToUnitFrame(2), cells), split_corner->grid.tables(RunOn::kCpu));
  for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
    EXPECT_EQ(CheckEveryDirectBorderBlockIsNeeded(split_corner->grid, read_blocks, ToBlocks({1}), width), 0);
  }
  // A few unneeded bigger blocks at most: fewer than one border block in ten.
  EXPECT_LT(10 * unneeded_count, border_block_count);
}

}  // namespace
}  // namespace fluidity::grid
