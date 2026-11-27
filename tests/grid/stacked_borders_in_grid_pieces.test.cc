#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/border_block_cases.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// The random grids compared with the vector form, the random grids checked through pieces, and the blocks of a range.
constexpr std::size_t kVectorFormGridCount = 40;
constexpr std::size_t kPieceGridCount = 5;
constexpr std::uint32_t kRangeBlocks = 4;
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

std::vector<std::uint32_t> ToBlockNumberList(std::span<const CellBlockIndex> blocks) {
  std::vector<std::uint32_t> numbers;
  for (const CellBlockIndex block : blocks) {
    numbers.push_back(block.value);
  }
  return numbers;
}

// The blocks of `kept` and of every `step`-th block of the grid outside `range`, ascending. A step of 1 gives every
// block outside the range.
std::vector<CellBlockIndex> BuildBlocksWithEveryStepth(
    std::span<const CellBlockIndex> kept,
    CellBlockRange range,
    const StaggeredGrid& grid,
    std::uint32_t step
) {
  const std::uint32_t block_count = grid.counts().cell_blocks;
  std::vector<CellBlockIndex> chosen;
  for (std::uint32_t block = 0; block < block_count; block += step) {
    const bool is_in_range = block >= range.first_block && block - range.first_block < range.block_count;
    if (!is_in_range) {
      chosen.push_back({.value = block});
    }
  }
  std::vector<CellBlockIndex> blocks;
  std::ranges::set_union(kept, chosen, std::back_inserter(blocks), {}, &CellBlockIndex::value, &CellBlockIndex::value);
  return blocks;
}

// Checks that, at each layer of a stacked border of `range`, the buffer form gives the merge of the layers so far and
// the vector form on the range plus the layers so far.
void CheckBufferFormEqualsVectorForm(BorderSearchRig& rig, CellBlockRange range) {
  std::vector<CellBlockIndex> border_so_far;
  for (std::size_t layer = 0; layer < kLayerWidths.size(); ++layer) {
    SCOPED_TRACE(testing::Message() << "layer " << layer);
    const BorderWidth width = kLayerWidths.at(layer);
    const std::vector<CellBlockIndex> buffer_border = FindLayerOrFail(rig, range, border_so_far, width);
    const std::vector<CellBlockIndex> loaded_blocks = BuildMergedBlocks(ToRangeBlocks(range), border_so_far);
    const std::vector<CellBlockIndex> vector_border = FindBorderOrFail(rig.grid, loaded_blocks, width);
    EXPECT_EQ(ToBlockNumberList(buffer_border), ToBlockNumberList(BuildMergedBlocks(border_so_far, vector_border)));
    border_so_far = buffer_border;
  }
}

// The tested ranges of each of 40 random grids, at four layers.
TEST(FindBorderBlocks, StackedBufferForm_EqualsVectorFormLayerByLayer) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kVectorFormGridCount);
  for (std::size_t grid = 0; grid < kVectorFormGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(staggered_grid);
    ASSERT_TRUE(rig.has_value());
    for (const CellBlockRange range : BuildTestedRanges(staggered_grid, kRangeBlocks)) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
      CheckBufferFormEqualsVectorForm(*rig, range);
    }
  }
}

TEST(FindBorderBlocks, StackedLayersOfRandomGrids_AnswerAsTheWholeGrid) {
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kPieceGridCount);
  CheckStackedLayersAnswerAsTheWholeGrid(FindRandomGridsWithCheckers(), grids, kRangeBlocks);
}

// Two layers (touching, four widths) on the tested ranges of 16 blocks of the nested-ball grid.
TEST(FindBorderBlocks, StackedLayersOfNestedBallRanges_AnswerAsTheWholeGrid) {
  const StaggeredGrid& grid = FindNestedBallGrid().grid;
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid);
  ASSERT_TRUE(rig.has_value());
  for (const CellBlockRange range : BuildNestedBallTestedRanges(grid, kNestedBallRangeBlocks)) {
    SCOPED_TRACE(testing::Message() << "range from block " << range.first_block);
    CheckLayersAnswerAsWholeGrid(*rig, range, kNestedBallLayerWidths);
  }
}

// Checks, for sets A inside B that both hold the border of `range` at `width` (A is that border, B adds blocks), that
// every reader from every place of the range answers the same in the piece of B as in the piece of A, which answers as
// the whole grid. B adds the next layer at four widths, every third block outside the range, or every block outside it.
void CheckMorePiecesChangeNoAnswer(BorderSearchRig& rig, CellBlockRange range, BorderWidth width) {
  const GridTables whole = rig.grid.tables(RunOn::kCpu);
  const std::vector<CellBlockIndex> no_blocks;
  const std::vector<CellBlockIndex> set_a = FindLayerOrFail(rig, range, no_blocks, width);
  const std::vector<CellBlockIndex> range_blocks = ToRangeBlocks(range);
  const std::optional<std::string> difference_in_a = FindFirstReaderDifference(whole, range_blocks, set_a, width);
  EXPECT_FALSE(difference_in_a.has_value()) << difference_in_a.value_or("");
  const std::array<std::vector<CellBlockIndex>, 3> sets_b = {
      FindLayerOrFail(rig, range, set_a, BorderWidth::kFourCellWidths),
      BuildBlocksWithEveryStepth(set_a, range, rig.grid, 3),
      BuildBlocksWithEveryStepth(set_a, range, rig.grid, 1),
  };
  for (std::size_t variant = 0; variant < sets_b.size(); ++variant) {
    SCOPED_TRACE(testing::Message() << "set B variant " << variant);
    const std::vector<CellBlockIndex>& set_b = sets_b.at(variant);
    EXPECT_TRUE(std::ranges::includes(set_b, set_a, {}, &CellBlockIndex::value, &CellBlockIndex::value));
    const std::optional<std::string> difference_in_b = FindFirstReaderDifference(whole, range_blocks, set_b, width);
    EXPECT_FALSE(difference_in_b.has_value()) << difference_in_b.value_or("");
  }
}

// The property that CheckLayersAnswerAsWholeGrid relies on, on the tested ranges of 5 random grids spread by size, at
// both widths: a piece that already holds the range's border at a width answers every reader of that width from the
// range as the whole grid, and so does every bigger piece.
TEST(FindBorderBlocks, BiggerPieceAfterTheBorder_GivesTheSameAnswers) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  for (const std::size_t grid : FindRandomGridsSpreadBySize(kPieceGridCount)) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(staggered_grid);
    ASSERT_TRUE(rig.has_value());
    for (const CellBlockRange range : BuildTestedRanges(staggered_grid, kRangeBlocks)) {
      for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
        SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
        CheckMorePiecesChangeNoAnswer(*rig, range, width);
      }
    }
  }
}

}  // namespace
}  // namespace fluidity::grid
