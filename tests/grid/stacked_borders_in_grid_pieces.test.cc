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

// The widths of the layers of a pass of four stages of widths (touching, touching, four widths, touching), found from
// the last stage backwards: layer 0 serves the last stage, layer 3 the first.
constexpr std::array<BorderWidth, 4> kLayerWidths = {
    BorderWidth::kTouchingCells,
    BorderWidth::kFourCellWidths,
    BorderWidth::kTouchingCells,
    BorderWidth::kTouchingCells,
};

// The widths of the two layers of the nested-ball check: touching, then four widths.
constexpr std::array<BorderWidth, 2> kNestedBallLayerWidths = {
    BorderWidth::kTouchingCells,
    BorderWidth::kFourCellWidths,
};

// The random grids compared with the vector form, the random grids checked through pieces, and the blocks of a range.
constexpr std::size_t kVectorFormGridCount = 30;
constexpr std::size_t kPieceGridCount = 2;
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

// The blocks of two ascending lists that share no block, in ascending order.
std::vector<CellBlockIndex>
BuildMergedBlocks(std::span<const CellBlockIndex> first, std::span<const CellBlockIndex> second) {
  std::vector<CellBlockIndex> merged;
  std::ranges::merge(first, second, std::back_inserter(merged), {}, &CellBlockIndex::value, &CellBlockIndex::value);
  return merged;
}

std::vector<std::uint32_t> ToBlockNumberList(std::span<const CellBlockIndex> blocks) {
  std::vector<std::uint32_t> numbers;
  for (const CellBlockIndex block : blocks) {
    numbers.push_back(block.value);
  }
  return numbers;
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

TEST(FindBorderBlocks, StackedBufferForm_EqualsVectorFormLayerByLayer) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kVectorFormGridCount);
  for (std::size_t grid = 0; grid < kVectorFormGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(staggered_grid);
    ASSERT_TRUE(rig.has_value());
    const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(staggered_grid, kRangeBlocks);
    ASSERT_TRUE(ranges.has_value());
    ASSERT_FALSE(ranges->empty());
    const CellBlockRange range = ranges->at(ranges->size() / 2);
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
    CheckBufferFormEqualsVectorForm(*rig, range);
  }
}

// Checks, for each layer j of the stacked border of `range` at `layer_widths`, that the piece of the range and the
// layers before it answers every reader of layer j's width from every place of the range and those layers. Layer j is
// the buffer form's list B(j) without B(j - 1): the blocks the stage adds.
void CheckLayersAnswerAsWholeGrid(
    BorderSearchRig& rig,
    CellBlockRange range,
    std::span<const BorderWidth> layer_widths
) {
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
    const std::optional<std::string> difference =
        FindFirstReaderDifference(rig.grid.tables(RunOn::kCpu), written_blocks, layer_blocks, width);
    EXPECT_FALSE(difference.has_value()) << difference.value_or("");
    border_so_far = outer_border;
  }
}

// A pass of four stages, on the middle range of 4 blocks of each of the first 2 random grids. A layer that missed a
// block a reader walks through would fail here, though every place the readers return lay in the layers.
TEST(FindBorderBlocks, StackedLayersOfRandomGrids_AnswerAsTheWholeGrid) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kPieceGridCount);
  for (std::size_t grid = 0; grid < kPieceGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(staggered_grid);
    ASSERT_TRUE(rig.has_value());
    const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(staggered_grid, kRangeBlocks);
    ASSERT_TRUE(ranges.has_value());
    ASSERT_FALSE(ranges->empty());
    const CellBlockRange range = ranges->at(ranges->size() / 2);
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
    CheckLayersAnswerAsWholeGrid(*rig, range, kLayerWidths);
  }
}

// Two layers (touching, four widths) on the middle range of 16 blocks of the nested-ball grid.
TEST(FindBorderBlocks, StackedLayersOfNestedBallRanges_AnswerAsTheWholeGrid) {
  const StaggeredGrid& grid = FindNestedBallGrid().grid;
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid);
  ASSERT_TRUE(rig.has_value());
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(grid, kNestedBallRangeBlocks);
  ASSERT_TRUE(ranges.has_value());
  ASSERT_FALSE(ranges->empty());
  const CellBlockRange range = ranges->at(ranges->size() / 2);
  SCOPED_TRACE(testing::Message() << "range from block " << range.first_block);
  CheckLayersAnswerAsWholeGrid(*rig, range, kNestedBallLayerWidths);
}

}  // namespace
}  // namespace fluidity::grid
