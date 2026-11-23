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
#include "fluidity/grid/grid_tables.h"
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

// The ranges a grid is tested on: the first range, the middle range and the last range of BuildRanges(grid,
// range_blocks), each once (fewer when the grid has fewer ranges), then the range of the whole grid. The first and the
// last lie at the edges of the grid, where readers meet the end of the tables, and the last is often shorter than the
// others. The middle range has a border on every side. The whole-grid range has an empty border. Testing every range
// would take minutes, and the ranges between these differ from the middle one only in position.
std::vector<CellBlockRange> BuildTestedRanges(const StaggeredGrid& grid, std::uint32_t range_blocks) {
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
std::vector<CellBlockRange> BuildNestedBallTestedRanges(const StaggeredGrid& grid, std::uint32_t range_blocks) {
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

// The tested ranges of each of 30 random grids, at four layers.
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
void CheckLayersAnswerAsWholeGrid(
    BorderSearchRig& rig,
    CellBlockRange range,
    std::span<const BorderWidth> layer_widths
) {
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
TEST(FindBorderBlocks, StackedLayersOfRandomGrids_AnswerAsTheWholeGrid) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  for (const std::size_t grid : FindRandomGridsSpreadBySize(kPieceGridCount)) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(staggered_grid);
    ASSERT_TRUE(rig.has_value());
    for (const CellBlockRange range : BuildTestedRanges(staggered_grid, kRangeBlocks)) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
      CheckLayersAnswerAsWholeGrid(*rig, range, kLayerWidths);
    }
  }
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
