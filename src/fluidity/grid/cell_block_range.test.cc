#include "fluidity/grid/cell_block_range.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

std::vector<std::uint32_t> ToIndexes(std::span<const CellBlockIndex> blocks) {
  std::vector<std::uint32_t> indexes;
  for (const CellBlockIndex block : blocks) {
    indexes.push_back(block.value);
  }
  return indexes;
}

TEST(BuildRanges, TenBlocksAtMostFour_GiveRangesOfFourFourTwo) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(10);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(test_grid->grid, 4);
  ASSERT_TRUE(ranges.has_value());
  ASSERT_EQ(ranges->size(), 3U);
  EXPECT_EQ(ranges->at(0).first_block, 0U);
  EXPECT_EQ(ranges->at(0).block_count, 4U);
  EXPECT_EQ(ranges->at(1).first_block, 4U);
  EXPECT_EQ(ranges->at(1).block_count, 4U);
  EXPECT_EQ(ranges->at(2).first_block, 8U);
  EXPECT_EQ(ranges->at(2).block_count, 2U);
}

// Checks that the ranges of a cut follow each other from block 0 to the last block, each of 1 to max_blocks_per_range
// blocks.
void CheckRangesCoverEveryBlockOnce(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range) {
  SCOPED_TRACE(testing::Message() << "at most " << max_blocks_per_range << " blocks per range");
  // A failed cut gives no range, so the last check below fails for it.
  const std::vector<CellBlockRange> ranges =
      BuildRanges(grid, max_blocks_per_range).value_or(std::vector<CellBlockRange>{});
  std::uint32_t next_block = 0;
  for (const CellBlockRange range : ranges) {
    EXPECT_EQ(range.first_block, next_block);
    EXPECT_GE(range.block_count, 1U);
    EXPECT_LE(range.block_count, max_blocks_per_range);
    next_block = range.first_block + range.block_count;
  }
  EXPECT_EQ(next_block, grid.counts().cell_blocks);
}

TEST(BuildRanges, Cuts_CoverEveryBlockOnceInOrder) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(10);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  ASSERT_EQ(test_grid->grid.counts().cell_blocks, 10U);
  for (const std::uint32_t max_blocks_per_range : std::array<std::uint32_t, 6>{1, 3, 4, 9, 10, 100}) {
    CheckRangesCoverEveryBlockOnce(test_grid->grid, max_blocks_per_range);
  }
}

TEST(BuildRanges, GridOfNoCell_GivesNoRange) {
  const std::vector<CellId> cells;
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(test_grid->grid, 4);
  ASSERT_TRUE(ranges.has_value());
  EXPECT_TRUE(ranges->empty());
}

TEST(BuildRanges, NoBlockPerRange_IsBadInput) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(3);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(test_grid->grid, 0);
  ASSERT_FALSE(ranges.has_value());
  EXPECT_EQ(ranges.error().error, GridError::kBadInput);
}

TEST(FindBorderBlocks, PlaneFourPieceBeyondTouchingCell_IsInBorder) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> blocks = ToBlocks({0});
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  ASSERT_TRUE(border.has_value());
  // A border made of touching cells alone would be {1}: block 2 holds the +Y face piece of the cell {4,3,0}.
  EXPECT_EQ(ToIndexes(*border), (std::vector<std::uint32_t>{1, 2}));
}

TEST(FindBorderBlocks, FacePieceWhoseCellLowIsOutside_PutsThatBlockInBorder) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  // Block 2 holds {6,6,0}, which touches nothing, and the +Y face piece of {4,3,0}, whose cell_low is in block 1.
  const std::vector<CellBlockIndex> blocks = ToBlocks({2});
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  ASSERT_TRUE(border.has_value());
  // A border made from the cells of the set alone would be empty. Block 0 holds {3,3,0}, which touches that cell_low,
  // so its particles can list the face piece: transfer reads it.
  EXPECT_EQ(ToIndexes(*border), (std::vector<std::uint32_t>{0, 1}));
}

TEST(FindBorderBlocks, CornerKeptInBlockOfNoTouchingCell_IsInBorder) {
  const std::vector<CellId> cells = BuildCornerBeyondCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  ASSERT_EQ(test_grid->grid.counts().cell_blocks, 3U);
  const std::vector<CellBlockIndex> blocks = ToBlocks({0});
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  ASSERT_TRUE(border.has_value());
  // Block 1 holds the +X face piece of {-1,0,0}; block 2 holds its 4 corners at x = 0 m and no cell touching it. A
  // border made from the cells of the set alone would be {1}.
  EXPECT_EQ(ToIndexes(*border), (std::vector<std::uint32_t>{1, 2}));
}

TEST(FindBorderBlocks, DistantCornerBlock_IsInBorderOnlyAtFourCellWidths) {
  const std::vector<CellId> cells = BuildDistantCornerCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> blocks = ToBlocks({0});
  const std::expected<std::vector<CellBlockIndex>, GridFailure> touching_border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  const std::expected<std::vector<CellBlockIndex>, GridFailure> wide_border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kFourCellWidths);
  ASSERT_TRUE(touching_border.has_value());
  ASSERT_TRUE(wide_border.has_value());
  EXPECT_TRUE(touching_border->empty());
  EXPECT_EQ(ToIndexes(*wide_border), (std::vector<std::uint32_t>{1}));
}

TEST(FindBorderBlocks, SmallBlockOfSplitCorner_HoldsTheSizeLevelZeroBlock) {
  const std::vector<CellId> cells = BuildSplitCornerCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> blocks = ToBlocks({1});
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  ASSERT_TRUE(border.has_value());
  // Every place a reader returns from block 1 is kept at size level 1, but FindCellAt walks down from block 0.
  ASSERT_FALSE(border->empty());
  EXPECT_EQ(border->front().value, 0U);
}

TEST(FindBorderBlocks, PieceOfSplitCornerBlockWithBorder_FindsEveryCellAtItsCentre) {
  const std::vector<CellId> cells = BuildSplitCornerCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables whole = test_grid->grid.tables(RunOn::kCpu);
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, ToBlocks({1}), BorderWidth::kTouchingCells);
  ASSERT_TRUE(border.has_value());
  std::vector<CellBlockIndex> loaded_blocks = *border;
  loaded_blocks.push_back({.value = 1});
  std::ranges::sort(loaded_blocks, {}, &CellBlockIndex::value);
  const PieceTables piece = BuildPieceTables(whole, loaded_blocks);
  const GridTables piece_tables = piece.ToTables();
  const auto piece_block =
      static_cast<std::uint32_t>(std::ranges::find(loaded_blocks, 1U, &CellBlockIndex::value) - loaded_blocks.begin());
  // Every cell place of block 1 is active; each centre must give that place of the piece's copy of block 1.
  std::vector<std::uint32_t> found_places;
  std::vector<std::uint32_t> expected_places;
  for (std::uint32_t place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
    const CellPlaceIndex whole_cell = {.value = kCellPlacesPerBlock + place_in_block};
    found_places.push_back(FindCellAt(piece_tables, ComputeCellCenter(whole, whole_cell)).value);
    expected_places.push_back((piece_block * kCellPlacesPerBlock) + place_in_block);
  }
  EXPECT_EQ(found_places, expected_places);
}

TEST(FindBorderBlocks, OutputOneShort_WritesFirstEntriesReturnsTotal) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> blocks = ToBlocks({0});
  std::array<CellBlockIndex, 2> exact_output = {CellBlockIndex{99}, CellBlockIndex{99}};
  std::array<CellBlockIndex, 1> short_output = {CellBlockIndex{99}};
  const std::uint64_t count_before_exact_call = ComputeAllocationCount();
  const std::expected<std::uint32_t, GridFailure> exact_total =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells, exact_output);
  const std::uint64_t count_after_exact_call = ComputeAllocationCount();
  const std::expected<std::uint32_t, GridFailure> short_total =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells, short_output);
  const std::uint64_t count_after_short_call = ComputeAllocationCount();
  ASSERT_TRUE(exact_total.has_value());
  ASSERT_TRUE(short_total.has_value());
  EXPECT_EQ(*exact_total, 2U);
  EXPECT_EQ(exact_output.at(0).value, 1U);
  EXPECT_EQ(exact_output.at(1).value, 2U);
  EXPECT_EQ(*short_total, 2U);
  EXPECT_EQ(short_output.at(0).value, 1U);
  EXPECT_EQ(count_after_exact_call - count_before_exact_call, 0U);
  EXPECT_EQ(count_after_short_call - count_after_exact_call, 0U);
}

TEST(FindBorderBlocks, UnsortedSet_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> descending_blocks = ToBlocks({2, 0});
  const std::vector<CellBlockIndex> repeated_blocks = ToBlocks({1, 1});
  const std::expected<std::vector<CellBlockIndex>, GridFailure> descending_border =
      FindBorderBlocks(test_grid->grid, descending_blocks, BorderWidth::kTouchingCells);
  const std::expected<std::vector<CellBlockIndex>, GridFailure> repeated_border =
      FindBorderBlocks(test_grid->grid, repeated_blocks, BorderWidth::kTouchingCells);
  ASSERT_FALSE(descending_border.has_value());
  ASSERT_FALSE(repeated_border.has_value());
  EXPECT_EQ(descending_border.error().error, GridError::kBadInput);
  EXPECT_EQ(repeated_border.error().error, GridError::kBadInput);
}

TEST(FindBorderBlocks, IndexBeyondTheGrid_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> blocks = ToBlocks({0, 3});
  std::array<CellBlockIndex, 4> output = {};
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  const std::expected<std::uint32_t, GridFailure> total =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells, output);
  ASSERT_FALSE(border.has_value());
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(border.error().error, GridError::kBadInput);
  EXPECT_EQ(total.error().error, GridError::kBadInput);
}

TEST(FindBorderBlocks, OutputOverTheSet_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  // The set {0} is the first entry of the array, and the output is the whole array, so the walk would write over it.
  std::array<CellBlockIndex, 3> shared = {CellBlockIndex{0}, CellBlockIndex{99}, CellBlockIndex{99}};
  const std::expected<std::uint32_t, GridFailure> total = FindBorderBlocks(
      test_grid->grid,
      std::span<const CellBlockIndex>(shared.data(), 1),
      BorderWidth::kTouchingCells,
      std::span<CellBlockIndex>(shared)
  );
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(total.error().error, GridError::kBadInput);
  EXPECT_EQ(shared.at(0).value, 0U);
  EXPECT_EQ(shared.at(1).value, 99U);
}

TEST(BuildRangeWithBorder, RangeOfTheMiddleBlock_HasBothOthersAsBorder) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const RangeWithBorder range_with_border = BuildRangeWithBorder(test_grid->grid, {.first_block = 1, .block_count = 1});
  EXPECT_EQ(range_with_border.range.first_block, 1U);
  EXPECT_EQ(range_with_border.range.block_count, 1U);
  // Block 0 holds the cell {3,3,0} that touches {4,3,0}; block 2 holds the +Y face piece of {4,3,0}.
  EXPECT_EQ(ToIndexes(range_with_border.border_blocks), (std::vector<std::uint32_t>{0, 2}));
}

TEST(BuildRangeWithBorder, EveryBlockOfTheGrid_HasNoBorder) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const RangeWithBorder range_with_border = BuildRangeWithBorder(test_grid->grid, test_grid->grid.all_blocks());
  EXPECT_TRUE(range_with_border.border_blocks.empty());
}

// Checks the border that FindBorderGeometrically finds for `blocks` in a grid of `cells` at kTouchingCells.
void CheckGeometricBorder(
    int size_level_count,
    const std::vector<CellId>& cells,
    std::initializer_list<std::uint32_t> blocks,
    const std::vector<std::uint32_t>& expected_border
) {
  const GridFrame frame = ToUnitFrame(size_level_count);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GeometricReadBlocks read_blocks =
      BuildGeometricReadBlocks(GeometricGrid(frame, cells), test_grid->grid.tables(RunOn::kCpu));
  EXPECT_EQ(
      ToIndexes(FindBorderGeometrically(read_blocks, ToBlocks(blocks), BorderWidth::kTouchingCells)),
      expected_border
  );
}

TEST(FindBorderGeometrically, PlaneFourGridFromBlockA_HasBothOtherBlocks) {
  CheckGeometricBorder(1, BuildPlaneFourCells(), {0}, {1, 2});
}

TEST(FindBorderGeometrically, PlaneFourGridFromBlockC_HasTheBlocksOfTheCellsNearTheFacePiece) {
  CheckGeometricBorder(1, BuildPlaneFourCells(), {2}, {0, 1});
}

TEST(FindBorderGeometrically, SplitCornerGridFromSmallBlock_HasTheSizeLevelZeroBlock) {
  const std::vector<CellId> cells = BuildSplitCornerCells();
  const GridFrame frame = ToUnitFrame(2);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GeometricReadBlocks read_blocks =
      BuildGeometricReadBlocks(GeometricGrid(frame, cells), test_grid->grid.tables(RunOn::kCpu));
  const std::vector<CellBlockIndex> border =
      FindBorderGeometrically(read_blocks, ToBlocks({1}), BorderWidth::kTouchingCells);
  ASSERT_FALSE(border.empty());
  EXPECT_EQ(border.front().value, 0U);
}

TEST(FindBorderGeometrically, CornerBeyondGrid_HasTheBlockKeepingTheCorners) {
  CheckGeometricBorder(2, BuildCornerBeyondCells(), {0}, {1, 2});
}

}  // namespace
}  // namespace fluidity::grid
