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

#include "fluidity/base/axis.h"
#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
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

TEST(FindBorderBlocks, VectorForm_AllocatesTwice) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> blocks = ToBlocks({0});
  const std::uint64_t count_before_call = ComputeAllocationCount();
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  const std::uint64_t count_after_call = ComputeAllocationCount();
  ASSERT_TRUE(border.has_value());
  ASSERT_EQ(border->size(), 2U);
  // The marks, and the list sized once.
  EXPECT_EQ(count_after_call - count_before_call, 2U);
}

TEST(BuildRangeWithBorder, Call_AllocatesTwice) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::uint64_t count_before_call = ComputeAllocationCount();
  const RangeWithBorder range_with_border = BuildRangeWithBorder(test_grid->grid, {.first_block = 0, .block_count = 1});
  const std::uint64_t count_after_call = ComputeAllocationCount();
  ASSERT_EQ(range_with_border.border_blocks.size(), 2U);
  EXPECT_EQ(count_after_call - count_before_call, 2U);
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
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(test_grid->grid, blocks, BorderWidth::kTouchingCells);
  ASSERT_FALSE(border.has_value());
  EXPECT_EQ(border.error().error, GridError::kBadInput);
}

// Writes `blocks` to the start of `buffer`. It fails the calling test when the write fails.
void WriteBlocksOrFail(Buffer<CellBlockIndex>& buffer, std::span<const CellBlockIndex> blocks) {
  EXPECT_TRUE(WriteBuffer<CellBlockIndex>(blocks, buffer, 0).has_value());
}

// The buffer form on the rig's buffers: `so_far_count` entries of its first buffer are the set found so far, and the
// border goes to its second buffer.
std::expected<std::uint32_t, GridFailure>
FindOnRig(BorderSearchRig& rig, CellBlockRange range, std::uint32_t so_far_count, BorderWidth width) {
  return FindBorderBlocks(
      rig.grid,
      range,
      rig.so_far_buffer,
      so_far_count,
      width,
      rig.search_buffers,
      rig.border_buffer
  );
}

TEST(FindBorderBlocks, StackedBufferFormOnPlaneFourGrid_GivesBlocksOneTwo) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::expected<std::uint32_t, GridFailure> count =
      FindOnRig(*rig, {.first_block = 0, .block_count = 1}, 0, BorderWidth::kTouchingCells);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 2U);
  EXPECT_EQ(rig->border_buffer.data()[0].value, 1U);
  EXPECT_EQ(rig->border_buffer.data()[1].value, 2U);
}

TEST(FindBorderBlocks, BorderSoFar_IsInTheList) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  // Block 2 alone has the border {0, 1}. Asked for range {0, 1} with block 2 so far, the layer is {2}: block 2 is in
  // the list as border so far, and nothing else lies outside range plus it.
  const std::vector<CellBlockIndex> so_far = ToBlocks({2});
  const std::vector<CellBlockIndex> layer =
      FindLayerOrFail(*rig, {.first_block = 0, .block_count = 2}, so_far, BorderWidth::kTouchingCells);
  EXPECT_EQ(ToIndexes(layer), (std::vector<std::uint32_t>{2}));
}

TEST(FindBorderBlocks, StackedBufferFormShortOutput_WritesFirstEntriesReturnsTotal) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  std::expected<Buffer<CellBlockIndex>, BufferError> short_output = BuildBuffer<CellBlockIndex>(RunOn::kCpu, 1);
  ASSERT_TRUE(short_output.has_value());
  const std::expected<std::uint32_t, GridFailure> total = FindBorderBlocks(
      test_grid->grid,
      {.first_block = 0, .block_count = 1},
      rig->so_far_buffer,
      0,
      BorderWidth::kTouchingCells,
      rig->search_buffers,
      *short_output
  );
  ASSERT_TRUE(total.has_value());
  EXPECT_EQ(*total, 2U);
  EXPECT_EQ(short_output->data()[0].value, 1U);
}

TEST(FindBorderBlocks, StackedBufferFormSameBufferTwice_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::vector<CellBlockIndex> so_far = ToBlocks({2});
  WriteBlocksOrFail(rig->so_far_buffer, so_far);
  const std::expected<std::uint32_t, GridFailure> total = FindBorderBlocks(
      test_grid->grid,
      {.first_block = 0, .block_count = 1},
      rig->so_far_buffer,
      1,
      BorderWidth::kTouchingCells,
      rig->search_buffers,
      rig->so_far_buffer
  );
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(total.error().error, GridError::kBadInput);
  EXPECT_EQ(rig->so_far_buffer.data()[0].value, 2U);
}

TEST(FindBorderBlocks, BorderSoFarInsideRange_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::vector<CellBlockIndex> so_far = ToBlocks({1});
  WriteBlocksOrFail(rig->so_far_buffer, so_far);
  const std::expected<std::uint32_t, GridFailure> total =
      FindOnRig(*rig, {.first_block = 0, .block_count = 2}, 1, BorderWidth::kTouchingCells);
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(total.error().error, GridError::kBadInput);
}

TEST(FindBorderBlocks, BorderSoFarNotAscending_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  for (const std::vector<CellBlockIndex>& so_far : {ToBlocks({2, 1}), ToBlocks({1, 1})}) {
    WriteBlocksOrFail(rig->so_far_buffer, so_far);
    const std::expected<std::uint32_t, GridFailure> total =
        FindOnRig(*rig, {.first_block = 0, .block_count = 1}, 2, BorderWidth::kTouchingCells);
    ASSERT_FALSE(total.has_value());
    EXPECT_EQ(total.error().error, GridError::kBadInput);
  }
}

TEST(FindBorderBlocks, BorderSoFarIndexBeyondTheGrid_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::vector<CellBlockIndex> so_far = ToBlocks({1, 3});
  WriteBlocksOrFail(rig->so_far_buffer, so_far);
  const std::expected<std::uint32_t, GridFailure> total =
      FindOnRig(*rig, {.first_block = 0, .block_count = 1}, 2, BorderWidth::kTouchingCells);
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(total.error().error, GridError::kBadInput);
}

TEST(FindBorderBlocks, BorderSoFarCountAboveBufferSize_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::uint32_t count_above_size = static_cast<std::uint32_t>(rig->so_far_buffer.size()) + 1;
  const std::expected<std::uint32_t, GridFailure> total =
      FindOnRig(*rig, {.first_block = 0, .block_count = 1}, count_above_size, BorderWidth::kTouchingCells);
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(total.error().error, GridError::kBadInput);
}

TEST(FindBorderBlocks, RangeBeyondTheGrid_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::expected<std::uint32_t, GridFailure> total =
      FindOnRig(*rig, {.first_block = 2, .block_count = 2}, 0, BorderWidth::kTouchingCells);
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(total.error().error, GridError::kBadInput);
}

TEST(FindBorderBlocks, SearchBuffersTooSmall_IsTooManyCellBlocks) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  std::expected<RunMemory, BufferError> small_memory = BuildRunMemory(RunOn::kCpu, 256);
  ASSERT_TRUE(small_memory.has_value());
  std::expected<BorderSearchBuffers, GridFailure> small_buffers =
      BuildBorderSearchBuffers(RunOn::kCpu, 2, *small_memory);
  ASSERT_TRUE(small_buffers.has_value());
  const std::expected<std::uint32_t, GridFailure> total = FindBorderBlocks(
      test_grid->grid,
      {.first_block = 0, .block_count = 1},
      rig->so_far_buffer,
      0,
      BorderWidth::kTouchingCells,
      *small_buffers,
      rig->border_buffer
  );
  ASSERT_FALSE(total.has_value());
  EXPECT_EQ(total.error().error, GridError::kTooManyCellBlocks);
}

TEST(FindBorderBlocks, StackedBufferForm_AllocatesNothing) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::vector<CellBlockIndex> so_far = ToBlocks({2});
  WriteBlocksOrFail(rig->so_far_buffer, so_far);
  const std::uint64_t count_before_call = ComputeAllocationCount();
  const std::expected<std::uint32_t, GridFailure> total =
      FindOnRig(*rig, {.first_block = 0, .block_count = 1}, 1, BorderWidth::kFourCellWidths);
  const std::uint64_t count_after_call = ComputeAllocationCount();
  ASSERT_TRUE(total.has_value());
  EXPECT_EQ(count_after_call - count_before_call, 0U);
}

// The cells {3,0,0}, {4,0,0} and {8,0,0} of 1 m cells, in the blocks 0, 1 and 2 along x:
// block 0 holds x 0..3, block 1 holds x 4..7 and block 2 holds x 8..11.
std::vector<CellId> BuildThreeCellLineCells() {
  return {ToCell(0, 3, 0, 0), ToCell(0, 4, 0, 0), ToCell(0, 8, 0, 0)};
}

// The first call marks block 2, which the border of block 0 does not hold. If the second call did not clear the marks,
// block 2 would join its list.
TEST(FindBorderBlocks, SecondCallAfterMarkingOtherBlock_EqualsFreshBuffers) {
  const std::vector<CellId> cells = BuildThreeCellLineCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> used_rig = BuildBorderSearchRig(test_grid->grid);
  std::expected<BorderSearchRig, GridFailure> fresh_rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(used_rig.has_value());
  ASSERT_TRUE(fresh_rig.has_value());
  const CellBlockRange middle_range = {.first_block = 1, .block_count = 1};
  const CellBlockRange first_range = {.first_block = 0, .block_count = 1};
  const std::vector<CellBlockIndex> no_blocks;
  const std::vector<CellBlockIndex> middle_layer =
      FindLayerOrFail(*used_rig, middle_range, no_blocks, BorderWidth::kFourCellWidths);
  const std::vector<CellBlockIndex> second_layer =
      FindLayerOrFail(*used_rig, first_range, no_blocks, BorderWidth::kFourCellWidths);
  const std::vector<CellBlockIndex> fresh_layer =
      FindLayerOrFail(*fresh_rig, first_range, no_blocks, BorderWidth::kFourCellWidths);
  EXPECT_EQ(ToIndexes(middle_layer), (std::vector<std::uint32_t>{0, 2}));
  EXPECT_EQ(ToIndexes(fresh_layer), (std::vector<std::uint32_t>{1}));
  EXPECT_EQ(ToIndexes(second_layer), (std::vector<std::uint32_t>{1}));
}

// Block 0 alone has the border {1} at both widths, since block 2 is 4 m or more from every face piece of block 0. A
// pass of two stages, the last touching and the one before it four widths wide, needs {1, 2}: the earlier stage reads
// from the places of block 1 as well, and from block 1 the four widths cover block 2.
TEST(FindBorderBlocks, ThreeCellLineStackOfTwoLayers_IsWiderThanTheWideBorder) {
  const std::vector<CellId> cells = BuildThreeCellLineCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const CellBlockRange range = {.first_block = 0, .block_count = 1};
  const std::vector<CellBlockIndex> no_blocks;
  const std::vector<CellBlockIndex> last_stage_layer =
      FindLayerOrFail(*rig, range, no_blocks, BorderWidth::kTouchingCells);
  const std::vector<CellBlockIndex> stack =
      FindLayerOrFail(*rig, range, last_stage_layer, BorderWidth::kFourCellWidths);
  const std::vector<CellBlockIndex> wide_border = FindLayerOrFail(*rig, range, no_blocks, BorderWidth::kFourCellWidths);
  const std::vector<CellBlockIndex> blocks_of_range = ToBlocks({0});
  const std::expected<std::vector<CellBlockIndex>, GridFailure> wide_border_by_vector =
      FindBorderBlocks(test_grid->grid, blocks_of_range, BorderWidth::kFourCellWidths);
  ASSERT_TRUE(wide_border_by_vector.has_value());
  EXPECT_EQ(ToIndexes(last_stage_layer), (std::vector<std::uint32_t>{1}));
  EXPECT_EQ(ToIndexes(stack), (std::vector<std::uint32_t>{1, 2}));
  EXPECT_EQ(ToIndexes(wide_border), (std::vector<std::uint32_t>{1}));
  EXPECT_EQ(ToIndexes(*wide_border_by_vector), (std::vector<std::uint32_t>{1}));
}

// The bytes of the estimate's entries that live in the run's memory, as the arrays take them.
std::uint64_t ComputeEstimateBytes(const std::vector<MemoryEstimateEntry>& estimate) {
  return ComputeMemoryBytes(estimate, MemoryCounts(), MemoryWhere::kRunOn);
}

TEST(BuildBorderSearchMemoryEstimate, MatchesAllocatedBytes) {
  const std::vector<MemoryEstimateEntry> estimate = BuildBorderSearchMemoryEstimate(RunOn::kCpu, 1000);
  ASSERT_EQ(estimate.size(), 1U);
  EXPECT_EQ(ComputeEstimateBytes(estimate), 1000U);
  std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(RunOn::kCpu, ComputeRunMemoryBytes(estimate, MemoryCounts(), MemoryWhere::kRunOn));
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(search_buffers.has_value());
  EXPECT_EQ(search_buffers->marks().size(), 1000U);
  EXPECT_EQ(sizeof(search_buffers->marks().data()[0]), 1U);
}

TEST(BuildBorderSearchBuffers, IntoRunMemory_HandsOutExactlyTheEstimate) {
  const std::vector<MemoryEstimateEntry> estimate = BuildBorderSearchMemoryEstimate(RunOn::kCpu, 1000);
  std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(RunOn::kCpu, ComputeRunMemoryBytes(estimate, MemoryCounts(), MemoryWhere::kRunOn));
  ASSERT_TRUE(run_memory.has_value());
  ASSERT_EQ(run_memory->reserved_bytes(), 1024U);
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(search_buffers.has_value());
  EXPECT_EQ(search_buffers->max_cell_blocks(), 1000U);
  EXPECT_EQ(search_buffers->run_on(), RunOn::kCpu);
  EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes());
}

TEST(BuildBorderSearchMemoryEstimate, Gpu_CountsEveryArray) {
  const std::vector<MemoryEstimateEntry> estimate = BuildBorderSearchMemoryEstimate(RunOn::kGpu, 1000);
  const OrderBuffersSize order_buffers_size = ComputeOrderBuffersSize(RunOn::kGpu);
  ASSERT_EQ(estimate.size(), 4U);
  EXPECT_EQ(
      ComputeEstimateBytes(estimate),
      1000U + 4004U + (std::uint64_t{order_buffers_size.bytes_per_item} * 1001U) + order_buffers_size.fixed_bytes
          + sizeof(GridBuildStatus)
  );
  std::uint64_t piece_bytes = 0;
  for (const MemoryEstimateEntry& entry : estimate) {
    piece_bytes += ComputeRunMemoryPieceBytes(entry.given_count * entry.bytes_each);
  }
  EXPECT_EQ(piece_bytes, ComputeRunMemoryBytes(estimate, MemoryCounts(), MemoryWhere::kRunOn));
}

TEST(BuildBorderSearchBuffers, ZeroBlocks_IsBadInput) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 1024);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kCpu, 0, *run_memory);
  ASSERT_FALSE(search_buffers.has_value());
  EXPECT_EQ(search_buffers.error().error, GridError::kBadInput);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

TEST(BuildBorderSearchBuffers, AboveTheLimit_IsBadInput) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 1024);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kCpu, kMaxCellBlocksLimit + 1, *run_memory);
  ASSERT_FALSE(search_buffers.has_value());
  EXPECT_EQ(search_buffers.error().error, GridError::kBadInput);
}

TEST(BuildBorderSearchBuffers, RunMemoryTooSmall_IsOutOfMemoryWithTheArrayBytes) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 256);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_FALSE(search_buffers.has_value());
  EXPECT_EQ(search_buffers.error().error, GridError::kOutOfMemory);
  EXPECT_EQ(search_buffers.error().magnitude, 1000U);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
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

// The block that holds the active cell at `position` of the tables.
CellBlockIndex FindBlockAt(const GridTables& tables, Vec3 position) {
  return ToCellBlockIndex(FindCellAt(tables, position));
}

// Says whether a list holds a face piece kept in `block`; the entry is written to `position_in_list` when it does.
bool FindListedFaceIn(const NearbyFaceList& list, CellBlockIndex block, int& position_in_list) {
  for (int entry = 0; entry < list.count; ++entry) {
    if (ToCellBlockIndex(list.faces[entry]).value == block.value) {
      position_in_list = entry;
      return true;
    }
  }
  return false;
}

// A particle's sums need face pieces beyond the near border of the range that collects from it. The range is the
// size-level-1 block of x in [2,4); the particle lies in the cell [0,1), a cell that can contribute to the face piece
// at x = 2, and its list has the face piece at x = -0.5, kept in the block of x in [-2,0): not in the range, and not in
// the near border. So the sums are made where the particle's cell is in the set, and carried.
TEST(BuildRangeWithBorder, ParticleSums_NeedAFaceBeyondTheNearBorder) {
  const std::vector<CellId> cells = BuildCarriedSumsCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const CellBlockIndex range_block = FindBlockAt(tables, {.x = 2.25, .y = 0.25, .z = 0.25});
  const CellBlockIndex far_block = FindBlockAt(tables, {.x = -0.25, .y = 0.25, .z = 0.25});
  const RangeWithBorder range_with_border =
      BuildRangeWithBorder(test_grid->grid, {.first_block = range_block.value, .block_count = 1});
  const std::vector<std::uint32_t> border = ToIndexes(range_with_border.border_blocks);
  ASSERT_NE(far_block.value, range_block.value);
  EXPECT_EQ(std::ranges::count(border, far_block.value), 0);
  const Vec3 position = {.x = 0.95, .y = 0.5, .z = 0.5};
  GridError error = GridError::kNone;
  const NearbyFaceList list = FindNearbyFaces(tables, position, Axis::kX, CellBlockIndex{kNoIndex}, &error);
  ASSERT_EQ(error, GridError::kNone);
  int range_position = -1;
  int far_position = -1;
  ASSERT_TRUE(FindListedFaceIn(list, range_block, range_position));
  ASSERT_TRUE(FindListedFaceIn(list, far_block, far_position));
  const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, list, position, ToMaskOfAll(list));
  EXPECT_GT(ComputeFaceWeight(tables, weights, far_position).weight, 0.0);
  // In the piece of the range and its border the particle's cell is there, yet its sums differ: the far face piece is
  // not.
  std::vector<CellBlockIndex> loaded_blocks = range_with_border.border_blocks;
  loaded_blocks.push_back(range_block);
  std::ranges::sort(loaded_blocks, {}, &CellBlockIndex::value);
  const PieceTables piece = BuildPieceTables(tables, loaded_blocks);
  const GridTables piece_tables = piece.ToTables();
  const NearbyFaceList piece_list = FindNearbyFaces(piece_tables, position, Axis::kX, CellBlockIndex{kNoIndex}, &error);
  ASSERT_EQ(error, GridError::kNone);
  const NearbyFaceWeights piece_weights =
      ComputeNearbyFaceWeights(piece_tables, piece_list, position, ToMaskOfAll(piece_list));
  EXPECT_LT(piece_list.count, list.count);
  EXPECT_NE(piece_weights.weight_sum, weights.weight_sum);
}

TEST(BuildBorderSearchStackEstimate, OnCpu_CountsNothing) {
  const std::expected<MemoryEstimateEntry, GridFailure> entry = BuildBorderSearchStackEstimate(RunOn::kCpu);
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->where, MemoryWhere::kDriverStack);
  EXPECT_EQ(entry->kept_for, KeptFor::kWholeRun);
  EXPECT_EQ(ComputeMemoryBytes(std::span<const MemoryEstimateEntry>(&*entry, 1), MemoryCounts()), 0U);
}

}  // namespace
}  // namespace fluidity::grid
