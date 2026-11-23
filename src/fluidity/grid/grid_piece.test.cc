#include "fluidity/grid/grid_piece.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
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

TEST(BuildGridPiece, PlaneFourGridRangeOfBlockC_EqualsPieceTables) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const StaggeredGrid& whole = test_grid->grid;
  const RangeWithBorder range_with_border = BuildRangeWithBorder(whole, {.first_block = 2, .block_count = 1});
  ASSERT_EQ(range_with_border.border_blocks.size(), 2U);
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(whole, range_with_border.range, range_with_border.border_blocks, buffers);
  ASSERT_TRUE(piece.has_value());
  const std::vector<CellBlockIndex> loaded_blocks = {{.value = 0}, {.value = 1}, {.value = 2}};
  const PieceTables expected = BuildPieceTables(whole.tables(RunOn::kCpu), loaded_blocks);
  const std::optional<std::string> difference = FindFirstTableDifference(*piece, expected);
  EXPECT_FALSE(difference.has_value()) << difference.value_or("");
  EXPECT_EQ(piece->range().first_block, 2U);
  EXPECT_EQ(piece->range().block_count, 1U);
}

TEST(BuildGridPiece, RangeBetweenBorderEntries_KeepsTheWholeGridsOrder) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(10);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(6);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::vector<CellBlockIndex> border = ToBlocks({1, 2, 8});
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(test_grid->grid, {.first_block = 4, .block_count = 3}, border, buffers);
  ASSERT_TRUE(piece.has_value());
  EXPECT_EQ(ToIndexes(piece->block_in_whole_grid()), (std::vector<std::uint32_t>{1, 2, 4, 5, 6, 8}));
  EXPECT_EQ(piece->range().first_block, 2U);
  EXPECT_EQ(piece->range().block_count, 3U);
  EXPECT_EQ(piece->range_in_whole_grid().first_block, 4U);
  EXPECT_EQ(piece->range_in_whole_grid().block_count, 3U);
  EXPECT_EQ(piece->grid().counts().cell_blocks, 6U);
  EXPECT_EQ(piece->grid().version().value, test_grid->grid.version().value);
  EXPECT_EQ(piece->block_in_whole_grid_buffer().size(), 6U);
}

TEST(BuildGridPiece, RangeOfWholeGridWithoutBorder_EqualsWholeGrid) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const StaggeredGrid& whole = test_grid->grid;
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(whole.counts().cell_blocks);
  ASSERT_TRUE(room.has_value());
  const std::expected<GridPiece, GridFailure> piece = BuildGridPiece(
      whole,
      {.first_block = 0, .block_count = whole.counts().cell_blocks},
      std::vector<CellBlockIndex>{},
      room->buffers
  );
  ASSERT_TRUE(piece.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(piece->grid(), whole));
  EXPECT_EQ(ToIndexes(piece->block_in_whole_grid()), (std::vector<std::uint32_t>{0, 1, 2}));
  EXPECT_EQ(piece->range().first_block, 0U);
  EXPECT_EQ(piece->range().block_count, 3U);
}

TEST(BuildGridPiece, AboveBuffersRoom_IsTooManyCellBlocks) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(2);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::vector<CellBlockIndex> border = ToBlocks({0, 1});
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(test_grid->grid, {.first_block = 2, .block_count = 1}, border, buffers);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error().error, GridError::kTooManyCellBlocks);
  EXPECT_EQ(piece.error().cell.value, 0U);
  EXPECT_EQ(buffers.max_cell_blocks(), 2U);
}

TEST(BuildGridPiece, BorderBlockInsideRange_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::vector<CellBlockIndex> border = ToBlocks({0, 2});
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(test_grid->grid, {.first_block = 2, .block_count = 1}, border, buffers);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error().error, GridError::kBadInput);
}

TEST(BuildGridPiece, BorderNotAscending_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::vector<CellBlockIndex> border = ToBlocks({1, 0});
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(test_grid->grid, {.first_block = 2, .block_count = 1}, border, buffers);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error().error, GridError::kBadInput);
}

TEST(BuildGridPiece, BorderBeyondTheGrid_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::vector<CellBlockIndex> border = ToBlocks({0, 3});
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(test_grid->grid, {.first_block = 2, .block_count = 1}, border, buffers);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error().error, GridError::kBadInput);
}

TEST(BuildGridPiece, RangeBeyondTheGrid_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(test_grid->grid, {.first_block = 2, .block_count = 2}, std::vector<CellBlockIndex>{}, buffers);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error().error, GridError::kBadInput);
}

TEST(BuildGridPiece, EmptyRangeWithBorder_IsBadInput) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const std::expected<GridPiece, GridFailure> piece = BuildGridPiece(
      test_grid->grid,
      {.first_block = 1, .block_count = 0},
      std::vector<CellBlockIndex>{{.value = 0}},
      buffers
  );
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error().error, GridError::kBadInput);
}

TEST(FindCellAt, InPieceOutsideLoadedBlocks_IsNoIndex) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  // The piece of block 2 alone, with an empty border. It shows what a border that is too small does: the build
  // succeeds, and the piece answers none where the whole grid has a cell, here in block 0, which it did not load.
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(test_grid->grid, {.first_block = 2, .block_count = 1}, std::vector<CellBlockIndex>{}, buffers);
  ASSERT_TRUE(piece.has_value());
  const Vec3 position_in_block_zero = {.x = 3.5, .y = 3.5, .z = 0.5};
  EXPECT_NE(FindCellAt(test_grid->grid.tables(RunOn::kCpu), position_in_block_zero).value, kNoIndex);
  EXPECT_EQ(FindCellAt(piece->grid().tables(RunOn::kCpu), position_in_block_zero).value, kNoIndex);
}

TEST(TakeBuffers, Piece_GivesBuffersForTheNextPiece) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  std::expected<GridPiece, GridFailure> first =
      BuildGridPiece(test_grid->grid, {.first_block = 0, .block_count = 3}, std::vector<CellBlockIndex>{}, buffers);
  ASSERT_TRUE(first.has_value());
  GridBuffers taken = TakeBuffers(std::move(*first));
  EXPECT_EQ(taken.max_cell_blocks(), 3U);
  EXPECT_EQ(first->grid().version().value, 0U);
  const std::expected<GridPiece, GridFailure> second =
      BuildGridPiece(test_grid->grid, {.first_block = 1, .block_count = 1}, ToBlocks({0}), taken);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(ToIndexes(second->block_in_whole_grid()), (std::vector<std::uint32_t>{0, 1}));
}

// Counts the allocations of three calls on the same buffers: the span form, the buffer form, and a call that fails.
// Each call and each give-back of the buffers allocates nothing.
TEST(BuildGridPiece, AnyPiece_AllocatesNothing) {
  const std::vector<CellId> cells = BuildPlaneFourCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(3);
  ASSERT_TRUE(room.has_value());
  GridBuffers& buffers = room->buffers;
  const CellBlockRange range = {.first_block = 2, .block_count = 1};
  const std::vector<CellBlockIndex> border = ToBlocks({0, 1});
  std::expected<Buffer<CellBlockIndex>, BufferError> border_buffer = BuildBuffer<CellBlockIndex>(RunOn::kCpu, 2);
  ASSERT_TRUE(border_buffer.has_value());
  ASSERT_TRUE(WriteBuffer<CellBlockIndex>(border, *border_buffer, 0).has_value());
  const std::vector<CellBlockIndex> border_inside_range = ToBlocks({0, 2});
  const std::uint64_t count_before_span_form = ComputeAllocationCount();
  std::expected<GridPiece, GridFailure> span_piece = BuildGridPiece(test_grid->grid, range, border, buffers);
  const std::uint64_t count_after_span_form = ComputeAllocationCount();
  ASSERT_TRUE(span_piece.has_value());
  buffers = TakeBuffers(std::move(*span_piece));
  const std::uint64_t count_before_buffer_form = ComputeAllocationCount();
  std::expected<GridPiece, GridFailure> buffer_piece =
      BuildGridPiece(test_grid->grid, range, *border_buffer, 2, buffers);
  const std::uint64_t count_after_buffer_form = ComputeAllocationCount();
  ASSERT_TRUE(buffer_piece.has_value());
  buffers = TakeBuffers(std::move(*buffer_piece));
  const std::uint64_t count_before_failure = ComputeAllocationCount();
  const std::expected<GridPiece, GridFailure> failed =
      BuildGridPiece(test_grid->grid, range, border_inside_range, buffers);
  const std::uint64_t count_after_failure = ComputeAllocationCount();
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().error, GridError::kBadInput);
  EXPECT_EQ(count_after_span_form - count_before_span_form, 0U);
  EXPECT_EQ(count_after_buffer_form - count_before_buffer_form, 0U);
  EXPECT_EQ(count_after_failure - count_before_failure, 0U);
}

}  // namespace
}  // namespace fluidity::grid
