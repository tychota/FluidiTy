#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <new>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

TEST(BuildTestGrid, DefaultSettings_SizedToTheList) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(9);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_EQ(test_grid->grid.max_cell_blocks(), 9U);
  EXPECT_EQ(test_grid->grid.counts().cell_blocks, 9U);
}

TEST(StaggeredGrid, CpuGrid_HasNoGpuTables) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(2);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables gpu_tables = test_grid->grid.tables(RunOn::kGpu);
  EXPECT_EQ(gpu_tables.version.value, 0U);
  EXPECT_EQ(gpu_tables.block_ids, nullptr);
  EXPECT_EQ(gpu_tables.same_size_level_blocks, nullptr);
  EXPECT_EQ(gpu_tables.bigger_block, nullptr);
  EXPECT_EQ(gpu_tables.smaller_blocks, nullptr);
  EXPECT_EQ(gpu_tables.cell_states, nullptr);
  EXPECT_EQ(gpu_tables.face_piece_kinds, nullptr);
}

// A request whose aligned block would need more than SIZE_MAX bytes fails as its form says, counted, rather than
// wrapping the size to a small block: null from the nothrow form, std::bad_alloc from the throwing one.
TEST(ComputeAllocationCount, AlignedRequestNearSizeMax_FailsWithoutWrapping) {
  // Read through a volatile, so the compiler sees no constant size to warn about.
  const volatile std::size_t near_size_max = std::numeric_limits<std::size_t>::max() - 8;
  const std::size_t byte_count = near_size_max;
  constexpr std::align_val_t kAlignment{64};
  const std::uint64_t count_before_no_throw_request = ComputeAllocationCount();
  void* const memory = ::operator new(byte_count, kAlignment, std::nothrow);
  const std::uint64_t count_after_no_throw_request = ComputeAllocationCount();
  EXPECT_EQ(memory, nullptr);
  EXPECT_EQ(count_after_no_throw_request - count_before_no_throw_request, 1U);
  // The throwing form is counted once too, but a debug runtime may allocate while it raises std::bad_alloc, and that
  // request is counted as well: so at least one.
  EXPECT_THROW(::operator delete(::operator new(byte_count, kAlignment), kAlignment), std::bad_alloc);
  EXPECT_GE(ComputeAllocationCount() - count_after_no_throw_request, 1U);
}

// Size level 0 {0,0,0} beside the 8 size-level-1 cells of size level 0 {1,0,0}, 3 size levels. Blocks in the fixed
// order: 0 is size level 0 (0,0,0), holding the active cell at place 0 and the split cell {1,0,0} at place 1; 1 is
// size level 1 (0,0,0), whose cells x = 0..3 hold the small cells at x = 2..3, y and z 0..1. Block 1's bigger block is
// 0; block 0's smaller block of eighth 0 is 1, its other eighths none. Block 0: 1 active, 1 split, 62 absent places.
// Block 1: 8 active; the 8 places x = 0..1, y and z 0..1 lie inside the active big cell; 48 absent. Face pieces: the
// big cell's 5 whole sides, the 4 quarters between it and the small cells, 12 sides between small cells and their 20
// other outer sides: 41. Active cells 9: 1 of size level 0, 8 of size level 1. In a room of 8 blocks the array of
// bigger and smaller blocks holds the 8 bigger blocks first, so block 0's smaller blocks start at entry 8.
TEST(BuildGrid, OneSplitCell_GivesTheWorkedTables) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const std::vector<CellId> small_cells = ToSmallCells(ToCell(0, 1, 0, 0));
  cells.insert(cells.end(), small_cells.begin(), small_cells.end());
  // A room of 8 blocks, more than the 2 used, so the smaller blocks start after the room's 8 bigger blocks.
  const std::expected<TestGrid, GridFailure> test_grid =
      BuildTestGrid(ToInputs(ToUnitFrame(3), cells), {.max_cell_blocks = 8});
  ASSERT_TRUE(test_grid.has_value());
  const StaggeredGrid& grid = test_grid->grid;
  EXPECT_EQ(grid.version().value, 1U);
  GridCounts expected_counts{.cell_blocks = 2, .active_cells_per_size_level = {}, .active_cells = 9, .face_pieces = 41};
  expected_counts.active_cells_per_size_level[0] = 1;
  expected_counts.active_cells_per_size_level[1] = 8;
  EXPECT_EQ(ToGridCountFields(grid.counts()), ToGridCountFields(expected_counts));

  const GridTables tables = grid.tables(RunOn::kCpu);
  ASSERT_EQ(tables.block_count, 2U);
  CheckOneSplitCellBlockTables(tables);
  CheckOneSplitCellStates(tables);
}

TEST(BuildGrid, EmptyList_GivesEmptyGrid) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), {}));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_EQ(test_grid->grid.version().value, 1U);
  EXPECT_EQ(ToGridCountFields(test_grid->grid.counts()), ToGridCountFields(GridCounts{}));
  EXPECT_EQ(test_grid->grid.all_blocks().first_block, 0U);
  EXPECT_EQ(test_grid->grid.all_blocks().block_count, 0U);
}

// The 7 size-level-1 cells of size level 0 {0,0,0} without eighth 5, size level 1 {1,0,1}: eighth 5 is not eighth 0,
// so the failure names it and its big cell.
TEST(BuildGrid, LoneSmallCell_IsSiblingCellMissing) {
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 5);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  CheckFailure(
      BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers),
      GridError::kSiblingCellMissing,
      ToCell(1, 1, 0, 1),
      ToCell(0, 0, 0, 0)
  );
}

// 9 blocks into a room of 8: step 5 of <proc-build-grid> counts the blocks against the room. The failed build leaves
// the buffers with their arrays, and they then build the first 8 cells.
TEST(BuildGrid, AboveBuffersRoom_IsTooManyCellBlocks) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(9);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  CheckFailure(BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers), GridError::kTooManyCellBlocks);
  EXPECT_EQ(buffers.max_cell_blocks(), 8U);
  const std::expected<StaggeredGrid, GridFailure> grid =
      BuildGrid(ToInputs(ToUnitFrame(3), std::span<const CellId>(cells).first(8)), buffers);
  ASSERT_TRUE(grid.has_value());
  EXPECT_EQ(grid->counts().cell_blocks, 8U);
}

TEST(BuildGrid, MoreCellsThanPlaces_IsTooManyCellBlocks) {
  const std::vector<CellId> cells = BuildMoreCellsThanEightBlocksHold();
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  CheckFailure(BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers), GridError::kTooManyCellBlocks);
}

// The same list with an id whose bit 63 is clear and one at kMaxCoord + 1: the length check still comes first, so the
// bad ids are never read.
TEST(BuildGrid, TooLongListWithBadId_IsTooManyCellBlocks) {
  std::vector<CellId> cells = BuildMoreCellsThanEightBlocksHold();
  cells.at(3) = CellId{.value = cells.at(3).value & ~(std::uint64_t{1} << 63U)};
  cells.at(7) = ToCell(0, kMaxCoord + 1, 0, 0);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  CheckFailure(BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers), GridError::kTooManyCellBlocks);
}

// The 64 size-level-0 cells {0..3, 0..3, 0..3} fill the 64 cell places of one block, with no split cell: 64 cells, not
// above 64 x 1, so a room of 1 builds. 1 block, 64 active cells, and 3 axes x 5 planes x 16 = 240 face pieces.
TEST(BuildGrid, CellPlacesExactlyFull_Builds) {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 4; ++z) {
    for (std::int32_t y = 0; y < 4; ++y) {
      for (std::int32_t x = 0; x < 4; ++x) {
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(2), {}), {.max_cell_blocks = 1});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  const std::expected<StaggeredGrid, GridFailure> grid = BuildGrid(ToInputs(ToUnitFrame(2), cells), buffers);
  ASSERT_TRUE(grid.has_value());
  GridCounts expected_counts{
      .cell_blocks = 1,
      .active_cells_per_size_level = {},
      .active_cells = 64,
      .face_pieces = 240,
  };
  expected_counts.active_cells_per_size_level[0] = 64;
  EXPECT_EQ(ToGridCountFields(grid->counts()), ToGridCountFields(expected_counts));
}

}  // namespace
}  // namespace fluidity::grid
