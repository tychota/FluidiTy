#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_read_cases.h"

namespace fluidity::grid {
namespace {

// Size-level-0 cells {4 i, 0, 0} for i = 0..count - 1: each alone in a block of its own.
std::vector<CellId> BuildCellsInBlocksOfTheirOwn(std::int32_t count) {
  std::vector<CellId> cells;
  cells.reserve(static_cast<std::size_t>(count));
  for (std::int32_t index = 0; index < count; ++index) {
    cells.push_back(ToCellId(SizeLevel{0}, {.x = 4 * index, .y = 0, .z = 0}));
  }
  return cells;
}

// 9 blocks into room for 8: no table is written and the buffers still build an 8-block grid.
TEST(ReadGrid, BuffersTooSmall_IsTooManyCellBlocks) {
  const GridFrame frame = ToUnitFrame(3);
  const std::vector<CellId> nine_cells = BuildCellsInBlocksOfTheirOwn(9);
  const std::expected<TestGrid, GridFailure> source = BuildTestGrid(ToInputs(frame, nine_cells));
  ASSERT_TRUE(source.has_value());
  ASSERT_EQ(source->grid.counts().cell_blocks, 9U);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(source->grid, buffers);

  ASSERT_FALSE(copy.has_value());
  EXPECT_EQ(copy.error().error, GridError::kTooManyCellBlocks);
  EXPECT_EQ(copy.error().cell.value, 0U);
  EXPECT_EQ(copy.error().other_cell.value, 0U);
  const std::span<const CellId> eight_cells = std::span<const CellId>(nine_cells).first(8);
  const std::expected<StaggeredGrid, GridFailure> built = BuildGrid(ToInputs(frame, eight_cells), buffers);
  ASSERT_TRUE(built.has_value());
  EXPECT_EQ(built->counts().cell_blocks, 8U);
}

// Buffers that were moved from have room 0 and no arrays: the build returns kBadInput for them and so does a read.
TEST(ReadGrid, MovedFromBuffers_IsBadInput) {
  const GridFrame frame = ToUnitFrame(3);
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(2);
  const std::expected<TestGrid, GridFailure> source = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(source.has_value());
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  const GridBuffers moved_to = std::move(buffers);

  const std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(source->grid, buffers);

  ASSERT_FALSE(copy.has_value());
  EXPECT_EQ(copy.error().error, GridError::kBadInput);
}

// A grid whose buffers were taken back holds no tables: version 0, so there is nothing to copy.
TEST(ReadGrid, GridWithoutTables_IsBadInput) {
  const GridFrame frame = ToUnitFrame(3);
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(2);
  std::expected<TestGrid, GridFailure> source = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(source.has_value());
  const GridBuffers taken_back = TakeBuffers(std::move(source->grid));
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(source->grid, buffers);

  ASSERT_FALSE(copy.has_value());
  EXPECT_EQ(copy.error().error, GridError::kBadInput);
  EXPECT_EQ(buffers.max_cell_blocks(), 8U);
}

// A grid of no cells has no blocks: the copy has its version and its zero counts, and no table is read.
TEST(ReadGrid, GridOfNoCells_CopiesVersionAndCounts) {
  const GridFrame frame = ToUnitFrame(3);
  const std::expected<TestGrid, GridFailure> source = BuildTestGrid(ToInputs(frame, {}));
  ASSERT_TRUE(source.has_value());
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(source->grid, buffers);

  ASSERT_TRUE(copy.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*copy, source->grid));
  EXPECT_EQ(copy->counts().cell_blocks, 0U);
}

// A grid of version 7 in a frame with an origin, a width and a number of size levels none of which is the default: the
// copy carries all of them.
TEST(ReadGrid, GridOfLaterVersionInOtherFrame_CopiesVersionAndFrame) {
  const GridFrame frame = {
      .origin = {.x = 1.5, .y = -2.0, .z = 0.25},
      .biggest_cell_width = 0.5,
      .size_level_count = 3,
  };
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(3);
  const std::expected<TestGrid, GridFailure> source =
      BuildTestGrid({.frame = frame, .active_cells = cells, .version = {.value = 7}});
  ASSERT_TRUE(source.has_value());
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(source->grid, buffers);

  ASSERT_TRUE(copy.has_value());
  EXPECT_EQ(copy->version().value, 7U);
  const GridFrame copied_frame = copy->tables(RunOn::kCpu).frame;
  EXPECT_EQ(copied_frame.origin.x, 1.5);
  EXPECT_EQ(copied_frame.origin.y, -2.0);
  EXPECT_EQ(copied_frame.origin.z, 0.25);
  EXPECT_EQ(copied_frame.biggest_cell_width, 0.5);
  EXPECT_EQ(copied_frame.size_level_count, 3U);
  EXPECT_TRUE(ComputeIsSameGrid(*copy, source->grid));
}

// A call with the namespace written out, as the step and the tools in other namespaces write it. A function declared
// only as a friend is not found that way, so this test fails to compile if ReadGrid has no declaration of its own.
TEST(ReadGrid, CallWithNamespace_CopiesTheGrid) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(2);
  const std::expected<TestGrid, GridFailure> source = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(source.has_value());
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::expected<StaggeredGrid, GridFailure> copy = fluidity::grid::ReadGrid(source->grid, buffers);

  ASSERT_TRUE(copy.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*copy, source->grid));
}

}  // namespace
}  // namespace fluidity::grid
