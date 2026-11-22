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
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_read_cases.h"

namespace fluidity::grid {
namespace {

// The number of random grids each test reads. The 200 random lists are too many for a debug build to read one by one,
// so the tests take this many at evenly spaced ranks of their size (FindRandomGridsSpreadBySize), from the grid with
// the most cells to the one with the fewest.
constexpr std::size_t kGridCount = 30;

// A room far above every random grid's block count (kMaxRandomGridCellBlocks).
constexpr std::uint32_t kBiggerRoom = 4096;

// What reading the grids in turn found.
struct ReadsInTurn {
  std::size_t failed_count = 0;
  std::size_t different_count = 0;  // Grids whose copy differs from the source in version, counts or used tables.
  std::size_t wrong_room_count = 0;  // Copies whose room is not the destination's.
};

// Reads each source grid into `buffers` in turn, compares the copy with its source and takes the buffers back, adding
// what it finds to `found`. Allocates nothing.
void ApplyReadsInTurn(const std::vector<TestGrid>& sources, GridBuffers& buffers, ReadsInTurn& found) {
  for (const TestGrid& source : sources) {
    const std::uint32_t room = buffers.max_cell_blocks();
    std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(source.grid, buffers);
    if (!copy.has_value()) {
      ++found.failed_count;
      continue;
    }
    const bool is_same_version = copy->version().value == source.grid.version().value;
    found.different_count += is_same_version && ComputeIsSameGrid(*copy, source.grid) ? 0 : 1;
    found.wrong_room_count += copy->max_cell_blocks() == room && copy->run_on() == RunOn::kCpu ? 0 : 1;
    buffers = TakeBuffers(std::move(*copy));
  }
}

// The random grids, each in a room of kMaxRandomGridCellBlocks, and the owner of the buffers they are read into.
struct SourcesAndBuffers {
  std::vector<TestGrid> sources;
  std::expected<TestGrid, GridFailure> destination_owner;
};

// Builds the spread of random grids and, as owner, an empty grid of room `room`, whose buffers a test takes out.
SourcesAndBuffers BuildSourcesAndOwner(std::uint32_t room) {
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  return {
      .sources = BuildReferenceGrids(grids),
      .destination_owner =
          BuildTestGrid(ToInputs(ToUnitFrame(kRandomGridSizeLevelCount), {}), {.max_cell_blocks = room}),
  };
}

TEST(ReadGrid, CpuGrid_CopiesIntoBiggerRoom) {
  SourcesAndBuffers set = BuildSourcesAndOwner(kBiggerRoom);
  ASSERT_EQ(set.sources.size(), kGridCount);
  GridBuffers buffers = TakeEmptyGridBuffers(set.destination_owner);

  ReadsInTurn found;
  ApplyReadsInTurn(set.sources, buffers, found);

  EXPECT_EQ(found.failed_count, 0U);
  EXPECT_EQ(found.different_count, 0U);
  EXPECT_EQ(found.wrong_room_count, 0U);
}

// The room is the source's own, 252 blocks: no part of the layout moves.
TEST(ReadGrid, CpuGrid_CopiesIntoEqualRoom) {
  SourcesAndBuffers set = BuildSourcesAndOwner(kMaxRandomGridCellBlocks);
  ASSERT_EQ(set.sources.size(), kGridCount);
  GridBuffers buffers = TakeEmptyGridBuffers(set.destination_owner);

  ReadsInTurn found;
  ApplyReadsInTurn(set.sources, buffers, found);

  EXPECT_EQ(found.failed_count, 0U);
  EXPECT_EQ(found.different_count, 0U);
  EXPECT_EQ(found.wrong_room_count, 0U);
}

// Buffers whose room is exactly the grid's block count: the last block of every table is the last entry of its array,
// and the smaller blocks start at the block count, not at the source's room of 252.
TEST(ReadGrid, RoomOfExactlyTheBlockCount_CopiesTheGrid) {
  const std::vector<TestGrid> sources = BuildReferenceGrids(FindRandomGridsSpreadBySize(kGridCount));
  ASSERT_EQ(sources.size(), kGridCount);
  std::size_t failed_count = 0;
  std::size_t different_count = 0;
  for (const TestGrid& source : sources) {
    SCOPED_TRACE(source.grid.counts().cell_blocks);
    std::expected<TestGrid, GridFailure> owner = BuildTestGrid(
        ToInputs(ToUnitFrame(kRandomGridSizeLevelCount), {}),
        {.max_cell_blocks = source.grid.counts().cell_blocks}
    );
    GridBuffers buffers = TakeEmptyGridBuffers(owner);
    const std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(source.grid, buffers);
    if (!copy.has_value()) {
      ++failed_count;
      continue;
    }
    different_count += ComputeIsSameGrid(*copy, source.grid) ? 0 : 1;
    EXPECT_EQ(copy->max_cell_blocks(), source.grid.counts().cell_blocks);
  }
  EXPECT_EQ(failed_count, 0U);
  EXPECT_EQ(different_count, 0U);
}

// The buffers held the biggest random grid; the smallest is read into them. Every entry of the used part of the copy
// comes from the smallest grid, and the entries beyond it are the biggest grid's, which the copy does not report.
TEST(ReadGrid, BuffersThatHeldAnotherGrid_HaveNoStaleEntryInTheUsedPart) {
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  const std::vector<TestGrid> sources = BuildReferenceGrids({grids.front(), grids.back()});
  ASSERT_EQ(sources.size(), 2U);
  const TestGrid& biggest = sources.front();
  const TestGrid& smallest = sources.back();
  ASSERT_GT(biggest.grid.counts().cell_blocks, smallest.grid.counts().cell_blocks);
  std::expected<TestGrid, GridFailure> owner =
      BuildTestGrid(ToInputs(ToUnitFrame(kRandomGridSizeLevelCount), {}), {.max_cell_blocks = kBiggerRoom});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  std::expected<StaggeredGrid, GridFailure> first_copy = ReadGrid(biggest.grid, buffers);
  ASSERT_TRUE(first_copy.has_value());
  buffers = TakeBuffers(std::move(*first_copy));
  const std::expected<StaggeredGrid, GridFailure> second_copy = ReadGrid(smallest.grid, buffers);

  ASSERT_TRUE(second_copy.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*second_copy, smallest.grid));
}

// A copy builds nothing, so the buffers keep their loan of builder buffers: they are taken back and build the next
// grid, and that grid equals the same grid built with memory of its own.
TEST(ReadGrid, CopiedGrid_BuffersBuildNextGridAfterTakeBuffers) {
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  const std::vector<TestGrid> sources = BuildReferenceGrids({grids.front(), grids.at(grids.size() / 2)});
  ASSERT_EQ(sources.size(), 2U);
  std::expected<TestGrid, GridFailure> owner =
      BuildTestGrid(ToInputs(ToUnitFrame(kRandomGridSizeLevelCount), {}), {.max_cell_blocks = kBiggerRoom});
  ASSERT_TRUE(owner.has_value());
  const GridBuilderBuffers& builder_buffers = *owner->builder_buffers;
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ASSERT_TRUE(ComputeHasLoanOf(buffers, builder_buffers));

  std::expected<StaggeredGrid, GridFailure> copy = ReadGrid(sources.front().grid, buffers);
  ASSERT_TRUE(copy.has_value());
  buffers = TakeBuffers(std::move(*copy));
  EXPECT_TRUE(ComputeHasLoanOf(buffers, builder_buffers));
  EXPECT_EQ(builder_buffers.lent_count(), 1U);

  const GridInputs next_inputs =
      ToInputs(ToUnitFrame(kRandomGridSizeLevelCount), BuildShuffledRandomGrids().at(grids.at(grids.size() / 2)));
  const std::expected<StaggeredGrid, GridFailure> next_grid = BuildGrid(next_inputs, buffers);
  ASSERT_TRUE(next_grid.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*next_grid, sources.back().grid));
}

// 30 random grids spread by size (FindRandomGridsSpreadBySize, from the grid with the most cells to the one with the
// fewest, the spread that staggered_grid.test.cc uses in place of all 200) are read in turn into buffers of 4,096
// blocks, compared with their sources and taken back. The sources are built first, outside the counted part.
TEST(ReadGrid, AnyGrid_AllocatesNothing) {
  SourcesAndBuffers set = BuildSourcesAndOwner(kBiggerRoom);
  ASSERT_EQ(set.sources.size(), kGridCount);
  GridBuffers buffers = TakeEmptyGridBuffers(set.destination_owner);
  ReadsInTurn found;

  const std::uint64_t allocation_count_before = ComputeAllocationCount();
  ApplyReadsInTurn(set.sources, buffers, found);
  const std::uint64_t allocation_count_after = ComputeAllocationCount();

  EXPECT_EQ(allocation_count_after - allocation_count_before, 0U);
  EXPECT_EQ(found.failed_count, 0U);
  EXPECT_EQ(found.different_count, 0U);
}

}  // namespace
}  // namespace fluidity::grid
